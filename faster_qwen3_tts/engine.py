"""Concurrent multi-request generation engine.

This module implements a vLLM-style batched generation API for FasterQwen3TTS.
Multiple TTS requests are prepared, left-padded into a common batch, and
decoded in a single call using the talker's batched forward (dynamic-cache
path, identical in numerics to ``parity_mode=True``).

**Scope for this PR (Phase 0 + Phase 1 MVP):**

* ``Request`` / ``Result`` dataclasses describe the public API surface.
* ``generate_batch`` drives the talker with batched inputs and returns one
  ``Result`` per submitted ``Request``.
* The underlying decode uses ``talker.generate`` (HF) with an attention-mask
  batch of shape ``[B, L]``. This is mathematically equivalent to running
  each request independently modulo sampling RNG state sharing across the
  batch call. See ``tests/test_engine_equivalence.py`` for per-step
  validation.

**Deferred (see plan's Phase 2/3):**

* Batched ``PredictorGraph`` / ``TalkerGraph`` CUDA-graph capture (the
  single-request path still uses CUDA graphs; only the batched path runs
  through the dynamic-cache branch today).
* Continuous batching / long-lived engine with streaming handles. The
  ``generate_batch`` call is currently "offline": submit N, get N back.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


# =============================================================================
# Public dataclasses
# =============================================================================

VoiceMode = str  # one of {"voice_clone", "custom_voice", "voice_design"}


@dataclass
class Request:
    """A single TTS generation request to be run concurrently with others.

    Fields mirror the keyword arguments of ``FasterQwen3TTS.generate_voice_clone``
    and siblings. The ``mode`` field selects which generation head is used; all
    requests in a single ``generate_batch`` call must share the same mode.

    Attributes:
        text: Text to synthesize.
        language: Target language (e.g. ``"English"``, ``"Chinese"``, ``"Auto"``).
        mode: One of ``"voice_clone"``, ``"custom_voice"``, ``"voice_design"``.
        ref_audio: Reference audio path for voice cloning. Required for
            ``voice_clone`` unless ``voice_clone_prompt`` is supplied.
        ref_text: Transcription of ``ref_audio`` (ICL mode).
        speaker: Speaker id for ``custom_voice`` mode.
        instruct: Optional instruction text. Required for ``voice_design``.
        voice_clone_prompt: Precomputed voice-clone prompt (see
            ``FasterQwen3TTS.generate_voice_clone``).
        xvec_only: Use only the x-vector (speaker embedding) instead of the
            full ICL prompt in voice_clone mode.
        append_silence: Append a small silence to the reference audio before
            tokenizing (voice_clone mode).
        non_streaming_mode: See ``FasterQwen3TTS.generate_voice_clone``.
            ``None`` uses the method-specific default.
        max_new_tokens / min_new_tokens / temperature / top_k / top_p /
        do_sample / repetition_penalty: Sampling hyperparameters. In the
        batched path, sampling params are applied per batch element by HF
        generate; pass the same values across requests to keep the batched
        call well-defined (HF does not support per-row sampling params in
        ``generate``).
        request_id: Optional caller-supplied id echoed back in the result.
    """

    text: str
    language: str = "English"
    mode: VoiceMode = "voice_clone"

    # Voice-clone inputs
    ref_audio: Optional[Union[str, Path]] = None
    ref_text: str = ""
    xvec_only: bool = False
    append_silence: bool = True
    voice_clone_prompt: Optional[Union[Dict[str, Any], List[Any]]] = None

    # Custom voice / voice design inputs
    speaker: Optional[str] = None
    instruct: Optional[str] = None

    # Sampling / length
    max_new_tokens: int = 2048
    min_new_tokens: int = 2
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    do_sample: bool = True
    repetition_penalty: float = 1.05
    non_streaming_mode: Optional[bool] = None

    request_id: Optional[str] = None

    def __post_init__(self) -> None:
        valid_modes = ("voice_clone", "custom_voice", "voice_design")
        if self.mode not in valid_modes:
            raise ValueError(
                f"Request.mode must be one of {valid_modes}, got {self.mode!r}"
            )
        if self.mode == "voice_clone":
            if self.ref_audio is None and self.voice_clone_prompt is None:
                raise ValueError(
                    "voice_clone requests require either ref_audio or voice_clone_prompt"
                )
        elif self.mode == "custom_voice":
            if not self.speaker:
                raise ValueError("custom_voice requests require speaker")
        elif self.mode == "voice_design":
            if not self.instruct:
                raise ValueError("voice_design requests require instruct")


@dataclass
class Result:
    """Output for one ``Request``.

    Attributes:
        audio: 1-D float32 numpy array of the generated waveform.
        sample_rate: Waveform sample rate in Hz.
        codec_ids: ``[T, 16]`` long tensor of generated codec ids (before
            codec decoding). Useful for inspection and for streaming audio
            externally. May be ``None`` if generation produced no tokens.
        timing: Dict with at least ``prefill_ms``, ``decode_s``, ``steps``,
            ``ms_per_step``. Batched runs attribute decode time in aggregate
            (same ``decode_s`` on every result) since all rows advance in
            lockstep.
        request_id: Echo of the ``Request.request_id`` for dispatch.
    """

    audio: np.ndarray
    sample_rate: int
    codec_ids: Optional[torch.Tensor] = None
    timing: Dict[str, float] = field(default_factory=dict)
    request_id: Optional[str] = None


# =============================================================================
# Batched generation primitive (dynamic-cache / HF generate path)
# =============================================================================


@torch.inference_mode()
def batched_talker_generate(
    *,
    talker,
    talker_input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    trailing_text_hiddens: torch.Tensor,
    tts_pad_embed: torch.Tensor,
    config,
    max_new_tokens: int = 2048,
    min_new_tokens: int = 2,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    do_sample: bool = True,
    repetition_penalty: float = 1.05,
) -> Tuple[List[Optional[torch.Tensor]], Dict[str, float]]:
    """Run ``talker.generate`` on a batch and split results per row.

    Inputs:
        talker_input_embeds: ``[B, L, H]`` left-padded batch.
        attention_mask: ``[B, L]`` with zeros in the left-padding region.
        trailing_text_hiddens: per-request trailing text embeddings.
        tts_pad_embed: pad embedding.

    Returns:
        (codes_list, timing) where ``codes_list[i]`` is ``[T_i, 16]`` or
        ``None`` if row ``i`` produced no codec tokens.
    """
    import time as _time

    eos_id = config.codec_eos_token_id
    vocab_size = config.vocab_size
    suppress_start = max(0, vocab_size - 1024)
    suppress_tokens = [i for i in range(suppress_start, vocab_size) if i != eos_id]

    t0 = _time.time()
    talker_result = talker.generate(
        inputs_embeds=talker_input_embeds,
        attention_mask=attention_mask,
        trailing_text_hidden=trailing_text_hiddens,
        tts_pad_embed=tts_pad_embed,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        eos_token_id=eos_id,
        suppress_tokens=suppress_tokens,
        subtalker_dosample=do_sample,
        subtalker_top_k=top_k,
        subtalker_top_p=top_p,
        subtalker_temperature=temperature,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )

    # Stack the per-step codec outputs: hidden_states[step][-1] is [B, 1, 16]
    # (see fast_generate parity branch).
    talker_codes = torch.stack(
        [hid[-1] for hid in talker_result.hidden_states if hid[-1] is not None],
        dim=1,
    )  # [B, T, 16]

    first_codebook = talker_codes[:, :, 0]  # [B, T]
    is_stop = first_codebook == eos_id
    has_stop = is_stop.any(dim=1)
    stop_idx = torch.argmax(is_stop.int(), dim=1)
    effective_lengths = torch.where(has_stop, stop_idx, talker_codes.shape[1])

    codes_list: List[Optional[torch.Tensor]] = []
    for i, length in enumerate(effective_lengths.tolist()):
        if length <= 0:
            codes_list.append(None)
        else:
            codes_list.append(talker_codes[i, :length, :])

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = _time.time() - t0
    steps = int(talker_codes.shape[1])
    timing = {
        "prefill_ms": 0.0,
        "decode_s": total_time,
        "steps": steps,
        "ms_per_step": (total_time / steps * 1000) if steps > 0 else 0.0,
        "steps_per_s": (steps / total_time) if total_time > 0 else 0.0,
    }
    return codes_list, timing


# =============================================================================
# TTSEngine — long-lived scheduler with submit() / RequestHandle streaming API
# =============================================================================


class RequestHandle:
    """Result handle returned by :meth:`TTSEngine.submit`.

    Iterating over the handle yields ``(audio_chunk, sample_rate, timing)``
    tuples, mirroring the existing streaming APIs. A final sentinel tuple
    ``(None, sample_rate, timing)`` is *not* emitted; iteration simply stops
    once the request is complete.

    The handle is also a future-like: :meth:`result` waits for the request to
    finish and returns the full :class:`Result`.
    """

    def __init__(self, request: "Request") -> None:
        self.request = request
        self._result: Optional["Result"] = None
        # A list of pre-chunked audio segments. The Phase-2 lockstep-cohort
        # engine fills this in one shot after the batched decode; the
        # continuous-batching engine (future) would push chunks here as they
        # become available.
        self._chunks: List[Tuple[np.ndarray, int, Dict[str, float]]] = []
        self._done: bool = False
        self._error: Optional[BaseException] = None

    # Internal API used by TTSEngine ---------------------------------------

    def _push_chunk(
        self, audio: np.ndarray, sample_rate: int, timing: Dict[str, float]
    ) -> None:
        self._chunks.append((audio, sample_rate, timing))

    def _set_result(self, result: "Result") -> None:
        self._result = result
        self._done = True

    def _set_error(self, err: BaseException) -> None:
        self._error = err
        self._done = True

    # Public API -----------------------------------------------------------

    @property
    def done(self) -> bool:
        return self._done

    def __iter__(self):
        for chunk in self._chunks:
            yield chunk
        if self._error is not None:
            raise self._error

    def result(self) -> "Result":
        """Return the final :class:`Result` for this request.

        Raises whatever exception the engine recorded for this request, if any.
        """
        if not self._done:
            raise RuntimeError(
                "RequestHandle.result() called before the engine finished "
                "the request. In the lockstep-cohort engine, result() is "
                "only valid after TTSEngine.flush() or .run_pending() returns."
            )
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


class TTSEngine:
    """Concurrent request scheduler around a :class:`FasterQwen3TTS` instance.

    The engine runs in "lockstep cohort" mode (the plan's Phase 2 option (b)):
    callers submit requests, which are queued; :meth:`run_pending` pops up to
    ``max_batch_size`` same-mode requests and executes them together through
    :meth:`FasterQwen3TTS.generate_batch`. Results are dispatched back to
    their ``RequestHandle``.

    Continuous batching (option (a) in the plan — admitting new requests
    mid-decode) is a future extension, which will be built on top of
    :class:`BatchedStaticCache`. The public submit/handle API is designed to
    remain unchanged when that switch happens.

    Args:
        model: A ready :class:`FasterQwen3TTS` instance.
        max_batch_size: Maximum number of requests per cohort. Defaults to 4.

    Example:
        >>> engine = TTSEngine(model, max_batch_size=4)
        >>> h1 = engine.submit(Request(text="hello", mode="custom_voice", speaker="A"))
        >>> h2 = engine.submit(Request(text="world", mode="custom_voice", speaker="B"))
        >>> engine.run_pending()
        >>> print(h1.result().audio.shape, h2.result().audio.shape)
    """

    def __init__(self, model, *, max_batch_size: int = 4) -> None:
        if max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
        self.model = model
        self.max_batch_size = int(max_batch_size)
        self._pending: List[Tuple["Request", RequestHandle]] = []

    # ------------------------------------------------------------------
    # Submission / queue management
    # ------------------------------------------------------------------

    def submit(self, request: "Request") -> RequestHandle:
        """Enqueue a request. Returns a handle; actual work is triggered by
        :meth:`run_pending` (or :meth:`flush`)."""
        if not isinstance(request, Request):
            raise TypeError(
                f"TTSEngine.submit expects a Request, got {type(request).__name__}"
            )
        handle = RequestHandle(request)
        self._pending.append((request, handle))
        return handle

    @property
    def pending(self) -> int:
        return len(self._pending)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_pending(self) -> int:
        """Run one cohort of pending requests through the batched model.

        Groups the queue by ``mode`` (since ``generate_batch`` requires a
        single mode) and dispatches the *first* mode group, up to
        ``max_batch_size`` requests. Returns the number of requests that were
        executed (0 if the queue is empty).

        Call repeatedly — or use :meth:`flush` — to drain the queue.
        """
        if not self._pending:
            return 0

        # Peel off the first same-mode cohort at the head of the queue, up to
        # max_batch_size.
        head_mode = self._pending[0][0].mode
        cohort: List[Tuple["Request", RequestHandle]] = []
        remaining: List[Tuple["Request", RequestHandle]] = []
        for req, handle in self._pending:
            if len(cohort) < self.max_batch_size and req.mode == head_mode:
                cohort.append((req, handle))
            else:
                remaining.append((req, handle))
        self._pending = remaining

        reqs = [r for r, _ in cohort]
        handles = [h for _, h in cohort]

        try:
            results = self.model.generate_batch(reqs)
        except BaseException as err:  # noqa: BLE001 - propagate to handles
            for h in handles:
                h._set_error(err)
            raise

        for handle, res in zip(handles, results):
            # The lockstep engine delivers a single "chunk" per request: the
            # full audio. Future continuous-batching mode will emit multiple
            # chunks here, but the handle contract is already chunk-oriented.
            handle._push_chunk(res.audio, res.sample_rate, dict(res.timing))
            handle._set_result(res)
        return len(cohort)

    def flush(self) -> int:
        """Drain the queue by repeatedly calling :meth:`run_pending`.

        Returns total number of requests executed.
        """
        total = 0
        while self._pending:
            executed = self.run_pending()
            if executed == 0:
                # Defensive: avoid infinite loop if a bug leaves items pending.
                break
            total += executed
        return total
