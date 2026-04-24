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
