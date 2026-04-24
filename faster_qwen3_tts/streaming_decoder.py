"""Per-request streaming codec decoder with hybrid accumulated/sliding-window decode.

Encapsulates the bookkeeping previously inlined inside
``FasterQwen3TTS.generate_voice_clone_streaming`` and its siblings:

* Phase 1 (accumulated): decode the full set of codec frames (optionally prepended
  with reference audio codes for ICL) until we have enough frames to calibrate
  ``samples_per_frame``.
* Phase 2 (sliding window): decode only a bounded window (``context_frames``
  left-context + new frames) and drop the context slice from the output.

Extracting this logic lets the batched engine reuse it per-request without
duplicating the three identical copies that existed in ``model.py``.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import torch


def _to_numpy(audio) -> np.ndarray:
    if hasattr(audio, "cpu"):
        return audio.flatten().cpu().numpy()
    if hasattr(audio, "flatten"):
        return audio.flatten()
    return audio


class StreamingCodecDecoder:
    """Incremental audio decoder over a growing sequence of codec frames.

    The decoder is fed codec chunks via :meth:`push` and yields the
    newly-generated audio samples for each chunk.

    Args:
        decode_fn: Callable taking a ``[1, T, 16]`` tensor of codec ids and
            returning ``(audio_list, sample_rate)`` with ``audio_list[0]``
            shaped ``[samples]`` (tensor or numpy array).
        ref_codes: Optional ``[R, 16]`` tensor of reference audio codec ids to
            prepend during phase 1 (ICL mode). Pass ``None`` for non-ICL.
        chunk_size: Number of new codec frames per push. Used only to pick a
            calibration threshold.
        context_frames: Left-context window size (in codec frames) for phase 2
            sliding-window decode.
    """

    def __init__(
        self,
        decode_fn: Callable,
        ref_codes: Optional[torch.Tensor] = None,
        chunk_size: int = 12,
        context_frames: int = 25,
    ) -> None:
        self.decode_fn = decode_fn
        self.ref_codes = ref_codes
        self.context_frames = int(context_frames)
        self.min_calibration_frames = max(self.context_frames, int(chunk_size))

        self._all_codes: List[torch.Tensor] = []
        self._prev_gen_audio_len = 0
        self._samples_per_frame: Optional[float] = None
        self._sample_rate: Optional[int] = None

    @property
    def sample_rate(self) -> Optional[int]:
        return self._sample_rate

    @property
    def samples_per_frame(self) -> Optional[float]:
        return self._samples_per_frame

    def push(self, codec_chunk: torch.Tensor) -> "DecodedChunk":
        """Decode audio for a new codec chunk.

        Args:
            codec_chunk: ``[n_new, 16]`` tensor of newly generated codec frames.

        Returns:
            DecodedChunk with ``audio`` (numpy float32) and ``sample_rate``.
        """
        self._all_codes.append(codec_chunk)
        n_new = int(codec_chunk.shape[0])
        all_flat = torch.cat(self._all_codes, dim=0)
        n_total = int(all_flat.shape[0])

        if self._samples_per_frame is None:
            new_audio, sr = self._phase1_accumulated(all_flat, n_total)
        else:
            new_audio, sr = self._phase2_sliding(all_flat, n_total, n_new)

        self._sample_rate = sr
        return DecodedChunk(audio=new_audio, sample_rate=sr)

    # ---- internals -------------------------------------------------------

    def _phase1_accumulated(self, all_flat: torch.Tensor, n_total: int):
        if self.ref_codes is not None:
            codes_input = torch.cat([self.ref_codes.to(all_flat.device), all_flat], dim=0)
        else:
            codes_input = all_flat

        audio_list, sr = self.decode_fn({"audio_codes": codes_input.unsqueeze(0)})
        audio = _to_numpy(audio_list[0])

        if self.ref_codes is not None:
            ref_len = int(self.ref_codes.shape[0])
            total_len = int(codes_input.shape[0])
            ref_audio_cut = int(ref_len / max(total_len, 1) * len(audio))
            gen_audio = audio[ref_audio_cut:]
        else:
            gen_audio = audio

        new_audio = gen_audio[self._prev_gen_audio_len :]
        self._prev_gen_audio_len = len(gen_audio)

        if n_total >= self.min_calibration_frames:
            self._samples_per_frame = len(gen_audio) / n_total

        return new_audio, sr

    def _phase2_sliding(self, all_flat: torch.Tensor, n_total: int, n_new: int):
        ctx_start = max(0, n_total - n_new - self.context_frames)
        window = all_flat[ctx_start:]
        n_ctx = int(window.shape[0] - n_new)

        audio_list, sr = self.decode_fn({"audio_codes": window.unsqueeze(0)})
        audio = _to_numpy(audio_list[0])

        if n_ctx > 0:
            ctx_samples = int(round(n_ctx * self._samples_per_frame))
            new_audio = audio[ctx_samples:]
        else:
            new_audio = audio

        return new_audio, sr


class DecodedChunk:
    """Lightweight struct returned by :meth:`StreamingCodecDecoder.push`."""

    __slots__ = ("audio", "sample_rate")

    def __init__(self, audio: np.ndarray, sample_rate: Optional[int]) -> None:
        self.audio = audio
        self.sample_rate = sample_rate

    def __iter__(self):
        # Allow tuple-unpacking for backwards compat.
        yield self.audio
        yield self.sample_rate
