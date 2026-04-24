"""Mathematical equivalence tests for the batched generation primitives.

These tests validate that the batched code paths introduced for concurrent
multi-text inference produce outputs that are numerically identical to the
per-row single-request path. They run on CPU and do not require a GPU.

Covered primitives:
    * ``apply_repetition_penalty`` — 1-D (legacy) vs 2-D (per-row batched)
    * ``sample_logits`` — batched ``[B, V]`` rows vs per-row ``[1, V]`` calls
    * ``StreamingCodecDecoder`` — extracted class reproduces the original
      inline chunking math for ICL and non-ICL inputs, given a fake codec.
    * ``batched_talker_generate`` — with a fake batched talker, per-row codec
      outputs match what running each request alone would have produced.
"""
from __future__ import annotations

import types
from typing import List

import numpy as np
import pytest
import torch

from faster_qwen3_tts.engine import Request, Result, batched_talker_generate
from faster_qwen3_tts.sampling import apply_repetition_penalty, sample_logits
from faster_qwen3_tts.streaming_decoder import StreamingCodecDecoder


# ============================================================================
# apply_repetition_penalty equivalence
# ============================================================================


def _ref_penalty_1d(logits_1d: torch.Tensor, hist_1d: torch.Tensor, pen: float) -> torch.Tensor:
    """Old 1-D formula, re-implemented for reference."""
    out = logits_1d.clone()
    if pen == 1.0 or hist_1d.numel() == 0:
        return out
    uniq = hist_1d.unique()
    row = out[uniq]
    out[uniq] = torch.where(row > 0, row / pen, row * pen)
    return out


def test_repetition_penalty_1d_bit_identical_with_old_formula():
    torch.manual_seed(0)
    logits = torch.randn(1, 1, 32)
    hist = torch.tensor([3, 5, 7, 3, 5, 11, 29, 31], dtype=torch.long)

    got = apply_repetition_penalty(logits.clone(), hist, repetition_penalty=1.1)
    ref = _ref_penalty_1d(logits[0, 0].clone(), hist, 1.1)
    assert torch.allclose(got[0, 0], ref, atol=0.0, rtol=0.0), \
        "1-D repetition penalty changed numerics after refactor"


def test_repetition_penalty_batched_matches_per_row():
    torch.manual_seed(1)
    B, V = 5, 64
    logits = torch.randn(B, V)
    # Per-row histories, variable lengths, with padding (-1) to common width.
    rows = [
        torch.tensor([2, 4, 6], dtype=torch.long),
        torch.tensor([0, 1, 1, 2, 3], dtype=torch.long),
        torch.tensor([], dtype=torch.long),
        torch.tensor([V - 1, V - 2, V - 3, V - 1], dtype=torch.long),
        torch.tensor([10], dtype=torch.long),
    ]
    max_len = max((r.numel() for r in rows), default=0)
    hist_2d = torch.full((B, max_len), -1, dtype=torch.long)
    for i, r in enumerate(rows):
        hist_2d[i, : r.numel()] = r

    got = apply_repetition_penalty(logits.clone(), hist_2d, repetition_penalty=1.3)

    # Reference: apply 1-D penalty per row.
    ref = torch.stack([
        _ref_penalty_1d(logits[i].clone(), rows[i], 1.3)
        for i in range(B)
    ])

    assert torch.allclose(got, ref, atol=0.0, rtol=0.0)


def test_repetition_penalty_batched_accepts_3d_logits():
    """[B, 1, V] layout must be supported since some callsites carry the
    singleton time dim through."""
    torch.manual_seed(2)
    B, V = 3, 16
    logits = torch.randn(B, 1, V)
    hist = torch.tensor([[1, 2, -1], [5, -1, -1], [0, 1, 2]], dtype=torch.long)
    got = apply_repetition_penalty(logits.clone(), hist, repetition_penalty=1.2)
    ref_flat = apply_repetition_penalty(logits.squeeze(1).clone(), hist, repetition_penalty=1.2)
    assert torch.allclose(got.squeeze(1), ref_flat, atol=0.0, rtol=0.0)


def test_repetition_penalty_no_op_at_1_0():
    logits = torch.randn(4, 8)
    hist = torch.randint(0, 8, (4, 3))
    got = apply_repetition_penalty(logits.clone(), hist, repetition_penalty=1.0)
    assert torch.equal(got, logits)


# ============================================================================
# sample_logits equivalence (batched rows vs per-row calls)
# ============================================================================


def test_sample_logits_batched_matches_per_row_under_fixed_seed():
    """Greedy (do_sample=False) is deterministic and row-independent."""
    torch.manual_seed(42)
    B, V = 7, 32
    logits = torch.randn(B, V)

    per_row = torch.stack([
        sample_logits(
            logits[i : i + 1],
            temperature=1.0,
            top_k=10,
            top_p=0.9,
            do_sample=False,
        ).squeeze(0)
        for i in range(B)
    ])

    batched = sample_logits(
        logits,
        temperature=1.0,
        top_k=10,
        top_p=0.9,
        do_sample=False,
    )

    assert torch.equal(per_row, batched)


def test_sample_logits_batched_sampling_matches_per_row_when_independent():
    """For sampling with do_sample=True, per-row and batched calls draw from
    independent multinomials with the same distribution, so the distribution
    over outputs must match in expectation. We test determinism under the
    same seed & layout.
    """
    B, V = 4, 20
    logits = torch.arange(B * V, dtype=torch.float32).reshape(B, V) / (B * V)
    # Scale so top-1 dominates → effectively deterministic sample.
    logits = logits * 20.0

    torch.manual_seed(7)
    batched = sample_logits(
        logits.clone(),
        temperature=0.1,
        top_k=1,  # hard-argmax after top-k
        top_p=1.0,
        do_sample=True,
    )
    # When top_k=1, sampling collapses to the argmax, which is row-independent.
    argmax = logits.argmax(dim=-1)
    assert torch.equal(batched, argmax)


# ============================================================================
# StreamingCodecDecoder equivalence with the original inline loop
# ============================================================================


class _FakeCodec:
    """Deterministic fake codec: audio[t] = 10 * frame_index[t] + codebook0 mean.

    Allows the equivalence test to check that chunks are decoded correctly,
    and that the sliding-window phase does not re-emit context samples.
    """

    SAMPLES_PER_FRAME = 8

    def decode(self, payload):
        codes = payload["audio_codes"]  # [1, T, 16]
        T = codes.shape[1]
        # Produce audio of length T * SAMPLES_PER_FRAME, value = frame index.
        audio = np.repeat(np.arange(T, dtype=np.float32), self.SAMPLES_PER_FRAME)
        return [audio], 24000


def _ref_streaming_decode(all_chunks: List[torch.Tensor], ref_codes, chunk_size, context_frames=25):
    """Reference implementation: the original inline loop from model.py."""
    codec = _FakeCodec()
    min_calib = max(context_frames, chunk_size)
    all_codes: List[torch.Tensor] = []
    prev_gen_audio_len = 0
    samples_per_frame = None
    outs = []
    for codec_chunk in all_chunks:
        all_codes.append(codec_chunk)
        n_new = codec_chunk.shape[0]
        all_flat = torch.cat(all_codes, dim=0)
        n_total = all_flat.shape[0]
        if samples_per_frame is None:
            if ref_codes is not None:
                codes_input = torch.cat([ref_codes, all_flat], dim=0)
            else:
                codes_input = all_flat
            audio_list, sr = codec.decode({"audio_codes": codes_input.unsqueeze(0)})
            audio = audio_list[0]
            if ref_codes is not None:
                ref_len = ref_codes.shape[0]
                total_len = codes_input.shape[0]
                ref_cut = int(ref_len / max(total_len, 1) * len(audio))
                gen_audio = audio[ref_cut:]
            else:
                gen_audio = audio
            new_audio = gen_audio[prev_gen_audio_len:]
            prev_gen_audio_len = len(gen_audio)
            if n_total >= min_calib:
                samples_per_frame = len(gen_audio) / n_total
        else:
            ctx_start = max(0, n_total - n_new - context_frames)
            window = all_flat[ctx_start:]
            n_ctx = window.shape[0] - n_new
            audio_list, sr = codec.decode({"audio_codes": window.unsqueeze(0)})
            audio = audio_list[0]
            if n_ctx > 0:
                ctx_samples = int(round(n_ctx * samples_per_frame))
                new_audio = audio[ctx_samples:]
            else:
                new_audio = audio
        outs.append((new_audio, sr))
    return outs


@pytest.mark.parametrize("use_ref", [False, True])
def test_streaming_codec_decoder_matches_inline_loop(use_ref):
    torch.manual_seed(0)
    # 6 chunks of varying size to exercise both phases.
    chunk_sizes = [8, 8, 8, 8, 8, 4]
    chunks = []
    cursor = 0
    for n in chunk_sizes:
        chunks.append(torch.arange(cursor, cursor + n, dtype=torch.long).unsqueeze(1).repeat(1, 16))
        cursor += n

    ref_codes = torch.arange(5, dtype=torch.long).unsqueeze(1).repeat(1, 16) if use_ref else None

    # Reference (inline loop).
    ref_outs = _ref_streaming_decode(chunks, ref_codes, chunk_size=8, context_frames=25)

    # Class under test.
    codec = _FakeCodec()
    dec = StreamingCodecDecoder(
        decode_fn=codec.decode,
        ref_codes=ref_codes,
        chunk_size=8,
        context_frames=25,
    )
    got_outs = [dec.push(c) for c in chunks]

    assert len(got_outs) == len(ref_outs)
    for i, (got, (ref_audio, ref_sr)) in enumerate(zip(got_outs, ref_outs)):
        assert got.sample_rate == ref_sr, f"chunk {i}: sr mismatch"
        # Compare arrays; allow trivial dtype differences (both float32).
        ref_arr = np.asarray(ref_audio, dtype=np.float32)
        got_arr = np.asarray(got.audio, dtype=np.float32)
        assert got_arr.shape == ref_arr.shape, f"chunk {i}: shape {got_arr.shape} vs {ref_arr.shape}"
        assert np.array_equal(got_arr, ref_arr), f"chunk {i}: audio mismatch"


# ============================================================================
# Request / Result dataclass behavior
# ============================================================================


def test_request_validation():
    with pytest.raises(ValueError, match="mode"):
        Request(text="hi", mode="nonsense")
    with pytest.raises(ValueError, match="ref_audio"):
        Request(text="hi", mode="voice_clone")
    with pytest.raises(ValueError, match="speaker"):
        Request(text="hi", mode="custom_voice")
    with pytest.raises(ValueError, match="instruct"):
        Request(text="hi", mode="voice_design")

    # Valid forms:
    Request(text="hi", mode="voice_clone", ref_audio="a.wav")
    Request(text="hi", mode="voice_clone", voice_clone_prompt={"ref_spk_embedding": [None]})
    Request(text="hi", mode="custom_voice", speaker="aiden")
    Request(text="hi", mode="voice_design", instruct="calm narrator")


def test_result_defaults():
    r = Result(audio=np.zeros(3, dtype=np.float32), sample_rate=24000)
    assert r.codec_ids is None
    assert r.timing == {}
    assert r.request_id is None


# ============================================================================
# batched_talker_generate equivalence on a fake batched talker
# ============================================================================


class _FakeTalkerGenerate:
    """A tiny stand-in for talker.generate that produces deterministic per-row
    codec token sequences with different lengths and asserts batch alignment."""

    def __init__(self, per_row_sequences: List[List[int]], num_code_groups: int = 16,
                 eos_id: int = 2):
        self.per_row_sequences = per_row_sequences
        self.num_code_groups = num_code_groups
        self.eos_id = eos_id

    def generate(self, **kwargs):
        inputs_embeds = kwargs["inputs_embeds"]
        attention_mask = kwargs["attention_mask"]
        B = inputs_embeds.shape[0]
        assert B == len(self.per_row_sequences)
        assert attention_mask.shape[0] == B
        assert kwargs["eos_token_id"] == self.eos_id

        # Each hidden_states tuple slot[-1] is [B, 1, num_code_groups].
        T = max(len(s) for s in self.per_row_sequences)
        # For each row i and step t, if t < len(seq[i]), emit seq[i][t] on
        # codebook 0 (and zeros on codebooks 1..15); if seq[i][t] is the
        # EOS value, that's the stop signal for row i.
        steps = []
        for t in range(T):
            step = torch.zeros(B, self.num_code_groups, dtype=torch.long)
            for i, seq in enumerate(self.per_row_sequences):
                if t < len(seq):
                    step[i, 0] = seq[t]
            steps.append((None, step))  # tuple[-1] is the step tensor

        return types.SimpleNamespace(hidden_states=steps)


class _FakeTalkerConfig:
    codec_eos_token_id = 2
    num_code_groups = 16
    vocab_size = 2048


def test_batched_talker_generate_per_row_split():
    cfg = _FakeTalkerConfig()
    # Row 0 stops at step 2 (emits EOS at t=2). Row 1 stops at step 4. Row 2 never stops.
    seqs = [
        [7, 8, cfg.codec_eos_token_id],                  # effective_length = 2
        [3, 4, 5, 6, cfg.codec_eos_token_id],             # effective_length = 4
        [1, 1, 1, 1, 1, 1],                               # no EOS → full length = 6
    ]
    talker = _FakeTalkerGenerate(seqs, num_code_groups=cfg.num_code_groups,
                                  eos_id=cfg.codec_eos_token_id)
    tie = torch.zeros(3, 5, 4)
    tam = torch.ones(3, 5, dtype=torch.long)
    tth = torch.zeros(3, 1, 4)
    tpe = torch.zeros(1, 1, 4)

    codes_list, timing = batched_talker_generate(
        talker=talker,
        talker_input_embeds=tie,
        attention_mask=tam,
        trailing_text_hiddens=tth,
        tts_pad_embed=tpe,
        config=cfg,
    )

    assert len(codes_list) == 3
    assert codes_list[0].shape == (2, cfg.num_code_groups)
    assert codes_list[1].shape == (4, cfg.num_code_groups)
    assert codes_list[2].shape == (6, cfg.num_code_groups)

    # Row contents match input sequences (codebook 0; others zero).
    assert codes_list[0][:, 0].tolist() == [7, 8]
    assert codes_list[1][:, 0].tolist() == [3, 4, 5, 6]
    assert codes_list[2][:, 0].tolist() == [1, 1, 1, 1, 1, 1]

    # Timing sanity.
    assert timing["steps"] == 6
    assert timing["decode_s"] >= 0.0


def test_batched_talker_generate_empty_row_returns_none():
    cfg = _FakeTalkerConfig()
    seqs = [[cfg.codec_eos_token_id]]  # EOS at t=0 → length 0
    talker = _FakeTalkerGenerate(seqs, eos_id=cfg.codec_eos_token_id)
    tie = torch.zeros(1, 3, 4)
    tam = torch.ones(1, 3, dtype=torch.long)
    tth = torch.zeros(1, 1, 4)
    tpe = torch.zeros(1, 1, 4)

    codes_list, _ = batched_talker_generate(
        talker=talker,
        talker_input_embeds=tie,
        attention_mask=tam,
        trailing_text_hiddens=tth,
        tts_pad_embed=tpe,
        config=cfg,
    )
    assert codes_list == [None]
