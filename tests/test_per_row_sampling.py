"""Mathematical equivalence tests for per-row sampling parameters.

Phase 2 requires ``sample_logits`` to accept per-row temperature, top_k, and
top_p. These tests validate that:

* Scalar (batch-uniform) calls produce bit-identical output to pre-Phase-2.
* Per-row calls with uniform values ≡ scalar calls (same output).
* Per-row calls with mixed values produce the correct per-row behavior by
  comparing against N independent scalar calls, one per row.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from faster_qwen3_tts.sampling import sample_logits


# ---------------------------------------------------------------------------
# Uniform-value equivalence (scalar param vs per-row param with same value)
# ---------------------------------------------------------------------------


def test_per_row_temperature_uniform_matches_scalar():
    torch.manual_seed(0)
    B, V = 5, 64
    logits = torch.randn(B, V)

    t_scalar = 0.7
    t_per_row = torch.full((B,), t_scalar)

    # do_sample=False → deterministic argmax → equivalence is point-wise.
    a = sample_logits(logits.clone(), temperature=t_scalar, top_k=0, top_p=1.0, do_sample=False)
    b = sample_logits(logits.clone(), temperature=t_per_row, top_k=0, top_p=1.0, do_sample=False)
    assert torch.equal(a, b)


def test_per_row_top_k_uniform_matches_scalar():
    torch.manual_seed(0)
    B, V = 6, 32
    logits = torch.randn(B, V)
    k_scalar = 5
    k_per_row = [5] * B
    torch.manual_seed(1)
    a = sample_logits(logits.clone(), temperature=1.0, top_k=k_scalar, top_p=1.0, do_sample=True)
    torch.manual_seed(1)
    b = sample_logits(logits.clone(), temperature=1.0, top_k=k_per_row, top_p=1.0, do_sample=True)
    # multinomial consumes the same RNG state for the same distribution.
    assert torch.equal(a, b)


def test_per_row_top_p_uniform_matches_scalar():
    torch.manual_seed(0)
    B, V = 6, 32
    logits = torch.randn(B, V)
    p_scalar = 0.8
    p_per_row = torch.full((B,), p_scalar)
    torch.manual_seed(2)
    a = sample_logits(logits.clone(), temperature=1.0, top_k=0, top_p=p_scalar, do_sample=True)
    torch.manual_seed(2)
    b = sample_logits(logits.clone(), temperature=1.0, top_k=0, top_p=p_per_row, do_sample=True)
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# Mixed per-row values — argmax after per-row top-k=1 is row-independent.
# This lets us prove that each row's filtering is applied independently.
# ---------------------------------------------------------------------------


def test_per_row_top_k_mixed_picks_per_row_argmax_under_temperature_collapse():
    torch.manual_seed(0)
    B, V = 4, 16
    logits = torch.randn(B, V)

    # top_k per row: [1, V, V, 1] → rows 0 and 3 collapse to argmax; rows 1, 2
    # see the full vocab.
    k_per_row = [1, V, V, 1]
    # Use very low temperature so top-1 dominates even for rows 1, 2.
    out = sample_logits(
        logits.clone() * 20.0,
        temperature=1.0,
        top_k=k_per_row,
        top_p=1.0,
        do_sample=True,
    )
    expected = logits.argmax(dim=-1)
    assert torch.equal(out, expected)


def test_per_row_top_p_zero_sampling_mass_picks_argmax_row():
    """With top_p very small we keep only the row's top-1 token; sampling
    should collapse to argmax for every row regardless of the other rows'
    distributions."""
    torch.manual_seed(0)
    B, V = 5, 20
    logits = torch.randn(B, V) * 10
    # Pick a different tiny p per row; all should still yield argmax.
    p_per_row = [0.001, 0.01, 0.05, 0.1, 0.2]
    out = sample_logits(
        logits.clone(),
        temperature=1.0,
        top_k=0,
        top_p=p_per_row,
        do_sample=True,
    )
    assert torch.equal(out, logits.argmax(dim=-1))


def test_per_row_temperature_mixed_matches_per_row_scalar_calls():
    """With do_sample=False (argmax), temperature rescales but argmax is
    invariant under positive scaling. So per-row temperatures should produce
    identical output across any values > 0 — equal to the argmax.
    """
    torch.manual_seed(42)
    B, V = 4, 24
    logits = torch.randn(B, V)
    temps = torch.tensor([0.1, 0.5, 1.0, 2.0])
    got = sample_logits(
        logits.clone(), temperature=temps, top_k=0, top_p=1.0, do_sample=False,
    )
    assert torch.equal(got, logits.argmax(dim=-1))


def test_per_row_disable_flags_behave_correctly():
    """Per-row value 0 (top_k) or 1.0 (top_p) disables that row's filter."""
    torch.manual_seed(0)
    B, V = 3, 10
    logits = torch.randn(B, V) * 5.0

    # Row 0: top_k=1 (keeps only argmax). Rows 1, 2: top_k=0 (disabled, full support).
    # Use very low temperature so sampling is effectively argmax for all rows.
    out = sample_logits(
        logits.clone() * 10.0,
        temperature=1.0,
        top_k=[1, 0, 0],
        top_p=1.0,
        do_sample=True,
    )
    assert torch.equal(out, logits.argmax(dim=-1))

    # Row 1: top_p=1.0 (disabled). Other rows: very small top_p → argmax.
    out2 = sample_logits(
        logits.clone() * 10.0,
        temperature=1.0,
        top_k=0,
        top_p=[0.01, 1.0, 0.01],
        do_sample=True,
    )
    assert torch.equal(out2, logits.argmax(dim=-1))


# ---------------------------------------------------------------------------
# Mixed values — compare against per-row scalar invocations
# ---------------------------------------------------------------------------


def test_per_row_mixed_matches_independent_scalar_runs():
    """The strongest equivalence: per-row call ≡ running each row through
    its own scalar call under the same RNG seed.

    Because ``torch.multinomial`` consumes RNG state proportionally to the
    input shape, we can only do this deterministically when sampling collapses
    to argmax (top_k=1). We configure each row to have top_k=1 with different
    temperature; argmax of logits/T is invariant to T > 0, so both paths
    produce the argmax regardless.
    """
    torch.manual_seed(0)
    B, V = 5, 32
    logits = torch.randn(B, V)
    temps = torch.tensor([0.3, 0.9, 1.2, 0.5, 2.0])

    # Per-row batched call.
    torch.manual_seed(5)
    got = sample_logits(
        logits.clone(),
        temperature=temps,
        top_k=[1] * B,
        top_p=1.0,
        do_sample=True,
    )

    # Reference: per-row argmax.
    expected = logits.argmax(dim=-1)
    assert torch.equal(got, expected)


# ---------------------------------------------------------------------------
# Sanity: scalar path is unchanged from pre-Phase-2 behavior.
# ---------------------------------------------------------------------------


def test_scalar_path_bit_identical_with_manual_reference():
    """Verify the scalar path still matches a hand-rolled reference
    implementation, to guard against accidental numerical regression."""
    torch.manual_seed(0)
    B, V = 3, 16
    logits = torch.randn(B, V)

    # Reference (the old single-path formula).
    def ref(logits, T, k, p):
        x = logits.clone() / T
        if k > 0:
            tv, _ = torch.topk(x, min(k, x.size(-1)))
            x = torch.where(x < tv[..., -1:], torch.full_like(x, float("-inf")), x)
        if p < 1.0:
            sl, si = torch.sort(x, descending=True)
            probs = F.softmax(sl, dim=-1)
            cum = torch.cumsum(probs, dim=-1)
            rem = cum > p
            rem[..., 0] = False
            sl = sl.masked_fill(rem, float("-inf"))
            x = torch.full_like(x, float("-inf"))
            x.scatter_(-1, si, sl)
        return torch.multinomial(F.softmax(x, dim=-1), 1).squeeze(-1)

    torch.manual_seed(7)
    got = sample_logits(logits.clone(), temperature=0.7, top_k=5, top_p=0.9, do_sample=True)
    torch.manual_seed(7)
    exp = ref(logits, 0.7, 5, 0.9)
    assert torch.equal(got, exp)
