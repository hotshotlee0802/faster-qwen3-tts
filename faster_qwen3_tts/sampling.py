"""Shared sampling helpers for talker and predictor generation."""
from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn.functional as F


def apply_repetition_penalty(
    logits: torch.Tensor,
    token_history: torch.Tensor,
    repetition_penalty: float,
) -> torch.Tensor:
    """Apply repetition penalty to logits in-place and return them.

    Args:
        logits: Tensor shaped [1, 1, vocab], [1, vocab], or [B, vocab].
        token_history: Token ids previously generated.
            - 1-D tensor of shape [T]: shared history applied to every row of
              `logits` (legacy single-request behavior).
            - 2-D tensor of shape [B, T]: per-row histories. Row ``b`` of
              `logits` is penalized only by `token_history[b]`. Rows whose
              history is shorter than T may be padded with a negative id
              (``< 0``); negative ids are ignored. This shape requires
              `logits.shape[0] == B`.
        repetition_penalty: HF-style repetition penalty (>1.0). No-op at 1.0.

    Returns:
        The same ``logits`` tensor, with repetition penalty applied in-place.
    """
    if repetition_penalty == 1.0 or token_history.numel() == 0:
        return logits

    if token_history.dim() == 1:
        unique_toks = token_history.unique()
        tok_logits = logits[..., unique_toks]
        logits[..., unique_toks] = torch.where(
            tok_logits > 0, tok_logits / repetition_penalty, tok_logits * repetition_penalty
        )
        return logits

    if token_history.dim() != 2:
        raise ValueError(
            f"token_history must be 1-D or 2-D, got shape {tuple(token_history.shape)}"
        )

    # Per-row history path. We need logits to be [B, vocab] to align with history rows.
    if logits.dim() == 3 and logits.shape[1] == 1:
        # Collapse the middle singleton so we can index [B, V] consistently, then restore.
        flat = logits.squeeze(1)
        _apply_per_row_penalty_(flat, token_history, repetition_penalty)
        return logits  # view into logits; mutation already applied

    if logits.dim() == 2:
        _apply_per_row_penalty_(logits, token_history, repetition_penalty)
        return logits

    raise ValueError(
        "For 2-D token_history, logits must be [B, V] or [B, 1, V]; "
        f"got shape {tuple(logits.shape)}"
    )


def _apply_per_row_penalty_(
    logits_2d: torch.Tensor, token_history_2d: torch.Tensor, repetition_penalty: float
) -> None:
    """In-place per-row repetition penalty. logits_2d: [B, V]; history: [B, T]."""
    if logits_2d.shape[0] != token_history_2d.shape[0]:
        raise ValueError(
            f"Batch size mismatch: logits {logits_2d.shape[0]} vs history "
            f"{token_history_2d.shape[0]}"
        )
    B, V = logits_2d.shape
    # Build a boolean seen-mask [B, V]. Negative ids in history are ignored.
    # Clamp negatives to 0 (a valid index) but mask them out before scatter.
    hist = token_history_2d
    valid = hist >= 0
    safe = hist.clamp(min=0)
    seen = torch.zeros(B, V, dtype=torch.bool, device=logits_2d.device)
    # scatter True into seen at positions safe[b, t] where valid[b, t]
    seen.scatter_(1, safe, valid)
    # Apply penalty: divide when logit > 0 else multiply.
    penalized = torch.where(
        logits_2d > 0, logits_2d / repetition_penalty, logits_2d * repetition_penalty
    )
    logits_2d.copy_(torch.where(seen, penalized, logits_2d))


def sample_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    do_sample: bool,
    suppress_mask: Optional[torch.Tensor] = None,
    suppress_tokens: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    """Sample a token from logits.

    Mirrors HF order: suppress -> temperature -> top-k -> top-p -> sample.
    """
    logits = logits.clone()
    if suppress_mask is not None:
        logits[..., suppress_mask] = float("-inf")
    if suppress_tokens:
        logits[..., list(suppress_tokens)] = float("-inf")
    if not do_sample:
        return torch.argmax(logits, dim=-1)
    logits = logits / temperature
    if top_k > 0:
        topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = torch.where(logits < topk_vals[..., -1:], torch.full_like(logits, float("-inf")), logits)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 0] = False
        sorted_logits[sorted_indices_to_remove] = float("-inf")
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(-1, sorted_indices, sorted_logits)
    return torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
