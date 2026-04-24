"""Batched KV cache with per-row write heads for continuous batching.

This cache is the foundation for Phase 2 (continuous batching) of the
concurrent inference plan. Unlike ``transformers.StaticCache`` — which uses a
single shared ``cache_position`` across the whole batch — ``BatchedStaticCache``
tracks an independent write head per row, so lanes that were admitted at
different wall-clock steps can advance through the KV tensor at different
absolute offsets without conflict.

Design
------
For each layer we keep two contiguous tensors

    key_cache[layer]   : [B, H, S, D]
    value_cache[layer] : [B, H, S, D]

where ``B`` is the static max batch size, ``H`` the number of KV heads, ``S``
the static max sequence length, and ``D`` the head dim. A ``write_heads``
tensor of shape ``[B]`` records the next write position per row.

:meth:`update` writes a length-``L`` chunk of new keys/values into the slot
``[write_heads[b] : write_heads[b] + L]`` for each row ``b`` independently,
then advances ``write_heads``. A ragged update (different ``L`` per row) is
supported by masking via a per-row ``seq_lens`` argument.

:meth:`reset_lane` zeroes both the KV tensors for a lane and its write head;
use this when admitting a new request into lane ``b``.

:meth:`copy_lane_from_prefill` bulk-copies the KV tensors produced by a
single-request HF prefill (``DynamicCache``-shaped: ``[1, H, L, D]``) into
lane ``b`` and sets ``write_heads[b] = L``.

Design notes vs alternatives
----------------------------
We deliberately do not subclass ``transformers.StaticCache``. Its public
contract is that ``cache_position`` is a 1-D tensor shared across the batch;
monkey-patching per-row writes would couple us tightly to its internals.
Instead this is a self-contained class; plumbing into the talker is handled
separately by the engine (which uses ``torch.scatter_`` on cache tensors it
owns, mirroring the vLLM pattern).

All operations are pure tensor ops, which makes them fully CPU-testable —
see ``tests/test_batched_cache.py``.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch


class BatchedStaticCache:
    """Per-row KV cache for continuous batching.

    Args:
        num_layers: Number of transformer layers.
        max_batch_size: Static maximum batch size (lanes). Must be known at
            graph-capture time in Phase 2.
        max_seq_len: Static maximum sequence length (per row).
        num_kv_heads: Number of key/value heads.
        head_dim: Head dimension.
        dtype: KV tensor dtype (typically ``torch.bfloat16`` for inference).
        device: Torch device.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        if num_layers <= 0 or max_batch_size <= 0 or max_seq_len <= 0:
            raise ValueError("num_layers, max_batch_size, max_seq_len must be positive")

        self.num_layers = int(num_layers)
        self.max_batch_size = int(max_batch_size)
        self.max_seq_len = int(max_seq_len)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.dtype = dtype
        self.device = torch.device(device)

        kv_shape = (
            self.max_batch_size,
            self.num_kv_heads,
            self.max_seq_len,
            self.head_dim,
        )
        self.key_cache: List[torch.Tensor] = [
            torch.zeros(kv_shape, dtype=dtype, device=self.device)
            for _ in range(self.num_layers)
        ]
        self.value_cache: List[torch.Tensor] = [
            torch.zeros(kv_shape, dtype=dtype, device=self.device)
            for _ in range(self.num_layers)
        ]
        self.write_heads: torch.Tensor = torch.zeros(
            self.max_batch_size, dtype=torch.long, device=self.device
        )

    # ------------------------------------------------------------------
    # Lifecycle: reset / copy-in
    # ------------------------------------------------------------------

    def reset_lane(self, lane_idx: int) -> None:
        """Zero out a single lane and rewind its write head to 0."""
        self._check_lane(lane_idx)
        for layer in range(self.num_layers):
            self.key_cache[layer][lane_idx].zero_()
            self.value_cache[layer][lane_idx].zero_()
        self.write_heads[lane_idx] = 0

    def reset_all(self) -> None:
        for layer in range(self.num_layers):
            self.key_cache[layer].zero_()
            self.value_cache[layer].zero_()
        self.write_heads.zero_()

    def copy_lane_from_prefill(
        self,
        lane_idx: int,
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Copy a single-request prefill into lane ``lane_idx``.

        Args:
            lane_idx: Destination lane.
            past_key_values: List of ``(k, v)`` layer tuples from a
                ``DynamicCache`` after an HF prefill. Each ``k`` / ``v`` is
                shaped ``[1, H, L, D]``. ``L`` must be ``<= max_seq_len``.
        """
        self._check_lane(lane_idx)
        if len(past_key_values) != self.num_layers:
            raise ValueError(
                f"past_key_values has {len(past_key_values)} layers, "
                f"expected {self.num_layers}"
            )
        # All layers should share the same prefill length.
        L = int(past_key_values[0][0].shape[2])
        if L > self.max_seq_len:
            raise ValueError(
                f"Prefill length {L} exceeds max_seq_len {self.max_seq_len}"
            )
        for layer, (k, v) in enumerate(past_key_values):
            if k.shape[0] != 1 or v.shape[0] != 1:
                raise ValueError(
                    f"Layer {layer}: expected batch=1 prefill, got k={tuple(k.shape)}"
                )
            if k.shape[2] != L or v.shape[2] != L:
                raise ValueError(
                    f"Layer {layer}: inconsistent seq dim ({k.shape[2]} vs first layer {L})"
                )
            self.key_cache[layer][lane_idx, :, :L, :].copy_(k[0].to(self.dtype))
            self.value_cache[layer][lane_idx, :, :L, :].copy_(v[0].to(self.dtype))
            # Zero out the rest of this lane's slots for determinism on later reads.
            if L < self.max_seq_len:
                self.key_cache[layer][lane_idx, :, L:, :].zero_()
                self.value_cache[layer][lane_idx, :, L:, :].zero_()
        self.write_heads[lane_idx] = L

    # ------------------------------------------------------------------
    # Update (decode step write)
    # ------------------------------------------------------------------

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *,
        seq_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Write per-row KV chunks into the cache at the current write heads.

        Args:
            layer_idx: Layer to write.
            key_states: ``[B, H, L, D]`` — new keys to append, per row.
            value_states: ``[B, H, L, D]`` — new values to append, per row.
            seq_lens: Optional ``[B]`` tensor of effective chunk lengths per
                row. Rows with ``seq_lens[b] < L`` only have their first
                ``seq_lens[b]`` positions written (the rest is ignored).
                Defaults to ``L`` for all rows.

        Returns:
            (key_cache_layer, value_cache_layer) — the full cache tensors for
            this layer, shaped ``[B, H, S, D]``. Callers typically slice these
            by ``write_heads`` / attention mask.

        Note:
            ``write_heads`` is advanced only on the **last** layer of each
            step (``layer_idx == num_layers - 1``). Callers are expected to
            iterate layers in order 0, 1, …, num_layers-1 within a single
            decode step so that every layer writes at the same position
            before the heads move forward.
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {self.num_layers})")
        if key_states.shape != value_states.shape:
            raise ValueError(
                f"key/value shape mismatch: {tuple(key_states.shape)} vs "
                f"{tuple(value_states.shape)}"
            )
        if key_states.dim() != 4:
            raise ValueError(
                f"Expected [B, H, L, D] key_states, got {tuple(key_states.shape)}"
            )
        B, H, L, D = key_states.shape
        if B != self.max_batch_size:
            raise ValueError(f"Batch dim {B} != max_batch_size {self.max_batch_size}")
        if H != self.num_kv_heads or D != self.head_dim:
            raise ValueError(
                f"Head mismatch: got H={H}, D={D}; cache expects H="
                f"{self.num_kv_heads}, D={self.head_dim}"
            )

        if seq_lens is None:
            eff_lens = torch.full((B,), L, dtype=torch.long, device=self.device)
        else:
            if seq_lens.shape != (B,):
                raise ValueError(
                    f"seq_lens must be [B]={B}, got {tuple(seq_lens.shape)}"
                )
            eff_lens = seq_lens.to(dtype=torch.long, device=self.device)
            if int(eff_lens.max().item()) > L:
                raise ValueError(
                    f"seq_lens max {int(eff_lens.max().item())} exceeds chunk length {L}"
                )

        # Overflow check: write_heads + eff_lens must be <= max_seq_len.
        end_heads = self.write_heads + eff_lens
        if int(end_heads.max().item()) > self.max_seq_len:
            raise RuntimeError(
                f"BatchedStaticCache overflow: some row would write past "
                f"max_seq_len={self.max_seq_len} "
                f"(write_heads={self.write_heads.tolist()}, "
                f"eff_lens={eff_lens.tolist()})"
            )

        # Per-row scatter. Vectorized implementation: build a [B, L] index
        # tensor where index[b, j] = write_heads[b] + j (clamped), plus a
        # [B, L] validity mask from eff_lens. We scatter along dim=2 (seq).
        j = torch.arange(L, device=self.device)  # [L]
        idx = self.write_heads.unsqueeze(1) + j.unsqueeze(0)  # [B, L]
        valid = j.unsqueeze(0) < eff_lens.unsqueeze(1)  # [B, L]
        # For rows where j >= eff_lens, clamp the index to write_heads[b] (a
        # valid in-range position) and skip via scatter_reduce below. We do
        # the simpler approach: explicit per-row copy_ in a Python loop. L
        # is tiny (==1 in the decode step hot path) so this is cheap.
        k_layer = self.key_cache[layer_idx]
        v_layer = self.value_cache[layer_idx]
        # Expand idx to [B, H, L, D] for gather/scatter along seq dim.
        idx_full = idx.view(B, 1, L, 1).expand(B, H, L, D).to(torch.long)
        # scatter_ writes src into self at positions specified by index along dim=2.
        # For masked-out positions we first set those src slots to the current
        # cache value, so the scatter is a no-op.
        # Easier: build a masked src that equals cache at the clamped idx for
        # invalid positions.
        if valid.all():
            k_layer.scatter_(2, idx_full, key_states.to(k_layer.dtype))
            v_layer.scatter_(2, idx_full, value_states.to(v_layer.dtype))
        else:
            # idx.clamp ensures valid indexing even for masked rows; we then
            # gather the current cache values to use as src for masked slots.
            idx_clamped = idx_full.clamp(max=self.max_seq_len - 1)
            current_k = k_layer.gather(2, idx_clamped)
            current_v = v_layer.gather(2, idx_clamped)
            mask4 = valid.view(B, 1, L, 1).expand(B, H, L, D)
            src_k = torch.where(mask4, key_states.to(k_layer.dtype), current_k)
            src_v = torch.where(mask4, value_states.to(v_layer.dtype), current_v)
            k_layer.scatter_(2, idx_clamped, src_k)
            v_layer.scatter_(2, idx_clamped, src_v)

        # Advance write heads only after all layers in this step have written.
        if layer_idx == self.num_layers - 1:
            self.write_heads = end_heads

        return k_layer, v_layer

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_write_heads(self) -> torch.Tensor:
        """Return a defensive copy of the per-row write head positions."""
        return self.write_heads.clone()

    def get_seq_length(self, lane_idx: int) -> int:
        self._check_lane(lane_idx)
        return int(self.write_heads[lane_idx].item())

    def _check_lane(self, lane_idx: int) -> None:
        if not (0 <= lane_idx < self.max_batch_size):
            raise IndexError(
                f"lane_idx {lane_idx} out of range [0, {self.max_batch_size})"
            )
