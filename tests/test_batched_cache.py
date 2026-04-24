"""Mathematical equivalence tests for ``BatchedStaticCache``.

All tests run on CPU. The cache is validated against a reference scalar
implementation: we run N independent single-row caches in parallel and
verify that a batched run produces bit-identical KV tensors and write-head
positions.
"""
from __future__ import annotations

import pytest
import torch

from faster_qwen3_tts.batched_cache import BatchedStaticCache


def _mk_cache(B=4, L=16, H=2, D=4, layers=3):
    return BatchedStaticCache(
        num_layers=layers,
        max_batch_size=B,
        max_seq_len=L,
        num_kv_heads=H,
        head_dim=D,
        dtype=torch.float32,
    )


def test_init_shapes_and_defaults():
    c = _mk_cache(B=5, L=10, H=3, D=7, layers=4)
    assert len(c.key_cache) == 4
    for layer in range(4):
        assert c.key_cache[layer].shape == (5, 3, 10, 7)
        assert c.value_cache[layer].shape == (5, 3, 10, 7)
        assert torch.equal(c.key_cache[layer], torch.zeros_like(c.key_cache[layer]))
    assert c.write_heads.tolist() == [0, 0, 0, 0, 0]


def test_init_rejects_bad_shapes():
    with pytest.raises(ValueError):
        BatchedStaticCache(num_layers=0, max_batch_size=1, max_seq_len=4, num_kv_heads=1, head_dim=1)


def test_copy_lane_from_prefill_places_kv_correctly():
    c = _mk_cache(B=3, L=16, H=2, D=4, layers=2)
    # Build a DynamicCache-shaped prefill of length 5 for lane 1 only.
    pkv = []
    for layer in range(2):
        k = torch.arange(1 * 2 * 5 * 4, dtype=torch.float32).reshape(1, 2, 5, 4) + layer * 100
        v = -k
        pkv.append((k, v))
    c.copy_lane_from_prefill(1, pkv)

    assert c.write_heads.tolist() == [0, 5, 0]
    # Lane 1 of each layer contains the prefill; lanes 0, 2 remain zero.
    for layer in range(2):
        assert torch.equal(c.key_cache[layer][1, :, :5, :], pkv[layer][0][0])
        assert torch.equal(c.key_cache[layer][0], torch.zeros(2, 16, 4))
        assert torch.equal(c.key_cache[layer][2], torch.zeros(2, 16, 4))
        # Trailing slots of lane 1 are zero.
        assert torch.equal(c.key_cache[layer][1, :, 5:, :], torch.zeros(2, 11, 4))


def test_copy_lane_rejects_oversized_prefill():
    c = _mk_cache(B=2, L=4)
    k = torch.zeros(1, 2, 5, 4)
    with pytest.raises(ValueError, match="exceeds"):
        c.copy_lane_from_prefill(0, [(k, k)] * c.num_layers)


def test_reset_lane_zeros_and_rewinds():
    c = _mk_cache(B=2, L=8)
    c.write_heads[0] = 3
    for layer in range(c.num_layers):
        c.key_cache[layer][0].fill_(7.0)
        c.value_cache[layer][0].fill_(-7.0)

    c.reset_lane(0)
    assert c.write_heads[0].item() == 0
    for layer in range(c.num_layers):
        assert torch.equal(c.key_cache[layer][0], torch.zeros(2, 8, 4))
        assert torch.equal(c.value_cache[layer][0], torch.zeros(2, 8, 4))


def test_update_advances_heads_only_on_last_layer():
    c = _mk_cache(B=2, L=8, H=1, D=3, layers=3)
    k = torch.randn(2, 1, 1, 3)
    v = torch.randn(2, 1, 1, 3)
    # Update all three layers in one step.
    c.update(0, k, v)
    assert c.write_heads.tolist() == [0, 0]  # not advanced yet
    c.update(1, k, v)
    assert c.write_heads.tolist() == [0, 0]
    c.update(2, k, v)
    assert c.write_heads.tolist() == [1, 1]  # advanced after last layer

    # Second decode step.
    c.update(0, k, v)
    c.update(1, k, v)
    c.update(2, k, v)
    assert c.write_heads.tolist() == [2, 2]


def test_update_equivalence_with_per_row_manual_scatter():
    """The core invariant: a batched update writes each row's new KV at its
    own write head, independently. Cross-check by running B separate
    single-lane caches and comparing the aggregate tensor."""
    torch.manual_seed(0)
    B, L_max, H, D, layers = 4, 32, 2, 4, 2
    batched = BatchedStaticCache(
        num_layers=layers, max_batch_size=B, max_seq_len=L_max,
        num_kv_heads=H, head_dim=D,
    )
    # Simulate staggered admission: each lane starts at a different position.
    initial_heads = [0, 5, 10, 2]
    for lane, pos in enumerate(initial_heads):
        if pos > 0:
            # Fill some synthetic "prefill" into this lane.
            pkv = []
            for layer in range(layers):
                k = torch.randn(1, H, pos, D)
                v = torch.randn(1, H, pos, D)
                pkv.append((k, v))
            batched.copy_lane_from_prefill(lane, pkv)
    assert batched.write_heads.tolist() == initial_heads

    # Reference: one independent [1, H, L_max, D] cache per lane.
    ref_k = [torch.zeros(H, L_max, D) for _ in range(B * layers)]
    ref_v = [torch.zeros(H, L_max, D) for _ in range(B * layers)]
    ref_heads = list(initial_heads)
    for lane in range(B):
        for layer in range(layers):
            idx = lane * layers + layer
            head = initial_heads[lane]
            ref_k[idx][:, :head, :] = batched.key_cache[layer][lane, :, :head, :].clone()
            ref_v[idx][:, :head, :] = batched.value_cache[layer][lane, :, :head, :].clone()

    # Run 12 decode steps of length-1 updates.
    for step in range(12):
        k = torch.randn(B, H, 1, D)
        v = torch.randn(B, H, 1, D)
        for layer in range(layers):
            batched.update(layer, k, v)
            # Update the reference per-lane caches.
            for lane in range(B):
                idx = lane * layers + layer
                pos = ref_heads[lane]
                ref_k[idx][:, pos : pos + 1, :] = k[lane]
                ref_v[idx][:, pos : pos + 1, :] = v[lane]
        for lane in range(B):
            ref_heads[lane] += 1

    # Compare.
    assert batched.write_heads.tolist() == ref_heads
    for lane in range(B):
        for layer in range(layers):
            idx = lane * layers + layer
            assert torch.equal(batched.key_cache[layer][lane], ref_k[idx]), \
                f"key mismatch at lane={lane} layer={layer}"
            assert torch.equal(batched.value_cache[layer][lane], ref_v[idx]), \
                f"value mismatch at lane={lane} layer={layer}"


def test_update_ragged_seq_lens_masks_invalid_rows():
    """If seq_lens[b] < L, only the first seq_lens[b] positions of row b
    are written; the rest of row b's cache stays unchanged."""
    c = _mk_cache(B=3, L=8, H=1, D=2, layers=1)
    # Pre-fill lane 1 with a marker at position 2..4 so we can detect any unwanted writes.
    c.key_cache[0][1, :, 2:5, :] = 9.0
    c.value_cache[0][1, :, 2:5, :] = -9.0
    c.write_heads[:] = torch.tensor([0, 2, 5])

    # Chunk length L_chunk=3; seq_lens says lane 0 writes 3, lane 1 writes 0, lane 2 writes 1.
    k = torch.arange(3 * 1 * 3 * 2, dtype=torch.float32).reshape(3, 1, 3, 2)
    v = k + 1000
    seq_lens = torch.tensor([3, 0, 1])
    c.update(0, k, v, seq_lens=seq_lens)

    # Heads advanced per eff_lens.
    assert c.write_heads.tolist() == [3, 2, 6]

    # Lane 0: positions 0..2 got k[0].
    assert torch.equal(c.key_cache[0][0, :, :3, :], k[0])
    # Lane 1: no write, slots 2..4 still the marker 9.0.
    assert torch.equal(c.key_cache[0][1, :, 2:5, :], torch.full((1, 3, 2), 9.0))
    # Lane 2: only position 5 got k[2, :, 0, :].
    assert torch.equal(c.key_cache[0][2, :, 5:6, :], k[2, :, 0:1, :])
    # Position 6 on lane 2 was NOT written (eff_len=1 < L_chunk=3).
    assert torch.equal(c.key_cache[0][2, :, 6:7, :], torch.zeros(1, 1, 2))


def test_update_rejects_overflow():
    c = _mk_cache(B=2, L=4, H=1, D=2, layers=1)
    c.write_heads[:] = torch.tensor([3, 0])
    # Trying to write 2 slots into lane 0 (head=3, L=4) overflows.
    k = torch.zeros(2, 1, 2, 2)
    with pytest.raises(RuntimeError, match="overflow"):
        c.update(0, k, k)


def test_update_rejects_shape_mismatch():
    c = _mk_cache(B=2, L=8, H=1, D=2, layers=1)
    # Wrong batch.
    k = torch.zeros(3, 1, 1, 2)
    with pytest.raises(ValueError, match="max_batch_size"):
        c.update(0, k, k)
    # Wrong head.
    k = torch.zeros(2, 2, 1, 2)
    with pytest.raises(ValueError, match="Head mismatch"):
        c.update(0, k, k)


def test_get_seq_length_and_write_heads_copy():
    c = _mk_cache(B=3, L=10)
    c.write_heads[:] = torch.tensor([2, 5, 7])
    assert c.get_seq_length(0) == 2
    assert c.get_seq_length(1) == 5
    assert c.get_seq_length(2) == 7
    w = c.get_write_heads()
    w[0] = 999  # must not mutate internal state.
    assert c.get_seq_length(0) == 2


def test_reset_all_zeros_heads_and_caches():
    c = _mk_cache(B=2, L=4, H=1, D=2, layers=2)
    c.update(0, torch.ones(2, 1, 2, 2), torch.ones(2, 1, 2, 2))
    c.update(1, torch.ones(2, 1, 2, 2), torch.ones(2, 1, 2, 2))
    c.reset_all()
    assert c.write_heads.tolist() == [0, 0]
    for layer in range(c.num_layers):
        assert torch.equal(c.key_cache[layer], torch.zeros_like(c.key_cache[layer]))
        assert torch.equal(c.value_cache[layer], torch.zeros_like(c.value_cache[layer]))
