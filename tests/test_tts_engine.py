"""Integration tests for ``TTSEngine`` submit/run/flush semantics.

Validates the Phase 2 streaming-engine skeleton end-to-end with a mocked
``FasterQwen3TTS`` so the tests run on CPU without loading real weights.

Covered behaviors:
    * submit() enqueues and returns a handle.
    * run_pending() batches up to max_batch_size requests of the same mode.
    * Mixed-mode queues are processed in FIFO cohorts by mode.
    * Results are dispatched to the correct handle; audio and timing
      round-trip through the handle API.
    * Errors during generate_batch propagate to every handle in the cohort.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from faster_qwen3_tts.engine import Request, Result, RequestHandle, TTSEngine


class _FakeModel:
    """Records calls; returns deterministic results."""

    def __init__(self):
        self.calls = []  # list of [modes] per batch

    def generate_batch(self, requests):
        self.calls.append([r.mode for r in requests])
        results = []
        for i, r in enumerate(requests):
            # Audio value encodes the request_id so tests can assert dispatch.
            rid = int(r.request_id) if r.request_id and r.request_id.isdigit() else i
            audio = np.full(4, float(rid), dtype=np.float32)
            results.append(Result(
                audio=audio,
                sample_rate=24000,
                codec_ids=torch.zeros(1, 16, dtype=torch.long),
                timing={"steps": 1, "decode_s": 0.001},
                request_id=r.request_id,
            ))
        return results


def _mk_req(idx, mode="custom_voice", **kwargs):
    """Build a valid Request for the given mode."""
    if mode == "custom_voice":
        kwargs.setdefault("speaker", f"spk_{idx}")
    elif mode == "voice_design":
        kwargs.setdefault("instruct", "calm narrator")
    elif mode == "voice_clone":
        kwargs.setdefault("ref_audio", f"/tmp/ref_{idx}.wav")
    return Request(text=f"text {idx}", mode=mode, request_id=str(idx), **kwargs)


# ---------------------------------------------------------------------------
# submit() / queueing
# ---------------------------------------------------------------------------


def test_submit_enqueues_and_returns_handle():
    engine = TTSEngine(_FakeModel(), max_batch_size=4)
    h = engine.submit(_mk_req(0))
    assert isinstance(h, RequestHandle)
    assert engine.pending == 1
    assert not h.done


def test_submit_rejects_non_request():
    engine = TTSEngine(_FakeModel())
    with pytest.raises(TypeError, match="Request"):
        engine.submit({"text": "hi"})


def test_engine_rejects_bad_max_batch_size():
    with pytest.raises(ValueError, match="max_batch_size"):
        TTSEngine(_FakeModel(), max_batch_size=0)


# ---------------------------------------------------------------------------
# run_pending() — basic batching
# ---------------------------------------------------------------------------


def test_run_pending_batches_same_mode_cohort():
    model = _FakeModel()
    engine = TTSEngine(model, max_batch_size=3)
    handles = [engine.submit(_mk_req(i)) for i in range(3)]

    executed = engine.run_pending()
    assert executed == 3
    assert engine.pending == 0
    assert model.calls == [["custom_voice", "custom_voice", "custom_voice"]]

    for i, h in enumerate(handles):
        assert h.done
        res = h.result()
        assert res.request_id == str(i)
        # Audio value encodes request_id → proves dispatch correctness.
        assert float(res.audio[0]) == float(i)


def test_run_pending_respects_max_batch_size():
    model = _FakeModel()
    engine = TTSEngine(model, max_batch_size=2)
    [engine.submit(_mk_req(i)) for i in range(5)]

    engine.run_pending()
    assert engine.pending == 3
    assert model.calls == [["custom_voice", "custom_voice"]]

    engine.run_pending()
    assert engine.pending == 1
    engine.run_pending()
    assert engine.pending == 0
    assert len(model.calls) == 3  # 2 + 2 + 1


def test_run_pending_on_empty_returns_zero():
    engine = TTSEngine(_FakeModel())
    assert engine.run_pending() == 0


# ---------------------------------------------------------------------------
# Mixed-mode queues: one cohort per mode at the queue head.
# ---------------------------------------------------------------------------


def test_mixed_mode_queue_processes_head_mode_only_per_cohort():
    model = _FakeModel()
    engine = TTSEngine(model, max_batch_size=4)
    # Queue: [custom, custom, design, custom, design]
    engine.submit(_mk_req(0, mode="custom_voice"))
    engine.submit(_mk_req(1, mode="custom_voice"))
    engine.submit(_mk_req(2, mode="voice_design"))
    engine.submit(_mk_req(3, mode="custom_voice"))
    engine.submit(_mk_req(4, mode="voice_design"))

    # First cohort: pulls custom_voice items in order 0, 1, 3 (skipping the
    # intervening design item but preserving queue order otherwise).
    executed = engine.run_pending()
    assert executed == 3
    assert model.calls[-1] == ["custom_voice", "custom_voice", "custom_voice"]
    assert engine.pending == 2  # design items remain

    # Second cohort: the two design items.
    executed = engine.run_pending()
    assert executed == 2
    assert model.calls[-1] == ["voice_design", "voice_design"]
    assert engine.pending == 0


# ---------------------------------------------------------------------------
# flush()
# ---------------------------------------------------------------------------


def test_flush_drains_queue():
    model = _FakeModel()
    engine = TTSEngine(model, max_batch_size=2)
    handles = [engine.submit(_mk_req(i)) for i in range(5)]

    total = engine.flush()
    assert total == 5
    assert engine.pending == 0
    for h in handles:
        assert h.done


# ---------------------------------------------------------------------------
# Handle iteration API
# ---------------------------------------------------------------------------


def test_handle_iter_yields_audio_chunks():
    model = _FakeModel()
    engine = TTSEngine(model)
    h = engine.submit(_mk_req(42))
    engine.run_pending()

    chunks = list(h)
    assert len(chunks) == 1
    audio, sr, timing = chunks[0]
    assert audio.shape == (4,)
    assert float(audio[0]) == 42.0
    assert sr == 24000
    assert "decode_s" in timing


def test_handle_result_before_done_raises():
    engine = TTSEngine(_FakeModel())
    h = engine.submit(_mk_req(0))
    with pytest.raises(RuntimeError, match="before the engine finished"):
        h.result()


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


class _FailingModel:
    def generate_batch(self, requests):
        raise RuntimeError("synthetic failure")


def test_generate_batch_failure_propagates_to_every_handle_in_cohort():
    engine = TTSEngine(_FailingModel(), max_batch_size=4)
    handles = [engine.submit(_mk_req(i)) for i in range(3)]
    with pytest.raises(RuntimeError, match="synthetic failure"):
        engine.run_pending()
    for h in handles:
        assert h.done
        with pytest.raises(RuntimeError, match="synthetic failure"):
            h.result()
