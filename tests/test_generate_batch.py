"""End-to-end mocked test for ``FasterQwen3TTS.generate_batch``.

Validates the full orchestration without touching GPU or real weights:
    * per-request input preparation and left-padded batching
    * dispatch of per-row codec outputs back to the correct ``Result``
    * per-lane effective length (EOS handling) yields correct per-request audio
    * mode validation and sampling-param mismatch warnings
"""
from __future__ import annotations

import logging
import types

import numpy as np
import pytest
import torch

from faster_qwen3_tts.engine import Request, Result
from faster_qwen3_tts.model import FasterQwen3TTS


def _build_dummy_model():
    base = types.SimpleNamespace()
    base.model = types.SimpleNamespace(
        talker=types.SimpleNamespace(rope_deltas=None),
        config=types.SimpleNamespace(talker_config=types.SimpleNamespace(
            codec_eos_token_id=2,
            num_code_groups=16,
            vocab_size=2048,
        )),
        tts_model_size="1b7",
        tts_model_type="base",
        speech_tokenizer=None,  # set by each test as needed
    )
    base._build_assistant_text = lambda text: text
    base._build_ref_text = lambda text: text
    base._build_instruct_text = lambda text: text
    # Each tokenize call produces a tensor with fixed width; keep it simple.
    base._tokenize_texts = lambda texts: [
        torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long) for _ in texts
    ]
    base._validate_languages = lambda _languages: None
    base._validate_speakers = lambda _speakers: None
    base.create_voice_clone_prompt = lambda *a, **kw: {
        "ref_code": [None],
        "ref_spk_embedding": [torch.zeros(1, 4)],
        "x_vector_only_mode": [True],
        "icl_mode": [False],
    }
    base._prompt_items_to_voice_clone_prompt = lambda prompt_items: {
        "ref_spk_embedding": [item.ref_spk_embedding for item in prompt_items],
    }

    model = FasterQwen3TTS(base, object(), object(), device="cpu", dtype=torch.float32)

    # Short-circuit input building to return deterministic shapes.
    def fake_build(**_kwargs):
        B = len(_kwargs["input_ids"])
        tie = torch.zeros(B, 10, 4, dtype=torch.float32)
        tam = torch.ones(B, 10, dtype=torch.long)
        tth = torch.zeros(B, 1, 4, dtype=torch.float32)
        tpe = torch.zeros(1, 1, 4, dtype=torch.float32)
        return tie, tam, tth, tpe

    model._build_talker_inputs_local = fake_build
    model._warmup = lambda _prefill_len: setattr(model, "_warmed_up", True)
    model._warmed_up = True  # skip warmup entirely
    return model


class _FakeTalker:
    """Emits one codec id per row, per step. Row i emits the i-th
    ``per_row_sequences`` value at step t, with EOS=2 terminating that row."""

    def __init__(self, per_row_sequences, num_code_groups=16, eos_id=2):
        self.per_row_sequences = per_row_sequences
        self.num_code_groups = num_code_groups
        self.eos_id = eos_id
        self.called_with = None

    def generate(self, **kwargs):
        self.called_with = kwargs
        B = kwargs["inputs_embeds"].shape[0]
        assert B == len(self.per_row_sequences)
        T = max(len(s) for s in self.per_row_sequences)
        steps = []
        for t in range(T):
            step = torch.zeros(B, self.num_code_groups, dtype=torch.long)
            for i, seq in enumerate(self.per_row_sequences):
                if t < len(seq):
                    step[i, 0] = seq[t]
            steps.append((None, step))
        return types.SimpleNamespace(hidden_states=steps)


class _FakeSpeechTokenizer:
    """Produces audio proportional to number of codec frames. Each request's
    audio value equals codebook-0 token * 10 + frame index, so we can verify
    which tokens ended up in which output."""

    def decode(self, payload):
        codes = payload["audio_codes"]  # [1, T, 16]
        T = codes.shape[1]
        first = codes[0, :, 0].tolist()
        # 4 samples per frame
        audio = np.array(
            [t * 10 + first[t] for t in range(T) for _ in range(4)],
            dtype=np.float32,
        )
        return [audio], 24000


def test_generate_batch_dispatches_per_row_results_correctly():
    """Batch 3 requests with distinct EOS positions; verify each Result
    contains exactly the codec tokens for its row."""
    model = _build_dummy_model()
    model.sample_rate = 24000
    model.model.model.speech_tokenizer = _FakeSpeechTokenizer()

    # Row 0: 3 tokens then EOS. Row 1: 5 tokens then EOS. Row 2: 4 tokens, no EOS.
    talker = _FakeTalker(
        per_row_sequences=[
            [11, 12, 13, 2],
            [21, 22, 23, 24, 25, 2],
            [31, 32, 33, 34],
        ],
    )
    model.model.model.talker = types.SimpleNamespace(generate=talker.generate, rope_deltas=None)

    requests = [
        Request(text="alpha",  mode="custom_voice", speaker="spk_a", request_id="A"),
        Request(text="beta",   mode="custom_voice", speaker="spk_b", request_id="B"),
        Request(text="gamma",  mode="custom_voice", speaker="spk_c", request_id="C"),
    ]
    # Mark model as custom_voice capable so mode validation passes.
    model.model.model.tts_model_type = "custom_voice"

    results = model.generate_batch(requests)

    assert len(results) == 3
    # Row 0 codes: first codebook [11, 12, 13]
    assert results[0].codec_ids is not None
    assert results[0].codec_ids.shape == (3, 16)
    assert results[0].codec_ids[:, 0].tolist() == [11, 12, 13]
    # Row 1 codes: [21, 22, 23, 24, 25]
    assert results[1].codec_ids.shape == (5, 16)
    assert results[1].codec_ids[:, 0].tolist() == [21, 22, 23, 24, 25]
    # Row 2 codes: [31, 32, 33, 34] then 2 zero-padded steps since it never
    # emitted EOS; batched generation runs to the global max T = 6.
    assert results[2].codec_ids.shape == (6, 16)
    assert results[2].codec_ids[:, 0].tolist() == [31, 32, 33, 34, 0, 0]

    # request_id round-tripped.
    assert [r.request_id for r in results] == ["A", "B", "C"]

    # Audio shapes reflect 4 samples per codec frame from the fake tokenizer.
    assert results[0].audio.shape == (3 * 4,)
    assert results[1].audio.shape == (5 * 4,)
    assert results[2].audio.shape == (6 * 4,)

    # Sample content reflects the right row's codes (codebook-0 values).
    # Row 0 frame 0: 0*10 + 11 = 11; frame 1: 1*10 + 12 = 22; frame 2: 2*10 + 13 = 33
    assert results[0].audio[0] == 11.0
    assert results[0].audio[4] == 22.0
    assert results[0].audio[8] == 33.0


def test_generate_batch_empty_row_returns_silence_result():
    model = _build_dummy_model()
    model.sample_rate = 24000
    model.model.model.speech_tokenizer = _FakeSpeechTokenizer()
    model.model.model.tts_model_type = "custom_voice"

    # Row emits EOS immediately at step 0 → empty codes.
    talker = _FakeTalker(per_row_sequences=[[2]])
    model.model.model.talker = types.SimpleNamespace(generate=talker.generate, rope_deltas=None)

    results = model.generate_batch([
        Request(text="hi", mode="custom_voice", speaker="spk_a"),
    ])

    assert len(results) == 1
    assert results[0].codec_ids is None
    assert results[0].audio.shape == (1,)
    assert results[0].audio[0] == 0.0
    assert results[0].sample_rate == 24000


def test_generate_batch_rejects_mixed_modes():
    model = _build_dummy_model()
    with pytest.raises(ValueError, match="same mode"):
        model.generate_batch([
            Request(text="hi", mode="custom_voice", speaker="spk_a"),
            Request(text="yo", mode="voice_design", instruct="calm"),
        ])


def test_generate_batch_rejects_non_request_input():
    model = _build_dummy_model()
    with pytest.raises(TypeError, match="Request"):
        model.generate_batch([{"text": "hi"}])


def test_generate_batch_warns_on_mismatched_sampling_params(caplog):
    model = _build_dummy_model()
    model.sample_rate = 24000
    model.model.model.speech_tokenizer = _FakeSpeechTokenizer()
    model.model.model.tts_model_type = "custom_voice"
    talker = _FakeTalker(per_row_sequences=[[1, 2], [3, 4, 2]])
    model.model.model.talker = types.SimpleNamespace(generate=talker.generate, rope_deltas=None)

    reqs = [
        Request(text="a", mode="custom_voice", speaker="s1", temperature=0.9),
        Request(text="b", mode="custom_voice", speaker="s2", temperature=0.5),
    ]
    with caplog.at_level(logging.WARNING, logger="faster_qwen3_tts.model"):
        model.generate_batch(reqs)

    assert any("temperature" in r.message for r in caplog.records), \
        "Expected a warning about mismatched temperature"


def test_generate_batch_empty_input_returns_empty_list():
    model = _build_dummy_model()
    assert model.generate_batch([]) == []


def test_generate_batch_rejects_custom_voice_on_base_model():
    model = _build_dummy_model()
    # Base model does not support custom_voice.
    model.model.model.tts_model_type = "base"
    with pytest.raises(ValueError, match="custom voice"):
        model.generate_batch([
            Request(text="hi", mode="custom_voice", speaker="spk_a"),
        ])
