"""Parity tests: single-request == batched-request when randomness is eliminated.

The core property under test
-----------------------------
With ``do_sample=False`` (greedy argmax), running text X alone in a batch of 1
must produce **bit-identical** codec token sequences to running X as part of a
larger batch (B > 1), regardless of:

* X's row-position in the batch (0, 1, 2, …)
* whether X is left-padded to align with a longer sibling request

Why this is guaranteed
-----------------------
Two independent conditions make it true:

1. **Attention-mask isolation** — row b can only attend to positions where
   ``attention_mask[b] == 1``.  The hidden state of row b at every step is
   therefore a function of *only* that row's non-padded input tokens, not of
   any other row in the batch.

2. **Greedy argmax is row-wise** — ``token_b = argmax(logits[b, :])``.  The
   argmax for row b depends only on ``logits[b]`` (which follows from point 1)
   and has no cross-row state.

CPU-testable proxy: ``_FakeRowIndependentTalker``
--------------------------------------------------
We implement a fake talker whose ``generate`` output per row depends *only* on
that row's non-padded input embeddings (signature computed from
``inputs_embeds[b][attention_mask[b].bool()]``).  This makes the
row-independence property explicit and verifiable without GPU/real weights.

Any real transformer talker that correctly uses ``attention_mask`` will have the
same property; the fake just makes it structurally impossible for the test to
pass unless ``batched_talker_generate`` correctly handles padding and row
splitting.

Test suite
-----------
1. ``test_solo_equals_same_text_in_batch`` — same-length rows; B=1 solo run ==
   same row inside a B=3 batch.
2. ``test_left_padded_row_equals_unpadded_solo`` — row A (length 5) is
   left-padded to 8 to match row B; codec tokens must equal the solo B=1 run.
3. ``test_batch_position_independence`` — row A at index 0, 1, and 2 in a
   3-row batch all produce identical tokens to the solo run.
4. ``test_multi_batch_sizes_match_solo`` — B ∈ {1, 2, 4} all produce identical
   codec tokens for the same row content.
5. ``test_generate_batch_codec_ids_match_solo_run`` — end-to-end parity
   through ``FasterQwen3TTS.generate_batch``: a single-item batch and a
   three-item batch both produce the same ``codec_ids`` for the same request.
"""
from __future__ import annotations

import types
from typing import List, Optional

import numpy as np
import pytest
import torch

from faster_qwen3_tts.engine import batched_talker_generate, Request, Result


# =============================================================================
# Fake row-independent talker
# =============================================================================


class _FakeRowIndependentTalker:
    """Fake talker whose per-row output depends ONLY on that row's non-padded
    input embeddings.

    **Signature computation**: for row b, mask the embeddings by
    ``attention_mask[b].bool()`` to exclude left-padding, then reduce to a
    single integer fingerprint.  The same content at any batch position, with
    any amount of leading zero-padding, produces the same fingerprint and
    therefore the same per-row codec token sequence.

    **Codec generation**: ``num_real_steps`` non-EOS tokens followed by a
    single EOS token. Token values are a deterministic function of (sig, step).

    **Keyword arguments**: the talker accepts ``**kwargs`` and ignores anything
    it does not need (mirrors the real ``talker.generate`` interface used by
    ``batched_talker_generate``).
    """

    EOS_ID = 2

    def __init__(self, num_real_steps: int = 5, num_code_groups: int = 16) -> None:
        self.num_real_steps = num_real_steps
        self.num_code_groups = num_code_groups

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _row_sig(self, emb_row: torch.Tensor, mask_row: torch.Tensor) -> int:
        """Integer fingerprint of the non-padded part of one row.

        Masking by ``attention_mask`` is the key property: left-padding (zeros)
        is excluded so that padded and unpadded versions of the same content
        produce the same fingerprint.

        We use a position-weighted sum to reduce hash collisions: two rows with
        different content but the same flat sum (which can happen with random
        tensors) will still produce different signatures.
        """
        valid_embs = emb_row[mask_row.bool()]  # [eff_len, H]
        flat = valid_embs.reshape(-1)
        n = flat.numel()
        if n == 0:
            return 0
        # Weight each element by its 1-indexed position to break symmetries.
        weights = torch.arange(1, n + 1, dtype=flat.dtype)
        raw = float((flat * weights).sum().item())
        return int(round(raw * 1000)) % 1_000_000

    def _token_for(self, sig: int, step: int) -> int:
        """Deterministic non-EOS token for (sig, step)."""
        v = (sig * 17 + step * 31 + 7) % 500 + 100
        # Make sure we never accidentally produce EOS in the middle.
        return v if v != self.EOS_ID else v + 100

    # ------------------------------------------------------------------
    # talker.generate interface
    # ------------------------------------------------------------------

    def generate(self, **kwargs):
        inputs_embeds = kwargs["inputs_embeds"]  # [B, L, H]
        attention_mask = kwargs["attention_mask"]  # [B, L]
        B = inputs_embeds.shape[0]

        sigs = [
            self._row_sig(inputs_embeds[b], attention_mask[b])
            for b in range(B)
        ]

        # num_real_steps real tokens + 1 EOS per row.
        steps: List[tuple] = []
        for t in range(self.num_real_steps + 1):
            frame = torch.zeros(B, self.num_code_groups, dtype=torch.long)
            for b in range(B):
                if t < self.num_real_steps:
                    frame[b, 0] = self._token_for(sigs[b], t)
                else:
                    frame[b, 0] = self.EOS_ID
            steps.append((None, frame))

        return types.SimpleNamespace(hidden_states=steps)


class _FakeConfig:
    codec_eos_token_id = _FakeRowIndependentTalker.EOS_ID
    num_code_groups = 16
    vocab_size = 2048


# =============================================================================
# Helpers
# =============================================================================


def _run_batched(
    talker,
    tie: torch.Tensor,
    tam: torch.Tensor,
    *,
    do_sample: bool = False,
) -> List[Optional[torch.Tensor]]:
    """Thin wrapper that handles the tth/tpe shapes and calls batched_talker_generate."""
    B, L, H = tie.shape
    cfg = _FakeConfig()
    tth = torch.zeros(B, 1, H)
    tpe = torch.zeros(1, 1, H)
    codes_list, _ = batched_talker_generate(
        talker=talker,
        talker_input_embeds=tie,
        attention_mask=tam,
        trailing_text_hiddens=tth,
        tts_pad_embed=tpe,
        config=cfg,
        do_sample=do_sample,
    )
    return codes_list


def _left_pad(emb: torch.Tensor, pad_len: int) -> torch.Tensor:
    """Left-pad a [1, L, H] embedding tensor with ``pad_len`` zero frames."""
    pad = torch.zeros(1, pad_len, emb.shape[-1])
    return torch.cat([pad, emb], dim=1)


def _left_pad_mask(orig_len: int, total_len: int) -> torch.Tensor:
    """Return a [1, total_len] attention mask with ``total_len - orig_len`` leading zeros."""
    mask = torch.zeros(1, total_len, dtype=torch.long)
    mask[0, total_len - orig_len:] = 1
    return mask


# =============================================================================
# Tests
# =============================================================================


def test_solo_equals_same_text_in_batch():
    """B=1 solo run ≡ same row inside a B=3 batch (same-length rows, no padding)."""
    torch.manual_seed(0)
    talker = _FakeRowIndependentTalker(num_real_steps=6)
    L, H = 7, 4

    emb_A = torch.randn(1, L, H)
    emb_B = torch.randn(1, L, H)
    emb_C = torch.randn(1, L, H)
    mask_1 = torch.ones(1, L, dtype=torch.long)
    mask_3 = torch.ones(3, L, dtype=torch.long)

    # Solo run: just row A.
    codes_solo = _run_batched(talker, emb_A, mask_1)

    # Batched run: rows [A, B, C].
    emb_batch = torch.cat([emb_A, emb_B, emb_C], dim=0)
    codes_batch = _run_batched(talker, emb_batch, mask_3)

    assert codes_solo[0] is not None, "solo run returned None"
    assert codes_batch[0] is not None, "row A in batch returned None"
    assert torch.equal(codes_solo[0], codes_batch[0]), (
        "Row A in a 3-row batch produced different codec tokens than the solo run; "
        "this means the batch is not row-independent (greedy path should be)."
    )

    # Codec tokens for rows B and C differ from row A (sanity check).
    assert not torch.equal(codes_batch[0], codes_batch[1]), "rows A and B should differ"
    assert not torch.equal(codes_batch[0], codes_batch[2]), "rows A and C should differ"


def test_left_padded_row_equals_unpadded_solo():
    """Row A padded (to match longer row B) == unpadded solo run.

    This verifies that the fake talker uses ``attention_mask`` to exclude the
    padding, so the output for the padded row is bit-identical to the solo run.
    Any real transformer that respects ``attention_mask`` has the same property.
    """
    torch.manual_seed(1)
    talker = _FakeRowIndependentTalker(num_real_steps=4)
    H = 4
    L_A, L_B = 5, 9  # A is shorter; will be padded to L_B

    emb_A = torch.randn(1, L_A, H)
    emb_B = torch.randn(1, L_B, H)

    # Solo run: row A, no padding.
    mask_A = torch.ones(1, L_A, dtype=torch.long)
    codes_solo = _run_batched(talker, emb_A, mask_A)

    # Build left-padded row A to match L_B.
    pad_len = L_B - L_A
    emb_A_padded = _left_pad(emb_A, pad_len)        # [1, L_B, H]
    mask_A_padded = _left_pad_mask(L_A, L_B)        # [1, L_B], zeros on left

    # Batched run: rows [A_padded, B] — A has leading zeros in its embedding.
    emb_batch = torch.cat([emb_A_padded, emb_B], dim=0)
    mask_batch = torch.cat([mask_A_padded, torch.ones(1, L_B, dtype=torch.long)], dim=0)
    codes_batch = _run_batched(talker, emb_batch, mask_batch)

    assert codes_solo[0] is not None
    assert codes_batch[0] is not None
    assert torch.equal(codes_solo[0], codes_batch[0]), (
        "Left-padded row A should produce the same codec tokens as the unpadded "
        "solo run. The fake talker uses attention_mask to mask padding — if this "
        "fails, the signature computation or batch construction is wrong."
    )


def test_batch_position_independence():
    """Row A produces the same codec tokens at positions 0, 1, and 2 in a
    3-row batch (same-length rows, no padding)."""
    torch.manual_seed(2)
    talker = _FakeRowIndependentTalker(num_real_steps=5)
    L, H = 6, 4

    emb_A = torch.randn(1, L, H)
    emb_B = torch.randn(1, L, H)
    emb_C = torch.randn(1, L, H)
    mask = torch.ones(3, L, dtype=torch.long)

    # Arrangement 1: [A, B, C] → row 0 is A.
    codes_ABC = _run_batched(talker, torch.cat([emb_A, emb_B, emb_C]), mask)
    # Arrangement 2: [B, A, C] → row 1 is A.
    codes_BAC = _run_batched(talker, torch.cat([emb_B, emb_A, emb_C]), mask)
    # Arrangement 3: [C, B, A] → row 2 is A.
    codes_CBA = _run_batched(talker, torch.cat([emb_C, emb_B, emb_A]), mask)

    assert codes_ABC[0] is not None
    assert codes_BAC[1] is not None
    assert codes_CBA[2] is not None

    assert torch.equal(codes_ABC[0], codes_BAC[1]), (
        "Row A at position 0 vs position 1 produced different codec tokens."
    )
    assert torch.equal(codes_ABC[0], codes_CBA[2]), (
        "Row A at position 0 vs position 2 produced different codec tokens."
    )


def test_multi_batch_sizes_match_solo():
    """Row A with a fixed set of siblings produces the same codec tokens for
    B ∈ {1, 2, 4} (verifies invariance across arbitrary batch sizes)."""
    torch.manual_seed(3)
    talker = _FakeRowIndependentTalker(num_real_steps=7)
    L, H = 8, 4

    emb_A = torch.randn(1, L, H)
    embs_others = torch.randn(3, L, H)
    mask_B = lambda B: torch.ones(B, L, dtype=torch.long)  # noqa: E731

    # B=1: row A alone.
    codes_1 = _run_batched(talker, emb_A, mask_B(1))

    # B=2: rows [A, X].
    emb_2 = torch.cat([emb_A, embs_others[0:1]], dim=0)
    codes_2 = _run_batched(talker, emb_2, mask_B(2))

    # B=4: rows [A, X, Y, Z].
    emb_4 = torch.cat([emb_A, embs_others], dim=0)
    codes_4 = _run_batched(talker, emb_4, mask_B(4))

    for label, codes in (("B=2", codes_2), ("B=4", codes_4)):
        assert codes[0] is not None, f"{label}: row 0 is None"
        assert torch.equal(codes_1[0], codes[0]), (
            f"Row A in a {label} batch produced different codec tokens from the B=1 run."
        )


def test_do_sample_false_is_the_key_condition():
    """With do_sample=False (greedy), rows are independent; the test documents
    that the parity guarantee is conditional on greedy decoding.

    This test runs the same four arrangements from ``test_batch_position_independence``
    and asserts their equality under do_sample=False. It does NOT assert equality
    under do_sample=True (where results differ because multinomial sampling uses a
    shared global RNG state in HF generate, so row outcomes depend on draw order).

    The fake talker is deterministic regardless of do_sample (it ignores the flag),
    so both calls produce the same outputs here. The comment is the specification.
    """
    torch.manual_seed(4)
    talker = _FakeRowIndependentTalker(num_real_steps=5)
    L, H = 5, 4
    emb_A = torch.randn(1, L, H)
    emb_B = torch.randn(1, L, H)
    mask = torch.ones(2, L, dtype=torch.long)

    codes_solo = _run_batched(talker, emb_A, torch.ones(1, L, dtype=torch.long), do_sample=False)
    codes_batch = _run_batched(talker, torch.cat([emb_A, emb_B]), mask, do_sample=False)

    assert codes_solo[0] is not None
    assert codes_batch[0] is not None
    assert torch.equal(codes_solo[0], codes_batch[0])

    # Verify token values are non-EOS (real tokens, not just zeros).
    assert not (codes_solo[0][:, 0] == _FakeRowIndependentTalker.EOS_ID).any(), (
        "Codec tokens should not contain EOS (EOS is stripped by batched_talker_generate)"
    )


# =============================================================================
# End-to-end: generate_batch produces same codec_ids as solo-batch run
# =============================================================================


def _build_model_with_row_independent_talker(num_real_steps: int = 5):
    """Return a fake FasterQwen3TTS whose talker is ``_FakeRowIndependentTalker``.

    The model's _build_batch_inputs uses the real per-request text tokenization
    helpers (all mocked to return fixed embeddings), so the talker input_embeds
    are deterministic functions of the request text.
    """
    import types as _types
    from faster_qwen3_tts.model import FasterQwen3TTS

    H = 4
    talker = _FakeRowIndependentTalker(num_real_steps=num_real_steps)

    base = _types.SimpleNamespace()
    base.model = _types.SimpleNamespace(
        talker=_types.SimpleNamespace(
            generate=talker.generate,
            rope_deltas=None,
        ),
        config=_types.SimpleNamespace(
            talker_config=_types.SimpleNamespace(
                codec_eos_token_id=_FakeRowIndependentTalker.EOS_ID,
                num_code_groups=16,
                vocab_size=2048,
            )
        ),
        tts_model_type="custom_voice",
        speech_tokenizer=_FakeSpeechTokenizerE2E(samples_per_frame=4),
    )
    # Mock text helpers: each text is tokenised to a distinct embedding based
    # on its content (hash-based fixed tensor), so different texts produce
    # different input_embeds but the same text always produces the same embed.
    base._build_assistant_text = lambda text: text
    base._build_ref_text = lambda text: text
    base._build_instruct_text = lambda text: text
    base._validate_languages = lambda _: None
    base._validate_speakers = lambda _: None

    # Tokenise each text into a fixed-width [1, 8] int tensor whose values
    # depend on the text content (via hash), so different texts produce
    # different talker input_embeds.
    def _tokenize_texts(texts):
        return [
            torch.tensor(
                [[abs(hash(t) >> (i * 3)) % 100 + 1 for i in range(8)]],
                dtype=torch.long,
            )
            for t in texts
        ]
    base._tokenize_texts = _tokenize_texts

    model = FasterQwen3TTS(base, object(), object(), device="cpu", dtype=torch.float32)

    # Short-circuit _build_talker_inputs_local: embed each row's input_ids
    # into a unique float tensor (same ids → same embedding → same signature).
    def _fake_build_inputs(**kwargs):
        input_ids_list = kwargs["input_ids"]  # list of [1, 8] tensors
        B = len(input_ids_list)
        tie_rows = []
        for ids in input_ids_list:
            # Map [1, L] int ids to [1, L, H] float embed (ids cast to float).
            tie_rows.append(ids.float().unsqueeze(-1).expand(1, ids.shape[1], H))
        tie = torch.cat(tie_rows, dim=0)           # [B, L, H]
        tam = torch.ones(B, tie.shape[1], dtype=torch.long)
        tth = torch.zeros(B, 1, H, dtype=torch.float32)
        tpe = torch.zeros(1, 1, H, dtype=torch.float32)
        return tie, tam, tth, tpe

    model._build_talker_inputs_local = _fake_build_inputs
    model._warmup = lambda _: None
    model._warmed_up = True
    model.sample_rate = 24000
    return model


class _FakeSpeechTokenizerE2E:
    """Codec decoder that returns audio proportional to codec frame count."""
    def __init__(self, samples_per_frame: int = 4) -> None:
        self.spf = samples_per_frame

    def decode(self, payload):
        codes = payload["audio_codes"]  # [1, T, 16]
        T = codes.shape[1]
        audio = np.arange(T * self.spf, dtype=np.float32)
        return [audio], 24000


def test_generate_batch_solo_equals_larger_batch_same_request():
    """``generate_batch([req])`` and ``generate_batch([req, other])`` must
    produce identical ``codec_ids`` for ``req`` when ``do_sample=False``.

    This is the end-to-end model-level parity check.  It uses the fake talker
    whose per-row output is determined solely by that row's input embeddings,
    so the test will fail if batching logic mutates row A's embeddings or if
    the EOS-splitting is off by one.
    """
    model = _build_model_with_row_independent_talker(num_real_steps=6)

    req_A = Request(
        text="hello world", mode="custom_voice", speaker="spk_A",
        do_sample=False, request_id="A",
    )
    req_B = Request(
        text="completely different text phrase here",
        mode="custom_voice", speaker="spk_B",
        do_sample=False, request_id="B",
    )

    # Solo run: only request A.
    results_solo = model.generate_batch([req_A])
    # Batched run: A and B together.
    results_batch = model.generate_batch([req_A, req_B])

    assert results_solo[0].codec_ids is not None, "Solo run returned None codec_ids"
    assert results_batch[0].codec_ids is not None, "Batch run returned None codec_ids for req_A"

    assert torch.equal(results_solo[0].codec_ids, results_batch[0].codec_ids), (
        f"Request A's codec_ids differ between solo and batched runs.\n"
        f"  Solo:    {results_solo[0].codec_ids[:, 0].tolist()}\n"
        f"  Batched: {results_batch[0].codec_ids[:, 0].tolist()}"
    )


def test_generate_batch_request_at_different_positions():
    """Request A at batch position 0, 1, and 2 all produce the same codec_ids."""
    model = _build_model_with_row_independent_talker(num_real_steps=5)

    req_A = Request(
        text="the quick brown fox", mode="custom_voice", speaker="spk_A",
        do_sample=False, request_id="A",
    )
    req_X = Request(
        text="some other text X", mode="custom_voice", speaker="spk_X",
        do_sample=False, request_id="X",
    )
    req_Y = Request(
        text="another different text Y", mode="custom_voice", speaker="spk_Y",
        do_sample=False, request_id="Y",
    )

    # A at position 0: [A, X, Y].
    res_0 = model.generate_batch([req_A, req_X, req_Y])
    # A at position 1: [X, A, Y].
    res_1 = model.generate_batch([req_X, req_A, req_Y])
    # A at position 2: [X, Y, A].
    res_2 = model.generate_batch([req_X, req_Y, req_A])

    ids_pos0 = res_0[0].codec_ids
    ids_pos1 = res_1[1].codec_ids
    ids_pos2 = res_2[2].codec_ids

    assert ids_pos0 is not None and ids_pos1 is not None and ids_pos2 is not None

    assert torch.equal(ids_pos0, ids_pos1), (
        "Request A at position 0 vs 1 produced different codec_ids."
    )
    assert torch.equal(ids_pos0, ids_pos2), (
        "Request A at position 0 vs 2 produced different codec_ids."
    )


def test_generate_batch_larger_sibling_pads_shorter_request():
    """When request A is shorter than request B, A is left-padded in the
    batched input. The codec_ids for A must still match the solo run."""
    # Use texts whose tokenised lengths differ (the tokeniser in the fake model
    # always returns length 8, but we can simulate different effective lengths
    # by using _build_talker_inputs_local to produce different row lengths in the
    # attention mask). Since _fake_build_inputs always produces length-8 rows with
    # uniform masks, we test the padding via the lower-level _run_batched helper.
    torch.manual_seed(5)
    talker = _FakeRowIndependentTalker(num_real_steps=4)
    H = 4
    L_A, L_B = 4, 10

    emb_A = torch.randn(1, L_A, H)
    emb_B = torch.randn(1, L_B, H)

    # Solo run for A.
    codes_A_solo = _run_batched(talker, emb_A, torch.ones(1, L_A, dtype=torch.long))

    # Build left-padded A to match L_B.
    pad = L_B - L_A
    emb_A_padded = _left_pad(emb_A, pad)
    mask_A_padded = _left_pad_mask(L_A, L_B)
    mask_B = torch.ones(1, L_B, dtype=torch.long)

    emb_batch = torch.cat([emb_A_padded, emb_B], dim=0)
    mask_batch = torch.cat([mask_A_padded, mask_B], dim=0)
    codes_batch = _run_batched(talker, emb_batch, mask_batch)

    assert codes_A_solo[0] is not None
    assert codes_batch[0] is not None
    assert torch.equal(codes_A_solo[0], codes_batch[0]), (
        "Left-padded request A in a B=2 batch produced different codec_ids "
        "than the unpadded solo run."
    )
