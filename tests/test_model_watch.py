"""Runs the whole pipeline on a tiny random Gemma 2 and a random SAE. No downloads.

python -m unittest discover -s tests -v
"""
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

os.environ.setdefault("MODEL_WATCH_CACHE", tempfile.mkdtemp(prefix="mw-cache-"))

from transformers import Gemma2Config, Gemma2ForCausalLM  # noqa: E402

from model_watch import (  # noqa: E402
    JumpReLUSAE,
    ModelWatcher,
    NeuronpediaLabels,
    export_html,
    format_step,
    settled_layer,
    step_html,
    trace_to_html,
)
from model_watch.labels import parse_label  # noqa: E402
from model_watch.render import Style, bar, show_token  # noqa: E402

D, LAYERS, VOCAB, FEATURES = 32, 4, 97, 50


class FakeTokenizer:
    eos_token_id = 1

    def __call__(self, text, return_tensors="pt"):
        ids = [2] + [3 + (ord(c) % 90) for c in text]
        return SimpleNamespace(input_ids=torch.tensor([ids]))

    def decode(self, ids):
        return "".join(f" t{i}" for i in ids)


def tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = Gemma2Config(
        vocab_size=VOCAB, hidden_size=D, intermediate_size=64, num_hidden_layers=LAYERS,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=128,
        attn_implementation="eager",
    )
    model = Gemma2ForCausalLM(cfg).eval()
    # Gemma initializes norm weights to zero; give them values so the tests exercise the real norm.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.normal_(0, 0.3)
    return model


def random_sae(seed=0):
    g = torch.Generator().manual_seed(seed)
    return JumpReLUSAE(
        W_enc=torch.randn(D, FEATURES, generator=g),
        W_dec=torch.randn(FEATURES, D, generator=g),
        b_enc=torch.zeros(FEATURES),
        b_dec=torch.zeros(D),
        threshold=torch.full((FEATURES,), 0.5),
    )


def make_watcher(**kw):
    labeler = NeuronpediaLabels("test-model", "2-test-sae", enabled=False, log=lambda *_: None)
    return ModelWatcher(tiny_model(), FakeTokenizer(), random_sae(), sae_layer=2, labeler=labeler,
                        top_k_tokens=5, top_k_features=6, meta={"model": "tiny"}, **kw)


class TestSettledLayer(unittest.TestCase):
    def test_cases(self):
        self.assertEqual(settled_layer([5, 5, 5], 5), 0)
        self.assertEqual(settled_layer([1, 5, 2, 5, 5], 5), 3)
        self.assertEqual(settled_layer([1, 2, 5], 5), 2)
        self.assertIsNone(settled_layer([5, 5, 2], 5))


class TestWatcher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = make_watcher()
        cls.ids = cls.w.encode_prompt("hello world")

    def test_captured_layers_match_hidden_states(self):
        """Hooked layer outputs equal transformers' own hidden states (except the last, which HF stores post-norm)."""
        _, resid = self.w._forward(self.ids)
        with torch.no_grad():
            hs = self.w.model(self.ids, output_hidden_states=True).hidden_states
        self.assertEqual(resid.shape, (LAYERS, D))
        for layer in range(LAYERS - 1):
            torch.testing.assert_close(resid[layer], hs[layer + 1][0, -1])

    def test_last_layer_lens_equals_real_prediction(self):
        record, chosen = self.w.step(self.ids)
        with torch.no_grad():
            probs = torch.softmax(self.w.model(self.ids).logits[0, -1].float(), -1)
        self.assertEqual(chosen, int(probs.argmax()))
        self.assertAlmostEqual(record["lens"][-1]["chosen_prob"], float(probs[chosen]), places=4)
        self.assertAlmostEqual(record["prob"], float(probs[chosen]), places=4)
        self.assertEqual(record["settled_layer"] is not None, True)
        self.assertEqual(record["lens"][-1]["top_token"], record["token"])

    def test_record_shape(self):
        record, _ = self.w.step(self.ids)
        self.assertEqual(len(record["lens"]), LAYERS)
        self.assertEqual(len(record["alternatives"]), 5)
        probs = [a["prob"] for a in record["alternatives"]]
        self.assertEqual(probs, sorted(probs, reverse=True))
        for row in record["lens"]:
            self.assertGreaterEqual(row["top_prob"], row["chosen_prob"] - 1e-6)

    def test_features_sorted_and_bounded(self):
        record, _ = self.w.step(self.ids)
        feats = record["features"]
        self.assertLessEqual(len(feats), 6)
        self.assertGreaterEqual(record["active_count"], len(feats))
        acts = [f["activation"] for f in feats]
        self.assertEqual(acts, sorted(acts, reverse=True))
        self.assertTrue(all(a > 0 for a in acts))
        self.assertTrue(all(f["url"].startswith("https://www.neuronpedia.org/test-model/2-test-sae/") for f in feats))

    def test_features_match_manual_sae(self):
        _, resid = self.w._forward(self.ids)
        acts = self.w.sae.encode(resid[2].float().unsqueeze(0))[0]
        record, _ = self.w.step(self.ids)
        if record["features"]:
            self.assertEqual(record["features"][0]["index"], int(acts.argmax()))

    def test_trace_is_json_ready(self):
        trace = self.w.trace("hi", max_new_tokens=4, stop_at_eos=False)
        self.assertEqual(len(trace["steps"]), 4)
        self.assertEqual([s["index"] for s in trace["steps"]], [0, 1, 2, 3])
        self.assertEqual(trace["meta"]["n_layers"], LAYERS)
        json.dumps(trace)

    def test_generation_matches_greedy(self):
        trace = self.w.trace("abc", max_new_tokens=3, stop_at_eos=False)
        ids = self.w.encode_prompt("abc")
        for step in trace["steps"]:
            with torch.no_grad():
                nxt = int(self.w.model(ids).logits[0, -1].argmax())
            self.assertEqual(step["token_id"], nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]])], 1)

    def test_stops_at_eos(self):
        w = make_watcher()
        record, chosen = w.step(w.encode_prompt("x"))
        w.eos_ids = {chosen}
        self.assertEqual(len(list(w.watch("x", max_new_tokens=5))), 1)

    def test_sae_layer_zero_allowed(self):
        ModelWatcher(tiny_model(), FakeTokenizer(), random_sae(), sae_layer=0)
        with self.assertRaises(ValueError):
            ModelWatcher(tiny_model(), FakeTokenizer(), random_sae(), sae_layer=LAYERS)


class TestSAE(unittest.TestCase):
    def test_jump_threshold(self):
        sae = JumpReLUSAE(torch.eye(2), torch.eye(2), torch.zeros(2), torch.zeros(2), torch.tensor([0.5, 0.5]))
        out = sae.encode(torch.tensor([[0.4, 0.9]]))
        torch.testing.assert_close(out, torch.tensor([[0.0, 0.9]]))

    def test_npz_roundtrip(self):
        sae = random_sae()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "params.npz")
            np.savez(path, **{k: v.detach().numpy() for k, v in sae.state_dict().items()})
            loaded = JumpReLUSAE.from_npz(path)
        x = torch.randn(3, D)
        torch.testing.assert_close(loaded.encode(x), sae.encode(x))

    def test_npz_missing_key(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "params.npz")
            np.savez(path, W_enc=np.zeros((2, 2)))
            with self.assertRaises(ValueError):
                JumpReLUSAE.from_npz(path)


class TestLabels(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_label({"explanations": [{"description": " capital cities "}]}), "capital cities")
        self.assertEqual(parse_label({"explanations": []}), None)
        self.assertEqual(parse_label({"explanation": "x"}), "x")
        self.assertIsNone(parse_label(None))

    def test_disabled_makes_no_requests(self):
        lab = NeuronpediaLabels("m", "s", enabled=False, log=lambda *_: None)
        self.assertIsNone(lab.get(5))
        self.assertEqual(lab.url(5), "https://www.neuronpedia.org/m/s/5")

    def test_failures_disable_and_are_not_cached(self):
        lab = NeuronpediaLabels("m", "s-fail", enabled=True, timeout=0.01, max_failures=2, log=lambda *_: None)
        lab._fetch = lambda i: (False, None)
        lab.get(1)
        self.assertNotIn("1", lab._cache)


class TestRendering(unittest.TestCase):
    def test_helpers(self):
        self.assertEqual(show_token(" a\n"), "␣a↵")
        self.assertEqual(show_token(""), "∅")
        self.assertEqual(len(bar(0.37, 20)), 20)
        self.assertEqual(bar(1.0, 10), "█" * 10)

    def test_format_and_html(self):
        w = make_watcher()
        record, _ = w.step(w.encode_prompt("hi"))
        text = format_step(record, "hi", 2, style=Style(False))
        self.assertIn("How the guess formed", text)
        self.assertIn("Settled at layer", text)
        self.assertIn("<div", step_html(record, "hi", 2, n_steps=3))

    def test_export_escapes_script_breakout(self):
        trace = make_watcher().trace("hi", max_new_tokens=2, stop_at_eos=False)
        trace["steps"][0]["token"] = "</script><b>x"
        page = trace_to_html(trace)
        self.assertNotIn("</script><b>", page)
        self.assertNotIn("/*__TRACE_JSON__*/null", page)
        with tempfile.TemporaryDirectory() as d:
            path = export_html(trace, os.path.join(d, "t.html"))
            self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
