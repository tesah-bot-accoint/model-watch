"""Runs the whole pipeline on tiny random Gemma 2, Gemma 3 and Qwen3 models with random SAEs. No downloads.

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

from transformers import (  # noqa: E402
    Gemma2Config,
    Gemma2ForCausalLM,
    Gemma3Config,
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
    Gemma3TextConfig,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from model_watch import (  # noqa: E402
    PRESETS,
    JumpReLUSAE,
    ModelWatcher,
    NeuronpediaLabels,
    WatchConfig,
    describe,
    export_html,
    format_step,
    settled_layer,
    step_html,
    trace_to_html,
)
from model_watch.labels import parse_label  # noqa: E402
from model_watch.render import Style, bar, show_token  # noqa: E402
from model_watch.sae import normalize_params  # noqa: E402

D, LAYERS, VOCAB, FEATURES = 32, 6, 97, 50
TEXT = dict(vocab_size=VOCAB, hidden_size=D, intermediate_size=64, num_hidden_layers=LAYERS, num_attention_heads=2,
            num_key_value_heads=1, head_dim=16, max_position_embeddings=256, sliding_window=4)


class FakeTokenizer:
    eos_token_id = 1
    unk_token_id = 0

    def __call__(self, text, return_tensors="pt"):
        return SimpleNamespace(input_ids=torch.tensor([[2] + [3 + (ord(c) % 90) for c in text]]))

    def decode(self, ids):
        return "".join(f" t{i}" for i in ids)

    def convert_tokens_to_ids(self, name):
        return {"<end_of_turn>": 96}.get(name, 0)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True,
                            **template_kwargs):
        self.template_kwargs = template_kwargs
        ids = [2]
        for m in messages:  # like real chat formats, every finished turn ends with the end-of-turn marker (96)
            ids += self(f"[{m['role']}]{m['content']}").input_ids[0, 1:].tolist() + [96]
        if add_generation_prompt:
            ids += self("[model]").input_ids[0, 1:].tolist()
        return {"input_ids": torch.tensor([ids])}


def _random_norms(model):
    # Gemma initializes norm weights to zero; give them values so tests exercise the real norm.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.normal_(0, 0.3)
    return model.eval()


def tiny(kind="gemma2", seed=0):
    torch.manual_seed(seed)
    if kind == "gemma2":
        return _random_norms(Gemma2ForCausalLM(Gemma2Config(**TEXT, attn_implementation="eager")))
    if kind == "gemma3":
        return _random_norms(Gemma3ForCausalLM(Gemma3TextConfig(**TEXT, attn_implementation="eager")))
    if kind == "qwen3":
        return _random_norms(Qwen3ForCausalLM(Qwen3Config(**TEXT, attn_implementation="eager")))
    cfg = Gemma3Config(text_config=TEXT, mm_tokens_per_image=4, attn_implementation="eager",
                       vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                          num_attention_heads=2, image_size=28, patch_size=14))
    return _random_norms(Gemma3ForConditionalGeneration(cfg))


def random_sae(seed=0, threshold=0.5):
    g = torch.Generator().manual_seed(seed)
    return JumpReLUSAE(torch.randn(D, FEATURES, generator=g), torch.randn(FEATURES, D, generator=g),
                       torch.zeros(FEATURES), torch.zeros(D), torch.full((FEATURES,), threshold))


def random_topk_sae(seed=0, k=5):
    g = torch.Generator().manual_seed(seed)
    return JumpReLUSAE(torch.randn(D, FEATURES, generator=g), torch.randn(FEATURES, D, generator=g),
                       torch.zeros(FEATURES), torch.zeros(D), k=k)


def make_watcher(kind="gemma2", chat=False, layers=(1, 4), thinking=None):
    labelers = {4: NeuronpediaLabels("test-model", "4-test-sae", enabled=False, log=lambda *_: None)} if 4 in layers else {}
    make = random_topk_sae if kind == "qwen3" else random_sae
    return ModelWatcher(tiny(kind), FakeTokenizer(), {l: make(l) for l in layers}, labelers, chat=chat,
                        top_k_tokens=5, top_k_features=6, promote_tokens=4, meta={"model": kind}, thinking=thinking)


class TestSettledLayer(unittest.TestCase):
    def test_cases(self):
        self.assertEqual(settled_layer([5, 5, 5], 5), 0)
        self.assertEqual(settled_layer([1, 5, 2, 5, 5], 5), 3)
        self.assertEqual(settled_layer([1, 2, 5], 5), 2)
        self.assertIsNone(settled_layer([5, 5, 2], 5))


class TestConfig(unittest.TestCase):
    def test_presets_resolve(self):
        for name in PRESETS:
            cfg = WatchConfig(preset=name).resolved()
            self.assertTrue(cfg.model_id)
            if cfg.sae_layers:
                self.assertTrue(cfg.sae_repo and cfg.sae_format)
            for layer in cfg.neuronpedia_sources:
                self.assertIn(layer, cfg.sae_layers)

    def test_overrides_win(self):
        cfg = WatchConfig(preset="gemma-3-1b-it", sae_layers=(13,), chat=False).resolved()
        self.assertEqual(cfg.sae_layers, (13,))
        self.assertFalse(cfg.chat)

    def test_unknown_preset(self):
        with self.assertRaises(ValueError):
            WatchConfig(preset="nope").resolved()


class TestWatcher(unittest.TestCase):
    KINDS = ("gemma2", "gemma3", "gemma3-mm", "qwen3")

    def test_captured_layers_match_hidden_states(self):
        """Hooked layer outputs equal transformers' hidden states (except the last, which HF stores post-norm)."""
        for kind in self.KINDS:
            w = make_watcher(kind)
            ids = w.encode("hello world")
            _, resid, _ = w._forward(ids)
            with torch.no_grad():
                hs = w.model(input_ids=ids, output_hidden_states=True).hidden_states
            self.assertEqual(resid.shape, (LAYERS, D))
            for layer in range(LAYERS - 1):
                torch.testing.assert_close(resid[layer], hs[layer + 1][0, -1], msg=kind)

    def test_last_layer_lens_equals_real_prediction(self):
        for kind in self.KINDS:
            w = make_watcher(kind)
            ids = w.encode("hello world")
            record, chosen = w.step(ids)
            with torch.no_grad():
                probs = torch.softmax(w.model(input_ids=ids).logits[0, -1].float(), -1)
            self.assertEqual(chosen, int(probs.argmax()))
            self.assertAlmostEqual(record["lens"][-1]["chosen_prob"], float(probs[chosen]), places=4, msg=kind)
            self.assertEqual(record["lens"][-1]["top_token"], record["token"])
            self.assertIsNotNone(record["settled_layer"])

    def test_cache_matches_full_recompute(self):
        for kind in self.KINDS:
            w = make_watcher(kind)
            fast = list(w.watch("a longer prompt here", 10, stop_at_eos=False, use_cache=True))
            slow = list(w.watch("a longer prompt here", 10, stop_at_eos=False, use_cache=False))
            self.assertEqual([s["token_id"] for s in fast], [s["token_id"] for s in slow], kind)
            for a, b in zip(fast, slow):
                for la, lb in zip(a["lens"], b["lens"]):
                    self.assertAlmostEqual(la["chosen_prob"], lb["chosen_prob"], places=4)
                self.assertEqual({k: [f["index"] for f in v] for k, v in a["features"].items()},
                                 {k: [f["index"] for f in v] for k, v in b["features"].items()})

    def test_generation_matches_greedy(self):
        w = make_watcher("gemma3")
        trace = w.trace("abc", max_new_tokens=4, stop_at_eos=False)
        ids = w.encode("abc")
        for step in trace["steps"]:
            with torch.no_grad():
                nxt = int(w.model(input_ids=ids).logits[0, -1].argmax())
            self.assertEqual(step["token_id"], nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]])], 1)

    def test_multi_layer_features(self):
        w = make_watcher("gemma3", layers=(0, 2, 5))
        record, _ = w.step(w.encode("hi there"))
        self.assertEqual(sorted(record["features"], key=int), ["0", "2", "5"])
        _, resid, _ = w._forward(w.encode("hi there"))
        for layer in (0, 2, 5):
            rows = record["features"][str(layer)]
            acts = [f["activation"] for f in rows]
            self.assertEqual(acts, sorted(acts, reverse=True))
            self.assertLessEqual(len(rows), 6)
            self.assertGreaterEqual(record["active_count"][str(layer)], len(rows))
            if rows:
                manual = w.saes[layer].encode(resid[layer].float().unsqueeze(0))[0]
                self.assertEqual(rows[0]["index"], int(manual.argmax()))

    def test_feature_info_and_promotes(self):
        w = make_watcher("gemma2")
        trace = w.trace("hi", 3, stop_at_eos=False)
        used = {f"{l}:{f['index']}" for s in trace["steps"] for l, rows in s["features"].items() for f in rows}
        self.assertEqual(set(trace["feature_info"]), used)
        some = next(iter(trace["feature_info"].values()))
        direction = w.saes[some["layer"]].W_dec[some["index"]]
        best = int((w.unembed.weight @ direction.to(w.unembed.weight.dtype)).argmax())
        self.assertEqual(some["promotes"][0], w._decode(best).strip())
        layer4 = [v for v in trace["feature_info"].values() if v["layer"] == 4]
        layer1 = [v for v in trace["feature_info"].values() if v["layer"] == 1]
        self.assertTrue(all(v["url"].startswith("https://www.neuronpedia.org/test-model/4-test-sae/") for v in layer4))
        self.assertTrue(all(v["url"] is None for v in layer1))

    def test_chat_mode(self):
        w = make_watcher("gemma3-mm", chat=True)
        self.assertIn(96, w.stop_ids)
        one = w.encode("hello")
        conv = w.encode([{"role": "user", "content": "hello"}])
        torch.testing.assert_close(one, conv)
        trace = w.trace([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"},
                         {"role": "user", "content": "why"}], 3, stop_at_eos=False)
        self.assertEqual(trace["prompt"], "why")
        self.assertEqual(len(trace["messages"]), 3)
        self.assertTrue(trace["meta"]["chat"])

    def test_plain_mode_rejects_messages(self):
        with self.assertRaises(ValueError):
            make_watcher().encode([{"role": "user", "content": "x"}])

    def test_stops_and_reply_text(self):
        w = make_watcher()
        record, chosen = w.step(w.encode("x"))
        w.stop_ids = {chosen}
        trace = w.trace("x", max_new_tokens=5)
        self.assertEqual(len(trace["steps"]), 1)
        self.assertTrue(trace["steps"][0]["stop"])
        self.assertEqual(trace["reply"], "")

    def test_trace_is_json_ready(self):
        trace = make_watcher("gemma3").trace("hi", max_new_tokens=4, stop_at_eos=False)
        self.assertEqual(trace["version"], 2)
        self.assertEqual([s["index"] for s in trace["steps"]], [0, 1, 2, 3])
        self.assertEqual(trace["meta"]["sae_layers"], [1, 4])
        self.assertEqual(trace["meta"]["labeled_layers"], [4])
        json.dumps(trace)

    def test_add_labels_uses_cache_and_limit(self):
        w = make_watcher()
        trace = w.trace("hi", 3, stop_at_eos=False)
        lab = w.labelers[4]
        lab.enabled = True
        calls = []
        lab._fetch = lambda i: (calls.append(i), (True, f"concept {i}"))[1]
        n = w.add_labels(trace, max_lookups=2, workers=2, log=lambda *_: None)
        self.assertEqual(n, len(calls))
        self.assertLessEqual(n, 2)
        labeled = [v for v in trace["feature_info"].values() if v["label"]]
        self.assertEqual(len(labeled), n)
        self.assertTrue(all(v["layer"] == 4 for v in labeled))

    def test_thinking_reaches_chat_template(self):
        w = make_watcher("qwen3", chat=True, thinking=False)
        w.encode("hi")
        self.assertEqual(w.tok.template_kwargs, {"enable_thinking": False})
        plain = make_watcher("gemma3", chat=True)
        plain.encode("hi")
        self.assertEqual(plain.tok.template_kwargs, {})

    def test_no_saes(self):
        """A preset without concept dictionaries still records the choice and the per-layer guesses."""
        w = make_watcher("qwen3", chat=True, layers=())
        trace = w.trace("hi", 3, stop_at_eos=False)
        self.assertEqual(trace["meta"]["sae_layers"], [])
        self.assertEqual(trace["feature_info"], {})
        self.assertTrue(all(s["features"] == {} and len(s["lens"]) == LAYERS for s in trace["steps"]))
        json.dumps(trace)
        self.assertIn("Chose", format_step(trace["steps"][0], "", w.feature_info, style=Style(False)))
        self.assertIn("Watching", step_html(trace["steps"][0], "", w.feature_info))
        self.assertIn('"sae_layers": []', trace_to_html(trace))

    def test_bad_sae_layer(self):
        with self.assertRaises(ValueError):
            ModelWatcher(tiny(), FakeTokenizer(), {LAYERS: random_sae()})


class TestSAE(unittest.TestCase):
    def test_jump_threshold(self):
        sae = JumpReLUSAE(torch.eye(2), torch.eye(2), torch.zeros(2), torch.zeros(2), torch.tensor([0.5, 0.5]))
        torch.testing.assert_close(sae.encode(torch.tensor([[0.4, 0.9]])), torch.tensor([[0.0, 0.9]]))

    def test_topk(self):
        sae = JumpReLUSAE(torch.eye(3), torch.eye(3), torch.zeros(3), torch.zeros(3), k=2)
        torch.testing.assert_close(sae.encode(torch.tensor([[0.5, -1.0, 2.0], [-3.0, -1.0, -2.0]])),
                                   torch.tensor([[0.5, 0.0, 2.0], [0.0, 0.0, 0.0]]))
        with self.assertRaises(ValueError):
            JumpReLUSAE(torch.eye(2), torch.eye(2), torch.zeros(2), torch.zeros(2))

    def test_qwen_scope_layout(self):
        """Qwen-Scope stores W_enc as [n, d] and W_dec as [d, n] with no threshold (per SAELens's loader)."""
        sae = random_topk_sae()
        raw = {"W_enc": sae.W_enc.T.contiguous(), "W_dec": sae.W_dec.T.contiguous(),
               "b_enc": sae.b_enc.clone(), "b_dec": sae.b_dec.clone()}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "layer3.sae.pt")
            torch.save({k: v.detach() for k, v in raw.items()}, path)
            loaded = JumpReLUSAE.from_params(torch.load(path, weights_only=True), k=5)
        x = torch.randn(3, D)
        torch.testing.assert_close(loaded.encode(x), sae.encode(x))
        self.assertEqual(int((loaded.encode(x) > 0).sum(-1).max()), 5)
        with self.assertRaises(ValueError):
            JumpReLUSAE.from_params(raw)  # no threshold and no k

    def test_npz_roundtrip(self):
        sae = random_sae()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "params.npz")
            np.savez(path, **{k: v.detach().numpy() for k, v in sae.state_dict().items()})
            loaded = JumpReLUSAE.from_npz(path)
        x = torch.randn(3, D)
        torch.testing.assert_close(loaded.encode(x), sae.encode(x))

    def test_safetensors_lowercase_and_transposed(self):
        from safetensors.torch import save_file

        sae = random_sae()
        raw = {"w_enc": sae.W_enc.T.contiguous(), "w_dec": sae.W_dec.T.contiguous(), "b_enc": sae.b_enc.clone(),
               "b_dec": sae.b_dec.clone(), "threshold": sae.threshold.clone()}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "params.safetensors")
            save_file({k: v.detach() for k, v in raw.items()}, path)
            loaded = JumpReLUSAE.from_safetensors(path)
        x = torch.randn(3, D)
        torch.testing.assert_close(loaded.encode(x), sae.encode(x))

    def test_missing_and_mismatched(self):
        with self.assertRaises(ValueError):
            normalize_params({"W_enc": np.zeros((2, 2))})
        with self.assertRaises(ValueError):
            normalize_params({"W_enc": np.zeros((4, 3)), "W_dec": np.zeros((5, 4)), "b_enc": np.zeros(3),
                              "b_dec": np.zeros(4), "threshold": np.zeros(3)})


class TestLabels(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_label({"explanations": [{"description": " capital cities "}]}), "capital cities")
        self.assertIsNone(parse_label({"explanations": []}))
        self.assertEqual(parse_label({"explanation": "x"}), "x")
        self.assertIsNone(parse_label(None))

    def test_disabled_makes_no_requests(self):
        lab = NeuronpediaLabels("m", "s", enabled=False, log=lambda *_: None)
        self.assertIsNone(lab.get(5))
        self.assertEqual(lab.url(5), "https://www.neuronpedia.org/m/s/5")

    def test_failures_not_cached(self):
        lab = NeuronpediaLabels("m", "s-fail", enabled=True, max_failures=2, log=lambda *_: None)
        lab._fetch = lambda i: (False, None)
        lab.get(1)
        self.assertFalse(lab.is_cached(1))

    def test_describe(self):
        self.assertEqual(describe({"label": "dogs", "promotes": ["x"]}), "dogs")
        self.assertEqual(describe({"label": None, "promotes": ["a", "b"]}), "pushes toward: a, b")
        self.assertEqual(describe(None), "")


class TestRendering(unittest.TestCase):
    def test_helpers(self):
        self.assertEqual(show_token(" a\n"), "␣a↵")
        self.assertEqual(show_token(""), "∅")
        self.assertEqual(len(bar(0.37, 20)), 20)
        self.assertEqual(bar(1.0, 10), "█" * 10)

    def test_format_and_html(self):
        w = make_watcher("gemma3")
        record, _ = w.step(w.encode("hi"))
        text = format_step(record, "hi", w.feature_info, style=Style(False))
        self.assertIn("How the guess formed", text)
        self.assertIn("Concepts at layer 4", text)
        compact = format_step(record, "hi", w.feature_info, style=Style(False), show_layers=False)
        self.assertNotIn("How the guess formed", compact)
        self.assertIn("<div", step_html(record, "hi", w.feature_info, n_steps=3))

    def test_export_escapes_script_breakout(self):
        trace = make_watcher().trace("hi", max_new_tokens=2, stop_at_eos=False)
        trace["steps"][0]["token"] = "</script><b>x"
        page = trace_to_html(trace)
        self.assertNotIn("</script><b>", page)
        self.assertNotIn("/*__TRACE_JSON__*/null", page)
        with tempfile.TemporaryDirectory() as d:
            self.assertGreater(export_html(trace, os.path.join(d, "t.html")).stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
