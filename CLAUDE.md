# CLAUDE.md

Context for Claude Code working in this repo.

## What this is

Model Watch lets a person chat with a Gemma model (or a Qwen3 reasoning model) and watch each reply form token by token: the choice and runners-up, a per-layer logit lens, and SAE features at four layers (Gemma Scope for Gemma, Qwen-Scope for `qwen3-1.7b`). A one-screen replay viewer adds a whole-reply concept timeline that can highlight where any concept was active in the text.

The owner wants to **observe the model while it runs**, not ablate or steer it, and wants everything **free to run** (Colab T4, Kaggle 2x T4, or CPU). Keep the default experience observe-only; interventions can be added later as an opt-in feature. The viewer must stay a **single screen** on laptops (no page scroll; panels scroll internally). The owner rejected a long scrolling layout.

## Commands

```bash
pip install -r requirements.txt            # or: pip install -e .
python -m unittest discover -s tests -v    # offline; test_model_watch.py: tiny random Gemma 2, Gemma 3 text, Gemma 3 multimodal and Qwen3; test_preview.py: no torch needed
python watch.py --prompt "Is AI a black box?"                   # needs HF_TOKEN + Gemma licenses accepted
python watch.py --interactive --compact --html chat.html
```

The notebook `notebooks/model_watch.ipynb` is generated plain nbformat JSON; edit carefully and keep every code cell parseable.

## Layout

- `model_watch/core.py`: `PRESETS`, `WatchConfig` (preset defaults fill any `None` field; `.resolved()`), `ModelWatcher` (`load`, `encode`, `step`, `watch`, `trace`, `reply_text`, `add_labels`, `promotes`), `settled_layer`, `feature_key`.
- `model_watch/sae.py`: `JumpReLUSAE` (despite the name, also TopK: pass `threshold` for JumpReLU or `k` for TopK); `from_gemma_scope_1` (npz), `from_gemma_scope_2` (safetensors), `from_qwen_scope` (`.pt`, loaded with `weights_only=True`); `normalize_params` accepts key spellings like `w_enc`/`W_enc` and fixes transposed matrices using `b_dec`'s length.
- `model_watch/labels.py`: `NeuronpediaLabels` (disk cache in `~/.cache/model-watch`, thread-safe, failures never cached, disables itself after 3 network failures).
- `model_watch/render.py`: `format_step` (terminal), `step_html` (notebook live panel), `describe`.
- `model_watch/export.py`: `save_json`, `export_html`, `trace_to_html` (injects JSON at `/*__TRACE_JSON__*/null`).
- `model_watch/viewer.html`: one-screen replay viewer; opened directly it shows a made-up sample (`meta.sample: true`) with a dismissible "Start here" guide (Play, tap a word, tap a concept, Open trace.json). Reads trace versions 1 and 2. The one-screen layout applies at 1000x620 and up; narrower windows (phones) stack the panels and the page scrolls. On touch screens (`pointer: coarse`) transport buttons, speed menu and file picker grow to 44px, pills to 36px, concept rows and search to 40px, and timeline rows to 30px (`--row`); anything shown on hover must also work on tap (tapping a timeline cell selects that concept, jumps to that word and shows its activation). The concepts panel says "Active here does not prove it caused the word"; keep it.
- `docs/index.html`: the preview README tells people to start with (GitHub Pages from `/docs`). It must be an exact copy of `viewer.html`; after editing the viewer run `cp model_watch/viewer.html docs/index.html` (`tests/test_preview.py` checks).
- `watch.py`: CLI (`--preset`, `--interactive`, `--compact`, `--quiet`, `--json`, `--html`, `--device-map auto`, `--sae-layers`; `--help` lists all).
- `tests/test_model_watch.py` (torch + transformers, tiny random models) and `tests/test_preview.py` (standard library only; docs copy and sample placeholder).

## Decisions and gotchas

- **No TransformerLens.** 4.0 removed `HookedTransformer` entirely (including `from_pretrained`; its replacement is `TransformerBridge`). Plain `transformers` with forward hooks on each decoder layer captures resid_post at the last position. This matches Gemma Scope 2's `hf_hook_point_in: model.layers.N.output`.
- **Finding layers.** Gemma 3 4B/12B load as multimodal `Gemma3ForConditionalGeneration` (layers at `model.language_model.layers`); 1B is text-only `Gemma3ForCausalLM`. `_decoder()` handles both. `load()` tries `AutoModelForCausalLM`, then `AutoModelForImageTextToText`.
- **Do not use `output_hidden_states` for the last layer**: HF stores it post-norm. Tests check hooked outputs equal `hidden_states[1:-1]`.
- **Logit lens** = final norm, then `lm_head`, then soft-cap if `get_text_config().final_logit_softcapping` is set (Gemma 2: 30; Gemma 3: none), then softmax. Matches the real prediction at the last layer (tested).
- **KV cache.** `watch(use_cache=True)` feeds only the new token after the first step. Tests confirm identical tokens, lens and features versus full recompute on all three tiny model types, including sliding-window attention.
- **Multi-GPU.** With `device_map="auto"`, hooks move captured activations to `self.home` (the `lm_head` device); SAEs live there too. Untested on real multi-GPU hardware.
- `dtype="auto"`: bfloat16 on any CUDA GPU (emulated on T4), float32 on CPU/MPS. Never float16 automatically.
- **Gemma Scope 2 paths**: `google/gemma-scope-2-{size}-it/resid_post/layer_{L}_width_16k_l0_medium/params.safetensors` (each folder also has a large `examples.safetensors` we do not download). Main `resid_post` has 4 layers per model; `resid_post_all` has every layer but only l0 small/big. Keys inside `params.safetensors` are `w_enc` [d, n], `w_dec`, `b_enc`, `b_dec`, `threshold` (checked against SAELens's `gemma_3_sae_huggingface_loader`, October 2026; not yet loaded from the real files here). `normalize_params` handles these and raises with the actual key list if none match.
- **Neuronpedia** sources verified to match the medium-L0 16k SAEs: `gemma-3-4b-it/17-gemmascope-2-res-16k`, `gemma-3-1b-it/13-...`, `gemma-3-12b-it/12-...`. Many 4B features have no explanation yet, which is why `promotes` (top tokens of `W_dec[f] @ W_U`) exists as a free label at every layer. API: `GET /api/feature/{model}/{source}/{index}`; the route returns the feature with `explanations[].description` (checked against Neuronpedia's source, October 2026); `parse_label` reads `explanations[0].description` with fallbacks. SAELens also maps the other Gemma Scope 2 layers (e.g. 4B 9/22/29, 1B 7/17/22) to Neuronpedia sources; whether they have explanations is unchecked, so presets only query one layer.
- Labels are fetched after generation by `add_labels` (parallel, capped, most-active first) so live display never waits on the network.
- Stop tokens: tokenizer EOS plus `<end_of_turn>` and `<eos>` when the tokenizer has them. Steps carry `stop: true`; `reply` excludes them.
- **Qwen3 reasoning presets.** `qwen3-1.7b` uses Qwen-Scope (`Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50/layer{L}.sae.pt`): TopK k=50 read from the repo name, hook = `model.layers.L` output (same site as Gemma Scope 2), `W_enc` stored [n, d] and `W_dec` [d, n] (transposed by `normalize_params`), no threshold; trained on the Base model; no Neuronpedia source. Format from SAELens's `qwen_scope_sae_huggingface_loader`, not from the real files. `qwen3-4b` has `sae_layers=()` (no dictionaries); `load()` refuses `--sae-layers` on such presets. `WatchConfig.thinking` / `ModelWatcher.thinking` passes `enable_thinking` to the chat template only when not `None`. Stop names also include `<|im_end|>` and `<|endoftext|>`. watch.py defaults to 1000 tokens when thinking is on.
- Tests must stay offline on tiny random models. Do not add tests that download weights.

## Trace JSON (version 2)

```
{ version: 2, created, prompt, messages|null, prompt_tokens, reply,
  meta: {preset, model, sae, sae_layers, labeled_layers, neuronpedia_model, neuronpedia_sources, device, dtype, chat, thinking, n_layers},
        # preset, model, sae, neuronpedia_*, device, dtype come from ModelWatcher.load(); the viewer's built-in sample adds sample: true
  steps: [ { index, token, token_id, prob, stop,
             alternatives: [{token, token_id, prob}],
             lens: [{layer, top_token, top_prob, chosen_prob}],
             settled_layer,                               # int or null
             features: {"<layer>": [{index, activation}]},
             active_count: {"<layer>": int} } ],
  feature_info: {"<layer>:<index>": {layer, index, label|null, promotes: [str], url|null}} }
```

The viewer depends on these names; `normalize()` in `viewer.html` upgrades version 1 traces. If you change the schema, update the viewer and bump `version`.

## Known gaps

README.md has a "Known gaps" section written for the owner: it explains why closed models (Claude, ChatGPT, Copilot, Gemini) can't be watched, how the scale compares with lab tools, and what's unverified. Write it plainly, without jargon or hype. When a gap is closed (for example after the first real-model run, or after adding attribution graphs), update that section in the same change.

## Ideas for next steps

- Live streaming into the full viewer (local web server with server-sent events; in Colab, `google.colab.output.serve_kernel_port_as_iframe`).
- Neuronpedia-labeled concepts for Qwen3: mwhanna's Qwen3 transcoders (0.6B–14B, in SAELens and circuit-tracer) have Neuronpedia sources like `qwen3-1.7b/{L}-transcoder-hp-lowl0`, but they read the MLP input (`post_attention_layernorm` output), not the layer output, and their width and memory cost weren't checked.
- Per-word attribution graphs via circuit-tracer (github.com/decoderesearch/circuit-tracer), which supports Gemma 3 transcoders from Gemma Scope 2.
- Use each SAE folder's `examples.safetensors` (top activating examples) as an offline label source.
- Tuned lens for more faithful early-layer readings.
- Side-by-side comparison of two prompts or two layers.
