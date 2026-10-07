# CLAUDE.md

Context for Claude Code working in this repo.

## What this is

Model Watch lets a person watch Gemma-2 2B generate text token by token and see why each token was chosen: the final choice and runners-up, a per-layer logit lens, and the top Gemma Scope SAE features at one layer with Neuronpedia labels.

The owner wants to **observe the model while it runs**, not ablate or steer it. Keep the default experience observe-only. Interventions can be added later as a separate, opt-in feature.

## Commands

```bash
pip install -r requirements.txt            # or: pip install -e .
python -m unittest discover -s tests -v    # no downloads, runs on a tiny random Gemma 2
python watch.py --prompt "The capital of France is"   # needs HF_TOKEN and Gemma license accepted
python watch.py --prompt "..." --html run.html --json run.json
```

## Layout

- `model_watch/core.py`: `WatchConfig`, `ModelWatcher` (`load`, `step`, `watch`, `trace`), `settled_layer`.
- `model_watch/sae.py`: `JumpReLUSAE` loading Gemma Scope `params.npz` (keys `W_enc, W_dec, b_enc, b_dec, threshold`).
- `model_watch/labels.py`: `NeuronpediaLabels`, best-effort labels with a disk cache in `~/.cache/model-watch` (override with `MODEL_WATCH_CACHE`).
- `model_watch/render.py`: `format_step` (terminal) and `step_html` (Jupyter/Colab live panel).
- `model_watch/export.py`: `save_json`, `export_html`, `trace_to_html` (injects JSON into `viewer.html` at the `/*__TRACE_JSON__*/null` placeholder).
- `model_watch/viewer.html`: single-file replay viewer. Opened directly, it shows an illustrative sample with a banner.
- `watch.py`: CLI. `notebooks/model_watch_colab.ipynb`: Colab walkthrough (generated; edit carefully, it is plain nbformat JSON).

## Decisions and gotchas

- **No TransformerLens.** TransformerLens 4.0 removed `HookedTransformer.from_pretrained`. We use plain `transformers` with forward hooks on each decoder layer, which captures the residual stream after each layer (resid_post) at the last position.
- **Do not use `output_hidden_states` for the last layer.** In HF Gemma 2 the final `hidden_states` entry already has the final norm applied; the others do not. Hooks avoid the mismatch. A test checks hooked outputs equal `hidden_states[1:-1]`.
- **Logit lens** = `final_norm` -> `lm_head` -> Gemma's final logit soft-cap (`config.final_logit_softcapping`, 30.0) -> softmax. At the last layer it matches the real prediction exactly (tested).
- Load Gemma 2 with `attn_implementation="eager"`. `transformers` renamed `torch_dtype` to `dtype` in 4.56; `_dtype_kwarg()` handles both.
- `dtype="auto"` picks bfloat16 on GPUs that support it, else float32. Avoid float16 for Gemma 2 (overflow).
- **SAE choice.** Default is `google/gemma-scope-2b-pt-res`, `layer_20/width_16k/average_l0_71`. The repo's `canonical` folders were deleted. Neuronpedia's `20-gemmascope-res-16k` is assumed to match L0 71 (closest to 100); this was not verified from source. If labels look unrelated to what fires, try other L0 values.
- **Neuronpedia API**: `GET https://www.neuronpedia.org/api/feature/{model}/{source}/{index}`. The response schema was not verified; `parse_label` reads `explanations[0].description` with fallbacks. After 3 network failures labels switch off for the session. Failures are never cached.
- Gemma Scope SAEs were trained on the base model, so default to `google/gemma-2-2b`, not `-it`.
- Generation is greedy and re-runs the full sequence each step (no KV cache). Simple and fine for short prompts.
- Tests must keep running offline on the tiny random model. Do not add tests that download weights.

## Trace JSON (version 1)

```
{ version, created, meta: {model, sae, sae_layer, neuronpedia_source, device, dtype, n_layers},
  prompt, prompt_tokens: [str],
  steps: [ { index, token, token_id, prob,
             alternatives: [{token, token_id, prob}],
             lens: [{layer, top_token, top_prob, chosen_prob}],
             settled_layer,                 # int or null
             features: [{index, activation, label, url}],
             active_count } ] }
```

The viewer depends on these field names. If you change them, update `viewer.html` and bump `version`.

## Ideas for next steps

- KV-cache generation for speed on longer outputs.
- Features at several layers (for example 6, 12, 20) to show concepts forming over depth.
- A live local web viewer for the home lab (stream steps over a websocket instead of replaying after).
- Tuned lens for more faithful early-layer readings.
- Per-token attribution graphs via `circuit-tracer` (github.com/decoderesearch/circuit-tracer) for "pause and inspect this word".
- Side-by-side comparison of two prompts.
