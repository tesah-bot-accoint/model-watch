# Model Watch

Watch Gemma-2 2B write, one word at a time, and see why each word was chosen.

For every word the model generates, Model Watch shows three things:

1. **The choice.** The word picked, its probability, and the runners-up.
2. **How the guess formed.** Each of the model's 26 layers, read as if the model stopped there (the "logit lens"). You see the answer firm up and the layer where it **settled**, meaning it became the top guess and never changed again.
3. **Active concepts.** Which features fired at layer 20, from Google DeepMind's Gemma Scope sparse autoencoder, with human-readable labels from Neuronpedia.

It only observes. Nothing inside the model is switched off or changed.

## Three ways to run it

| Where | How | Needs |
|---|---|---|
| Google Colab | Open `notebooks/model_watch_colab.ipynb`, pick a T4 GPU, run top to bottom | A free Google account |
| Terminal (home lab or any Linux box) | `python watch.py --prompt "..."` | A GPU is best; CPU works, slowly |
| Replay page | Open the `trace.html` a run saves | Any browser, offline |

### One-time Hugging Face setup

Gemma is gated. Before the first run:

1. Accept the license at <https://huggingface.co/google/gemma-2-2b> (and at <https://huggingface.co/google/gemma-scope-2b-pt-res> if asked).
2. Create a read token at <https://huggingface.co/settings/tokens>.
3. Export it: `export HF_TOKEN=hf_...` (terminal) or add it as a Colab secret named `HF_TOKEN`.

### Terminal quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=hf_...

python watch.py --prompt "The capital of France is"
python watch.py --prompt $'Q: What is 17 + 25?\nA:' --step         # press Enter for each word
python watch.py --prompt "def is_even(n):" --tokens 30 --html run.html --json run.json
```

Useful flags: `--step` (pause on each word), `--delay 1.0`, `--layer-every 2` (shorter layer list), `--no-labels` (offline), `--device cpu`, `--dtype float32`.

### Hardware (approximate)

| Setup | Memory needed | Notes |
|---|---|---|
| GPU with bfloat16 (RTX 30xx or newer, A-series) | ~6 GB VRAM | Picked automatically |
| Older GPU (T4, P100, P40, RTX 20xx) | ~11 GB VRAM | Runs in float32 |
| CPU only | ~12 GB RAM | Works, a few seconds per word |

float16 is not used by default because Gemma 2 can overflow in it.

## How to read what you see

- **Settled at layer N.** Factual answers often settle in the middle-to-late layers. Small grammar words can settle early, or stay undecided until the end.
- **Early layers look strange.** The logit lens reads every layer through the model's final output step, which early layers were not trained for. Watch the trend across layers, not the first few rows.
- **Watching shows what is active.** That is strong evidence of why the model chose a word, and usually enough to see the pattern. It is still correlation. If a single feature ever matters for a decision, test it by intervention.
- **Labels are a starting point.** Neuronpedia labels are written by an AI from examples where each feature fires. Click a feature number to see those examples.

### First-run check

On `"The capital of France is"`, the layer-20 concepts should look geographic or factual. If the labels look unrelated, the SAE and the labels may come from different dictionaries. Try another L0 value: `--sae-l0 38` or `--sae-l0 139` (layer 20, width 16k has 22, 38, 71, 139 and 294).

## Project layout

```
model_watch/
  core.py      ModelWatcher: runs the model one token at a time and records each layer
  sae.py       JumpReLU sparse autoencoder (Gemma Scope's format), about 60 lines
  labels.py    Neuronpedia label lookups with a disk cache
  render.py    Terminal and notebook display of one step
  export.py    Save a trace as JSON or a self-contained replay page
  viewer.html  The replay page template (shows an illustrative sample if opened directly)
watch.py       Terminal command
notebooks/     Colab notebook
tests/         Runs everything on a tiny random model, no downloads
```

Run the tests with `python -m unittest discover -s tests -v`.

## Built on

- Gemma 2 and Gemma Scope by Google DeepMind: <https://huggingface.co/google/gemma-scope-2b-pt-res>
- Neuronpedia feature pages and labels: <https://www.neuronpedia.org/gemma-2-2b>
- Hugging Face `transformers`. This project does not use TransformerLens, whose 4.0 release removed the `HookedTransformer.from_pretrained` loader most tutorials still use.
