# Model Watch

Chat with Gemma 3 and watch it think, one word at a time. Free to run.

For every word the model writes, Model Watch shows:

1. **The choice.** The word picked, its probability, and the runners-up.
2. **How the guess formed.** Every layer read as if the model stopped there (the "logit lens"), and the layer where the answer **settled**, meaning it became the top guess and never changed again.
3. **Concepts at four depths.** Which features fired at four layers, from Google DeepMind's Gemma Scope 2 sparse autoencoders. Each feature gets a free offline label (the words it pushes the model toward), plus a Neuronpedia label where one exists.

After each reply, a **one-screen replay viewer** shows the whole answer with a **concept timeline**: rows are concepts, columns are words. Click a concept and every word where it was active lights up in the text, so you can compare what the model says with what was active while it said it.

It only observes. Nothing inside the model is switched off or changed.

## Free ways to run it

| Where | Setup | Model |
|---|---|---|
| Google Colab | Open `notebooks/model_watch.ipynb`, Runtime → T4 GPU | Gemma 3 4B chat (default) |
| Kaggle | Same notebook; Accelerator → GPU T4 x2, Internet on (needs a phone-verified account) | Gemma 3 4B, or 12B split across both GPUs |
| Your own Linux machine | `python watch.py ...` | Any preset your hardware fits |
| No GPU at all | `python watch.py --preset gemma-3-1b-it --device cpu ...` | Gemma 3 1B chat (slow, works) |

### One-time Hugging Face setup

Gemma is gated. Before the first run:

1. Accept the license on the model page, for example <https://huggingface.co/google/gemma-3-4b-it>, and on the matching Gemma Scope 2 repo if asked, for example <https://huggingface.co/google/gemma-scope-2-4b-it>.
2. Create a read token at <https://huggingface.co/settings/tokens>.
3. Provide it as `HF_TOKEN`: `export HF_TOKEN=hf_...` in a terminal, or a notebook secret named `HF_TOKEN` (Colab: key icon; Kaggle: Add-ons → Secrets).

### Terminal

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=hf_...

python watch.py --prompt "Is AI a black box?"                       # full live view
python watch.py --prompt "Is AI a black box?" --compact             # choice + concepts only
python watch.py --prompt "Why is the sky blue?" --quiet --html run.html   # just the text, then open run.html
python watch.py --interactive --compact --html chat.html            # keep chatting; one replay per reply
python watch.py --preset gemma-3-12b-it --device-map auto --prompt "..."  # split across GPUs
```

Useful flags: `--step` (Enter for each word), `--delay 0.5`, `--tokens 300`, `--sae-layers 9,22`, `--no-labels` (offline), `--device cpu`.

### Presets

| Preset | Concept layers | Neuronpedia labels | Rough GPU memory |
|---|---|---|---|
| `gemma-3-1b-it` | 7, 13, 17, 22 of 26 | layer 13 | ~3 GB (or CPU) |
| `gemma-3-4b-it` (default) | 9, 17, 22, 29 of 34 | layer 17 | ~10 GB |
| `gemma-3-12b-it` | 12, 24, 31, 41 of 48 | layer 12 | ~26 GB (2× T4 with `--device-map auto`) |
| `gemma-2-2b` | 20 of 26 | layer 20 | ~6–11 GB, base model, no chat |

Memory figures are estimates for bfloat16 weights plus four 16k-feature dictionaries. On a GPU without native bfloat16 (T4 and older), PyTorch emulates it: slower, same answers. float16 is never picked automatically because Gemma can overflow in it.

## How to read it

- **Concept timeline.** Switch layers with the pills. Early layers track wording and grammar; middle and late layers track topics and intent. Use "Find a concept" to search labels.
- **Labels.** Black text is a Neuronpedia label. Gray "pushes toward" text is computed from the model's own weights and costs nothing. It is usually clearer at later layers.
- **Settled at layer N.** Factual words often settle mid-to-late; small grammar words can settle early or stay undecided until the end.
- **Says vs. active.** Ask the model to explain its reasoning, then light up concepts across that explanation. Agreement is evidence the explanation is faithful; mismatches are the gap current research studies.
- **Limits.** Watching shows what is active: strong evidence about why, not proof. Concrete topics read clearly; abstract multi-step reasoning is the hardest case for every current tool. Early-layer logit-lens guesses are rough.

## Project layout

```
model_watch/
  core.py      Presets, ModelWatcher: chat, cached generation, logit lens, multi-layer concepts, labels
  sae.py       JumpReLU sparse autoencoder; loads Gemma Scope 1 (.npz) and 2 (.safetensors)
  labels.py    Neuronpedia label lookups with a disk cache
  render.py    Terminal and notebook live display of one word
  export.py    Save a trace as JSON or a self-contained replay page
  viewer.html  One-screen replay viewer (shows an illustrative sample if opened directly)
watch.py       Terminal command
notebooks/     Colab and Kaggle notebook
tests/         Runs everything on tiny random Gemma 2 and Gemma 3 models, no downloads
```

Run the tests with `python -m unittest discover -s tests -v`.

## Built on

- Gemma 3 and Gemma Scope 2 by Google DeepMind: <https://huggingface.co/google/gemma-scope-2>
- Neuronpedia feature pages and labels: <https://www.neuronpedia.org>
- Hugging Face `transformers`. This project does not use TransformerLens, whose 4.0 release removed the `HookedTransformer.from_pretrained` loader most tutorials still use.
