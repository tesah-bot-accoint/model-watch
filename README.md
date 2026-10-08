# Model Watch

Chat with Gemma 3 and watch it think, one word at a time. Free to run.

For every word the model writes, Model Watch shows:

1. **The choice.** The word picked, its probability, and the runners-up.
2. **How the guess formed.** Every layer read as if the model stopped there (the "logit lens"), and the layer where the answer **settled**, meaning it became the top guess and never changed again.
3. **Concepts at four depths.** Which features fired at four layers, from Google DeepMind's Gemma Scope 2 sparse autoencoders. Each feature gets a free offline label (the words it pushes the model toward), plus a Neuronpedia label where one exists.

After each reply, a **one-screen replay viewer** (on a phone, its panels stack) shows the whole answer with a **concept timeline**: rows are concepts, columns are words. Click a concept and every word where it was active lights up in the text, so you can compare what the model says with what was active while it said it.

It only observes. Nothing inside the model is switched off or changed.

## Start here: try the preview

Before installing anything, open the preview. It's the replay viewer loaded with a sample reply, so you can see what Model Watch shows. It runs in any browser, on a phone or a computer, with nothing to install and no account.

- **On the web:** <https://tesah-bot-accoint.github.io/model-watch/> (works once GitHub Pages is turned on; see below).
- **From this repo:** download `docs/index.html` and open it in a browser.

The sample's numbers and labels are made up to show the layout. A short guide at the top of the page walks you through it: press Play, tap a word, tap a concept. To look at a real run, choose **Open trace.json** and pick a file saved by `watch.py --json` or the notebook.

To put the preview on the web, the repo owner turns on GitHub Pages once: **Settings → Pages → Build and deployment → Deploy from a branch → `main`, folder `/docs` → Save**. The address above goes live a minute or two later.

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
python watch.py --preset qwen3-1.7b --prompt "Is 391 prime?" --compact     # a reasoning model: watch it think
```

Useful flags: `--step` (Enter for each word), `--delay 0.5`, `--tokens 300` (default 200, or 1000 for reasoning models), `--no-thinking` (reasoning models answer directly), `--sae-layers 9,22`, `--json run.json` (save the trace for the viewer), `--no-labels` (offline), `--device cpu`. Run `python watch.py --help` for the full list.

### Presets

| Preset | Concept layers | Neuronpedia labels | Rough GPU memory |
|---|---|---|---|
| `gemma-3-1b-it` | 7, 13, 17, 22 of 26 | layer 13 | ~3 GB (or CPU) |
| `gemma-3-4b-it` (default) | 9, 17, 22, 29 of 34 | layer 17 | ~10 GB |
| `gemma-3-12b-it` | 12, 24, 31, 41 of 48 | layer 12 | ~26 GB (2× T4 with `--device-map auto`) |
| `gemma-2-2b` | 20 of 26 | layer 20 | ~6–11 GB, base model, no chat |
| `qwen3-1.7b` (reasoning) | 7, 14, 18, 24 of 28 | none | ~6 GB (CPU works but slow, about 10 GB of RAM) |
| `qwen3-4b` (reasoning) | none: choice and layers only | none | ~9 GB |

Memory figures are estimates for bfloat16 weights plus the preset's dictionaries (four 16k-feature ones for Gemma 3, one for Gemma 2, four 32k-feature ones for `qwen3-1.7b`). On a GPU without native bfloat16 (T4 and older), PyTorch emulates it: slower, same answers. float16 is never picked automatically because Gemma can overflow in it.

### Watching a reasoning model

The two `qwen3` presets are reasoning models: before answering, they write out their thinking between `<think>` and `</think>`, then give the answer. Model Watch traces every word of both parts, so you can see things like whether the final answer was already the model's top guess partway through its thinking. Qwen models don't need a license click or `HF_TOKEN`.

- **`qwen3-1.7b`** shows concepts from Qwen-Scope, Qwen's own published dictionaries. They were trained on the plain version of the model, before it learned to think step by step, so they may miss concepts that only the reasoning version uses. They have no Neuronpedia labels yet, so each concept shows only its gray "pushes toward" words.
- **`qwen3-4b`** is the stronger thinker, but there's no matching dictionary for it, so it shows the choice and the layer-by-layer guesses without concepts.

Thinking makes replies long, so these presets allow 1000 tokens per reply by default. Add `--no-thinking` to ask for a direct answer instead.

## How to read it

- **Concept timeline.** Switch layers with the pills. Early layers track wording and grammar; middle and late layers track topics and intent. Use "Find a concept" to search labels. Tap or click a square in the timeline to jump to that word, light up that concept, and see how strong it was there.
- **Labels.** Text in the normal color is a Neuronpedia label. Gray "pushes toward" text is computed from the model's own weights and costs nothing. It is usually clearer at later layers.
- **Settled at layer N.** Factual words often settle mid-to-late; small grammar words can settle early or stay undecided until the end.
- **Says vs. active.** Ask the model to explain its reasoning, then light up concepts across that explanation. Agreement is evidence the explanation is faithful; mismatches are the gap current research studies.
- **Limits.** See [Known gaps](#known-gaps) below.

## Known gaps

What Model Watch can't do, how it compares to the tools AI labs use on their own models, and why.

### It can't watch Claude, ChatGPT, Copilot or Gemini

Watching a model's layers requires its weights, the files that make up the model. Anthropic, OpenAI, Google and Microsoft keep the weights of their chat products private, so nobody outside those companies can look inside them. From outside you only get the text a model writes. Some services also report how likely each word was, which could fill the "choice" panel and nothing else.

A model's visible "thinking" or reasoning text doesn't help here either. It's more text the model wrote, not a view of what happened inside. Whether that text matches what's really going on is an open research question, and it's one of the things Model Watch helps you check on open models.

### The method is the same, but the scale is far smaller

The labs use the same kind of tools on their own models. Anthropic, for example, has published this sort of analysis on Claude. The difference is size:

| | Model Watch (Gemma 3 4B) | Lab tools on their own models |
|---|---|---|
| Model size | 4 billion parameters | Not disclosed, likely far larger |
| Concepts per layer | About 16,000, at 4 layers | Millions (Anthropic reported 34 million on Claude 3 Sonnet in 2024) |
| Labels | Some written labels, plus "pushes toward" words | Large-scale labeling, checked by researchers |

With fewer concepts, each one is broader. Model Watch might show a single "AI and technology" concept where a lab tool would separate "interpretability research" from "AI policy".

### It shows what's active, not what caused the word

When a concept lights up as a word is written, that's strong evidence it played a part, but it isn't proof. Two things can happen together without one causing the other. Researchers prove cause by switching a concept off and checking whether the word changes. Model Watch deliberately doesn't change the model, so it can't run that test.

### It doesn't show how concepts connect

You can see which concepts were active, but not which ones fed into which. The labs' wiring diagrams, called attribution graphs, show that. The public tool for building them is circuit-tracer (github.com/decoderesearch/circuit-tracer), and it supports Gemma 3. It isn't built into Model Watch yet.

### Labels can be wrong or missing

- Neuronpedia labels are written by an AI from examples where each concept fired. They are usually close, sometimes wrong, and many concepts have none. For the 4B model, Model Watch currently asks Neuronpedia only at layer 17. The `qwen3` presets have no written labels at all yet.
- The gray "pushes toward" words are calculated from the model itself, so they're always available. But they only describe which words a concept makes more likely, not what it means. They're noisy at early layers, where concepts are more about spelling and grammar than meaning.

### Early layers give rough guesses

The "how the guess formed" chart reads every layer as if it were the last one. Later layers are built for that and read well. Early layers aren't, so their guesses can look random even when the layer is doing useful work. Look at the overall trend, not the first few rows.

### Complex thinking is the hardest to read

Concrete things like places, names, code and dates show up clearly. Abstract, multi-step reasoning, like weighing an argument or planning an answer, spreads across many concepts and many words. That's the hardest case for every current tool, including the labs' own.

### Not yet run on the real models

Everything has been tested on tiny, randomly built versions of Gemma 2, Gemma 3 and Qwen3. These have the same structure as the real models, but they're small enough to run without downloading anything. The first real run may surface small problems:

- **Dictionary file format.** The names inside Google's dictionary files have been checked against SAELens, a widely used library that loads the same files, and match what the loader expects. They haven't been loaded from the real files here yet. If they ever differ, the error message lists them, and the fix is one line in `model_watch/sae.py`.
- **12B on two GPUs.** Splitting the 12B model across two GPUs (Kaggle) is written but untested.
- **Qwen-Scope dictionaries.** Their file layout comes from SAELens's loader, not from opening the real files. They were trained on the plain Qwen3 1.7B, and how well they read the thinking version is untested.

### Closest stand-ins for the big closed models

| Open model | Why it's the closest | Free? |
|---|---|---|
| Gemma 3 27B chat | Full Gemma Scope 2 dictionaries, plus Anthropic's Natural Language Autoencoders on Neuronpedia, which turn a layer's activity into plain sentences | No, needs a large rented GPU |
| GPT-OSS 20B | OpenAI's open-weight model, so the nearest look at an OpenAI-made model. circuit-tracer has a published dictionary for it | Unconfirmed; Model Watch would also need support for that dictionary format |
| Llama 3.3 70B | Natural Language Autoencoders available on Neuronpedia | No, too large |

## Why it's written in Python

Not all of it is. The replay viewer (`model_watch/viewer.html`, copied to `docs/index.html`) is a web page written in HTML, CSS and JavaScript, so it runs on any phone or computer. The part that runs the model is Python, for four reasons:

1. **Gemma ships for Python.** Google publishes Gemma through Hugging Face's `transformers` library, which runs on PyTorch. Both are Python.
2. **Model Watch needs to see inside the model while it runs.** It attaches a small listener to every layer to copy out that layer's numbers as each word forms. PyTorch makes this simple. Most other ways of running a model only hand back the finished words.
3. **The concept dictionaries are built for Python.** Google's Gemma Scope files and the research tools around them (Neuronpedia, circuit-tracer) all work in Python, so new tools can be added later without translating them.
4. **The free GPUs are Python notebooks.** Colab and Kaggle run Python, and they're what keeps Model Watch free to run.

So the work is split. Python runs the model on a GPU and records everything to a file, and the web page replays that file anywhere.

**Could it all run in a browser?** Only partly, for now. Tools such as Transformers.js can run a small model like Gemma 3 1B in a browser, but the ready-made browser versions of the model only hand back the finished words, not each layer's numbers. The concept dictionaries are also large: each one for the 4B model is a few hundred megabytes, and Model Watch uses four, which is too much for most phones. A browser-only version with the 1B model and one dictionary is possible as an experiment, but it would be slow. The more practical next step is streaming a live Colab run into the web viewer, so you watch the reply form in the viewer while Python does the work on a free GPU.

## Project layout

```
model_watch/
  core.py      Presets, ModelWatcher: chat, cached generation, logit lens, multi-layer concepts, labels
  sae.py       JumpReLU sparse autoencoder; loads Gemma Scope 1 (.npz) and 2 (.safetensors)
  labels.py    Neuronpedia label lookups with a disk cache
  render.py    Terminal and notebook live display of one word
  export.py    Save a trace as JSON or a self-contained replay page
  viewer.html  One-screen replay viewer (opened directly, shows a made-up sample with a "Start here" guide)
watch.py       Terminal command
notebooks/     Colab and Kaggle notebook
docs/          The preview: index.html, an exact copy of viewer.html, served by GitHub Pages
tests/
  test_model_watch.py  Runs everything on tiny random Gemma 2, Gemma 3 and Qwen3 models, no downloads
  test_preview.py      Checks docs/index.html still matches viewer.html (needs no PyTorch)
requirements.txt, pyproject.toml   What to install
```

Run the tests with `python -m unittest discover -s tests -v`. They run offline. After changing `model_watch/viewer.html`, copy it to `docs/index.html` or `test_preview.py` fails.

## Built on

- Gemma 3 and Gemma Scope 2 by Google DeepMind: <https://huggingface.co/google/gemma-scope-2>
- Neuronpedia feature pages and labels: <https://www.neuronpedia.org>
- Hugging Face `transformers`. This project does not use TransformerLens, whose 4.0 release removed `HookedTransformer` and its `from_pretrained` loader, which most tutorials still use.
