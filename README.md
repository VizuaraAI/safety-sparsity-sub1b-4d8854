# safety-sparsity-sub1b

**Does safety-token sparsity survive below 1B parameters?**

Pradyuman Dash, Vizuara SLM Bootcamp research track.

---

## The question

A lot of recent interpretability work says the same thing in different
coordinate systems: safety behaviour in an aligned language model is **sparse**.
It lives in the first few generated tokens, in about 5 percent of MLP neurons,
in a few contiguous middle layers, or along a single direction in the residual
stream. That sparsity is the whole reason localized alignment is attractive: if
the safety mechanism is small and findable, you can harden it with a surgical
edit instead of a full retrain, and you pay almost nothing in capability.

Every one of those results was measured at 7B parameters or above, where there
is plenty of spare capacity for a dedicated safety subnetwork to live in.

Your project asks the obvious next question that nobody has actually run:
**does any of it still hold at 0.5B?** The capacity argument says maybe not. A
narrow model is under much heavier superposition pressure, so safety features
may be packed into the same neurons and the same directions as capability
features. If so, there is nothing clean to localize into, and the surgical edit
stops being free exactly where it is most needed, since sub-1B models are the
ones shipping on phones and inside agent loops with no large safety filter
behind them.

Either answer is publishable. "Sparsity survives down to X" is a validated
operating range. "Sparsity collapses below X" is a warning with a number
attached. That symmetry is why this is a good eight-week project.

## What this repo measures

Three localization axes, one model ladder, two composite scores.

| Axis | Question | Output |
| --- | --- | --- |
| A, token depth | How many generated tokens does the refusal decision span? | `k*` |
| B, neuron support | What fraction of MLP neurons must be ablated to kill refusal? | `p*` |
| C, subspace rank | How many residual-stream directions carry refusal? | `r*` |

Those three are not comparable across models on their own, which is exactly why
the scaling question has never been asked. So the repo normalizes each by the
largest support that model could have used and combines them:

```
SLI = 1 - (k_hat * p_hat * r_hat) ** (1/3)
```

near 1 means safety is tightly localized, near 0 means it is smeared across the
model. Alongside it, the **Safety-Capability Entanglement** score is the Jaccard
overlap between safety-critical and capability-critical neurons. A rising SCE as
parameters fall is the direct fingerprint of superposition.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Install `torch` matched to your CUDA version from the PyTorch site if the plain
wheel is wrong for your machine. A single consumer GPU is plenty; the smoke test
also runs on CPU, just slowly.

## The first three things to run

**1. Smoke test on the smallest model (a few minutes).**

```bash
python -m src.evaluate --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --n-harmful 6 --n-benign 6 --run-name smoke
```

This downloads SmolLM2-135M, runs all three axes on a handful of prompts, and
writes `runs/smoke-.../results.json` plus an aggregated `runs/table.md` whose
columns are the columns of Table 1 in your paper draft. If this works, every
hook in the repo works.

**2. One real model, real sample sizes.**

```bash
python -m src.evaluate --model Qwen/Qwen2.5-0.5B-Instruct \
  --n-harmful 100 --n-benign 100 --run-name v1
```

**3. The whole ladder, then look at the shape.**

```bash
bash scripts/run_ladder.sh v1
python -m src.evaluate --table
```

Six rows. Plot SLI against parameter count. That plot is Figure 2(a) of your
paper, and the first time you see it with real numbers in it is the day the
project becomes real.

Then the intervention test, which is what turns an interpretability measurement
into a deployment claim:

```bash
bash scripts/run_intervention.sh Qwen/Qwen2.5-0.5B-Instruct
```

## Layout

```
configs/default.yaml      every knob, with the smoke-test sizes as defaults
src/config.py             config loading with dotted access
src/modeling.py           model ladder loading, chat formatting, generation
src/data.py               harmful / benign / capability probe corpora
src/refusal.py            refusal classification (start cheap, upgrade later)
src/activations.py        hooks: capture, neuron ablation, directional ablation
src/axis_token.py         Axis A -> k*, plus token distribution shift
src/axis_neuron.py        Axis B -> p*, and the SCE overlap
src/axis_direction.py     Axis C -> r*, plus the singular value spectrum
src/capability.py         log-likelihood accuracy on ARC and HellaSwag
src/metrics.py            SLI, SCE, and the results table
src/evaluate.py           entry point: all three axes for one model
src/train.py              entry point: full SFT vs LoRA vs localized edit
scripts/                  run scripts for the ladder and the intervention
experiments/log_template.md   one row per run, mapped to the paper's tables
```

## Three things that will bite you, in the order they will

**1. Your smallest models may have no refusal behaviour at all.** When you run
the smoke test on SmolLM2-135M you will likely see a baseline refusal rate near
zero. That is not a bug. There is nothing to localize, so `src/metrics.py`
deliberately returns `sli: None` with a reason rather than a high number. Watch
for this: a naive pipeline reports "perfectly localized safety" for a model that
simply never refuses, and that single artifact could sink your whole result.

**2. The refusal classifier is the weakest link.** `src/refusal.py` ships with
substring matching, which is deterministic, free, and wrong in predictable ways:
it scores "I cannot help, but here is how" as a refusal, and it scores any
incoherent 20-character output as a successful jailbreak. At sub-1B scale the
second failure mode is severe, because small models produce garbage that looks
like compliance. Upgrade to the HarmBench classifier and a StrongREJECT-style
usefulness score before you report anything, and keep the cheap numbers in your
logs so you can show the two agree or explain why they do not.

**3. HarmBench and AdvBench are gated on the Hub.** The default config uses
JailbreakBench, which is ungated, so the pipeline runs today. Request HarmBench
access, export `HF_TOKEN`, and switch `harmful_dataset` before the real run: the
paper's headline numbers should be on HarmBench for comparability.

## Responsible use

This repo measures refusal behaviour and deliberately removes it in order to
study where it lives. That is standard safety-interpretability methodology and
it is also why you should keep ablated checkpoints local, never push them to the
Hub, and report attack success rates on public benchmark prompts rather than
publishing new harmful content. When you write the paper, include a short
limitations and ethics paragraph saying exactly this. Reviewers at safety venues
look for it.

## Where this goes

The experiment log in `experiments/log_template.md` has one row per run, and the
columns are the columns of Table 1 and Table 2 in `../paper/main.tex`. Fill the
log as you go and the paper fills itself in. The placeholder draft already has
the structure; your job over eight weeks is to replace every `XX.X`.
