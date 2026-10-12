# CLAUDE.md

Instructions for the coding agent working in this repository with Pradyuman.

## Project goal

Determine whether the sparsity of safety behaviour in language models survives
below 1B parameters. Prior work (Qi et al. 2024 on token depth, Chen et al. 2024
on safety neurons, Arditi et al. 2024 on the refusal direction, Li et al. 2025 on
safety layers) establishes that safety is localized at 7B scale, which is what
makes surgical "localized alignment" cheap. All of it was measured on large
models. The hypothesis under test is a capacity argument: localization needs
spare capacity to localize into, and sub-1B models under superposition pressure
may not have any.

The deliverable is a workshop paper. The draft lives at `../paper/main.tex` and
already has the full structure with `XX.X` placeholders. Every experiment in
this repo maps to a specific table or figure in it.

## Repository layout

```
configs/default.yaml      all hyperparameters; nothing is hardcoded in src/
src/config.py             config loading, dotted access, run directories
src/modeling.py           model loading, chat formatting, greedy generation
src/data.py               probe corpora, with offline fallbacks
src/refusal.py            refusal classification
src/activations.py        ALL hooks live here (capture + ablation)
src/axis_token.py         Axis A: k*, prefill sweep, token distribution shift
src/axis_neuron.py        Axis B: p*, safety neuron scoring, SCE
src/axis_direction.py     Axis C: r*, difference-in-means, directional ablation
src/capability.py         log-likelihood multiple choice accuracy
src/metrics.py            SLI, SCE, results table assembly
src/evaluate.py           entry point for measurement
src/train.py              entry point for the intervention test
experiments/log_template.md   the running experiment log
```

## Conventions

- **Hooks go in `src/activations.py`.** Never register a hook inline in an
  experiment script. Every hook is wrapped in a context manager that removes its
  handles on exit. Leaked handles produce stale activations and cost a day.
- **Config over flags.** New knobs go in `configs/default.yaml` and are read via
  `cfg.get_path(...)`. CLI flags exist only for the few things swept per run.
- **Greedy decoding everywhere.** Sampling noise would be confounded with
  ablation effects. If you need sampling, add it as an explicit ablation.
- **Every harmful measurement needs its benign control in the same run.** A
  model that refuses everything looks perfectly safe. Over-refusal on XSTest
  safe prompts is reported next to every attack success rate, always.
- **Results are JSON, tables are derived.** `src/evaluate.py` writes
  `results.json`; `src/metrics.py` renders the markdown table. Never hand-edit a
  table.
- Python 3.10+ syntax. Type hints on function signatures. No new dependencies
  without asking: the stack is torch, transformers, datasets, peft, numpy.
- Comments explain *why*, especially where a choice is methodologically load
  bearing (last-token position, benign controls, the SLI guard). Do not add
  comments that restate the code.

## Methodological traps, do not let these through

1. **SLI is undefined when the model never refuses.** `MIN_BASELINE_REFUSAL` in
   `src/metrics.py` guards this. If baseline refusal is near zero then `k*` is 0
   and the naive formula reports SLI near 1.0, meaning "perfectly localized"
   when the truth is "no safety at all". The smallest models really do hit this.
   Never remove the guard; if a measurement needs a different threshold, make it
   a config value and log it.
2. **Attack success rate is inflated at small scale.** The substring classifier
   counts any non-refusal over 20 characters as compliance, and a 135M model
   emits a lot of coherent-looking nonsense. Pair every ASR number with a
   StrongREJECT-style usefulness score before reporting.
3. **Normalize by the right denominator.** At 135M parameters most weights are
   in the embedding matrices. `LoadedModel.n_params_non_embedding` exists for
   this reason, and every scaling curve should be produced both ways. The
   capacity floor can move substantially between the two.
4. **The Pythia control ladder is the scientific backbone.** Released instruct
   models differ in data, alignment recipe and tokenizer, so a trend across them
   is suggestive at best. Pythia holds data and data order fixed. If an
   experiment only has time for one ladder, it is this one.

## Current state

The measurement pipeline runs end to end and has been smoke tested on
SmolLM2-135M-Instruct. `src/train.py` is implemented but has not been run at
scale.

## Next steps, in order

1. Upgrade `src/refusal.py` to the HarmBench classifier, keeping the substring
   scores alongside for comparison.
2. Get HarmBench access (`HF_TOKEN`) and switch `data.harmful_dataset`.
3. Run the full six-model ladder and produce the first real SLI-versus-scale
   plot. This is Figure 2(a) of the paper.
4. Add the safety SFT run for the Pythia control ladder, then measure it.
5. Run the three-way intervention comparison (full SFT, LoRA, localized) at
   0.36B, 0.5B and 1.5B. This is Table 2.
6. Ablations: width versus depth at matched parameters, non-embedding parameter
   count, alignment strength sweep, SAE monosemanticity check.

## Working agreement

- When an experiment finishes, append a row to `experiments/log_template.md`
  before moving on. The paper is assembled from that log.
- Prefer a small honest experiment that runs today over a comprehensive one that
  runs next week. This is an eight-week project.
- When a result is surprising, assume a measurement artifact first and say which
  check would distinguish artifact from finding.
- Do not report a number without saying how many prompts it came from.
