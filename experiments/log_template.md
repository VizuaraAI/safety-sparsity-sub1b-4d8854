# Experiment log

One row per run. The columns are the columns of the paper's tables, so a filled
log assembles into `../../paper/main.tex` with no translation step. Copy the
template blocks, do not overwrite them.

Rules that keep this log worth keeping:

- Log the run **before** you look at the result, so that a disappointing number
  cannot quietly become an unlogged run.
- Always record `n_harmful` and `n_benign`. A number without a sample size is
  not a result.
- If `sli` comes back empty, write the reason in Notes. Usually it means the
  baseline refusal rate was below the threshold, which is a finding about the
  model, not a broken run.
- One sentence in Notes on what you expected before running. Over eight weeks
  this is the single most useful column you will have.

---

## Table 1: safety localization across the ladder

Maps to Table 1 and Figure 2(a) of the paper. Produced by
`python -m src.evaluate --model <model> --run-name <run>`.

| date | run | model | params | non-emb params | baseline refusal | k* | p* (%) | r* | SLI | SCE | ASR (%) | over-refusal (%) | capability (%) | n_harmful | n_benign | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-13 | smoke | SmolLM2-135M-Instruct | 135M | 106M | 0.17 | 0 | 0.05 | 1 |  | 0.18 | 100.0 | 16.7 | 36.5 | 6 | 6 | Smoke test only. Example row, replace with your own. Baseline refusal is far too low to trust; six prompts is not a measurement. |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

### Control ladder (Pythia, identical pretraining data and order)

Same columns. This ladder is the scientific backbone: it is the only way to
attribute a trend to parameter count rather than to alignment recipe.

| date | run | model | params | baseline refusal | k* | p* (%) | r* | SLI | SCE | ASR (%) | over-refusal (%) | capability (%) | SFT examples | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  | Pythia-160M |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  | Pythia-410M |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  | Pythia-1.0B |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  | Pythia-1.4B |  |  |  |  |  |  |  |  |  |  |  |  |

---

## Table 2: cost of buying safety

Maps to Table 2 and Figure 2(b) of the paper. Produced by
`bash scripts/run_intervention.sh <model>`.

| date | model size | method | ASR (%) | StrongREJECT | over-refusal (%) | capability (%) | trainable params | wall clock | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | 0.36B | base |  |  |  |  | 0 |  |  |
|  | 0.36B | full SFT |  |  |  |  |  |  |  |
|  | 0.36B | LoRA |  |  |  |  |  |  |  |
|  | 0.36B | localized |  |  |  |  |  |  |  |
|  | 0.5B | base |  |  |  |  | 0 |  |  |
|  | 0.5B | full SFT |  |  |  |  |  |  |  |
|  | 0.5B | LoRA |  |  |  |  |  |  |  |
|  | 0.5B | localized |  |  |  |  |  |  |  |
|  | 1.5B | base |  |  |  |  | 0 |  |  |
|  | 1.5B | full SFT |  |  |  |  |  |  |  |
|  | 1.5B | LoRA |  |  |  |  |  |  |  |
|  | 1.5B | localized |  |  |  |  |  |  |  |

---

## Ablations

Maps to Section 5 of the paper. One block per ablation; each should answer
"would this alternative explanation also produce my headline curve?".

| date | ablation | variant | what changed | headline number | does the capacity floor move? | notes |
| --- | --- | --- | --- | --- | --- | --- |
|  | width vs depth | deep-thin |  |  |  |  |
|  | width vs depth | wide-shallow |  |  |  |  |
|  | parameter counting | non-embedding |  |  |  |  |
|  | pretraining tokens | Pythia ckpt @ X tokens |  |  |  |  |
|  | alignment strength | 128 / 512 / 2048 SFT examples |  |  |  |  |
|  | SLI definition | arithmetic instead of geometric mean |  |  |  |  |
|  | SLI thresholds | ASR 0.3 / 0.5 / 0.7 |  |  |  |  |
|  | monosemanticity | SAE latents for refusal |  |  |  |  |
|  | token-constrained defence | PACT-style objective per size |  |  |  |  |

---

## Decisions and dead ends

Not every run produces a table row. Log the judgement calls here, because in
week seven you will not remember why you chose the last token position, and a
reviewer will ask.

| date | decision or dead end | why | what it rules out |
| --- | --- | --- | --- |
|  |  |  |  |
|  |  |  |  |

---

## Weekly check-in

Three lines every Friday. Takes two minutes and is the thing that keeps an
eight-week project from quietly becoming a four-week project.

| week | what moved | what is blocked | what ships next week |
| --- | --- | --- | --- |
| 1 |  |  |  |
| 2 |  |  |  |
| 3 |  |  |  |
| 4 |  |  |  |
| 5 |  |  |  |
| 6 |  |  |  |
| 7 |  |  |  |
| 8 |  |  |  |
