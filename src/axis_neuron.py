"""Axis B: neuron-support sparsity, plus the Safety-Capability Entanglement score.

Protocol.
  1. Score every MLP neuron by how much more it activates on harmful prompts
     than on matched benign prompts. This is activation contrasting, the method
     the safety-neuron literature uses, and the benign control is what stops it
     from simply finding "long prompt" neurons.
  2. Ablate the top f fraction of neurons model-wide and re-measure refusal
     rate. Sweep f.
  3. p* is the smallest f whose ablation drops refusal rate by
     neuron_axis.refusal_drop_target relative to the unablated model.
  4. Independently score neurons by capability importance, then report the
     Jaccard overlap of the two top-m sets. That overlap is SCE, and a rising
     SCE as parameters fall is the signature of superposition.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .activations import ablate_neurons, mean_mlp_activation, topk_mask
from .data import MultipleChoiceItem
from .modeling import LoadedModel, build_prompt, generate
from .refusal import score_completions


@dataclass
class NeuronAxisResult:
    fractions: list[float]
    refusal_by_fraction: list[float]
    baseline_refusal: float
    p_star: float | None
    sce: float | None
    n_selected_at_p_star: int | None
    curve: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "axis": "neuron_support",
            "fractions": self.fractions,
            "refusal_by_fraction": [round(v, 4) for v in self.refusal_by_fraction],
            "baseline_refusal": round(self.baseline_refusal, 4),
            "p_star": self.p_star,
            "sce": None if self.sce is None else round(self.sce, 4),
            "n_selected_at_p_star": self.n_selected_at_p_star,
            "curve": self.curve,
        }


@torch.no_grad()
def safety_scores(
    lm: LoadedModel, harmful: list[str], benign: list[str]
) -> torch.Tensor:
    """(n_layers, d_mlp) contrast score. Higher means more safety-selective."""
    harmful_prompts = [build_prompt(lm, p) for p in harmful]
    benign_prompts = [build_prompt(lm, p) for p in benign]
    act_harm = mean_mlp_activation(lm, harmful_prompts)
    act_benign = mean_mlp_activation(lm, benign_prompts)
    return act_harm - act_benign


@torch.no_grad()
def capability_scores(
    lm: LoadedModel, items: list[MultipleChoiceItem], limit: int = 64
) -> torch.Tensor:
    """(n_layers, d_mlp) importance on the capability set.

    Magnitude of activation as the importance proxy, which is the same family
    of criterion the pruning literature uses. Scored on capability text only,
    so the two score matrices are genuinely independent measurements.
    """
    texts = [item.context + item.choices[item.answer_index] for item in items[:limit]]
    return mean_mlp_activation(lm, texts)


def jaccard(mask_a: dict[int, torch.Tensor], mask_b: dict[int, torch.Tensor]) -> float:
    inter = sum(int((mask_a[i] & mask_b[i]).sum()) for i in mask_a)
    union = sum(int((mask_a[i] | mask_b[i]).sum()) for i in mask_a)
    return inter / union if union else 0.0


def _top_m_mask(scores: torch.Tensor, m: int) -> dict[int, torch.Tensor]:
    fraction = min(1.0, m / scores.numel())
    return topk_mask(scores, fraction)


@torch.no_grad()
def run(
    lm: LoadedModel,
    harmful: list[str],
    benign: list[str],
    capability_items: list[MultipleChoiceItem],
    cfg,
) -> NeuronAxisResult:
    fractions = list(cfg.neuron_axis.fractions)
    target = float(cfg.neuron_axis.refusal_drop_target)
    max_new = int(cfg.model.max_new_tokens)

    print("[axis-B] scoring safety neurons by harmful-vs-benign activation contrast")
    s_scores = safety_scores(lm, harmful, benign)

    harmful_prompts = [build_prompt(lm, p) for p in harmful]
    baseline = score_completions(
        generate(lm, harmful_prompts, max_new_tokens=max_new)
    ).refusal_rate
    print(f"[axis-B] baseline refusal rate on harmful prompts: {baseline:.3f}")

    refusal_by_fraction: list[float] = []
    curve: list[dict] = []
    for fraction in fractions:
        mask = topk_mask(s_scores, fraction)
        n_selected = sum(int(v.sum()) for v in mask.values())
        with ablate_neurons(lm, mask):
            stats = score_completions(
                generate(lm, harmful_prompts, max_new_tokens=max_new)
            )
        refusal_by_fraction.append(stats.refusal_rate)
        curve.append({"fraction": fraction, "n_neurons": n_selected, **stats.as_dict()})
        print(f"[axis-B] f={fraction:<8} n={n_selected:<7} refusal={stats.refusal_rate:.3f}")

    p_star = None
    n_at_p_star = None
    if baseline > 0:
        for fraction, refusal, row in zip(fractions, refusal_by_fraction, curve):
            if (baseline - refusal) / baseline >= target:
                p_star, n_at_p_star = fraction, row["n_neurons"]
                break

    sce = None
    if capability_items:
        print("[axis-B] scoring capability-critical neurons for the SCE overlap")
        c_scores = capability_scores(lm, capability_items)
        m = int(cfg.neuron_axis.entanglement_top_m)
        if n_at_p_star:
            m = n_at_p_star  # compare the sets at the size that actually matters
        sce = jaccard(_top_m_mask(s_scores, m), _top_m_mask(c_scores, m))
        print(f"[axis-B] SCE (Jaccard overlap at m={m}): {sce:.4f}")

    return NeuronAxisResult(
        fractions=fractions,
        refusal_by_fraction=refusal_by_fraction,
        baseline_refusal=baseline,
        p_star=p_star,
        sce=sce,
        n_selected_at_p_star=n_at_p_star,
        curve=curve,
    )
