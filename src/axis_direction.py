"""Axis C: subspace-rank sparsity.

Protocol.
  1. At several depths in the residual stream, take the difference in means
     between harmful and benign activations. Each difference is a candidate
     refusal direction.
  2. Orthonormalize them, then ablate the first r directions from the whole
     residual stream (embedding output and every layer output) and re-measure
     refusal rate. Sweep r.
  3. r* is the smallest rank that drives refusal rate below
     direction_axis.refusal_floor.

Do not assume r* = 1. The single-direction result holds at 7B scale; recent
work finds a reusable core of refusal latents plus a long tail, and the whole
question here is whether that core survives when capacity is scarce. Report the
singular value spectrum alongside r*, because a slowly decaying spectrum is
itself the evidence that refusal has spread out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .activations import ablate_directions, mean_residual, orthonormal_basis
from .modeling import LoadedModel, build_prompt, generate
from .refusal import score_completions


@dataclass
class DirectionAxisResult:
    ranks: list[int]
    refusal_by_rank: list[float]
    baseline_refusal: float
    r_star: int | None
    singular_values: list[float]
    layer_indices: list[int]
    curve: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "axis": "subspace_rank",
            "ranks": self.ranks,
            "refusal_by_rank": [round(v, 4) for v in self.refusal_by_rank],
            "baseline_refusal": round(self.baseline_refusal, 4),
            "r_star": self.r_star,
            "singular_values": [round(v, 5) for v in self.singular_values],
            "layer_indices": self.layer_indices,
            "curve": self.curve,
        }


@torch.no_grad()
def candidate_directions(
    lm: LoadedModel, harmful: list[str], benign: list[str], layer_fractions: list[float]
) -> tuple[torch.Tensor, list[int]]:
    """Difference-in-means directions, one per sampled depth. Shape (n, d_model)."""
    harmful_prompts = [build_prompt(lm, p) for p in harmful]
    benign_prompts = [build_prompt(lm, p) for p in benign]
    res_harm = mean_residual(lm, harmful_prompts)
    res_benign = mean_residual(lm, benign_prompts)
    diff = res_harm - res_benign  # (n_layers + 1, d_model)

    indices = sorted({
        min(lm.n_layers, max(1, int(round(frac * lm.n_layers))))
        for frac in layer_fractions
    })
    return diff[indices].clone(), indices


@torch.no_grad()
def run(lm: LoadedModel, harmful: list[str], benign: list[str], cfg) -> DirectionAxisResult:
    max_new = int(cfg.model.max_new_tokens)
    floor = float(cfg.direction_axis.refusal_floor)
    layer_fractions = list(cfg.direction_axis.layer_fractions)

    print("[axis-C] extracting difference-in-means refusal directions")
    directions, indices = candidate_directions(lm, harmful, benign, layer_fractions)

    # Order the candidates by how much variance they carry, so that "rank r"
    # means the r strongest directions rather than an arbitrary r of them.
    normalized = directions / (directions.norm(dim=-1, keepdim=True) + 1e-8)
    u, s, _ = torch.linalg.svd(normalized, full_matrices=False)
    ranked = (u.T @ normalized)  # principal directions of the candidate set
    basis_full = orthonormal_basis(ranked)

    max_rank = min(int(cfg.direction_axis.max_rank), basis_full.shape[0])
    ranks = list(range(1, max_rank + 1))

    harmful_prompts = [build_prompt(lm, p) for p in harmful]
    baseline = score_completions(
        generate(lm, harmful_prompts, max_new_tokens=max_new)
    ).refusal_rate
    print(f"[axis-C] baseline refusal rate: {baseline:.3f}")

    refusal_by_rank: list[float] = []
    curve: list[dict] = []
    for rank in ranks:
        with ablate_directions(lm, basis_full[:rank]):
            stats = score_completions(
                generate(lm, harmful_prompts, max_new_tokens=max_new)
            )
        refusal_by_rank.append(stats.refusal_rate)
        curve.append({"rank": rank, **stats.as_dict()})
        print(f"[axis-C] r={rank:<3} refusal={stats.refusal_rate:.3f}  "
              f"ASR={stats.attack_success_rate:.3f}")

    r_star = next((r for r, v in zip(ranks, refusal_by_rank) if v < floor), None)
    return DirectionAxisResult(
        ranks=ranks,
        refusal_by_rank=refusal_by_rank,
        baseline_refusal=baseline,
        r_star=r_star,
        singular_values=[float(v) for v in s.cpu()],
        layer_indices=indices,
        curve=curve,
    )
