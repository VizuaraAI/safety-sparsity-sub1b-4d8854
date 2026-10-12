"""Capability evaluation by log-likelihood scoring.

Why not just prompt the model and parse an answer letter? Because a 135M model
cannot reliably follow that instruction, and you would end up measuring format
compliance instead of ability. Log-likelihood scoring over the answer choices
is the standard harness behaviour and it has usable dynamic range below 1B,
which is exactly where this study lives.

Once the pipeline works, replace this with lm-eval for comparability with
published numbers, and keep these numbers in your logs as the fast inner loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import MultipleChoiceItem
from .modeling import LoadedModel


@dataclass
class CapabilityResult:
    n: int
    accuracy: float
    accuracy_length_normalized: float

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "accuracy_length_normalized": round(self.accuracy_length_normalized, 4),
        }


@torch.no_grad()
def _choice_logprob(lm: LoadedModel, context: str, choice: str) -> tuple[float, int]:
    ctx_ids = lm.tokenizer(context, return_tensors="pt")["input_ids"]
    full_ids = lm.tokenizer(context + choice, return_tensors="pt")["input_ids"]
    n_choice = full_ids.shape[1] - ctx_ids.shape[1]
    if n_choice <= 0:
        return float("-inf"), 1

    full_ids = full_ids.to(lm.device)
    logits = lm.model(full_ids).logits.float()
    log_probs = torch.log_softmax(logits[0, :-1], dim=-1)
    targets = full_ids[0, 1:]
    chosen = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(chosen[-n_choice:].sum()), n_choice


@torch.no_grad()
def evaluate_capability(lm: LoadedModel, items: list[MultipleChoiceItem]) -> CapabilityResult:
    correct = 0
    correct_norm = 0
    for item in items:
        scores = [_choice_logprob(lm, item.context, c) for c in item.choices]
        totals = [s for s, _ in scores]
        norms = [s / max(n, 1) for s, n in scores]
        if max(range(len(totals)), key=lambda i: totals[i]) == item.answer_index:
            correct += 1
        if max(range(len(norms)), key=lambda i: norms[i]) == item.answer_index:
            correct_norm += 1
    n = max(len(items), 1)
    return CapabilityResult(
        n=len(items),
        accuracy=correct / n,
        accuracy_length_normalized=correct_norm / n,
    )
