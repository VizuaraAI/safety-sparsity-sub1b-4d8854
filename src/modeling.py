"""Model loading, chat formatting and generation for the sub-1B ladder.

Every model in the ladder (SmolLM2, Qwen2.5, Qwen3, Llama 3.2) uses the
Llama-family MLP with gate_proj / up_proj / down_proj, which is why the
activation hooks in activations.py work uniformly across all of them. If you
add a model with a different MLP, check that assumption first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class LoadedModel:
    name: str
    model: torch.nn.Module
    tokenizer: "AutoTokenizer"
    device: torch.device

    @property
    def layers(self) -> Sequence[torch.nn.Module]:
        """The decoder layer list, which is where every hook attaches."""
        return self.model.model.layers

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def d_model(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def d_mlp(self) -> int:
        return int(self.model.config.intermediate_size)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @property
    def n_params_non_embedding(self) -> int:
        """Needed for the embedding-share ablation: at 135M params the
        embedding matrices dominate, which shifts every scaling curve."""
        embed = self.model.get_input_embeddings().weight.numel()
        head = self.model.get_output_embeddings()
        untied = 0 if head is None or head.weight is self.model.get_input_embeddings().weight \
            else head.weight.numel()
        return self.n_params - embed - untied


def load_model(name: str, dtype: str = "float32", device: str = "auto") -> LoadedModel:
    resolved = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left padding keeps generation aligned

    model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=DTYPES.get(dtype, torch.float32),
        attn_implementation="eager",  # hooks and attention probes need eager
    )
    model.to(resolved)
    model.eval()
    model.config.use_cache = True
    return LoadedModel(name=name, model=model, tokenizer=tokenizer, device=resolved)


def build_prompt(lm: LoadedModel, user_text: str, assistant_prefill: str = "") -> str:
    """Render one user turn, optionally with the assistant turn prefilled.

    The prefill is the whole mechanism of Axis A: we hand the model the first
    few tokens of a compliant answer and ask whether it still refuses.
    """
    messages = [{"role": "user", "content": user_text}]
    if getattr(lm.tokenizer, "chat_template", None):
        text = lm.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # Base checkpoints have no chat template. Use a minimal, fixed format
        # so base-vs-aligned comparisons stay apples to apples.
        text = f"User: {user_text}\nAssistant:"
    return text + assistant_prefill


@torch.no_grad()
def generate(
    lm: LoadedModel,
    prompts: Sequence[str],
    max_new_tokens: int = 48,
    batch_size: int = 8,
) -> list[str]:
    """Greedy decoding. Greedy, not sampled, so that ablation effects are not
    confounded by sampling noise."""
    completions: list[str] = []
    for start in range(0, len(prompts), batch_size):
        chunk = list(prompts[start : start + batch_size])
        enc = lm.tokenizer(chunk, return_tensors="pt", padding=True).to(lm.device)
        out = lm.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=lm.tokenizer.pad_token_id,
        )
        new_tokens = out[:, enc["input_ids"].shape[1] :]
        completions.extend(
            lm.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        )
    return completions


def batched(items: Iterable, size: int):
    """Small helper used by the axis scripts."""
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
