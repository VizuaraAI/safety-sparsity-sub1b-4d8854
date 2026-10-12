"""Entry point: the intervention test. Three ways of buying safety.

    python -m src.train --method full_sft
    python -m src.train --method lora
    python -m src.train --method localized      # updates only the safety units

`localized` is the method the paper is actually about. It runs Axis B first to
find the safety-critical neurons, freezes everything else, and trains only the
rows of each down_proj that correspond to those neurons. If safety really is
sparse, this buys the same safety as full SFT at a fraction of the capability
cost. The prediction under test is that the advantage shrinks or inverts below
1B parameters.

After training, run `python -m src.evaluate --model <output_dir>` on the saved
checkpoint to fill in the corresponding row of Table 2.
"""

from __future__ import annotations

import argparse
import sys

import torch
from torch.utils.data import DataLoader, Dataset

from . import axis_neuron
from .activations import topk_mask
from .config import load_config, run_dir
from .data import build_probe_set
from .modeling import LoadedModel, build_prompt, load_model

REFUSAL_TEMPLATE = (
    "I can't help with that. If you are in a situation where this feels "
    "urgent, I would rather point you to something safe and useful instead."
)

METHODS = ("full_sft", "lora", "localized")


class SafetySFTDataset(Dataset):
    """Harmful prompt in, canonical refusal out. Loss on the response only.

    Masking the prompt tokens out of the loss matters here: with only a few
    hundred examples, letting the model train on the harmful prompts themselves
    teaches it to produce them.
    """

    def __init__(self, lm: LoadedModel, prompts: list[str], max_len: int):
        self.lm = lm
        self.prompts = prompts
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict:
        prompt = build_prompt(self.lm, self.prompts[idx])
        prompt_ids = self.lm.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = self.lm.tokenizer(
            REFUSAL_TEMPLATE, add_special_tokens=False
        )["input_ids"] + [self.lm.tokenizer.eos_token_id]

        input_ids = (prompt_ids + answer_ids)[: self.max_len]
        labels = ([-100] * len(prompt_ids) + answer_ids)[: self.max_len]
        return {"input_ids": input_ids, "labels": labels}


def collate(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(item["input_ids"]) for item in batch)
    input_ids, labels, attention = [], [], []
    for item in batch:
        pad = width - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_id] * pad)
        labels.append(item["labels"] + [-100] * pad)
        attention.append([1] * len(item["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention),
    }


def freeze_all_but_safety_neurons(lm: LoadedModel, mask: dict[int, torch.Tensor]) -> int:
    """Train only the down_proj columns belonging to the selected neurons.

    Implemented with a gradient hook rather than by slicing the weight, so the
    optimizer sees a normal parameter and the saved checkpoint stays a plain
    model that `from_pretrained` can load.
    """
    for param in lm.model.parameters():
        param.requires_grad_(False)

    n_trainable = 0
    for idx, layer in enumerate(lm.layers):
        if idx not in mask or not bool(mask[idx].any()):
            continue
        weight = layer.mlp.down_proj.weight  # (d_model, d_mlp)
        weight.requires_grad_(True)
        keep = mask[idx].to(weight.device).view(1, -1).to(weight.dtype)
        weight.register_hook(lambda grad, keep=keep: grad * keep)
        n_trainable += int(mask[idx].sum())
    return n_trainable


def apply_lora(lm: LoadedModel, cfg) -> None:
    from peft import LoraConfig, get_peft_model

    lora = LoraConfig(
        r=int(cfg.intervention.lora_r),
        lora_alpha=int(cfg.intervention.lora_alpha),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    lm.model = get_peft_model(lm.model, lora)
    lm.model.print_trainable_parameters()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--method", default=None, choices=METHODS)
    parser.add_argument("--model", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    for flag, key in (
        (args.model, "model.name"),
        (args.method, "intervention.method"),
        (args.epochs, "intervention.epochs"),
        (args.lr, "intervention.learning_rate"),
    ):
        if flag is not None:
            cfg = cfg.override(key, flag)

    method = str(cfg.intervention.method)
    print(f"[train] method={method}  model={cfg.model.name}")

    lm = load_model(cfg.model.name, dtype=cfg.model.dtype, device=cfg.model.device)
    lm.model.config.use_cache = False
    probes = build_probe_set(cfg)

    if method == "localized":
        print("[train] running Axis B to locate the safety-critical neurons")
        scores = axis_neuron.safety_scores(lm, probes.harmful, probes.benign)
        # Once `src.evaluate` has reported a p* for this model, set
        # intervention.localized_fraction to it in the config. Until then the
        # largest fraction in the Axis B sweep is a conservative upper bound on
        # the safety support.
        fraction = float(
            cfg.get_path("intervention.localized_fraction")
            or cfg.neuron_axis.fractions[-1]
        )
        mask = topk_mask(scores, fraction)
        n_trainable = freeze_all_but_safety_neurons(lm, mask)
        print(f"[train] training {n_trainable} safety neurons "
              f"({100.0 * fraction:.2f}% of all MLP neurons)")
    elif method == "lora":
        apply_lora(lm, cfg)
    else:
        for param in lm.model.parameters():
            param.requires_grad_(True)

    dataset = SafetySFTDataset(lm, probes.harmful, int(cfg.intervention.max_seq_len))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.intervention.batch_size),
        shuffle=True,
        collate_fn=lambda b: collate(b, lm.tokenizer.pad_token_id),
    )
    trainable = [p for p in lm.model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters; check the freezing logic")
    optimizer = torch.optim.AdamW(trainable, lr=float(cfg.intervention.learning_rate))

    lm.model.train()
    accum = max(1, int(cfg.intervention.grad_accum))
    step = 0
    for epoch in range(int(cfg.intervention.epochs)):
        for i, batch in enumerate(loader):
            batch = {k: v.to(lm.device) for k, v in batch.items()}
            loss = lm.model(**batch).loss / accum
            loss.backward()
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                print(f"[train] epoch={epoch} step={step} "
                      f"loss={float(loss) * accum:.4f}")
    lm.model.eval()

    out = run_dir(cfg, suffix=method)
    lm.model.save_pretrained(out)
    lm.tokenizer.save_pretrained(out)
    print(f"[train] saved to {out}")
    print(f"[train] next: python -m src.evaluate --model {out} "
          f"--run-name {method}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
