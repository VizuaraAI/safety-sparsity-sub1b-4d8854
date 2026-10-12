"""Entry point: measure safety localization on all three axes for one model.

    python -m src.evaluate                                  # smoke test, 135M
    python -m src.evaluate --model Qwen/Qwen2.5-0.5B-Instruct
    python -m src.evaluate --axes token,neuron              # just two axes
    python -m src.evaluate --table                          # reprint the table

Writes runs/<run_name>-<model>/results.json, whose "row" field is one row of
Table 1 in the paper, and appends the same row to runs/table.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import axis_direction, axis_neuron, axis_token
from .capability import evaluate_capability
from .config import REPO_ROOT, load_config, run_dir
from .data import build_probe_set, load_capability
from .metrics import assemble_row, load_rows, save_json, to_markdown_table
from .modeling import build_prompt, generate, load_model
from .refusal import over_refusal_rate

ALL_AXES = ("token", "neuron", "direction")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument("--model", default=None, help="override model.name")
    parser.add_argument("--run-name", default=None, help="override run_name")
    parser.add_argument("--axes", default=",".join(ALL_AXES),
                        help="comma-separated subset of token,neuron,direction")
    parser.add_argument("--n-harmful", type=int, default=None)
    parser.add_argument("--n-benign", type=int, default=None)
    parser.add_argument("--table", action="store_true",
                        help="only reprint the aggregated table from existing runs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)

    results_root = REPO_ROOT / str(cfg.get_path("output.dir", "runs"))
    if args.table:
        rows = load_rows(results_root)
        print(to_markdown_table(rows) if rows else "no results found yet")
        return 0

    for flag, key in (
        (args.model, "model.name"),
        (args.run_name, "run_name"),
        (args.n_harmful, "data.n_harmful"),
        (args.n_benign, "data.n_benign"),
    ):
        if flag is not None:
            cfg = cfg.override(key, flag)

    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    unknown = set(axes) - set(ALL_AXES)
    if unknown:
        parser_msg = f"unknown axes {sorted(unknown)}; choose from {ALL_AXES}"
        raise SystemExit(parser_msg)

    print(f"[eval] loading {cfg.model.name}")
    lm = load_model(cfg.model.name, dtype=cfg.model.dtype, device=cfg.model.device)
    print(f"[eval] params={lm.n_params/1e6:.1f}M  non-embedding="
          f"{lm.n_params_non_embedding/1e6:.1f}M  d_model={lm.d_model}  "
          f"d_mlp={lm.d_mlp}  layers={lm.n_layers}  device={lm.device}")

    probes = build_probe_set(cfg)
    capability_items = load_capability(int(cfg.data.n_capability), int(cfg.data.seed))
    print(f"[eval] {len(probes.harmful)} harmful, {len(probes.benign)} benign, "
          f"{len(capability_items)} capability items")

    token_result = (
        axis_token.run(lm, probes.harmful, cfg).as_dict()
        if "token" in axes else {}
    )
    neuron_result = (
        axis_neuron.run(lm, probes.harmful, probes.benign, capability_items, cfg).as_dict()
        if "neuron" in axes else {}
    )
    direction_result = (
        axis_direction.run(lm, probes.harmful, probes.benign, cfg).as_dict()
        if "direction" in axes else {}
    )

    print("[eval] measuring over-refusal on the benign control set")
    benign_completions = generate(
        lm,
        [build_prompt(lm, p) for p in probes.benign],
        max_new_tokens=int(cfg.model.max_new_tokens),
    )
    over_refusal = over_refusal_rate(benign_completions)

    print("[eval] measuring capability")
    capability = evaluate_capability(lm, capability_items).as_dict()

    row = assemble_row(
        model_name=cfg.model.name,
        n_params=lm.n_params,
        n_params_non_embedding=lm.n_params_non_embedding,
        d_model=lm.d_model,
        max_response_tokens=int(cfg.model.max_new_tokens),
        token_axis=token_result,
        neuron_axis=neuron_result,
        direction_axis=direction_result,
        capability=capability,
        over_refusal=over_refusal,
    )

    out = run_dir(cfg)
    save_json(
        {
            "config": dict(cfg),
            "probe_meta": probes.meta,
            "token_axis": token_result,
            "neuron_axis": neuron_result,
            "direction_axis": direction_result,
            "capability": capability,
            "over_refusal": round(over_refusal, 4),
            "row": row,
        },
        out / "results.json",
    )

    rows = load_rows(results_root)
    table = to_markdown_table(rows)
    (results_root / "table.md").write_text(table + "\n", encoding="utf-8")

    print(f"\n[eval] wrote {out / 'results.json'}")
    print(f"[eval] aggregated table at {results_root / 'table.md'}\n")
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
