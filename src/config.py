"""Config loading. One YAML file in, one nested dict out, with dotted access."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


class Config(dict):
    """A dict that also supports cfg.model.name and cfg.get_path('model.name')."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) else value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def override(self, dotted: str, value: Any) -> "Config":
        """Return a copy with one dotted key replaced. Used by the CLI flags."""
        out = copy.deepcopy(dict(self))
        node = out
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
        return Config(out)


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return Config(raw)


def run_dir(cfg: Config, suffix: str = "") -> Path:
    """Directory for one run's artifacts. Created on demand."""
    name = cfg.get_path("run_name", "run")
    model = str(cfg.get_path("model.name", "model")).replace("/", "__")
    parts = [str(name), model] + ([suffix] if suffix else [])
    out = REPO_ROOT / str(cfg.get_path("output.dir", "runs")) / "-".join(parts)
    out.mkdir(parents=True, exist_ok=True)
    return out
