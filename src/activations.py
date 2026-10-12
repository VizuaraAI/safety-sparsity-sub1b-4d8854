"""Activation capture and ablation.

Two read sites and two write sites, all implemented with plain PyTorch hooks so
that the code works on any Llama-family model without a wrapper library:

  read  MLP post-activation  forward_pre_hook on layer.mlp.down_proj
        (its input is the gated hidden of width intermediate_size, which is
        exactly the "neuron" unit the safety-neuron literature uses)
  read  residual stream      output_hidden_states=True on the forward pass

  write neuron ablation      forward_pre_hook that zeroes selected channels
  write directional ablation forward hook on every layer output that removes
        the component along a set of directions, applied everywhere in the
        residual stream so the direction cannot be re-introduced downstream

Every hook context manager removes its handles on exit. If you add a hook, add
it here, not inline in an experiment script, or you will leak handles and spend
an afternoon debugging activations that are stale by one run.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Sequence

import torch

from .modeling import LoadedModel


# ---------------------------------------------------------------- reads

@contextmanager
def capture_mlp_activations(lm: LoadedModel, store: dict[int, torch.Tensor]):
    """Fill `store[layer_idx]` with the MLP post-activation, shape (B, T, d_mlp)."""
    handles = []

    def make_hook(idx: int):
        def hook(_module, args):
            store[idx] = args[0].detach()
            return None
        return hook

    for idx, layer in enumerate(lm.layers):
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(make_hook(idx)))
    try:
        yield store
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def mean_mlp_activation(
    lm: LoadedModel, prompts: Sequence[str], batch_size: int = 8
) -> torch.Tensor:
    """Mean |activation| per neuron over the last token of each prompt.

    Shape (n_layers, d_mlp). The last token is used because that is the
    position at which the refusal decision is first expressed; sweep this to
    other positions once the pipeline runs, it is a real design choice and
    worth an ablation row.
    """
    totals = torch.zeros(lm.n_layers, lm.d_mlp, dtype=torch.float32)
    seen = 0
    for start in range(0, len(prompts), batch_size):
        chunk = list(prompts[start : start + batch_size])
        enc = lm.tokenizer(chunk, return_tensors="pt", padding=True).to(lm.device)
        store: dict[int, torch.Tensor] = {}
        with capture_mlp_activations(lm, store):
            lm.model(**enc)
        for idx in range(lm.n_layers):
            last = store[idx][:, -1, :].abs().float().cpu()
            totals[idx] += last.sum(dim=0)
        seen += len(chunk)
    return totals / max(seen, 1)


@torch.no_grad()
def mean_residual(
    lm: LoadedModel, prompts: Sequence[str], batch_size: int = 8
) -> torch.Tensor:
    """Mean residual-stream vector at the last token. Shape (n_layers + 1, d_model)."""
    totals = torch.zeros(lm.n_layers + 1, lm.d_model, dtype=torch.float32)
    seen = 0
    for start in range(0, len(prompts), batch_size):
        chunk = list(prompts[start : start + batch_size])
        enc = lm.tokenizer(chunk, return_tensors="pt", padding=True).to(lm.device)
        out = lm.model(**enc, output_hidden_states=True)
        for idx, hidden in enumerate(out.hidden_states):
            totals[idx] += hidden[:, -1, :].float().cpu().sum(dim=0)
        seen += len(chunk)
    return totals / max(seen, 1)


# ---------------------------------------------------------------- writes

@contextmanager
def ablate_neurons(lm: LoadedModel, mask: dict[int, torch.Tensor]):
    """Zero the neurons named in `mask[layer_idx]` (a bool tensor of width d_mlp)."""
    handles = []

    def make_hook(keep: torch.Tensor):
        def hook(_module, args):
            hidden = args[0]
            return (hidden * keep.to(hidden.device, hidden.dtype),)
        return hook

    for idx, layer in enumerate(lm.layers):
        if idx not in mask:
            continue
        keep = (~mask[idx]).view(1, 1, -1)
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(make_hook(keep)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _project_out(hidden: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Remove the component of `hidden` lying in the span of `basis`.

    basis is (r, d_model) and assumed orthonormal, which orthonormal_basis
    guarantees.
    """
    b = basis.to(hidden.device, hidden.dtype)
    coeffs = hidden @ b.T            # (..., r)
    return hidden - coeffs @ b       # (..., d_model)


def orthonormal_basis(directions: torch.Tensor) -> torch.Tensor:
    """Orthonormalize (r, d_model) candidate directions, dropping degenerate ones."""
    mat = directions.float()
    mat = mat / (mat.norm(dim=-1, keepdim=True) + 1e-8)
    q, _ = torch.linalg.qr(mat.T)    # (d_model, r)
    return q.T.contiguous()


@contextmanager
def ablate_directions(lm: LoadedModel, basis: torch.Tensor):
    """Project a direction subspace out of the residual stream everywhere.

    Applied at the embedding output and at every decoder layer output, because
    a direction removed at one layer is otherwise written back by the next one.
    This is the whole-stream version of directional ablation.
    """
    handles = []

    def embed_hook(_module, _args, output):
        return _project_out(output, basis)

    def layer_hook(_module, _args, output):
        if isinstance(output, tuple):
            return (_project_out(output[0], basis),) + output[1:]
        return _project_out(output, basis)

    handles.append(lm.model.get_input_embeddings().register_forward_hook(embed_hook))
    for layer in lm.layers:
        handles.append(layer.register_forward_hook(layer_hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def topk_mask(scores: torch.Tensor, fraction: float) -> dict[int, torch.Tensor]:
    """Turn a (n_layers, d_mlp) score matrix into a per-layer boolean mask.

    The top `fraction` of neurons model-wide are selected, not per layer, so
    that the selection is free to concentrate in a few layers. That freedom is
    the point: if safety really does live in a few middle layers, the mask
    should discover it rather than be told.
    """
    flat = scores.flatten()
    k = max(1, int(round(fraction * flat.numel())))
    threshold = torch.topk(flat, k).values.min()
    selected = scores >= threshold
    return {idx: selected[idx] for idx in range(scores.shape[0])}
