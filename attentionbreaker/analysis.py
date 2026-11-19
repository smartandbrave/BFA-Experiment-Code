"""Projection sensitivity analysis shared by the main and ablation workflows."""

from __future__ import annotations

import gc
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

import torch

BlockedIndices = dict[str, set[int]]
TARGET_PROJECTIONS = (
    ".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj", ".self_attn.o_proj",
    ".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj",
)


def normalize_module_name(parameter_name: str) -> str:
    return parameter_name.removesuffix(".weight")


def is_target_projection(name: str) -> bool:
    module_name = normalize_module_name(name)
    return any(module_name.endswith(suffix) for suffix in TARGET_PROJECTIONS)


@dataclass(frozen=True)
class MatrixResult:
    parameter_name: str
    flipped_count: int
    flipped_fraction: float
    perturbed_loss: float
    matrix_score: float


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def importance_score(weight: torch.Tensor, gradient: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    weight_abs = weight.float().abs()
    gradient_abs = gradient.float().abs()
    weight_norm = (weight_abs - weight_abs.min()) / (weight_abs.max() - weight_abs.min() + 1e-8)
    gradient_norm = (gradient_abs - gradient_abs.min()) / (
        gradient_abs.max() - gradient_abs.min() + 1e-8
    )
    return alpha * gradient_norm + (1.0 - alpha) * weight_norm


def flip_sign_bit(values: torch.Tensor) -> torch.Tensor:
    if values.dtype != torch.int8:
        raise TypeError("The released experiment workflow expects INT8 projection weights.")
    return torch.bitwise_xor(values, torch.tensor(-128, dtype=torch.int8, device=values.device))


def collect_projection_gradients(
    model, build_loss: Callable[[], torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Compute weight gradients for all MLP and attention projections.

    Quantized Linear8bitLt weights do not expose ordinary weight gradients. The
    function therefore retains each target module's input and output gradient
    and computes dW = dY^T X. The same rule is applied to all seven projection
    types, avoiding the MLP-only behavior of the experiment draft.
    """
    saved_inputs: dict[str, list[torch.Tensor]] = {}
    saved_outputs: dict[str, list[torch.Tensor]] = {}
    handles = []

    def hook_for(name: str):
        def hook(_module, inputs, output):
            saved_inputs.setdefault(name, []).append(inputs[0].detach())
            saved_outputs.setdefault(name, []).append(output)
            output.retain_grad()
        return hook

    for name, module in model.named_modules():
        if is_target_projection(name) and hasattr(module, "weight"):
            handles.append(module.register_forward_hook(hook_for(name)))

    if not handles:
        raise RuntimeError("No supported attention or MLP projection modules were found.")

    try:
        loss = build_loss()
        loss.backward()
        gradients: dict[str, torch.Tensor] = {}
        for name, outputs in saved_outputs.items():
            matrix_gradient = None
            for x_value, output in zip(saved_inputs[name], outputs):
                if output.grad is None:
                    continue
                x = x_value.reshape(-1, x_value.shape[-1]).float()
                dy = output.grad.reshape(-1, output.grad.shape[-1]).float()
                contribution = dy.T @ x
                matrix_gradient = (
                    contribution if matrix_gradient is None else matrix_gradient + contribution
                )
            if matrix_gradient is not None:
                gradients[name] = matrix_gradient.detach().cpu()
        return gradients
    finally:
        for handle in handles:
            handle.remove()


def _eligible_top_indices(
    parameter_name: str,
    score: torch.Tensor,
    count: int,
    blocked: BlockedIndices,
) -> torch.Tensor:
    blocked_set = blocked.get(parameter_name, set())
    if count <= 0:
        return torch.empty(0, dtype=torch.long)
    order = torch.argsort(score.reshape(-1), descending=True)
    if blocked_set:
        order = torch.tensor(
            [index for index in order.tolist() if index not in blocked_set],
            dtype=torch.long,
        )
    return order[:count]


def analyze_matrices(
    model,
    gradients: dict[str, torch.Tensor],
    evaluate_loss: Callable[[], float],
    blocked: BlockedIndices,
    rank_fraction: float,
    loss_threshold: float,
    alpha: float = 0.5,
) -> list[MatrixResult]:
    """Perturb the top-ranked weights of every target projection matrix."""
    results: list[MatrixResult] = []
    for parameter_name, parameter in model.named_parameters():
        if not is_target_projection(parameter_name) or parameter.dtype != torch.int8:
            continue
        module_name = normalize_module_name(parameter_name)
        gradient = gradients.get(module_name)
        if gradient is None or gradient.numel() != parameter.numel():
            continue

        flat_weight = parameter.data.reshape(-1)
        score = importance_score(flat_weight.detach().cpu(), gradient.reshape(-1), alpha)
        requested = max(1, int(rank_fraction * parameter.numel()))
        indices = _eligible_top_indices(parameter_name, score, requested, blocked)
        if indices.numel() == 0:
            continue

        original = flat_weight[indices.to(flat_weight.device)].clone()
        try:
            flat_weight[indices.to(flat_weight.device)] = flip_sign_bit(original)
            perturbed_loss = float(evaluate_loss())
        finally:
            flat_weight[indices.to(flat_weight.device)] = original

        sign = 1.0 if perturbed_loss >= loss_threshold else -1.0
        results.append(
            MatrixResult(
                parameter_name=parameter_name,
                flipped_count=int(indices.numel()),
                flipped_fraction=rank_fraction,
                perturbed_loss=perturbed_loss,
                matrix_score=sign * perturbed_loss / int(indices.numel()),
            )
        )
        clear_memory()
    return results


def results_as_dicts(results: Iterable[MatrixResult]) -> list[dict]:
    return [asdict(result) for result in results]


def select_sensitive_matrix(results: Iterable[MatrixResult], loss_threshold: float) -> MatrixResult | None:
    eligible = [result for result in results if result.perturbed_loss >= loss_threshold]
    return max(eligible, key=lambda result: result.matrix_score, default=None)


def build_candidate_pool(
    model,
    parameter_name: str,
    gradient: torch.Tensor,
    blocked: BlockedIndices,
    subspace_fraction: float,
    alpha: float = 0.5,
) -> list[int]:
    parameter = dict(model.named_parameters())[parameter_name]
    score = importance_score(parameter.data.reshape(-1).detach().cpu(), gradient.reshape(-1), alpha)
    requested = max(1, int(subspace_fraction * parameter.numel()))
    return _eligible_top_indices(parameter_name, score, requested, blocked).tolist()

