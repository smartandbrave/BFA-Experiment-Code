"""Progressive bit search with explicit blocked-index handling.

The code performs simulated INT8 sign-bit flips. It identifies and blocks
searched positions; it does not implement a physical protection mechanism.
"""

from __future__ import annotations

import inspect
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import torch

from index_store import BlockedIndices, add_indices, normalize_blocked_indices

DEFAULT_MODULE_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


@dataclass(frozen=True)
class SearchStep:
    module_name: str
    indices: tuple[int, ...]
    loss_before: float
    loss_after: float


@dataclass
class _Candidate:
    module_name: str
    indices: list[int]
    quantized_weight: torch.Tensor
    floating_weight: torch.Tensor
    loss: float = math.inf


def _call_loss(
    loss_function: Callable,
    input_ids: torch.Tensor,
    key_token_ids: Sequence[int],
    model,
    tokenizer,
    penalty_factor: float,
) -> torch.Tensor:
    """Call compatible upstream loss functions without assuming extra kwargs."""
    values = {
        "inputs_ids": input_ids,
        "input_ids": input_ids,
        "key_token_ids": key_token_ids,
        "chatglm_model": model,
        "model": model,
        "chatglm_tokenizer": tokenizer,
        "tokenizer": tokenizer,
        "penalty_factor": penalty_factor,
    }
    signature = inspect.signature(loss_function)
    if any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()):
        return loss_function(**values)
    kwargs = {name: values[name] for name in signature.parameters if name in values}
    return loss_function(**kwargs)


class ProgressiveBitSearcher:
    def __init__(
        self,
        backward_loss: Callable,
        evaluation_loss: Callable,
        *,
        k_top: int = 10,
        blocked_indices: Mapping[str, Iterable[int]] | None = None,
        module_suffixes: Sequence[str] = DEFAULT_MODULE_SUFFIXES,
        penalty_factor: float = 50.0,
        log_path: str | Path | None = None,
    ) -> None:
        if k_top <= 0:
            raise ValueError("k_top must be positive.")
        self.backward_loss = backward_loss
        self.evaluation_loss = evaluation_loss
        self.k_top = int(k_top)
        self.blocked_indices: BlockedIndices = normalize_blocked_indices(blocked_indices)
        self.module_suffixes = tuple(module_suffixes)
        self.penalty_factor = float(penalty_factor)
        self.logger = self._build_logger(log_path)

    @staticmethod
    def _build_logger(log_path: str | Path | None) -> logging.Logger:
        logger = logging.getLogger(f"silentstriker-screen-{id(log_path)}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not logger.handlers:
            handler: logging.Handler
            if log_path is None:
                handler = logging.StreamHandler()
            else:
                destination = Path(log_path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                handler = logging.FileHandler(destination, encoding="utf-8")
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            )
            logger.addHandler(handler)
        return logger

    @staticmethod
    def _row_major_to_ampere(flat_index: int, rows: int, cols: int) -> int:
        tile_size = 32
        subtile_size = 8
        row_index = flat_index // cols
        column_index = flat_index % cols
        base_row = (row_index // tile_size) * tile_size
        base_column = (column_index // tile_size) * tile_size
        subrow = row_index % tile_size
        local_row = (
            ((subrow % subtile_size) // 2) * subtile_size
            + (subrow // subtile_size) * 2
            + (subrow % 2)
        )
        global_offset = (
            (base_column // tile_size) * rows * tile_size
            + (base_row // tile_size) * tile_size**2
        )
        return global_offset + local_row * tile_size + (column_index % tile_size)

    @staticmethod
    def _flip_int8_sign(value: int) -> int:
        unsigned = int(value) & 0xFF
        flipped = unsigned ^ 0x80
        return flipped - 256 if flipped >= 128 else flipped

    def _is_candidate_module(self, name: str, module, quantized_module) -> bool:
        if not name.endswith(self.module_suffixes):
            return False
        if not hasattr(module, "weight") or module.weight.grad is None:
            return False
        state = getattr(quantized_module, "state", None)
        return state is not None and hasattr(state, "CxB") and hasattr(state, "SCB")

    def _selected_row_major_indices(self, name: str, module, quantized_module) -> list[int]:
        gradients = module.weight.grad.detach().reshape(-1)
        scores = gradients.abs().clone()
        blocked = self.blocked_indices.get(name, set())
        valid_blocked = [index for index in blocked if 0 <= index < scores.numel()]
        if valid_blocked:
            blocked_tensor = torch.tensor(valid_blocked, device=scores.device, dtype=torch.long)
            scores[blocked_tensor] = -torch.inf

        available = int(torch.isfinite(scores).sum().item())
        if available == 0:
            return []
        candidate_count = min(available, scores.numel(), self.k_top * 10)
        candidate_indices = torch.topk(scores, candidate_count).indices

        rows, columns = module.weight.shape
        ampere_indices = torch.tensor(
            [
                self._row_major_to_ampere(int(index), rows, columns)
                for index in candidate_indices.tolist()
            ],
            device=quantized_module.state.CxB.device,
            dtype=torch.long,
        )
        quantized_values = quantized_module.state.CxB.detach().reshape(-1)[ampere_indices]

        selected: list[int] = []
        flat_gradients = gradients.reshape(-1)
        for row_major_index, quantized_value in zip(candidate_indices.tolist(), quantized_values):
            gradient_sign = torch.sign(flat_gradients[row_major_index]).item()
            parameter_sign = torch.sign(quantized_value).item()
            if (gradient_sign < 0 and parameter_sign < 0) or (
                gradient_sign > 0 and parameter_sign > 0
            ):
                selected.append(int(row_major_index))
                if len(selected) == self.k_top:
                    break
        return selected

    def _make_candidate(self, name: str, module, quantized_module) -> _Candidate | None:
        selected = self._selected_row_major_indices(name, module, quantized_module)
        if not selected:
            return None

        rows, columns = module.weight.shape
        row_indices = torch.tensor(selected, device=module.weight.device, dtype=torch.long)
        ampere_indices = torch.tensor(
            [self._row_major_to_ampere(index, rows, columns) for index in selected],
            device=quantized_module.state.CxB.device,
            dtype=torch.long,
        )

        quantized_flat = quantized_module.state.CxB.detach().reshape(-1).clone()
        original_values = quantized_flat[ampere_indices]
        flipped_values = torch.tensor(
            [self._flip_int8_sign(int(value.item())) for value in original_values],
            dtype=quantized_flat.dtype,
            device=quantized_flat.device,
        )
        quantized_flat[ampere_indices] = flipped_values

        scales = (
            quantized_module.state.SCB.reshape(-1, 1)
            .expand(-1, columns)
            .detach()
            .reshape(-1)[row_indices.to(quantized_module.state.SCB.device)]
        )
        floating_values = (flipped_values.to(scales.device) * scales / 127).to(
            dtype=module.weight.dtype,
            device=module.weight.device,
        )
        floating_flat = module.weight.detach().reshape(-1).clone()
        floating_flat[row_indices] = floating_values

        return _Candidate(
            module_name=name,
            indices=selected,
            quantized_weight=quantized_flat.view_as(quantized_module.state.CxB),
            floating_weight=floating_flat.view_as(module.weight),
        )

    def _dataset_loss(self, model, tokenizer, encoded_inputs, key_token_ids) -> torch.Tensor:
        total = None
        for input_ids, keys in zip(encoded_inputs, key_token_ids):
            value = _call_loss(
                self.evaluation_loss,
                input_ids,
                keys,
                model,
                tokenizer,
                self.penalty_factor,
            )
            total = value if total is None else total + value
        if total is None:
            raise ValueError("The attack dataset is empty.")
        return total

    def search_once(
        self,
        full_precision_model,
        quantized_model,
        tokenizer,
        encoded_full_inputs: Sequence[torch.Tensor],
        encoded_quantized_inputs: Sequence[torch.Tensor],
        key_token_ids: Sequence[Sequence[int]],
    ) -> SearchStep | None:
        """Run one PBS step and apply only a strictly loss-decreasing candidate."""
        full_precision_model.eval()
        quantized_model.eval()
        full_precision_model.zero_grad(set_to_none=True)

        backward_total = None
        for input_ids, keys in zip(encoded_full_inputs, key_token_ids):
            value = _call_loss(
                self.backward_loss,
                input_ids,
                keys,
                full_precision_model,
                tokenizer,
                self.penalty_factor,
            )
            backward_total = value if backward_total is None else backward_total + value
        if backward_total is None:
            raise ValueError("The attack dataset is empty.")
        backward_total.backward()

        with torch.no_grad():
            current_loss_tensor = self._dataset_loss(
                quantized_model,
                tokenizer,
                encoded_quantized_inputs,
                key_token_ids,
            )
        current_loss = float(current_loss_tensor.item())
        quantized_modules = dict(quantized_model.named_modules())
        candidates: list[_Candidate] = []

        for name, module in full_precision_model.named_modules():
            quantized_module = quantized_modules.get(name)
            if quantized_module is None or not self._is_candidate_module(
                name, module, quantized_module
            ):
                continue
            candidate = self._make_candidate(name, module, quantized_module)
            if candidate is None:
                continue

            clean_quantized = quantized_module.state.CxB.detach().clone()
            clean_floating = module.weight.detach().clone()
            try:
                quantized_module.state.CxB = candidate.quantized_weight
                module.weight.data = candidate.floating_weight
                with torch.no_grad():
                    candidate.loss = float(
                        self._dataset_loss(
                            quantized_model,
                            tokenizer,
                            encoded_quantized_inputs,
                            key_token_ids,
                        ).item()
                    )
            finally:
                quantized_module.state.CxB = clean_quantized
                module.weight.data = clean_floating
            if math.isfinite(candidate.loss) and candidate.loss < current_loss:
                candidates.append(candidate)
            self.logger.info(
                "module=%s current_loss=%.6f candidate_loss=%.6f flips=%d",
                name,
                current_loss,
                candidate.loss,
                len(candidate.indices),
            )

        if not candidates:
            self.logger.info("No candidate produced a lower attack loss.")
            return None

        best = min(candidates, key=lambda item: item.loss)
        target_module = dict(full_precision_model.named_modules())[best.module_name]
        target_quantized_module = quantized_modules[best.module_name]
        target_module.weight.data = best.floating_weight
        target_quantized_module.state.CxB = best.quantized_weight
        add_indices(self.blocked_indices, best.module_name, best.indices)

        full_precision_model.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return SearchStep(
            module_name=best.module_name,
            indices=tuple(best.indices),
            loss_before=current_loss,
            loss_after=best.loss,
        )
