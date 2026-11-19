"""Command-line workflow for sensitivity analysis and GenBFA optimization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from analysis import (
    analyze_matrices,
    build_candidate_pool,
    collect_projection_gradients,
    normalize_module_name,
    results_as_dicts,
    select_sensitive_matrix,
)
from evaluation import accuracy, attack_loss, load_mmlu_records
from genbfa import optimize

BlockedIndices = dict[str, set[int]]


def load_blocked(path: str | Path | None) -> BlockedIndices:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(name): {int(index) for index in indices} for name, indices in payload.items()}


def save_blocked(path: str | Path, values: Mapping[str, Iterable[int]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: sorted({int(index) for index in indices}) for name, indices in values.items()}
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_int8_model(model_name: str, device_map: str = "auto"):
    configuration = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device_map, quantization_config=configuration, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False
    model.eval()
    return model, tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AttentionBreaker over attention and MLP projections.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", default="outputs/attentionbreaker")
    parser.add_argument("--blocked-indices")
    parser.add_argument("--subject", default="astronomy")
    parser.add_argument("--loss-samples", default="51,101,21,31")
    parser.add_argument("--rank-percent", type=float, default=5.0)
    parser.add_argument("--subspace-percent", type=float, default=2.0)
    parser.add_argument("--loss-threshold", type=float, default=7.0)
    parser.add_argument("--success-accuracy", type=float, default=0.01)
    parser.add_argument("--max-generations", type=int, default=700)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--initial-random-solutions", type=int, default=50)
    parser.add_argument("--population-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blocked = load_blocked(args.blocked_indices)
    model, tokenizer = load_int8_model(args.model_name)
    records = load_mmlu_records(args.subject)
    requested_indices = [int(value) for value in args.loss_samples.split(",") if value.strip()]
    sample_indices = [index for index in requested_indices if 0 <= index < len(records)]

    def differentiable_loss():
        return attack_loss(model, tokenizer, records, sample_indices)

    model.zero_grad(set_to_none=True)
    gradients = collect_projection_gradients(model, differentiable_loss)
    model.zero_grad(set_to_none=True)

    def scalar_loss() -> float:
        return float(differentiable_loss().detach().cpu())

    matrix_results = analyze_matrices(
        model=model,
        gradients=gradients,
        evaluate_loss=scalar_loss,
        blocked=blocked,
        rank_fraction=args.rank_percent / 100.0,
        loss_threshold=args.loss_threshold,
    )
    pd.DataFrame(results_as_dicts(matrix_results)).to_csv(
        output_dir / "matrix_sensitivity.csv", index=False
    )
    selected = select_sensitive_matrix(matrix_results, args.loss_threshold)
    if selected is None:
        (output_dir / "run_summary.json").write_text(
            json.dumps(
                {"status": "no_matrix_reached_loss_threshold", "config": vars(args)},
                indent=2,
            ),
            encoding="utf-8",
        )
        return

    module_name = normalize_module_name(selected.parameter_name)
    candidate_indices = build_candidate_pool(
        model=model,
        parameter_name=selected.parameter_name,
        gradient=gradients[module_name],
        blocked=blocked,
        subspace_fraction=args.subspace_percent / 100.0,
    )
    parameter = dict(model.named_parameters())[selected.parameter_name]
    solution, history = optimize(
        parameter=parameter,
        candidate_indices=candidate_indices,
        evaluate_loss=scalar_loss,
        loss_threshold=args.loss_threshold,
        initial_random_solutions=args.initial_random_solutions,
        population_size=args.population_size,
        max_generations=args.max_generations,
        patience=args.patience,
        seed=args.seed,
    )
    blocked.setdefault(selected.parameter_name, set()).update(solution.indices)
    save_blocked(output_dir / "blocked_indices_latest.json", blocked)
    pd.DataFrame(history).to_csv(output_dir / "optimization_history.csv", index=False)

    summary = {
        "status": "optimization_complete",
        "config": vars(args),
        "sensitive_matrix": selected.parameter_name,
        "critical_indices": list(solution.indices),
        "critical_count": len(solution.indices),
        "attack_loss": solution.loss,
        "loss_threshold_reached": solution.loss >= args.loss_threshold,
    }
    if solution.loss >= args.loss_threshold:
        from analysis import flip_sign_bit
        import torch

        flat = parameter.data.reshape(-1)
        device_indices = torch.tensor(solution.indices, device=flat.device, dtype=torch.long)
        original = flat[device_indices].clone()
        try:
            flat[device_indices] = flip_sign_bit(original)
            summary["task_accuracy"] = accuracy(model, tokenizer, records)
            summary["attack_success"] = summary["task_accuracy"] < args.success_accuracy
        finally:
            flat[device_indices] = original
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

