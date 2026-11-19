"""Sensitivity-percentage ablation over all attention and MLP projections."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis import analyze_matrices, collect_projection_gradients, results_as_dicts
from attentionbreaker import load_blocked, load_int8_model
from evaluation import attack_loss, load_mmlu_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AttentionBreaker sensitivity ablations.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", default="outputs/attentionbreaker/ablation.csv")
    parser.add_argument("--blocked-indices")
    parser.add_argument("--subject", default="astronomy")
    parser.add_argument("--loss-samples", default="51,101,21,31")
    parser.add_argument("--percentages", default="0.00001,0.0001,0.001,0.01,0.1,1,5,10")
    parser.add_argument("--loss-threshold", type=float, default=7.0)
    args = parser.parse_args()

    blocked = load_blocked(args.blocked_indices)
    model, tokenizer = load_int8_model(args.model_name)
    records = load_mmlu_records(args.subject)
    requested = [int(value) for value in args.loss_samples.split(",") if value.strip()]
    sample_indices = [index for index in requested if 0 <= index < len(records)]

    def differentiable_loss():
        return attack_loss(model, tokenizer, records, sample_indices)

    gradients = collect_projection_gradients(model, differentiable_loss)
    model.zero_grad(set_to_none=True)
    rows = []
    for percentage in (float(value) for value in args.percentages.split(",")):
        results = analyze_matrices(
            model=model,
            gradients=gradients,
            evaluate_loss=lambda: float(differentiable_loss().detach().cpu()),
            blocked=blocked,
            rank_fraction=percentage / 100.0,
            loss_threshold=args.loss_threshold,
        )
        rows.extend(results_as_dicts(results))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)


if __name__ == "__main__":
    main()

