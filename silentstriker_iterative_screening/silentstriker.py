"""Command-line runner for iterative SilentStriker parameter screening."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import spacy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from index_store import count_indices, load_blocked_indices, save_blocked_indices
from loss_1 import loss_func as backward_loss
from loss_2 import loss_func as evaluation_loss
from search import ProgressiveBitSearcher

EXCLUDED_POS = {
    "DET",
    "ADP",
    "AUX",
    "CONJ",
    "PRON",
    "PUNCT",
    "CCONJ",
    "SCONJ",
    "VERB",
    "PART",
    "ADV",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Iteratively search SilentStriker-sensitive INT8 parameters while "
            "excluding indices that were selected in earlier rounds."
        )
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset", required=True, help="JSON list of objects with a question field.")
    parser.add_argument("--output-dir", default="outputs/silentstriker_screening")
    parser.add_argument(
        "--blocked-indices",
        default=None,
        help="Optional JSON snapshot. Omit this option to start from an empty blocked set.",
    )
    parser.add_argument("--start-round", type=int, default=1)
    parser.add_argument("--num-rounds", type=int, default=1)
    parser.add_argument("--max-search-steps", type=int, default=300)
    parser.add_argument("--k-top", type=int, default=10)
    parser.add_argument("--early-stop-loss", type=float, default=15.0)
    parser.add_argument("--penalty-factor", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--key-token-generation-length", type=int, default=40)
    parser.add_argument("--evaluation-generation-length", type=int, default=100)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Dataset must be a non-empty JSON list.")
    for row in dataset:
        if not isinstance(row, dict) or not isinstance(row.get("question"), str):
            raise ValueError("Every dataset row must contain a string question field.")
    return dataset


def load_models(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    full_precision_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    quantized_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map="auto",
    )
    full_precision_model.config.pad_token_id = tokenizer.pad_token_id
    quantized_model.config.pad_token_id = tokenizer.pad_token_id
    full_precision_model.config.use_cache = False
    return full_precision_model, quantized_model, tokenizer


def _content_words(text: str, nlp) -> list[str]:
    return [token.text for token in nlp(text) if token.pos_ not in EXCLUDED_POS]


def extract_key_token_ids(
    questions: Sequence[str],
    quantized_model,
    tokenizer,
    nlp,
    max_new_tokens: int,
) -> list[list[int]]:
    all_ids: list[list[int]] = []
    for question in questions:
        encoded = tokenizer(question, return_tensors="pt", truncation=True)
        input_ids = encoded.input_ids.to(quantized_model.device)
        with torch.no_grad():
            generated = quantized_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                no_repeat_ngram_size=2,
                repetition_penalty=1.2,
            )
        answer_ids = generated[0, input_ids.shape[-1] :]
        answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
        question_words = set(_content_words(question, nlp))
        words = [word for word in _content_words(answer, nlp) if word not in question_words]

        variants: list[str] = []
        for word in dict.fromkeys(words):
            variants.extend((word, f" {word}", word.upper(), f" {word.upper()}"))
        token_ids: set[int] = set()
        for variant in variants:
            token_ids.update(tokenizer.encode(variant, add_special_tokens=False))
        token_ids.discard(tokenizer.pad_token_id)
        if tokenizer.unk_token_id is not None:
            token_ids.discard(tokenizer.unk_token_id)
        if not token_ids:
            raise RuntimeError(f"No key tokens were extracted for question: {question}")
        all_ids.append(sorted(token_ids))
    return all_ids


def encode_questions(questions: Sequence[str], tokenizer, device) -> list[torch.Tensor]:
    return [
        tokenizer(question, return_tensors="pt", truncation=True).input_ids.to(device)
        for question in questions
    ]


def generate_outputs(model, tokenizer, questions: Sequence[str], max_new_tokens: int) -> list[str]:
    outputs: list[str] = []
    for question in questions:
        input_ids = tokenizer(question, return_tensors="pt", truncation=True).input_ids.to(
            model.device
        )
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                no_repeat_ngram_size=2,
                repetition_penalty=1.2,
            )
        outputs.append(tokenizer.decode(generated[0], skip_special_tokens=True))
    return outputs


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> None:
    if args.start_round <= 0 or args.num_rounds <= 0:
        raise ValueError("start-round and num-rounds must be positive.")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.dataset)
    questions = [row["question"] for row in dataset]
    global_blocked = load_blocked_indices(args.blocked_indices)
    nlp = spacy.load(args.spacy_model)
    record_path = output_dir / "round_records.jsonl"

    run_config = vars(args).copy()
    run_config["hf_token_present"] = bool(os.environ.get("HF_TOKEN"))
    run_config.pop("blocked_indices", None)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    end_round = args.start_round + args.num_rounds
    for round_id in range(args.start_round, end_round):
        set_seed(args.seed + round_id)
        full_model, quantized_model, tokenizer = load_models(args.model_name)
        try:
            key_token_ids = extract_key_token_ids(
                questions,
                quantized_model,
                tokenizer,
                nlp,
                args.key_token_generation_length,
            )
            full_inputs = encode_questions(questions, tokenizer, full_model.device)
            quantized_inputs = encode_questions(questions, tokenizer, quantized_model.device)
            before = {name: set(indices) for name, indices in global_blocked.items()}
            searcher = ProgressiveBitSearcher(
                backward_loss,
                evaluation_loss,
                k_top=args.k_top,
                blocked_indices=before,
                penalty_factor=args.penalty_factor,
                log_path=output_dir / "logs" / f"round_{round_id:04d}.log",
            )

            steps: list[dict] = []
            stop_reason = "maximum_search_steps"
            for step_number in range(1, args.max_search_steps + 1):
                step = searcher.search_once(
                    full_model,
                    quantized_model,
                    tokenizer,
                    full_inputs,
                    quantized_inputs,
                    key_token_ids,
                )
                if step is None:
                    stop_reason = "no_loss_decreasing_candidate"
                    break
                steps.append(
                    {
                        "step": step_number,
                        "module": step.module_name,
                        "indices": list(step.indices),
                        "loss_before": step.loss_before,
                        "loss_after": step.loss_after,
                    }
                )
                if step.loss_after < args.early_stop_loss:
                    stop_reason = "loss_threshold"
                    break

            round_new: dict[str, list[int]] = {}
            for module_name, indices in searcher.blocked_indices.items():
                new_indices = sorted(indices - before.get(module_name, set()))
                if new_indices:
                    round_new[module_name] = new_indices
                global_blocked.setdefault(module_name, set()).update(indices)

            snapshot = output_dir / "blocked_indices" / f"round_{round_id:04d}.json"
            save_blocked_indices(snapshot, global_blocked)
            generated_outputs = generate_outputs(
                quantized_model,
                tokenizer,
                questions,
                args.evaluation_generation_length,
            )
            append_jsonl(
                record_path,
                {
                    "round": round_id,
                    "stop_reason": stop_reason,
                    "steps": steps,
                    "round_flips": round_new,
                    "new_blocked_count": sum(len(v) for v in round_new.values()),
                    "total_blocked_count": count_indices(global_blocked),
                    "generated_outputs": generated_outputs,
                },
            )
        finally:
            del full_model, quantized_model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    save_blocked_indices(output_dir / "blocked_indices_latest.json", global_blocked)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
