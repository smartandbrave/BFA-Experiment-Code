"""Compact multi-round TFL screening program."""

from __future__ import annotations

import argparse
import gc
import heapq
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from objective import first_keyword_stats, generate, keyword_token_ids, prefix_success_rate, target_loss
from utility import accuracy, ranking_score

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
BitKey = tuple[str, int, int]


@dataclass(frozen=True)
class Candidate:
    score: float
    layer: str
    index: int
    bit: int
    raw_before: int
    raw_after: int
    delta_weight: float
    gradient: float


@dataclass(frozen=True)
class AcceptedFlip:
    step: int
    layer: str
    index: int
    bit: int
    raw_before: int
    raw_after: int
    target_loss: float
    target_success: float
    utility_accuracy: float
    ranking_score: float


def load_blocked(path: Path) -> set[BitKey]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {(str(layer), int(index), int(bit)) for layer, index, bit in payload}


def save_blocked(path: Path, values: set[BitKey]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [[layer, index, bit] for layer, index, bit in sorted(values)]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_model(model_name: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def select_parameters(model) -> list[str]:
    names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.dtype == torch.bfloat16
        and name.endswith(".weight")
        and any(projection in name for projection in PROJECTIONS)
        and not any(excluded in name for excluded in ("norm", "embed_tokens", "lm_head"))
    ]
    if not names:
        raise RuntimeError("No BF16 attention or MLP projection weights were found.")
    selected = set(names)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in selected)
    return names


def signed_to_uint16(value: int) -> int:
    return int(value) & 0xFFFF


def uint16_to_signed(value: int) -> int:
    value = int(value) & 0xFFFF
    return value - 0x10000 if value >= 0x8000 else value


def raw_to_float(value: int) -> float:
    raw = torch.tensor([uint16_to_signed(value)], dtype=torch.int16)
    return float(raw.view(torch.bfloat16).float().item())


def flip_bit(model, layer: str, index: int, bit: int) -> tuple[int, int]:
    parameter = dict(model.named_parameters())[layer]
    if not parameter.data.is_contiguous():
        parameter.data = parameter.data.contiguous()
    raw = parameter.data.view(torch.int16).reshape(-1)
    old = signed_to_uint16(int(raw[index]))
    new = old ^ (1 << bit)
    with torch.no_grad():
        raw[index] = uint16_to_signed(new)
    return old, new


def collect_gradients(model, tokenizer, prompts, answer, keyword_ids, max_length, parameter_names):
    model.zero_grad(set_to_none=True)
    loss = target_loss(model, tokenizer, prompts, answer, keyword_ids, max_length)
    loss.backward()
    selected = set(parameter_names)
    gradients = {
        name: parameter.grad.detach().cpu()
        for name, parameter in model.named_parameters()
        if name in selected and parameter.grad is not None
    }
    model.zero_grad(set_to_none=True)
    return gradients


def build_candidates(model, gradients, parameter_names, bits, global_topk, per_bit_topk, blocked):
    """Keep Top-60 per matrix/bit group, followed by global Top-300."""
    heap, counter = [], 0
    parameters = dict(model.named_parameters())
    for name in parameter_names:
        parameter, gradient = parameters[name], gradients.get(name)
        if gradient is None:
            continue
        weights = parameter.detach().reshape(-1).cpu()
        gradient = gradient.reshape(-1).float().cpu()
        count = min(weights.numel(), gradient.numel())
        weights, gradient = weights[:count], gradient[:count]
        lower, upper = float(weights.float().min()), float(weights.float().max())
        raw_uint = torch.bitwise_and(weights.view(torch.int16).to(torch.int32), 0xFFFF)
        old_weight = weights.float()

        for bit in bits:
            new_uint = torch.bitwise_xor(raw_uint, 1 << bit)
            new_signed = torch.where(new_uint >= 0x8000, new_uint - 0x10000, new_uint).to(torch.int16)
            new_weight = new_signed.view(torch.bfloat16).float()
            delta = new_weight - old_weight
            scores = gradient.abs() * delta.abs()
            valid = (
                torch.isfinite(scores)
                & torch.isfinite(new_weight)
                & (new_weight >= lower)
                & (new_weight <= upper)
                & (scores > 0)
            )
            valid_indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
            if valid_indices.numel() == 0:
                continue
            local_count = min(per_bit_topk, int(valid_indices.numel()))
            values, positions = torch.topk(scores[valid_indices], local_count)
            for score, position in zip(values.tolist(), positions.tolist()):
                index = int(valid_indices[position])
                if (name, index, bit) in blocked:
                    continue
                old_raw = int(raw_uint[index])
                new_raw = old_raw ^ (1 << bit)
                candidate = Candidate(
                    score=float(score),
                    layer=name,
                    index=index,
                    bit=int(bit),
                    raw_before=old_raw,
                    raw_after=new_raw,
                    delta_weight=raw_to_float(new_raw) - raw_to_float(old_raw),
                    gradient=float(gradient[index]),
                )
                counter += 1
                item = (candidate.score, counter, candidate)
                if len(heap) < global_topk:
                    heapq.heappush(heap, item)
                else:
                    heapq.heappushpop(heap, item)
    return [item[2] for item in sorted(heap, reverse=True)]


def loss_value(model, tokenizer, prompts, answer, keyword_ids, max_length):
    with torch.no_grad():
        return float(target_loss(model, tokenizer, prompts, answer, keyword_ids, max_length))


def run_round(model, tokenizer, parameter_names, prompts, answer, keyword_ids, bits, blocked, args):
    accepted = []
    unavailable = set(blocked)
    initial_utility = accuracy(model, tokenizer, args.utility_tasks, args.utility_samples, args.max_length)

    for step in range(1, args.max_steps + 1):
        current_success = prefix_success_rate(
            model, tokenizer, prompts, answer, args.max_new_tokens, args.repetition_penalty
        )
        if current_success >= 1.0:
            break
        base_loss = loss_value(model, tokenizer, prompts, answer, keyword_ids, args.max_length)
        base_utility = accuracy(model, tokenizer, args.utility_tasks, args.utility_samples, args.max_length)
        base_probability, base_rank = first_keyword_stats(model, tokenizer, prompts, keyword_ids)
        gradients = collect_gradients(
            model, tokenizer, prompts, answer, keyword_ids, args.max_length, parameter_names
        )
        candidates = build_candidates(
            model, gradients, parameter_names, bits, args.candidate_k, args.per_bit_topk, unavailable
        )
        if not candidates:
            break

        best = None
        for candidate in candidates:
            flip_bit(model, candidate.layer, candidate.index, candidate.bit)
            try:
                flipped_loss = loss_value(
                    model, tokenizer, prompts, answer, keyword_ids, args.max_length
                )
                flipped_utility = accuracy(
                    model, tokenizer, args.utility_tasks, args.utility_samples, args.max_length
                )
                success = prefix_success_rate(
                    model, tokenizer, prompts, answer, args.max_new_tokens, args.repetition_penalty
                )
                score = ranking_score(base_loss, flipped_loss, base_utility, flipped_utility)
                probability, rank = first_keyword_stats(model, tokenizer, prompts, keyword_ids)
                score -= args.keyword_weight * (
                    (probability - base_probability) / max(abs(base_probability), 1e-12)
                    + (base_rank - rank) / max(abs(base_rank), 1.0)
                )
                score -= args.success_bonus * (success - current_success)
                improved = (
                    flipped_loss < base_loss
                    or success > current_success
                    or probability > base_probability
                    or rank < base_rank
                )
            finally:
                flip_bit(model, candidate.layer, candidate.index, candidate.bit)
            if improved and (best is None or score < best[0]):
                best = (score, flipped_loss, success, flipped_utility, candidate)

        if best is None:
            break
        score, flipped_loss, success, flipped_utility, candidate = best
        old, new = flip_bit(model, candidate.layer, candidate.index, candidate.bit)
        unavailable.add((candidate.layer, candidate.index, candidate.bit))
        accepted.append(
            AcceptedFlip(
                step, candidate.layer, candidate.index, candidate.bit, old, new,
                flipped_loss, success, flipped_utility, score
            )
        )

    success = prefix_success_rate(
        model, tokenizer, prompts, answer, args.max_new_tokens, args.repetition_penalty
    )
    return accepted, {
        "target_loss": loss_value(model, tokenizer, prompts, answer, keyword_ids, args.max_length),
        "target_success": success,
        "attack_success": success >= 1.0,
        "initial_utility_accuracy": initial_utility,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Iterative TFL screening on BF16 projections.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--target-answer", required=True)
    parser.add_argument("--target-keywords", required=True, help="Separate keywords with ||")
    parser.add_argument("--output-dir", default="outputs/tfl")
    parser.add_argument("--blocked-bits")
    parser.add_argument("--round-start", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--candidate-k", type=int, default=300)
    parser.add_argument("--per-bit-topk", type=int, default=60)
    parser.add_argument("--bits", default="15,14,13,12,11,10,9,8,7,6,5,4,3")
    parser.add_argument("--utility-tasks", default="astronomy")
    parser.add_argument("--utility-samples", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--keyword-weight", type=float, default=0.25)
    parser.add_argument("--success-bonus", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blocked_path = Path(args.blocked_bits) if args.blocked_bits else output_dir / "blocked_bits_latest.json"
    blocked = load_blocked(blocked_path)
    prompts = [value.strip() for value in args.prompt.split("||") if value.strip()]
    keywords = [value.strip() for value in args.target_keywords.split("||") if value.strip()]
    bits = [int(value) for value in args.bits.split(",") if value.strip()]
    args.utility_tasks = [value.strip() for value in args.utility_tasks.split(",") if value.strip()]

    for round_id in range(args.round_start, args.round_start + args.rounds):
        torch.manual_seed(args.seed + round_id)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + round_id)
        model = tokenizer = None
        try:
            model, tokenizer = load_model(args.model_name, args.device)
            parameter_names = select_parameters(model)
            keyword_ids = keyword_token_ids(tokenizer, args.target_answer, keywords)
            before = [
                generate(model, tokenizer, prompt, args.max_new_tokens, args.repetition_penalty)[0]
                for prompt in prompts
            ]
            accepted, final = run_round(
                model, tokenizer, parameter_names, prompts, args.target_answer,
                keyword_ids, bits, blocked, args
            )
            new_bits = {(flip.layer, flip.index, flip.bit) for flip in accepted}
            blocked.update(new_bits)
            save_blocked(blocked_path, blocked)
            after = [
                generate(model, tokenizer, prompt, args.max_new_tokens, args.repetition_penalty)[0]
                for prompt in prompts
            ]
            parameters = dict(model.named_parameters())
            records = []
            for flip in accepted:
                record = asdict(flip)
                columns = int(parameters[flip.layer].shape[-1])
                record["row"], record["column"] = divmod(flip.index, columns)
                records.append(record)
            result = {
                "round": round_id,
                "model": args.model_name,
                "config": {
                    key: value for key, value in vars(args).items()
                    if key != "blocked_bits"
                },
                "before_outputs": before,
                "after_outputs": after,
                "accepted_flips": records,
                "new_blocked_bit_count": len(new_bits),
                "cumulative_blocked_bit_count": len(blocked),
                "final": final,
            }
            (output_dir / f"round_{round_id:04d}.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        finally:
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

