"""Target-keyword loss and generation-success evaluation."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def keyword_token_ids(tokenizer, target_answer: str, keywords: Sequence[str]) -> list[int]:
    answer_ids = tokenizer(target_answer, add_special_tokens=False).input_ids
    pieces = [tokenizer.decode([token_id]).strip().lower() for token_id in answer_ids]
    normalized = [keyword.strip().lower() for keyword in keywords if keyword.strip()]
    matched = [
        token_id
        for token_id, piece in zip(answer_ids, pieces)
        if piece and any(keyword in piece or piece in keyword for keyword in normalized)
    ]
    if not matched:
        for keyword in keywords:
            matched.extend(tokenizer(" " + keyword.strip(), add_special_tokens=False).input_ids)
    if not matched:
        raise ValueError("No target-answer token matched --target-keywords.")
    return list(dict.fromkeys(int(token_id) for token_id in matched))


def target_loss(model, tokenizer, prompts, target_answer, keyword_ids, max_length):
    """Mean teacher-forced cross entropy at target-keyword positions."""
    device = next(model.parameters()).device
    keyword_set = set(int(token_id) for token_id in keyword_ids)
    losses = []
    for prompt in prompts:
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        answer_ids = tokenizer(target_answer, add_special_tokens=False).input_ids
        token_ids = (prompt_ids + answer_ids)[-max_length:]
        prompt_length = max(0, len(token_ids) - len(answer_ids))
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits.float()
        for position in range(prompt_length, len(token_ids)):
            token_id = token_ids[position]
            if token_id in keyword_set and position > 0:
                target = torch.tensor([token_id], dtype=torch.long, device=device)
                losses.append(F.cross_entropy(logits[:, position - 1, :], target))
    if not losses:
        raise ValueError("No keyword loss could be constructed.")
    return torch.stack(losses).mean()


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, repetition_penalty):
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated_ids = output[0, encoded.input_ids.shape[1] :].tolist()
    return tokenizer.decode(generated_ids, skip_special_tokens=True), generated_ids


def prefix_success_rate(model, tokenizer, prompts, target_text, max_new_tokens, repetition_penalty):
    target_ids = tokenizer(target_text, add_special_tokens=False).input_ids
    successes = 0
    for prompt in prompts:
        _, generated_ids = generate(model, tokenizer, prompt, max_new_tokens, repetition_penalty)
        successes += generated_ids[: len(target_ids)] == target_ids
    return successes / max(1, len(prompts))


@torch.no_grad()
def first_keyword_stats(model, tokenizer, prompts, keyword_ids):
    device = next(model.parameters()).device
    token_id = int(keyword_ids[0])
    probabilities, ranks = [], []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        logits = model(**encoded).logits[:, -1, :].float()
        probability = torch.softmax(logits, dim=-1)[0, token_id]
        rank = int((logits[0] > logits[0, token_id]).sum().item()) + 1
        probabilities.append(float(probability))
        ranks.append(rank)
    return sum(probabilities) / len(probabilities), sum(ranks) / len(ranks)

