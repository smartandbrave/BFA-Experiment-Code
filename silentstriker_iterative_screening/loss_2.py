"""Non-gradient SilentStriker attack loss used for candidate evaluation."""

from __future__ import annotations

import os
from collections import defaultdict

import torch
from transformers import (
    LogitsProcessorList,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
)


def entropy_weight_term(logits, len_q):
    """Mean answer-span entropy multiplied by ENTROPY_W."""
    weight = float(os.environ.get("ENTROPY_W", "8.0"))
    answer_logits = logits[:, len_q - 1 :, :]
    if answer_logits.size(1) == 0:
        answer_logits = logits[:, -1:, :]
    answer_logits = answer_logits.float()
    log_probabilities = torch.log_softmax(answer_logits, dim=-1)
    probabilities = torch.softmax(answer_logits, dim=-1)
    entropy = -(probabilities * log_probabilities).sum(dim=-1).mean()
    return weight * entropy


def loss_func(
    inputs_ids,
    key_token_ids,
    chatglm_model,
    chatglm_tokenizer,
    forbidden_chars=None,
    max_length=50,
    repetition_penalty=1.2,
    ngram_size=2,
):
    del forbidden_chars
    device = inputs_ids.device
    len_q = inputs_ids.size(-1)
    attention_mask = torch.ones(1, len_q, device=device)
    position_ids = torch.arange(0, len_q, device=device).unsqueeze(0)
    logits_processor = LogitsProcessorList(
        [
            NoRepeatNGramLogitsProcessor(ngram_size),
            RepetitionPenaltyLogitsProcessor(repetition_penalty),
        ]
    )

    key_loss = torch.tensor(0.0, device=device)
    repetition_loss = torch.tensor(0.0, device=device)
    token_counts = defaultdict(int)
    whitespace_total = 0
    raw_token_ids: list[int] = []
    past = None

    with torch.no_grad():
        for step in range(max_length):
            if past is None:
                inference_input = inputs_ids
                inference_position = position_ids
            else:
                inference_input = next_token_id
                inference_position = torch.tensor(
                    [[inputs_ids.size(-1) - 1]], device=device
                )

            output = chatglm_model(
                input_ids=inference_input,
                past_key_values=past,
                position_ids=inference_position,
                use_cache=True,
            )
            logits = output.logits[:, -1, :]
            raw_token_ids.append(int(torch.argmax(logits, dim=-1).item()))
            logits = logits_processor(inputs_ids, logits)
            probabilities = torch.softmax(logits, dim=-1)
            top5_token_ids = torch.topk(probabilities, 5, dim=-1).indices.squeeze(0).tolist()
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            next_past = output.past_key_values

            token_id = next_token_id.item()
            token_counts[token_id] += 1
            if not chatglm_tokenizer.decode([token_id]).strip():
                whitespace_total += 1
                if whitespace_total > 3:
                    repetition_loss = repetition_loss + torch.tensor(
                        50.0 * float(whitespace_total - 3),
                        dtype=torch.float32,
                        device=device,
                    )

            if any(key_id in top5_token_ids for key_id in key_token_ids):
                key_probabilities = probabilities[:, key_token_ids].sum()
                key_loss = key_loss + 2 * key_probabilities

            past = next_past
            inputs_ids = torch.cat([inputs_ids, next_token_id], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones(1, 1, device=device)], dim=-1
            )
            if token_id == chatglm_tokenizer.eos_token_id:
                break

        cross_entropy_input = inputs_ids[:, :-1]
        outputs = chatglm_model(cross_entropy_input, labels=cross_entropy_input)
        perplexity_weight = float(os.environ.get("PPL_W", "1.0"))
        perplexity_loss = perplexity_weight * torch.exp(outputs.loss)
        key_penalty_weight = 5.0
        minimum_loss = key_penalty_weight * ((key_loss + 1) ** 2 - 1)
        entropy_loss = entropy_weight_term(outputs.logits, len_q)

        repetition_weight = float(os.environ.get("REP_W", "30.0"))
        diversity_weight = float(os.environ.get("DIV_W", "30.0"))
        if len(raw_token_ids) >= 2:
            bigrams = list(zip(raw_token_ids, raw_token_ids[1:]))
            repeated_bigram_ratio = 1.0 - len(set(bigrams)) / len(bigrams)
            distinct_ratio = len(set(raw_token_ids)) / len(raw_token_ids)
        else:
            repeated_bigram_ratio = 0.0
            distinct_ratio = 1.0
        degeneration_penalty = torch.tensor(
            repetition_weight * repeated_bigram_ratio
            + diversity_weight * (1.0 - distinct_ratio),
            device=device,
        )
        return (
            minimum_loss
            + perplexity_loss
            + repetition_loss
            + entropy_loss
            + degeneration_penalty
        )
