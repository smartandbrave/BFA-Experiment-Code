"""Differentiable SilentStriker attack loss used for gradient calculation."""

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

    key_loss = 0.0
    repetition_loss = torch.tensor(0.0, device=device)
    token_counts = defaultdict(int)
    whitespace_total = 0
    past = None

    for step in range(max_length):
        with torch.no_grad():
            if past is None:
                inference_input = inputs_ids
                inference_position = position_ids
            else:
                inference_input = next_token_id
                inference_position = torch.tensor(
                    [[inputs_ids.size(-1) - 1]], device=device
                )

            no_grad_output = chatglm_model(
                input_ids=inference_input,
                past_key_values=past,
                position_ids=inference_position,
                use_cache=True,
            )
            no_grad_logits = no_grad_output.logits[:, -1, :]
            no_grad_logits = logits_processor(inputs_ids, no_grad_logits)
            no_grad_probabilities = torch.softmax(no_grad_logits, dim=-1)
            top5_token_ids = (
                torch.topk(no_grad_probabilities, 5, dim=-1)
                .indices.squeeze(0)
                .tolist()
            )
            next_token_id = torch.argmax(no_grad_logits, dim=-1, keepdim=True)
            next_past = no_grad_output.past_key_values

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
            gradient_output = chatglm_model(
                input_ids=inference_input,
                past_key_values=past,
                position_ids=inference_position,
                use_cache=True,
            )
            gradient_logits = gradient_output.logits[:, -1, :]
            gradient_logits = logits_processor(inputs_ids, gradient_logits)
            gradient_probabilities = torch.softmax(gradient_logits, dim=-1)
            key_probabilities = gradient_probabilities[:, key_token_ids].sum()
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
    return minimum_loss + perplexity_loss + repetition_loss + entropy_loss
