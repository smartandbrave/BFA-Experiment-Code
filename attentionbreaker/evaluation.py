"""MMLU prompt construction and loss/accuracy evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from datasets import load_dataset

CHOICES = "ABCD"


@dataclass(frozen=True)
class MMLURecord:
    prompt: str
    answer: str


def _format_question(row, subject: str, include_answer: bool) -> str:
    text = (
        f"The following is a multiple-choice question about {subject.replace('_', ' ')}. "
        "Answer with only A, B, C, or D.\n"
        f"{row['question']}"
    )
    for letter, choice in zip(CHOICES, row["choices"]):
        text += f"\n{letter}. {choice}"
    text += "\nAnswer:"
    if include_answer:
        text += f" {CHOICES[int(row['answer'])]}\n\n"
    return text


def load_mmlu_records(subject: str = "astronomy", few_shot: int = 5) -> list[MMLURecord]:
    dataset = load_dataset("cais/mmlu", subject)
    demonstrations = "".join(
        _format_question(row, subject, include_answer=True)
        for row in dataset["dev"].select(range(min(few_shot, len(dataset["dev"]))))
    )
    return [
        MMLURecord(
            prompt=demonstrations + _format_question(row, subject, include_answer=False),
            answer=CHOICES[int(row["answer"])],
        )
        for row in dataset["test"]
    ]


def _encode(tokenizer, prompt: str, device: torch.device):
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    return {key: value.to(device) for key, value in encoded.items()}


def attack_loss(model, tokenizer, records: list[MMLURecord], sample_indices: list[int]) -> torch.Tensor:
    """Mean autoregressive loss over the configured loss samples."""
    device = next(model.parameters()).device
    losses = []
    for index in sample_indices:
        encoded = _encode(tokenizer, records[index].prompt, device)
        outputs = model(**encoded, labels=encoded["input_ids"], use_cache=False)
        losses.append(outputs.loss.float())
    if not losses:
        raise ValueError("At least one valid loss-sample index is required.")
    return torch.stack(losses).mean()


@torch.no_grad()
def accuracy(model, tokenizer, records: list[MMLURecord]) -> float:
    device = next(model.parameters()).device
    correct = 0
    for record in records:
        encoded = _encode(tokenizer, record.prompt, device)
        output = model.generate(**encoded, max_new_tokens=1, do_sample=False)
        token = output[0, encoded["input_ids"].shape[1] :]
        prediction = tokenizer.decode(token, skip_special_tokens=True).strip()[:1]
        correct += prediction == record.answer
    return correct / len(records)

