"""Auxiliary five-shot MMLU accuracy."""

from __future__ import annotations

import torch
from datasets import load_dataset

CHOICES = "ABCD"


def _prompt(row, subject: str, include_answer: bool = False) -> str:
    text = f"The following is a multiple-choice question about {subject.replace('_', ' ')}.\n{row['question']}"
    for letter, choice in zip(CHOICES, row["choices"]):
        text += f"\n{letter}. {choice}"
    text += "\nAnswer:"
    if include_answer:
        text += f" {CHOICES[int(row['answer'])]}\n\n"
    return text


@torch.no_grad()
def accuracy(model, tokenizer, tasks, samples, max_length):
    device = next(model.parameters()).device
    correct = total = 0
    for task in tasks:
        dataset = load_dataset("cais/mmlu", task)
        demonstrations = [
            _prompt(row, task, include_answer=True)
            for row in dataset["dev"].select(range(min(5, len(dataset["dev"]))))
        ]
        rows = dataset["test"]
        for row in rows.select(range(min(samples, len(rows)))):
            question = _prompt(row, task)
            active_demonstrations = list(demonstrations)
            prompt = "".join(active_demonstrations) + question
            while active_demonstrations and len(tokenizer(prompt).input_ids) > max_length:
                active_demonstrations.pop(0)
                prompt = "".join(active_demonstrations) + question
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(device)
            output = model.generate(**encoded, max_new_tokens=1, do_sample=False)
            generated = output[0, encoded.input_ids.shape[1] :]
            prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()[:1]
            correct += prediction == CHOICES[int(row["answer"])]
            total += 1
    return correct / max(1, total)


def ranking_score(base_loss, flipped_loss, base_accuracy, flipped_accuracy):
    target_improvement = (base_loss - flipped_loss) / max(abs(base_loss), 1e-12)
    utility_drop = (base_accuracy - flipped_accuracy) / max(abs(base_accuracy), 1e-12)
    return -target_improvement + utility_drop

