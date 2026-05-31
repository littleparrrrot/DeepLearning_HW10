from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from torch import nn
from torch.utils.data import DataLoader

from hw.constants import IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN
from hw.dataset import MathVQADataset
from hw.model import MathVLM, ModelConfig
from hw.processor import MathVLMProcessor, ProcessorConfig


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SimpleTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 1

        self.vocab = {
            "<pad>": self.pad_token_id,
            "<eos>": self.eos_token_id,
            IMAGE_START_TOKEN: 2,
            IMAGE_TOKEN: 3,
            IMAGE_END_TOKEN: 4,
        }

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab[token]

    def __len__(self) -> int:
        return 5000

    def _token_to_id(self, token: str) -> int:
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab)

        return self.vocab[token]

    def __call__(self, text: str, max_length: int | None = None, truncation: bool = False) -> dict[str, list[int]]:
        tokens = text.split()

        input_ids = [self._token_to_id(token) for token in tokens]
        input_ids.append(self.eos_token_id)

        if truncation and max_length is not None:
            input_ids = input_ids[:max_length]

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class TinyVisionEncoder(nn.Module):
    def __init__(self, hidden_size: int, num_tokens: int) -> None:
        super().__init__()
        self.linear = nn.Linear(3, hidden_size)
        self.num_tokens = num_tokens

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        x = pixel_values.mean(dim=(2, 3))
        x = self.linear(x)
        x = x.unsqueeze(1).repeat(1, self.num_tokens, 1)

        return {"last_hidden_state": x}


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> Any:
        logits = self.head(inputs_embeds)

        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits}

    def generate(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 1,
        **kwargs: Any,
    ) -> torch.Tensor:
        logits = self.head(inputs_embeds)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        return next_token.repeat(1, max_new_tokens)


def train_one_step(model: torch.nn.Module, batch: dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> float:
    """Run one optimization step and return scalar loss."""
    model.train()
    optimizer.zero_grad()

    outputs = model(batch)

    if isinstance(outputs, dict):
        loss = outputs["loss"]
    else:
        loss = outputs.loss

    if not torch.isfinite(loss):
        raise ValueError("Loss is not finite")

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return float(loss.item())


def run_training(config: dict[str, Any], fast_train: bool = False) -> None:
    """Main training entry point."""
    device = torch.device("cpu")

    data_config = config.get("data", {})
    model_config = config.get("model", {})
    processor_config = config.get("processor", {})
    trainer_config = config.get("trainer", {})

    if fast_train:
        max_steps = 2
        max_samples = 8
    else:
        max_steps = int(trainer_config.get("max_steps", 3))
        max_samples = data_config.get("max_samples")

    batch_size = int(trainer_config.get("local_batch_size", 1))
    learning_rate = float(trainer_config.get("learning_rate", 5e-4))
    weight_decay = float(trainer_config.get("weight_decay", 0.0))

    global_batch_size = int(trainer_config.get("global_batch_size", batch_size))
    gradient_accumulation_steps = max(1, global_batch_size // batch_size)

    tokenizer = SimpleTokenizer()

    processor = MathVLMProcessor(
        tokenizer=tokenizer,
        config=ProcessorConfig(**processor_config),
    )

    dataset = MathVQADataset(
        manifest_path=data_config["train_manifest"],
        split=data_config.get("split", "train"),
        max_samples=max_samples,
    )

    def collate_samples(samples):
        items = [processor(sample) for sample in samples]
        return processor.collate(items)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(trainer_config.get("num_workers", 0)),
        collate_fn=collate_samples,
    )

    hidden_size = 32

    vision_encoder = TinyVisionEncoder(
        hidden_size=hidden_size,
        num_tokens=processor.config.num_image_tokens,
    )

    language_model = TinyLanguageModel(
        vocab_size=len(tokenizer),
        hidden_size=hidden_size,
    )

    vlm_config = ModelConfig(
        vision_hidden_size=hidden_size,
        text_hidden_size=hidden_size,
        num_image_tokens=processor.config.num_image_tokens,
        image_token_id=tokenizer.convert_tokens_to_ids(IMAGE_TOKEN),
    )

    model = MathVLM(
        vision_encoder=vision_encoder,
        language_model=language_model,
        config=vlm_config,
    ).to(device)

    if model_config.get("freeze_vision", True) or model_config.get("freeze_llm", True):
        model.freeze_backbones()

    optimizer = torch.optim.AdamW(
        model.adapter.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    model.train()
    optimizer.zero_grad()

    step = 0
    optimizer_step = 0

    while optimizer_step < max_steps:
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}

            outputs = model(batch)

            if isinstance(outputs, dict):
                loss = outputs["loss"]
            else:
                loss = outputs.loss

            if not torch.isfinite(loss):
                raise ValueError("Loss is not finite")

            loss = loss / gradient_accumulation_steps
            loss.backward()

            step += 1

            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

                optimizer_step += 1
                print(f"step {optimizer_step}: loss = {loss.item() * gradient_accumulation_steps:.4f}")

                if optimizer_step >= max_steps:
                    break

    save_path = trainer_config.get("save_checkpoint_path")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "adapter": model.adapter.state_dict(),
                "config": config,
            },
            save_path,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fast-train", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    run_training(config, fast_train=args.fast_train)


if __name__ == "__main__":
    main()
