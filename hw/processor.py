from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from hw.constants import IGNORE_INDEX, IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN
from hw.dataset import MathVQASample


@dataclass
class ProcessorConfig:
    image_size: int = 224
    num_tiles: int = 1
    tile_overlap: float = 0.0
    num_image_tokens: int = 49
    max_length: int = 512
    ignore_index: int = IGNORE_INDEX


class MathVLMProcessor:
    """Builds model inputs from MathVQASample.

    The processor owns all text/image preprocessing that must be deterministic
    across train and inference.
    """

    def __init__(self, tokenizer: Any, config: ProcessorConfig | None = None) -> None:
        self.tokenizer = tokenizer
        self.config = config or ProcessorConfig()

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Convert image to tensor with shape [num_tiles, 3, image_size, image_size].

        TODO:
            - convert to RGB;
            - resize/crop/pad;
            - split into tiles if num_tiles > 1;
            - normalize to float tensor.
        """
        image_size = self.config.image_size
        num_tiles = self.config.num_tiles

        image = image.convert("RGB")

        width, height = image.size
        max_side = max(width, height)

        square_image = Image.new("RGB", (max_side, max_side), color="white")

        left = (max_side - width) // 2
        top = (max_side - height) // 2
        square_image.paste(image, (left, top))

        square_image = square_image.resize((image_size, image_size))

        image_array = np.array(square_image)

        image_tensor = torch.tensor(image_array, dtype=torch.float32)
        image_tensor = image_tensor.permute(2, 0, 1)
        image_tensor = image_tensor / 255.0

        image_tensor = image_tensor.unsqueeze(0)

        if num_tiles > 1:
            image_tensor = image_tensor.repeat(num_tiles, 1, 1, 1)

        return image_tensor

    def build_prompt(self, sample: MathVQASample, include_answer: bool) -> str:
        """Build a text prompt with visual special tokens and options.

        For training, include_answer=True should append the assistant answer.
        For inference, include_answer=False should stop before the answer.
        """
        image_tokens = " ".join([IMAGE_TOKEN] * self.config.num_image_tokens)

        options_text = "\n".join(sample.options)

        prompt = (
            f"{IMAGE_START_TOKEN} {image_tokens} {IMAGE_END_TOKEN}\n"
            f"Question: {sample.question}\n"
            f"Options:\n"
            f"{options_text}\n"
            f"Answer:"
        )

        if include_answer:
            prompt += f" {sample.answer}"

        return prompt

    def tokenize_sample(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        """Return input_ids, attention_mask and labels for one sample.

        labels must be IGNORE_INDEX for prompt tokens and real token ids only
        for the assistant answer.
        """
        prompt_text = self.build_prompt(sample, include_answer=False)
        full_text = self.build_prompt(sample, include_answer=True)

        full_encoding = self.tokenizer(
            full_text,
            max_length=self.config.max_length,
            truncation=True,
        )

        prompt_encoding = self.tokenizer(
            prompt_text,
            max_length=self.config.max_length,
            truncation=True,
        )

        input_ids = torch.tensor(full_encoding["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(full_encoding["attention_mask"], dtype=torch.long)

        labels = input_ids.clone()

        prompt_length = len(prompt_encoding["input_ids"])
        prompt_length = min(prompt_length, len(labels))

        labels[:prompt_length] = self.config.ignore_index

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __call__(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        item = self.tokenize_sample(sample)
        item["pixel_values"] = self.preprocess_image(sample.image)
        return item

    def collate(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Pad text fields and stack pixel_values"""
        pad_token_id = self.tokenizer.pad_token_id

        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]
        pixel_values = [item["pixel_values"] for item in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=pad_token_id,
        )

        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0,
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=self.config.ignore_index,
        )

        pixel_values = torch.stack(pixel_values, dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
        }
