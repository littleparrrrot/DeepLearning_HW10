from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

from hw.constants import CHOICES
import torch

from hw.constants import IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN
from hw.dataset import MathVQADataset
from hw.model import MathVLM, ModelConfig
from hw.processor import MathVLMProcessor, ProcessorConfig
from hw.train import SimpleTokenizer, TinyLanguageModel, TinyVisionEncoder


def normalize_text(text: str) -> str:
    """Simple normalization for free-form answers."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_mc_answer(text: str, choices: tuple[str, ...] = CHOICES) -> str | None:
    """Extract multiple-choice answer letter from model output."""
    text = text.strip()
    choices_set = {choice.upper() for choice in choices}

    patterns = [
        r"^(?:\s*[\(\[]?\s*([A-Z])\s*[\)\].:]?\s*)$",
        r"(?:answer|ответ|correct answer is|правильный ответ)\s*[:\-]?\s*[\(\[]?\s*([A-Z])\s*[\)\].:]?",
        r"\(([A-Z])\)",
        r"\b([A-Z])\)",
        r"\b([A-Z])\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is not None:
            answer = match.group(1).upper()
            if answer in choices_set:
                return answer

    return None


def build_benchmark_prompt(question: str, options: list[str]) -> str:
    """Build prompt for multiple-choice visual math evaluation."""
    options_text = "\n".join(options)
    return (
        "Реши визуально-математическую задачу. "
        "Выбери один вариант ответа и в конце напиши только букву.\n\n"
        f"Вопрос: {question}\n"
        f"Варианты:\n{options_text}\n"
        "Ответ:"
    )


def compute_accuracy(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute overall and per-subject accuracy from prediction rows."""
    if not rows:
        return {"overall": 0.0}

    total = len(rows)
    correct = sum(int(r.get("prediction") == r.get("answer")) for r in rows)
    metrics = {"overall": correct / total}

    subjects = sorted({r.get("subject", "unknown") for r in rows})
    for subject in subjects:
        sub_rows = [r for r in rows if r.get("subject", "unknown") == subject]
        sub_correct = sum(int(r.get("prediction") == r.get("answer")) for r in sub_rows)
        metrics[f"subject/{subject}"] = sub_correct / max(1, len(sub_rows))
    return metrics


def run_benchmark(config: dict[str, Any], toy: bool = False) -> dict[str, float]:
    """Run evaluation loop."""
    device = torch.device("cpu")

    data_config = config.get("data", {})
    processor_config = config.get("processor", {})
    evaluation_config = config.get("evaluation", {})

    if toy:
        manifest_path = data_config.get(
            "toy_manifest",
            data_config.get(
                "toy_manifest_path",
                "assets/toy_math_vqa/manifest.jsonl",
            ),
        )
        split = data_config.get("toy_split", "dev")
    else:
        manifest_path = evaluation_config.get(
            "manifest_path",
            data_config.get("manifest_path", data_config.get("eval_manifest")),
        )
        split = evaluation_config.get("split", data_config.get("split", "dev"))

    max_samples = evaluation_config.get("max_samples", data_config.get("max_samples"))

    tokenizer = SimpleTokenizer()

    processor = MathVLMProcessor(
        tokenizer=tokenizer,
        config=ProcessorConfig(**processor_config),
    )

    dataset = MathVQADataset(
        manifest_path=manifest_path,
        split=split,
        max_samples=max_samples,
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

    model_config = ModelConfig(
        vision_hidden_size=hidden_size,
        text_hidden_size=hidden_size,
        num_image_tokens=processor.config.num_image_tokens,
        image_token_id=tokenizer.convert_tokens_to_ids(IMAGE_TOKEN),
    )

    model = MathVLM(
        vision_encoder=vision_encoder,
        language_model=language_model,
        config=model_config,
    ).to(device)

    model.eval()

    prediction_rows: list[dict[str, Any]] = []

    for sample in dataset:
        image_tokens = " ".join([IMAGE_TOKEN] * processor.config.num_image_tokens)

        prompt = (
            f"{IMAGE_START_TOKEN} {image_tokens} {IMAGE_END_TOKEN}\n"
            f"{build_benchmark_prompt(sample.question, sample.options)}"
        )

        encoded = tokenizer(
            prompt,
            max_length=processor.config.max_length,
            truncation=True,
        )

        batch = {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long).unsqueeze(0).to(device),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long).unsqueeze(0).to(device),
            "pixel_values": processor.preprocess_image(sample.image).unsqueeze(0).to(device),
        }

        with torch.no_grad():
            generated_ids = model.generate(batch, max_new_tokens=4)

        id_to_token = {idx: token for token, idx in tokenizer.vocab.items()}
        generated_text = " ".join(
            id_to_token.get(int(token_id), "")
            for token_id in generated_ids[0].detach().cpu().tolist()
        ).strip()

        prediction = parse_mc_answer(generated_text)

        if prediction is None:
            prediction = normalize_text(generated_text)

        answer = str(sample.answer).strip()

        if answer.upper() in CHOICES:
            answer = answer.upper()
        else:
            answer = normalize_text(answer)

        prediction_rows.append(
            {
                "id": sample.id,
                "subject": sample.subject,
                "answer": answer,
                "prediction": prediction,
                "raw_output": generated_text,
            }
        )

    output_path = evaluation_config.get("output_path", config.get("output_path"))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(prediction_rows, f, ensure_ascii=False, indent=2)

    return compute_accuracy(prediction_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--toy", action="store_true")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    metrics = run_benchmark(config, toy=args.toy)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
