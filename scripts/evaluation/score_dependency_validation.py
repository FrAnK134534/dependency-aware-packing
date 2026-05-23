#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score dependency-sensitive validation records with context-gain loss."
    )
    parser.add_argument("--model", required=True, help="Base or fully saved model path/name.")
    parser.add_argument("--adapter", help="Optional PEFT adapter path.")
    parser.add_argument("--tokenizer", help="Tokenizer path/name. Defaults to --model.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
        device_map="auto",
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = []
    with torch.no_grad():
        for record in _read_jsonl(args.input):
            loss_without, tokens_without = _target_loss(
                model,
                tokenizer,
                str(record["context_without_dependency"]),
                str(record["target_text"]),
                args.max_seq_length,
            )
            loss_with, tokens_with = _target_loss(
                model,
                tokenizer,
                str(record["context_with_dependency"]),
                str(record["target_text"]),
                args.max_seq_length,
            )
            rows.append(
                {
                    "sample_id": record["sample_id"],
                    "primary_relation": record["primary_relation"],
                    "relation": record["relation"],
                    "source_docid": record["source_docid"],
                    "target_docid": record["target_docid"],
                    "loss_without_dependency": loss_without,
                    "loss_with_dependency": loss_with,
                    "context_gain": loss_without - loss_with,
                    "target_tokens_without_dependency": tokens_without,
                    "target_tokens_with_dependency": tokens_with,
                }
            )

    _write_jsonl(args.output, rows)
    gains = [float(row["context_gain"]) for row in rows]
    mean_gain = sum(gains) / max(len(gains), 1)
    positive_rate = sum(1 for gain in gains if gain > 0) / max(len(gains), 1)
    print(f"records={len(rows)}")
    print(f"mean_context_gain={mean_gain:.6f}")
    print(f"positive_context_gain_rate={positive_rate:.6f}")
    print(f"output={args.output}")


def _target_loss(
    model,
    tokenizer,
    context: str,
    target: str,
    max_seq_length: int,
) -> tuple[float, int]:
    import torch

    context_ids = tokenizer.encode(context, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    if not target_ids:
        return 0.0, 0

    max_target_len = max_seq_length - 1
    if len(target_ids) > max_target_len:
        target_ids = target_ids[:max_target_len]
    context_budget = max(max_seq_length - len(target_ids), 0)
    if len(context_ids) > context_budget:
        context_ids = context_ids[-context_budget:] if context_budget > 0 else []

    input_ids = context_ids + target_ids
    labels = [-100] * len(context_ids) + target_ids
    device = next(model.parameters()).device
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    label_tensor = torch.tensor([labels], dtype=torch.long, device=device)
    loss = model(input_ids=input_tensor, labels=label_tensor).loss
    return float(loss.detach().cpu()), len(target_ids)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
