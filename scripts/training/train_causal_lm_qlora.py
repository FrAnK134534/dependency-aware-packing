#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal LoRA/QLoRA causal-LM training entrypoint for packed JSONL data."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--validation-file", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-name", default="dapacking_qlora_smoke")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--report-to", default="none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    tokenizer_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    device_map = None
    if args.load_in_4bit:
        compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        local_rank = os.environ.get("LOCAL_RANK")
        device_map = {"": int(local_rank)} if local_rank is not None else "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
        quantization_config=quantization_config,
        device_map=device_map,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[module.strip() for module in args.target_modules.split(",") if module.strip()],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = PackedJsonlDataset(
        args.train_file,
        tokenizer,
        args.max_seq_length,
        limit=args.max_train_samples,
    )
    eval_dataset = (
        PackedJsonlDataset(
            args.validation_file,
            tokenizer,
            args.max_seq_length,
            limit=args.max_validation_samples,
        )
        if args.validation_file
        else None
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        run_name=args.run_name,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        save_steps=args.save_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        optim="paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        report_to=[] if args.report_to == "none" else [args.report_to],
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(args.output_dir / "final_adapter"))


class PackedJsonlDataset:
    def __init__(
        self,
        path: Path,
        tokenizer: Any,
        max_seq_length: int,
        *,
        limit: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.records = _read_packed_texts(path, limit)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        encoded = self.tokenizer(
            self.records[index],
            truncation=True,
            max_length=self.max_seq_length,
            add_special_tokens=True,
        )
        return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}


def _read_packed_texts(path: Path, limit: int | None = None) -> list[str]:
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if limit is not None and len(texts) >= limit:
                break
            record = json.loads(line)
            content = record.get("content")
            if content is None:
                content = record.get("text")
            if content is None:
                raise ValueError(f"Packed record in {path} has no 'content' or 'text' field.")
            texts.append(str(content))
    if not texts:
        raise ValueError(f"No training records found in {path}.")
    return texts


if __name__ == "__main__":
    main()
