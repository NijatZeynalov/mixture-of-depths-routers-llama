import argparse
import os
from typing import Dict

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

from mod_router import RouterConfig, attach_mod_routers


def parse_args():
    parser = argparse.ArgumentParser(description="Router-tune LLaMA-1B (Mixture-of-Depths style)")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Base LLaMA-1B (e.g., TinyLlama/TinyLlama-1.1B-*)")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save router-tuned model")
    parser.add_argument("--dataset_name", type=str, default=None, help="HF dataset name (e.g., wikitext)")
    parser.add_argument("--dataset_config_name", type=str, default=None, help="HF dataset config (e.g., wikitext-2-raw-v1)")
    parser.add_argument("--train_file", type=str, default=None, help="Local text file for training")
    parser.add_argument("--validation_file", type=str, default=None, help="Local text file for eval")
    parser.add_argument("--sequence_length", type=int, default=512)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--router_hidden_size", type=int, default=256)
    parser.add_argument("--router_dropout", type=float, default=0.0)
    parser.add_argument("--router_pattern", type=str, default="all", choices=["all", "every_other"])
    parser.add_argument("--router_start", type=int, default=0, help="First layer (0-indexed) to wrap")
    parser.add_argument("--router_stop", type=int, default=None, help="Stop wrapping before this layer index")
    parser.add_argument("--router_lambda", type=float, default=0.01, help="Weight for compute penalty")
    parser.add_argument("--target_active_ratio", type=float, default=0.7, help="Target average gate (fraction of active layers)")
    parser.add_argument("--layer_norm_router_input", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--tf32", action="store_true")
    return parser.parse_args()


def load_text_datasets(args, tokenizer):
    if args.dataset_name:
        raw = load_dataset(args.dataset_name, args.dataset_config_name)
    elif args.train_file:
        data_files = {"train": args.train_file}
        if args.validation_file:
            data_files["validation"] = args.validation_file
        raw = load_dataset("text", data_files=data_files)
    else:
        raise ValueError("Provide either --dataset_name or --train_file")

    if "validation" not in raw:
        split = max(1, int(0.01 * len(raw["train"])))
        raw = raw.train_test_split(test_size=split, seed=42)
        raw["validation"] = raw["test"]

    def tokenize_function(examples: Dict[str, str]):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.sequence_length,
            padding="max_length",
            return_attention_mask=True,
        )

    keep_cols = ["text"]
    tokenized = raw.map(
        tokenize_function,
        batched=True,
        remove_columns=[c for c in raw["train"].column_names if c not in keep_cols],
    )
    return tokenized


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class RouterTrainer(Trainer):
    def __init__(self, router_lambda: float, target_active_ratio: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.router_lambda = router_lambda
        self.target_active_ratio = target_active_ratio

    def compute_loss(self, model, inputs, return_outputs=False):
        if hasattr(model, "router_usage_recorder"):
            model.router_usage_recorder.reset()
        outputs = model(**inputs)
        loss = outputs.loss
        if hasattr(model, "router_usage_recorder"):
            usage = model.router_usage_recorder.average()
            if usage is not None:
                penalty = torch.relu(usage - self.target_active_ratio)
                loss = loss + self.router_lambda * penalty
                outputs.router_usage = usage.detach()
                outputs.router_penalty = penalty.detach()
        return (loss, outputs) if return_outputs else loss


def main():
    args = parse_args()
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    router_cfg = RouterConfig(
        router_hidden_size=args.router_hidden_size,
        dropout=args.router_dropout,
        pattern=args.router_pattern,
        start_layer=args.router_start,
        stop_layer=args.router_stop,
        layer_norm_input=args.layer_norm_router_input,
    )
    attach_mod_routers(model, router_cfg)

    trainable = count_trainable_params(model)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({100*trainable/total:.3f}%)")

    datasets = load_text_datasets(args, tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_train_steps if args.max_train_steps else -1,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        report_to="none",
        bf16=args.bf16,
        remove_unused_columns=False,
        save_total_limit=2,
    )

    trainer = RouterTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=collator,
        tokenizer=tokenizer,
        router_lambda=args.router_lambda,
        target_active_ratio=args.target_active_ratio,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    router_cfg.save_json(os.path.join(args.output_dir, "router_config.json"))


if __name__ == "__main__":
    main()
