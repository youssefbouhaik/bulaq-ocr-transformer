# -*- coding: utf-8 -*-
"""
Bulaq 1280 AH ByT5 OCR Post-Correction Transformer (v2 Sentence-Level)
======================================================================
Fine-tunes google/byt5-small on 69,640 empirical sentence-level pairs.
Optimized for high-throughput execution on A100 / H100 / T4 GPUs.
"""

import os
import sys
import json
import gzip
import inspect
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    gz_path = os.path.join(base_dir, "data", "bulaq_sentence_pairs.jsonl.gz")
    jsonl_path = os.path.join(base_dir, "data", "bulaq_sentence_pairs.jsonl")
    
    pairs = []
    if os.path.exists(gz_path):
        print(f"Loading compressed dataset from: {gz_path}")
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    src = row.get("corrupt_ocr", "").strip()
                    tgt = row.get("ground_truth", "").strip()
                    if src and tgt and src != tgt:
                        pairs.append({"input_text": src, "target_text": tgt})
    elif os.path.exists(jsonl_path):
        print(f"Loading dataset from: {jsonl_path}")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    src = row.get("corrupt_ocr", "").strip()
                    tgt = row.get("ground_truth", "").strip()
                    if src and tgt and src != tgt:
                        pairs.append({"input_text": src, "target_text": tgt})
    else:
        raise FileNotFoundError("Could not find bulaq_sentence_pairs.jsonl(.gz) in data/")
        
    return pairs

def main():
    raw_data = load_data()
    print(f"Total sentence-level training pairs: {len(raw_data):,}")

    full_dataset = Dataset.from_list(raw_data)
    split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = split_dataset["train"]
    val_ds = split_dataset["test"]
    print(f"Train size: {len(train_ds):,}, Validation size: {len(val_ds):,}")

    model_name = "google/byt5-small"
    print(f"Initializing ByT5 tokenizer and model ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    max_src_len = 256
    max_tgt_len = 256

    def preprocess(batch):
        inputs = tokenizer(
            batch["input_text"],
            max_length=max_src_len,
            padding="max_length",
            truncation=True
        )
        targets = tokenizer(
            text_target=batch["target_text"],
            max_length=max_tgt_len,
            padding="max_length",
            truncation=True
        )
        labels = [
            [(t if t != tokenizer.pad_token_id else -100) for t in target]
            for target in targets["input_ids"]
        ]
        inputs["labels"] = labels
        return inputs

    print("Tokenizing datasets...")
    tokenized_train = train_ds.map(preprocess, batched=True, remove_columns=["input_text", "target_text"])
    tokenized_val = val_ds.map(preprocess, batched=True, remove_columns=["input_text", "target_text"])

    has_cuda = torch.cuda.is_available()
    use_bf16 = False
    use_fp16 = False
    if has_cuda:
        use_bf16 = torch.cuda.is_bf16_supported()
        use_fp16 = not use_bf16
        print(f"CUDA Accelerator: {torch.cuda.get_device_name(0)}")
        print(f"Precision Mode : {'BF16 (Native Ampere/Hopper)' if use_bf16 else 'FP16'}")
        # Enable gradient checkpointing for safety
        model.gradient_checkpointing_enable()
    else:
        print("Running on CPU")

    # High-throughput batch sizing (Auto-scales for H100/A100 vs T4)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if has_cuda else 0
    if vram_gb >= 70:      # H100 / A100 80GB
        batch_size = 32
        grad_accum = 1
    elif vram_gb >= 35:    # A100 40GB
        batch_size = 16
        grad_accum = 2
    else:                  # T4 / smaller
        batch_size = 4
        grad_accum = 8

    print(f"Detected {vram_gb:.1f} GB VRAM -> Setting batch_size={batch_size}, grad_accum={grad_accum}")

    training_args = Seq2SeqTrainingArguments(
        output_dir="./bulaq_byt5_checkpoints",
        optim="adamw_torch",
        learning_rate=3e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=3,
        warmup_steps=200,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=25,
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=has_cuda,
        predict_with_generate=False,
        report_to="none"
    )

    if hasattr(training_args, "eval_strategy"):
        training_args.eval_strategy = "epoch"
    elif hasattr(training_args, "evaluation_strategy"):
        training_args.evaluation_strategy = "epoch"

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_train,
        "eval_dataset": tokenized_val,
        "data_collator": data_collator,
    }
    trainer_sig = inspect.signature(Seq2SeqTrainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    print("\n🚀 Starting ByT5 Training on 69,640 Bulaq sentence pairs...")
    trainer.train()

    save_dir = "./bulaq_byt5_ocr_corrector"
    print(f"\nSaving final model weights to: {save_dir}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("✅ Training Complete and Model Saved!")

if __name__ == "__main__":
    main()
