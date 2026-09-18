# -*- coding: utf-8 -*-
"""
Bulaq 1280 AH ByT5 OCR Post-Correction Transformer
===================================================
Fine-tunes google/byt5-small on 28,468 empirical Bulaq OCR fallacy pairs.
Byte-level representation guarantees 0 out-of-vocabulary tokens for historical Arabic ligatures.
"""

import os
import sys
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

def load_data(jsonl_path):
    pairs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            src = row.get("corrupt_ocr", "").strip()
            tgt = row.get("ground_truth", "").strip()
            if src and tgt and src != tgt:
                pairs.append({"input_text": src, "target_text": tgt})
    return pairs

def main():
    data_path = os.path.join(os.path.dirname(__file__), "data", "bulaq_ocr_fallacies_dataset.jsonl")
    if not os.path.exists(data_path):
        data_path = "data/bulaq_ocr_fallacies_dataset.jsonl"
    
    print(f"Loading dataset from: {data_path}")
    raw_data = load_data(data_path)
    print(f"Total training pairs: {len(raw_data):,}")

    # Train/Validation Split (90/10)
    full_dataset = Dataset.from_list(raw_data)
    split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = split_dataset["train"]
    val_ds = split_dataset["test"]
    print(f"Train size: {len(train_ds):,}, Validation size: {len(val_ds):,}")

    model_name = "google/byt5-small"
    print(f"Initializing ByT5 tokenizer and model ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    max_src_len = 128
    max_tgt_len = 128

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

    # Determine optimal precision
    has_cuda = torch.cuda.is_available()
    use_bf16 = False
    use_fp16 = False
    if has_cuda:
        use_bf16 = torch.cuda.is_bf16_supported()
        use_fp16 = not use_bf16
        print(f"CUDA Available: {torch.cuda.get_device_name(0)}")
        print(f"Precision: {'BF16' if use_bf16 else 'FP16'}")
    else:
        print("Running on CPU / MPS (No CUDA detected)")

    training_args = Seq2SeqTrainingArguments(
        output_dir="./bulaq_byt5_checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-4,
        per_device_train_batch_size=32 if has_cuda else 4,
        per_device_eval_batch_size=32 if has_cuda else 4,
        gradient_accumulation_steps=2 if has_cuda else 4,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=50,
        warmup_ratio=0.05,
        report_to="none"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    print("Starting ByT5 Fine-Tuning on Bulaq Fallacies...")
    trainer.train()

    save_dir = "./bulaq_byt5_ocr_corrector"
    print(f"Saving final trained model to: {save_dir}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("Training Complete!")

if __name__ == "__main__":
    main()
