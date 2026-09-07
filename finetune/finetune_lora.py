#!/usr/bin/env python3
"""
finetune_lora.py
Fine-tunes CohereLabs/tiny-aya-earth on the Yoruba split of masakhane/african-ultrachat using QLoRA.
Supports local execution and Google Colab Pro, with optional direct export to Hugging Face Hub.
"""

import argparse
import os
import sys
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Import our template utility
try:
    from prompt_template import format_multiturn, format_single_turn
except ImportError:
    from finetune.prompt_template import format_multiturn, format_single_turn


def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for TinyAya on African languages (Yoruba focus)")
    parser.add_argument("--model_id", type=str, default="CohereLabs/tiny-aya-earth", help="Hugging Face model ID")
    parser.add_argument("--dataset_name", type=str, default="masakhane/african-ultrachat", help="Hugging Face dataset ID")
    parser.add_argument("--language", type=str, default="yo", help="Language code (e.g. 'yo' for Yoruba, 'ha' for Hausa, 'ig' for Igbo)")
    parser.add_argument("--output_dir", type=str, default="./output_lora", help="Output directory for LoRA adapter")
    parser.add_argument("--max_samples", type=int, default=5000, help="Maximum training samples to use")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device train batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--push_to_hub", action="store_true", help="Push trained adapter to Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str, default="husseinalamutu/tiny-aya-earth-yoruba-lora", help="Hugging Face Hub repository ID")
    parser.add_argument("--use_4bit", action="store_true", default=True, help="Use 4-bit quantization (QLoRA)")
    return parser.parse_args()


def prepare_dataset(dataset_name: str, language: str, max_samples: int):
    """
    Loads and normalizes the target language conversational dataset.
    """
    print(f"[*] Loading dataset '{dataset_name}' for language '{language}'...")
    try:
        # Attempt loading language-specific split/config
        ds = load_dataset(dataset_name, language, split="train")
    except Exception as e:
        print(f"[!] Direct language config load failed ({e}), attempting standard split load...")
        ds = load_dataset(dataset_name, split="train")
        # Filter by language column if present
        if "language" in ds.column_names:
            ds = ds.filter(lambda x: x["language"].lower() in [language.lower(), f"{language.lower()}_ng", "yoruba"])
        elif "lang" in ds.column_names:
            ds = ds.filter(lambda x: x["lang"].lower() in [language.lower(), "yoruba"])

    if len(ds) > max_samples:
        print(f"[*] Subsampling {max_samples} examples from {len(ds)} total records.")
        ds = ds.shuffle(seed=42).select(range(max_samples))
    else:
        print(f"[*] Using full dataset split ({len(ds)} records).")

    return ds


def formatting_prompts_func(example):
    """
    Extracts text from multi-turn or instruction format and applies TinyAya tokens.
    """
    texts = []
    # Determine structure of African-UltraChat records
    if "messages" in example:
        for messages in example["messages"]:
            texts.append(format_multiturn(messages))
    elif "conversations" in example:
        for conv in example["conversations"]:
            texts.append(format_multiturn(conv))
    elif "instruction" in example and "response" in example:
        for inst, resp in zip(example["instruction"], example["response"]):
            texts.append(format_single_turn(inst, resp))
    elif "prompt" in example and "completion" in example:
        for p, c in zip(example["prompt"], example["completion"]):
            texts.append(format_single_turn(p, c))
    else:
        # Fallback to text column if pre-formatted
        if "text" in example:
            return example["text"]
        raise ValueError(f"Unknown dataset structure with columns: {list(example.keys())}")
    return texts


def main():
    args = parse_args()
    print("=" * 60)
    print("Sovereign AI: Localized LLM Fine-Tuning Pipeline")
    print(f"Base Model:     {args.model_id}")
    print(f"Target Lang:    {args.language.upper()} (Yoruba)")
    print(f"Output Path:    {args.output_dir}")
    print(f"Push to Hub:    {args.push_to_hub} ({args.hub_model_id if args.push_to_hub else 'N/A'})")
    print("=" * 60)

    # 1. Tokenizer
    print("[*] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Model with QLoRA quantization
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    if args.use_4bit and torch.cuda.is_available():
        print(f"[*] Enabling 4-bit BitsAndBytes quantization (compute_dtype={compute_dtype})...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    print(f"[*] Loading base model '{args.model_id}'...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )

    if args.use_4bit and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    # 3. LoRA Configuration
    print("[*] Configuring LoRA parameters...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Dataset
    dataset = prepare_dataset(args.dataset_name, args.language, args.max_samples)

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        fp16=(compute_dtype == torch.float16),
        bf16=(compute_dtype == torch.bfloat16),
        max_grad_norm=0.3,
        report_to="none",
    )

    # 6. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        formatting_func=formatting_prompts_func,
    )

    print("[*] Starting training...")
    trainer.train()

    # 7. Save Adapter
    print(f"[*] Saving trained LoRA adapter to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 8. Optional Hugging Face Hub Upload
    if args.push_to_hub:
        print(f"[*] Uploading LoRA adapter to Hugging Face Hub: {args.hub_model_id}...")
        trainer.model.push_to_hub(args.hub_model_id)
        tokenizer.push_to_hub(args.hub_model_id)
        print("[+] Adapter successfully published to Hugging Face Hub!")

    print("[+] Fine-tuning workflow completed successfully.")


if __name__ == "__main__":
    main()
