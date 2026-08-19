import argparse
import json
import os

import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer


SYSTEM_PROMPT = (
    "You are a precise function-calling assistant. "
    "Given a user query and a list of available tools, respond with a JSON array "
    "containing all the tool calls needed to answer the query.\n\n"
    "Each element of the array must be a JSON object with exactly two keys:\n"
    "  \"name\"      : the tool name (string)\n"
    "  \"arguments\" : a dict of argument name → value\n\n"
    "Wrap your entire response in a single ```json ... ``` code block. "
    "Do not include any explanation outside the code block."
)



def build_messages(example: dict, tokenizer) -> dict:
    tools_str = example["tools"]
    query = example["query"]
    answers_str = example["answers"]

    try:
        answers = json.loads(answers_str)
        target = "```json\n" + json.dumps(answers, indent=2) + "\n```"
    except Exception:
        target = answers_str  # fallback

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\nAvailable tools:\n" + tools_str},
        {"role": "user", "content": query},
        {"role": "assistant", "content": target},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


def parse_args():
    p = argparse.ArgumentParser(description="SFT training on xLAM function-calling dataset")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--output_dir", type=str, default="outputs/sft_model")
    p.add_argument("--train_file", type=str, default=None,
                   help="Path to train JSONL (e.g. data/xlam_train.jsonl). "
                        "If omitted, falls back to full HF dataset.")
    p.add_argument("--eval_file", type=str, default=None,
                   help="Path to eval JSONL (e.g. data/xlam_eval.jsonl). "
                        "If omitted, no eval is run during training.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=500,
                   help="Run evaluation every N steps (only if --eval_file is set)")
    return p.parse_args()


def load_split(path: str | None, hf_split: str = "train") -> Dataset:
    if path and os.path.exists(path):
        print(f"[data] Loading from local file: {path}")
        return load_dataset("json", data_files={"data": path})["data"]
    print(f"[data] Local file not found. Downloading Salesforce/xlam-function-calling-60k ({hf_split}) …")
    return load_dataset("Salesforce/xlam-function-calling-60k", split=hf_split)


def main():
    args = parse_args()

    print(f"[SFT] Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    raw_train = load_split(args.train_file, hf_split="train")
    print(f"[SFT] Train size: {len(raw_train):,}")

    raw_eval = None
    if args.eval_file:
        raw_eval = load_split(args.eval_file, hf_split="validation")
        print(f"[SFT] Eval  size: {len(raw_eval):,}")
    else:
        print("[SFT] No eval file provided — skipping validation.")

    print("[SFT] Formatting dataset into chat template …")
    fmt = lambda ex: build_messages(ex, tokenizer)

    train_dataset = raw_train.map(fmt, remove_columns=raw_train.column_names, num_proc=4)
    eval_dataset  = raw_eval.map(fmt, remove_columns=raw_eval.column_names, num_proc=4) \
                    if raw_eval else None

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    do_eval = eval_dataset is not None
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        logging_steps=20,
        save_steps=args.save_steps,
        save_total_limit=10,
        eval_strategy="steps" if do_eval else "no",
        eval_steps=args.eval_steps if do_eval else None,
        load_best_model_at_end=do_eval,
        metric_for_best_model="eval_loss" if do_eval else None,
        max_length=args.max_seq_length,
        dataset_text_field="text",
        packing=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=args.model_name,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print("[SFT] Starting training …")
    trainer.train()

    print(f"[SFT] Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[SFT] Done.")


if __name__ == "__main__":
    main()
