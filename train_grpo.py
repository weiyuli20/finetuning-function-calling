import argparse
import json
import os

import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from rewards_xlam import (
    format_reward,
    schema_validity_reward,
    tool_name_recall_reward,
    argument_accuracy_reward,
    hallucination_reward,
    count_match_reward,
)


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



def build_prompt(example: dict) -> dict:
    """Convert an xLAM row into TRL GRPO prompt format (list of messages)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\nAvailable tools:\n" + example["tools"]},
        {"role": "user", "content": example["query"]},
    ]
    return {
        "prompt": messages,
        "answers": example["answers"],  
    }



def parse_args():
    p = argparse.ArgumentParser(description="GRPO training on xLAM function-calling dataset")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="Base model OR SFT checkpoint to start GRPO from")
    p.add_argument("--output_dir", type=str, default="outputs/grpo_model")
    p.add_argument("--train_file", type=str, default=None,
                   help="Path to train JSONL (e.g. data/xlam_train.jsonl)")
    p.add_argument("--eval_file", type=str, default=None,
                   help="Path to eval JSONL (e.g. data/xlam_eval.jsonl)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--max_completion_length", type=int, default=512)
    p.add_argument("--num_generations", type=int, default=4,
                   help="Number of completions per prompt (GRPO group size)")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=200,
                   help="Run reward-based eval every N steps (if --eval_file set)")
    p.add_argument("--warmup_ratio", type=float, default=0.03)

    p.add_argument("--w_format",      type=float, default=1.0)
    p.add_argument("--w_schema",      type=float, default=1.0)
    p.add_argument("--w_name_recall", type=float, default=2.0)
    p.add_argument("--w_arg_acc",     type=float, default=3.0)
    p.add_argument("--w_halluc",      type=float, default=1.5)
    p.add_argument("--w_count",       type=float, default=1.0)
    return p.parse_args()


def load_split(path: str | None, hf_split: str = "train") -> Dataset:
    if path and os.path.exists(path):
        print(f"[data] Loading from local file: {path}")
        return load_dataset("json", data_files={"data": path})["data"]
    print(f"[data] Local file not found. Downloading Salesforce/xlam-function-calling-60k ({hf_split}) …")
    return load_dataset("Salesforce/xlam-function-calling-60k", split=hf_split)


def main():
    args = parse_args()

    print(f"[GRPO] Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    raw_train = load_split(args.train_file, hf_split="train")
    print(f"[GRPO] Train size: {len(raw_train):,}")

    raw_eval = None
    if args.eval_file:
        raw_eval = load_split(args.eval_file, hf_split="validation")
        print(f"[GRPO] Eval  size: {len(raw_eval):,}")
    else:
        print("[GRPO] No eval file provided — skipping validation.")

    train_dataset = raw_train.map(build_prompt, remove_columns=raw_train.column_names, num_proc=4)
    eval_dataset  = raw_eval.map(build_prompt, remove_columns=raw_eval.column_names, num_proc=4) \
                    if raw_eval else None

    def w(fn, weight):
        def _wrapped(*a, **kw):
            return [r * weight for r in fn(*a, **kw)]
        _wrapped.__name__ = fn.__name__
        return _wrapped

    active_rewards = []
    if args.w_format      > 0: active_rewards.append(w(format_reward,           args.w_format))
    if args.w_schema      > 0: active_rewards.append(w(schema_validity_reward,  args.w_schema))
    if args.w_name_recall > 0: active_rewards.append(w(tool_name_recall_reward, args.w_name_recall))
    if args.w_arg_acc     > 0: active_rewards.append(w(argument_accuracy_reward,args.w_arg_acc))
    if args.w_halluc      > 0: active_rewards.append(w(hallucination_reward,    args.w_halluc))
    if args.w_count       > 0: active_rewards.append(w(count_match_reward,      args.w_count))

    print(f"[GRPO] Active reward functions: {len(active_rewards)}")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    do_eval = eval_dataset is not None
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if do_eval else "no",
        eval_steps=args.eval_steps if do_eval else None,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=args.model_name,
        reward_funcs=active_rewards,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    print("[GRPO] Starting training …")
    trainer.train()

    print(f"[GRPO] Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[GRPO] Done.")


if __name__ == "__main__":
    main()
