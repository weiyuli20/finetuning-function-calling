import argparse
import json
import os

import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, PeftModel
from swanlab.integration.transformers import SwanLabCallback
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

from rewards_xlam import *


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


def build_sft_example(example: dict, tokenizer) -> dict:
    try:
        answers = json.loads(example["answers"])
        target = "```json\n" + json.dumps(answers, indent=2) + "\n```"
    except Exception:
        target = example["answers"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\nAvailable tools:\n" + example["tools"]},
        {"role": "user", "content": example["query"]},
        {"role": "assistant", "content": target},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


def build_grpo_example(example: dict) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\nAvailable tools:\n" + example["tools"]},
        {"role": "user", "content": example["query"]},
    ]
    return {
        "prompt": messages,
        "answers": example["answers"],
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Two-phase SFT → GRPO training on xLAM function-calling dataset"
    )
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--output_dir", type=str, default="outputs/sft_grpo")
    p.add_argument("--sft_checkpoint", type=str, default=None,
                   help="Path to a pre-existing SFT checkpoint; skips Phase 1 if --skip_sft is set")
    p.add_argument("--skip_sft", action="store_true",
                   help="Skip SFT and go straight to GRPO using --sft_checkpoint or --model_name")

    p.add_argument("--train_file", type=str, default=None,
                   help="Path to train JSONL (e.g. data/xlam_train.jsonl)")
    p.add_argument("--eval_file", type=str, default=None,
                   help="Path to eval JSONL (e.g. data/xlam_eval.jsonl)")

    p.add_argument("--sft_epochs", type=int, default=3)
    p.add_argument("--sft_lr", type=float, default=2e-4)
    p.add_argument("--sft_max_seq_length", type=int, default=512)

    p.add_argument("--grpo_epochs", type=int, default=1)
    p.add_argument("--grpo_lr", type=float, default=5e-6)
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--max_completion_length", type=int, default=512)
    p.add_argument("--num_generations", type=int, default=8)

    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--eval_steps", type=int, default=500,
                   help="Evaluation frequency in steps (for both SFT and GRPO phases)")
    p.add_argument("--warmup_ratio", type=float, default=0.005)
    p.add_argument("--grpo_temp", type=float, default=1.4)

    p.add_argument("--w_format",        type=float, default=1.0)
    p.add_argument("--w_schema",        type=float, default=1.0)
    p.add_argument("--w_name_recall",   type=float, default=2.0)
    p.add_argument("--w_arg_acc",       type=float, default=3.0)
    p.add_argument("--w_halluc",        type=float, default=1.5)
    p.add_argument("--w_count",         type=float, default=1.0)

    p.add_argument("--w_required_args", type=float, default=1.0)
    p.add_argument("--w_extra_args",    type=float, default=0.5)
    p.add_argument("--w_value_acc",     type=float, default=2.0)
    p.add_argument("--w_type_acc",      type=float, default=0.5)
    p.add_argument("--w_order",         type=float, default=0.5)
    p.add_argument("--w_necessity",     type=float, default=1.0)

    return p.parse_args()


def load_split(path: str | None, hf_split: str = "train") -> Dataset:
    if path and os.path.exists(path):
        print(f"[data] Loading from local file: {path}")
        return load_dataset("json", data_files={"data": path})["data"]
    print(f"[data] Local file not found. Downloading Salesforce/xlam-function-calling-60k ({hf_split}) …")
    return load_dataset("Salesforce/xlam-function-calling-60k", split=hf_split)


def run_sft(args, raw_train, raw_eval, tokenizer) -> str:
    sft_output = os.path.join(args.output_dir, "sft")
    print(f"\n{'='*60}")
    print(f"PHASE 1: SFT  →  {sft_output}")
    print(f"{'='*60}\n")

    fmt = lambda ex: build_sft_example(ex, tokenizer)  # noqa: E731
    sft_train = raw_train.map(fmt, remove_columns=raw_train.column_names, num_proc=4)
    sft_eval  = raw_eval.map(fmt, remove_columns=raw_eval.column_names, num_proc=4) \
                if raw_eval else None

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    do_eval = sft_eval is not None
    sft_args = SFTConfig(
        output_dir=sft_output,
        num_train_epochs=args.sft_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.sft_lr,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        logging_steps=20,
        save_steps=args.save_steps,
        save_total_limit=2,
        evaluation_strategy="steps" if do_eval else "no",
        eval_steps=args.eval_steps if do_eval else None,
        load_best_model_at_end=do_eval,
        metric_for_best_model="eval_loss" if do_eval else None,
        max_seq_length=args.sft_max_seq_length,
        dataset_text_field="text",
        packing=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=args.model_name,
        args=sft_args,
        train_dataset=sft_train,
        eval_dataset=sft_eval,
        peft_config=peft_config,
        tokenizer=tokenizer,
        callbacks=[SwanLabCallback()],
    )

    trainer.train()
    trainer.save_model(sft_output)
    tokenizer.save_pretrained(sft_output)
    print(f"[SFT] Checkpoint saved → {sft_output}")
    return sft_output


def run_grpo(args, raw_train, raw_eval, grpo_base_model):
    grpo_output = os.path.join(args.output_dir, "grpo")

    print(f"\n{'='*60}")
    print(f"PHASE 2: GRPO  ({grpo_base_model})  →  {grpo_output}")
    print(f"{'='*60}\n")

    grpo_train = raw_train.map(
        build_grpo_example,
        remove_columns=raw_train.column_names,
        num_proc=4,
    )

    grpo_eval = raw_eval.map(
        build_grpo_example,
        remove_columns=raw_eval.column_names,
        num_proc=4,
    ) if raw_eval else None

    def w(fn, weight):
        def _wrapped(*a, **kw):
            return [r * weight for r in fn(*a, **kw)]
        _wrapped.__name__ = fn.__name__
        return _wrapped

    active_rewards = []

    if args.w_format > 0:
        active_rewards.append(
            w(format_reward, args.w_format)
        )

    if args.w_schema > 0:
        active_rewards.append(
            w(schema_validity_reward, args.w_schema)
        )

    if args.w_name_recall > 0:
        active_rewards.append(
            w(tool_name_recall_reward, args.w_name_recall)
        )

    if args.w_arg_acc > 0:
        active_rewards.append(
            w(argument_accuracy_reward, args.w_arg_acc)
        )

    if args.w_halluc > 0:
        active_rewards.append(
            w(hallucination_reward, args.w_halluc)
        )

    if args.w_count > 0:
        active_rewards.append(
            w(count_match_reward, args.w_count)
        )

    if args.w_required_args > 0:
        active_rewards.append(
            w(required_args_reward, args.w_required_args)
        )

    if args.w_extra_args > 0:
        active_rewards.append(
            w(extra_args_reward, args.w_extra_args)
        )

    if args.w_value_acc > 0:
        active_rewards.append(
            w(argument_value_reward, args.w_value_acc)
        )

    if args.w_type_acc > 0:
        active_rewards.append(
            w(argument_type_reward, args.w_type_acc)
        )

    if args.w_order > 0:
        active_rewards.append(
            w(tool_order_reward, args.w_order)
        )

    if args.w_necessity > 0:
        active_rewards.append(
            w(tool_necessity_reward, args.w_necessity)
        )

    print(f"[GRPO] Active reward functions: {len(active_rewards)}")

    print(f"[GRPO] Loading base model: {args.model_name}")

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=(
            torch.bfloat16
            if torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
            else torch.float16
        ),
        trust_remote_code=True,
    )

    print(f"[GRPO] Loading SFT adapter: {grpo_base_model}")

    model = PeftModel.from_pretrained(
        base_model,
        grpo_base_model,
        is_trainable=True,
    )

    do_eval = grpo_eval is not None

    grpo_config = GRPOConfig(
        output_dir=grpo_output,
        learning_rate=args.grpo_lr,
        lr_scheduler_type="cosine",
        warmup_steps=100,

        temperature=args.grpo_temp,

        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.grpo_epochs,

        bf16=torch.cuda.is_available()
            and torch.cuda.is_bf16_supported(),

        fp16=torch.cuda.is_available()
            and not torch.cuda.is_bf16_supported(),

        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=2,

        eval_strategy="steps" if do_eval else "no",
        eval_steps=args.eval_steps if do_eval else None,

        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,

        remove_unused_columns=False,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=active_rewards,
        args=grpo_config,
        train_dataset=grpo_train,
        eval_dataset=grpo_eval,
        callbacks=[SwanLabCallback()],
    )

    trainer.train()

    trainer.save_model(grpo_output)

    print(f"[GRPO] Final model saved → {grpo_output}")

    return grpo_output



def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Setup] Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    raw_train = load_split(args.train_file, hf_split="train")
    print(f"[Setup] Train size: {len(raw_train):,}")

    raw_eval = None
    if args.eval_file:
        raw_eval = load_split(args.eval_file, hf_split="validation")
        print(f"[Setup] Eval  size: {len(raw_eval):,}")
    else:
        print("[Setup] No eval file provided — skipping validation in both phases.")

    if args.skip_sft:
        sft_ckpt = args.sft_checkpoint or args.model_name
        print(f"[Setup] Skipping SFT. Using: {sft_ckpt}")
    else:
        sft_ckpt = run_sft(args, raw_train, raw_eval, tokenizer)

    run_grpo(args, raw_train, raw_eval, grpo_base_model=sft_ckpt)

    print("\n[Done] Both training phases complete.")
    print(f"  SFT checkpoint : {os.path.join(args.output_dir, 'sft')}")
    print(f"  GRPO checkpoint: {os.path.join(args.output_dir, 'grpo')}")



if __name__ == "__main__":
    main()

'''
python train_sft_grpo.py \
  --skip_sft \
  --model_name ./Qwen2.5-0.5B-Instruct \
  --sft_checkpoint outputs/sft_model \
  --train_file data/xlam_grpo_train.jsonl \
  --eval_file data/xlam_eval.jsonl \
  --num_generations 4
'''