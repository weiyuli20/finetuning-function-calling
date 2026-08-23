import argparse
import json
import os

import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer


SYSTEM_PROMPT = (
    "You are a precise function-calling assistant. "
    "Given a user query and a list of available tools, respond with a JSON array "
    "containing all the tool calls needed to answer the query.\n\n"
    "Each element of the array must be a JSON object with exactly two keys:\n"
    '  "name"      : the tool name (string)\n'
    '  "arguments" : a dict of argument name → value\n\n'
    "Wrap your entire response in a single ```json ... ``` code block. "
    "Do not include any explanation outside the code block."
)


def build_messages(
    example: dict,
    tokenizer,
    max_seq_length: int = 1024,
) -> dict:

    tools_str = example["tools"]
    query = example["query"]
    answers_str = example["answers"]

    try:
        answers = json.loads(answers_str)

        target = (
            "```json\n"
            + json.dumps(answers, indent=2)
            + "\n```"
        )

    except Exception:
        target = answers_str

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
            + "\n\nAvailable tools:\n"
            + tools_str,
        },
        {
            "role": "user",
            "content": query,
        },
        {
            "role": "assistant",
            "content": target,
        },
    ]

    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        truncation=True,
        max_length=max_seq_length,
    )

    encoded["labels"] = [
        token_id if is_assistant else -100
        for token_id, is_assistant in zip(
            encoded["input_ids"],
            encoded["assistant_masks"],
        )
    ]

    del encoded["assistant_masks"]

    return encoded


def parse_args():

    p = argparse.ArgumentParser(
        description="SFT training on xLAM function-calling dataset"
    )

    p.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
    )

    p.add_argument(
        "--output_dir",
        type=str,
        default="outputs/sft_model",
    )

    p.add_argument(
        "--train_file",
        type=str,
        default=None,
        help=(
            "Path to train JSONL. "
            "If omitted, falls back to full HF dataset."
        ),
    )

    p.add_argument(
        "--eval_file",
        type=str,
        default=None,
        help=(
            "Path to eval JSONL. "
            "If omitted, no eval is run during training."
        ),
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    p.add_argument(
        "--grad_accum",
        type=int,
        default=4,
    )

    p.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
    )

    p.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
    )

    p.add_argument(
        "--lora_r",
        type=int,
        default=64,
    )

    p.add_argument(
        "--lora_alpha",
        type=int,
        default=128,
    )

    p.add_argument(
        "--save_steps",
        type=int,
        default=500,
    )

    p.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help=(
            "Run evaluation every N steps "
            "(only if --eval_file is set)"
        ),
    )

    p.add_argument(
        "--train_lora_model",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to an existing LoRA adapter. "
            "When supplied, the existing adapter is loaded and "
            "continued training uses its original LoRA configuration."
        ),
    )

    return p.parse_args()


def load_split(
    path: str | None,
    hf_split: str = "train",
) -> Dataset:

    if path and os.path.exists(path):

        print(f"[data] Loading from local file: {path}")

        return load_dataset(
            "json",
            data_files={"data": path},
        )["data"]

    print(
        "[data] Local file not found. "
        f"Downloading Salesforce/xlam-function-calling-60k "
        f"({hf_split}) …"
    )

    return load_dataset(
        "Salesforce/xlam-function-calling-60k",
        split=hf_split,
    )


def main():

    args = parse_args()

    continue_lora = args.train_lora_model is not None

    if continue_lora:

        adapter_path = os.path.abspath(
            args.train_lora_model
        )

        print("=" * 70)
        print("[SFT] CONTINUING EXISTING LoRA MODEL")
        print(f"[SFT] Adapter: {adapter_path}")
        print("=" * 70)

        if not os.path.isdir(adapter_path):

            raise FileNotFoundError(
                f"LoRA adapter directory does not exist: "
                f"{adapter_path}"
            )

        adapter_config = os.path.join(
            adapter_path,
            "adapter_config.json",
        )

        if not os.path.isfile(adapter_config):

            raise FileNotFoundError(
                f"adapter_config.json not found in: "
                f"{adapter_path}\n"
                "Make sure --train_lora_model points to "
                "the LoRA adapter directory."
            )

    else:

        print("=" * 70)
        print("[SFT] TRAINING NEW LoRA MODEL")
        print(f"[SFT] LoRA rank: {args.lora_r}")
        print(f"[SFT] LoRA alpha: {args.lora_alpha}")
        print("=" * 70)

    print(
        f"[SFT] Loading tokenizer: "
        f"{args.model_name}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    tokenizer.chat_template = """{%- if tools %}
{{- '<|im_start|>system\\n' }}
{%- if messages[0]['role'] == 'system' %}
    {{- messages[0]['content'] }}
{%- else %}
    {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
{%- endif %}
{{- "\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
{%- for tool in tools %}
    {{- "\\n" }}
    {{- tool | tojson }}
{%- endfor %}
{{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}
{%- else %}
{%- if messages[0]['role'] == 'system' %}
    {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}
{%- else %}
    {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}
{%- endif %}
{%- endif %}

{%- for message in messages %}

{%- if (message.role == "user") or (message.role == "system" and not loop.first) %}

    {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}

{%- elif message.role == "assistant" and not message.tool_calls %}

    {{- '<|im_start|>' + message.role + '\\n' }}
    {%- generation %}
    {{- message.content + '<|im_end|>\\n' }}
    {%- endgeneration %}

{%- elif message.role == "assistant" %}

    {{- '<|im_start|>' + message.role }}

    {%- generation %}

    {%- if message.content %}
        {{- '\\n' + message.content }}
    {%- endif %}

    {%- for tool_call in message.tool_calls %}

        {%- if tool_call.function is defined %}
            {%- set tool_call = tool_call.function %}
        {%- endif %}

        {{- '\\n<tool_call>\\n{"name": "' }}
        {{- tool_call.name }}
        {{- '", "arguments": ' }}
        {{- tool_call.arguments | tojson }}
        {{- '}\\n</tool_call>' }}

    {%- endfor %}

    {{- '<|im_end|>\\n' }}

    {%- endgeneration %}

{%- elif message.role == "tool" %}

    {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
        {{- '<|im_start|>user' }}
    {%- endif %}

    {{- '\\n<tool_response>\\n' }}
    {{- message.content }}
    {{- '\\n</tool_response>' }}

    {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
        {{- '<|im_end|>\\n' }}
    {%- endif %}

{%- endif %}

{%- endfor %}

{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
{%- endif %}"""

    raw_train = load_split(
        args.train_file,
        hf_split="train",
    )

    print(
        f"[SFT] Train size: {len(raw_train):,}"
    )

    raw_eval = None

    if args.eval_file:

        raw_eval = load_split(
            args.eval_file,
            hf_split="validation",
        )

        print(
            f"[SFT] Eval size: {len(raw_eval):,}"
        )

    else:

        print(
            "[SFT] No eval file provided — "
            "skipping validation."
        )

    print(
        "[SFT] Formatting and tokenizing dataset …"
    )

    fmt = lambda ex: build_messages(
        ex,
        tokenizer,
        args.max_seq_length,
    )

    train_dataset = raw_train.map(
        fmt,
        remove_columns=raw_train.column_names,
        num_proc=4,
    )

    if raw_eval is not None:

        eval_dataset = raw_eval.map(
            fmt,
            remove_columns=raw_eval.column_names,
            num_proc=4,
        )

    else:

        eval_dataset = None

    print(
        f"[SFT] Loading base model: "
        f"{args.model_name}"
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if continue_lora:

        print(
            f"[SFT] Loading existing LoRA adapter "
            f"from: {adapter_path}"
        )

        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            is_trainable=True,
        )

        peft_config = None

        print(
            "[SFT] Existing LoRA configuration loaded."
        )

        print(
            f"[SFT] Trainable parameters: "
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )

    else:

        print(
            "[SFT] Creating new LoRA adapter..."
        )

        model = base_model

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,

            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],

            lora_dropout=0.1,
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

        bf16=(
            torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        ),

        fp16=(
            torch.cuda.is_available()
            and not torch.cuda.is_bf16_supported()
        ),

        logging_steps=20,

        save_steps=args.save_steps,

        save_total_limit=10,

        eval_strategy=("steps"
            if do_eval
            else "no"),

        eval_steps=(
            args.eval_steps
            if do_eval
            else None
        ),

        load_best_model_at_end=do_eval,

        metric_for_best_model=(
            "eval_loss"
            if do_eval
            else None
        ),

        max_length=args.max_seq_length,

        packing=False,

        report_to="none",
    )

    # TRAINER
    trainer = SFTTrainer(
        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=eval_dataset,

        peft_config=peft_config,

        processing_class=tokenizer,
    )

    print("[SFT] Starting training …")

    trainer.train()

    print(f"[SFT] Saving model to "f"{args.output_dir}")

    trainer.save_model(args.output_dir)

    tokenizer.save_pretrained(args.output_dir)

    print("[SFT] Done.")


if __name__ == "__main__":
    main()