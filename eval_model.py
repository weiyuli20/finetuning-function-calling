import argparse
import json
import os
import re
from collections import Counter
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import AutoPeftModelForCausalLM
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

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


def normalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): normalize_json(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, list):
        return [normalize_json(v) for v in value]
    return value


def extract_json(text: str):
    '''格式匹配，匹配```json ... ```,并提取内容'''
    text = text.strip()

    fenced = re.fullmatch(r"\s*```json\s*(.*?)\s*```\s*", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        payload = fenced.group(1).strip()
        try:
            return json.loads(payload), True
        except json.JSONDecodeError:
            return None, False

    fenced_generic = re.fullmatch(r"\s*```\s*(.*?)\s*```\s*", text, flags=re.DOTALL)
    if fenced_generic:
        payload = fenced_generic.group(1).strip()
        try:
            return json.loads(payload), False
        except json.JSONDecodeError:
            return None, False

    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        return None, False


def is_valid_call_obj(obj: Any) -> bool:
    '''
    验证每一个函数调用结构是否合法
    '''
    return (
        isinstance(obj, dict)
        and set(obj.keys()) == {"name", "arguments"}
        and isinstance(obj["name"], str)
        and isinstance(obj["arguments"], dict)
    )


def validate_prediction(parsed: Any):
    if not isinstance(parsed, list):
        return False, None
    if not all(is_valid_call_obj(x) for x in parsed):
        return False, None
    return True, normalize_json(parsed)


def expected_calls(row):
    answers = row["answers"]
    if isinstance(answers, str):
        answers = json.loads(answers)
    if not isinstance(answers, list):
        raise ValueError("Expected 'answers' to decode to a list")
    if not all(is_valid_call_obj(x) for x in answers):
        raise ValueError("Ground-truth contains malformed tool call(s)")
    return normalize_json(answers)


def call_tool_name_match(pred_calls, gold_calls, ignore_order=False):
    if ignore_order:
        pred_names = Counter(x["name"] for x in pred_calls)
        gold_names = Counter(x["name"] for x in gold_calls)
        return pred_names == gold_names
    return len(pred_calls) == len(gold_calls) and all(
        p["name"] == g["name"] for p, g in zip(pred_calls, gold_calls)
    )


def get_call_matches(pred_calls, gold_calls, ignore_order=False):
    if not ignore_order:
        n = min(len(pred_calls), len(gold_calls))
        return list(zip(pred_calls[:n], gold_calls[:n]))

    remaining = list(gold_calls)
    pairs = []
    for pred in pred_calls:
        idx = next((i for i, gold in enumerate(remaining) if gold["name"] == pred["name"]), None)
        if idx is not None:
            pairs.append((pred, remaining.pop(idx)))
    return pairs


def count_argument_fields(gold_calls, pred_calls, ignore_order=False):
    pairs = get_call_matches(pred_calls, gold_calls, ignore_order)
    correct_fields = 0
    total_fields = 0
    hallucinated_fields = 0

    for pred, gold in pairs:
        gold_args = gold["arguments"]
        pred_args = pred["arguments"]

        total_fields += len(gold_args)
        correct_fields += sum(
            key in pred_args and normalize_json(pred_args[key]) == normalize_json(value)
            for key, value in gold_args.items()
        )
        hallucinated_fields += sum(key not in gold_args for key in pred_args)

    if len(pred_calls) > len(pairs):
        for pred in pred_calls:
            if pred not in [p for p, _ in pairs]:
                hallucinated_fields += len(pred["arguments"])

    return correct_fields, total_fields, hallucinated_fields


def build_inputs(rows, tokenizer):
    messages_batch = []
    for row in rows:
        tools_str = row["tools"]
        query = row["query"]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\nAvailable tools:\n" + tools_str},
            {"role": "user", "content": query},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        messages_batch.append(text)
    return messages_batch


def load_model(model_path, tokenizer):
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    kwargs = {"torch_dtype": dtype if torch.cuda.is_available() else torch.float32}

    adapter_config = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config) and HAS_PEFT:
        print(f"[eval] Loading LoRA adapter: {model_path}")
        model = AutoPeftModelForCausalLM.from_pretrained(model_path, **kwargs)
    else:
        print(f"[eval] Loading full model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate an SFT tool-calling model on held-out xLAM eval data")
    p.add_argument("--model_name", type=str, required=True,
                   help="SFT checkpoint directory, e.g. outputs/sft_model")
    p.add_argument("--eval_file", type=str, default="data/xlam_eval.jsonl")
    p.add_argument("--output_file", type=str, default="outputs/sft_eval_metrics.json")
    p.add_argument("--predictions_file", type=str, default=None,
                   help="Optional JSONL file with per-example predictions and diagnostics")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_input_length", type=int, default=2048)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--ignore_order", action="store_true",
                   help="Match tool calls by tool name instead of requiring list order")
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate only the first N examples")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.eval_file):
        raise FileNotFoundError(f"Eval file not found: {args.eval_file}")

    print(f"[eval] Loading eval data: {args.eval_file}")
    dataset = load_dataset("json", data_files={"data": args.eval_file})["data"]
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    print(f"[eval] Examples: {len(dataset):,}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = load_model(args.model_name, tokenizer)

    stats = Counter()
    total_gold_calls = 0
    total_pred_calls = 0
    total_correct_tool_calls = 0
    total_exact_calls = 0
    total_gold_arg_fields = 0
    total_correct_arg_fields = 0
    total_hallucinated_fields = 0
    total_generated_tokens = 0
    per_example = []

    for start in range(0, len(dataset), args.batch_size):
        rows = [dataset[i] for i in range(start, min(start + args.batch_size, len(dataset)))]
        prompts = build_inputs(rows, tokenizer)
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_length,
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        for j, row in enumerate(rows):
            output_ids = generated[j][input_len:]
            text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            total_generated_tokens += len(output_ids)
            eos_completed = (tokenizer.eos_token_id is not None and tokenizer.eos_token_id in output_ids.tolist())

            parsed, fenced_json = extract_json(text)
            schema_ok, pred_calls = validate_prediction(parsed)
            gold_calls = expected_calls(row)

            n_gold = len(gold_calls)
            n_pred = len(pred_calls) if schema_ok else 0
            total_gold_calls += n_gold
            total_pred_calls += n_pred

            nonempty = bool(text)
            parse_ok = parsed is not None
            format_ok = bool(fenced_json and schema_ok)

            if nonempty:
                stats["nonempty"] += 1
            if eos_completed:
                stats["eos_completed"] += 1
            if parse_ok:
                stats["parseable"] += 1
            if format_ok:
                stats["format_ok"] += 1
            if schema_ok:
                stats["schema_ok"] += 1

            tool_ok = schema_ok and call_tool_name_match(pred_calls, gold_calls, args.ignore_order)
            if tool_ok:
                stats["tool_exact_example"] += 1

            exact = schema_ok and pred_calls == gold_calls
            if exact:
                stats["exact_match"] += 1

            if schema_ok:
                pairs = get_call_matches(pred_calls, gold_calls, args.ignore_order)
                total_correct_tool_calls += sum(p["name"] == g["name"] for p, g in pairs)
                total_exact_calls += sum(
                    p["name"] == g["name"] and p["arguments"] == g["arguments"]
                    for p, g in pairs
                )

                correct_fields, total_fields, hallucinated_fields = count_argument_fields(
                    gold_calls, pred_calls, args.ignore_order
                )
                total_correct_arg_fields += correct_fields
                total_gold_arg_fields += total_fields
                total_hallucinated_fields += hallucinated_fields

            per_example.append({
                "index": start + j,
                "query": row["query"],
                "gold": gold_calls,
                "prediction_text": text,
                "prediction": pred_calls if schema_ok else None,
                "nonempty": nonempty,
                "parseable_json": parse_ok,
                "correct_format": format_ok,
                "schema_valid": schema_ok,
                "correct_tools": tool_ok,
                "exact_match": exact,
            })

        done = min(start + len(rows), len(dataset))
        print(f"[eval] {done:,}/{len(dataset):,} examples")

    n = len(dataset)
    metrics = {
        "num_examples": n,
        "completion_percentage": 100.0 * stats["eos_completed"] / n if n else 0.0,
        "nonempty_response_percentage": 100.0 * stats["nonempty"] / n if n else 0.0,
        "parseable_json_percentage": 100.0 * stats["parseable"] / n if n else 0.0,
        "correct_format_percentage": 100.0 * stats["format_ok"] / n if n else 0.0,
        "schema_validity_rate": stats["schema_ok"] / n if n else 0.0,
        "correct_tool_percentage": 100.0 * stats["tool_exact_example"] / n if n else 0.0,
        "exact_match_percentage": 100.0 * stats["exact_match"] / n if n else 0.0,
        "tool_call_accuracy": total_correct_tool_calls / total_gold_calls if total_gold_calls else 0.0,
        "exact_tool_call_accuracy": total_exact_calls / total_gold_calls if total_gold_calls else 0.0,
        "argument_field_accuracy": total_correct_arg_fields / total_gold_arg_fields if total_gold_arg_fields else 0.0,
        "avg_hallucinated_fields_per_example": total_hallucinated_fields / n if n else 0.0,
        "avg_completion_tokens": total_generated_tokens / n if n else 0.0,
        "total_gold_tool_calls": total_gold_calls,
        "total_predicted_tool_calls": total_pred_calls,
    }

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if args.predictions_file:
        os.makedirs(os.path.dirname(args.predictions_file) or ".", exist_ok=True)
        with open(args.predictions_file, "w", encoding="utf-8") as f:
            for row in per_example:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n[eval] Results")
    print(f"  Examples                 : {n:,}")
    print(f"  Completion (EOS)         : {metrics['completion_percentage']:.2f}%")
    print(f"  Non-empty response       : {metrics['nonempty_response_percentage']:.2f}%")
    print(f"  Parseable JSON           : {metrics['parseable_json_percentage']:.2f}%")
    print(f"  Correct format           : {metrics['correct_format_percentage']:.2f}%")
    print(f"  Schema validity          : {metrics['schema_validity_rate']:.4f}")
    print(f"  Correct tools (examples) : {metrics['correct_tool_percentage']:.2f}%")
    print(f"  Tool-call name accuracy  : {metrics['tool_call_accuracy']:.4f}")
    print(f"  Exact tool-call accuracy : {metrics['exact_tool_call_accuracy']:.4f}")
    print(f"  Argument field accuracy  : {metrics['argument_field_accuracy']:.4f}")
    print(f"  Exact match              : {metrics['exact_match_percentage']:.2f}%")
    print(f"  Avg hallucinated fields  : {metrics['avg_hallucinated_fields_per_example']:.4f}")
    print(f"  Avg completion tokens    : {metrics['avg_completion_tokens']:.2f}")
    print(f"\n[eval] Metrics saved -> {args.output_file}")
    if args.predictions_file:
        print(f"[eval] Predictions saved -> {args.predictions_file}")


if __name__ == "__main__":
    main()

'''
cd /home/dyyuser/lq/finetuning-function-calling

CUDA_VISIBLE_DEVICES=2 python eval_model.py \
  --model_name outputs/sft_grpo/grpo \
  --eval_file data/xlam_eval.jsonl \
  --output_file outputs/grpo_eval_metrics.json \
  --predictions_file outputs/grpo_eval_predictions.jsonl

'''