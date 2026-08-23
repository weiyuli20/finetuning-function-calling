import json
import re
from typing import Any

def _get_content(comp: Any) -> str:
    if isinstance(comp, list):
        if comp and isinstance(comp[-1], dict):
            return comp[-1].get("content", "")
        return str(comp)
    if isinstance(comp, dict):
        return comp.get("content", "")
    return str(comp) if comp is not None else ""


def _extract_json_objects(text: str) -> list[dict]:
    results: list[dict] = []

    blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    for block in blocks:
        block = block.strip()
        try:
            obj = json.loads(block)
            if isinstance(obj, list):
                results.extend([o for o in obj if isinstance(o, dict)])
            elif isinstance(obj, dict):
                results.append(obj)
        except json.JSONDecodeError:
            pass

    if results:
        return results

    arr_match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if arr_match:
        try:
            obj = json.loads(arr_match.group(0))
            if isinstance(obj, list):
                results.extend([o for o in obj if isinstance(o, dict)])
                return results
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                results.append(obj)
        except json.JSONDecodeError:
            pass

    return results


def _normalize_value(v: Any) -> str:
    if isinstance(v, dict):
        return json.dumps(v, sort_keys=True, separators=(",", ":")).lower()

    if isinstance(v, list):
        return json.dumps(
            sorted(str(x).lower() for x in v),
            separators=(",", ":"),
        )

    if isinstance(v, bool):
        return str(v).lower()

    if v is None:
        return "null"

    return str(v).lower().strip()


def _tool_call_match(predicted: dict, expected: dict) -> bool:
    if not isinstance(predicted, dict) or not isinstance(expected, dict):
        return False

    if predicted.get("name") != expected.get("name"):
        return False

    pred_args = predicted.get("arguments", {})
    exp_args = expected.get("arguments", {})

    if not isinstance(pred_args, dict):
        return False

    if not isinstance(exp_args, dict):
        return False

    for key, val in exp_args.items():
        if key not in pred_args:
            return False

        if _normalize_value(pred_args[key]) != _normalize_value(val):
            return False

    return True


def format_reward(prompts, completions, **kwargs) -> list[float]:
    # for i, completion in enumerate(completions):
    #             print("\n" + "=" * 80)
    #             print(f"GENERATED COMPLETION {i + 1}")
    #             print("=" * 80)
    #             print(completion)
    rewards = []
    for comp in completions:
        text = _get_content(comp)
        has_fence = bool(re.search(r"```(?:json)?\s*[\[\{]", text, re.IGNORECASE))
        objs = _extract_json_objects(text)
        if has_fence and objs:
            rewards.append(1.0)
        elif objs:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


def schema_validity_reward(prompts, completions, **kwargs) -> list[float]:
    rewards = []
    for comp in completions:
        text = _get_content(comp)
        objs = _extract_json_objects(text)
        if not objs:
            rewards.append(0.0)
            continue
        valid = sum(
            1
            for o in objs
            if isinstance(o, dict)
            and isinstance(o.get("name"), str)
            and isinstance(o.get("arguments"), dict)
        )
        rewards.append(valid / max(len(objs), 1))
    return rewards


def tool_name_recall_reward(prompts, completions, answers, **kwargs) -> list[float]:
    rewards = []
    for comp, ans_raw in zip(completions, answers):
        text = _get_content(comp)
        predicted = _extract_json_objects(text)

        try:
            if isinstance(ans_raw, str):
                gt_calls: list[dict] = json.loads(ans_raw)
            else:
                gt_calls = list(ans_raw)
        except Exception:
            rewards.append(0.0)
            continue

        if not gt_calls:
            rewards.append(1.0)  
            continue

        gt_names = [c.get("name") for c in gt_calls]
        pred_names = [p.get("name") for p in predicted if isinstance(p.get("name"), str)]

        matched = sum(1 for gn in gt_names if gn in pred_names)
        rewards.append(matched / len(gt_names))
    return rewards


def argument_accuracy_reward(prompts, completions, answers, **kwargs) -> list[float]:
    rewards = []
    for comp, ans_raw in zip(completions, answers):
        text = _get_content(comp)
        predicted = _extract_json_objects(text)

        try:
            if isinstance(ans_raw, str):
                gt_calls: list[dict] = json.loads(ans_raw)
            else:
                gt_calls = list(ans_raw)
        except Exception:
            rewards.append(0.0)
            continue

        if not gt_calls:
            rewards.append(1.0)
            continue

        if not predicted:
            rewards.append(0.0)
            continue

        used = [False] * len(predicted)
        correct = 0
        for gt in gt_calls:
            for i, pred in enumerate(predicted):
                if not used[i] and _tool_call_match(pred, gt):
                    correct += 1
                    used[i] = True
                    break

        precision = correct / max(len(predicted), 1)
        recall = correct / max(len(gt_calls), 1)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        rewards.append(f1)
    return rewards

def hallucination_reward(prompts, completions, answers, **kwargs) -> list[float]:
    rewards = []

    for comp, ans_raw in zip(completions, answers):
        text = _get_content(comp)
        predicted = _extract_json_objects(text)

        try:
            if isinstance(ans_raw, str):
                gt_calls: list[dict] = json.loads(ans_raw)
            else:
                gt_calls = list(ans_raw)
        except Exception:
            rewards.append(0.0)
            continue

        penalty = 0.0

        extra_calls = max(0, len(predicted) - len(gt_calls))
        penalty -= 0.3 * extra_calls

        gt_names_set = {c.get("name") for c in gt_calls}

        gt_args_by_name: dict[str, set] = {}

        for c in gt_calls:
            args = c.get("arguments", {})

            if not isinstance(args, dict):
                args = {}

            gt_args_by_name.setdefault(c.get("name"), set()).update(args.keys())

        for pred in predicted:
            pname = pred.get("name")
            args = pred.get("arguments", {})

            if not isinstance(args, dict):
                penalty -= 0.1
                continue

            pargs = set(args.keys())

            if pname in gt_names_set:
                allowed = gt_args_by_name.get(pname, set())
                spurious = pargs - allowed
                penalty -= 0.1 * len(spurious)

        rewards.append(max(penalty, -1.0))

    return rewards


def count_match_reward(prompts, completions, answers, **kwargs) -> list[float]:
    rewards = []
    for comp, ans_raw in zip(completions, answers):
        text = _get_content(comp)
        predicted = _extract_json_objects(text)

        try:
            if isinstance(ans_raw, str):
                gt_calls: list[dict] = json.loads(ans_raw)
            else:
                gt_calls = list(ans_raw)
        except Exception:
            rewards.append(0.0)
            continue

        diff = abs(len(predicted) - len(gt_calls))
        if diff == 0:
            rewards.append(0.5)
        elif diff == 1:
            rewards.append(0.0)
        else:
            rewards.append(-0.5)
    return rewards

def required_args_reward(
    prompts, completions, answers, **kwargs
):
    
    rewards = []

    for comp, ans_raw in zip(completions, answers):
        text = _get_content(comp)
        predicted = _extract_json_objects(text)

        try:
            if isinstance(ans_raw, str):
                gt_calls = json.loads(ans_raw)
            else:
                gt_calls = list(ans_raw)
        except Exception:
            rewards.append(0.0)
            continue

        if not gt_calls:
            rewards.append(1.0)
            continue

        if not predicted:
            rewards.append(0.0)
            continue

        scores = []

        for gt in gt_calls:
            gt_name = gt.get("name")
            gt_args = gt.get("arguments", {})

            if not isinstance(gt_args, dict):
                gt_args = {}

            candidates = [
                p for p in predicted
                if p.get("name") == gt_name
                and isinstance(p.get("arguments"), dict)
            ]

            if not candidates:
                scores.append(0.0)
                continue

            best_score = 0.0

            for pred in candidates:
                pred_args = pred.get("arguments", {})

                if not gt_args:
                    score = 1.0
                else:
                    present = sum(
                        1 for key in gt_args
                        if key in pred_args
                    )
                    score = present / len(gt_args)

                best_score = max(best_score, score)

            scores.append(best_score)

        rewards.append(sum(scores) / len(scores))

    return rewards


def extra_args_reward(
    prompts, completions, answers, **kwargs
):
    
    rewards = []

    for comp, ans_raw in zip(completions, answers):
        text = _get_content(comp)
        predicted = _extract_json_objects(text)

        try:
            if isinstance(ans_raw, str):
                gt_calls = json.loads(ans_raw)
            else:
                gt_calls = list(ans_raw)
        except Exception:
            rewards.append(0.0)
            continue

        if not gt_calls:
            rewards.append(1.0 if not predicted else -1.0)
            continue

        if not predicted:
            rewards.append(0.0)
            continue

        total_extra = 0
        total_pred_args = 0

        allowed_args = {}

        for gt in gt_calls:
            name = gt.get("name")
            args = gt.get("arguments", {})

            if not isinstance(args, dict):
                args = {}

            allowed_args.setdefault(name, set()).update(args.keys())

        for pred in predicted:
            name = pred.get("name")
            args = pred.get("arguments", {})

            if not isinstance(args, dict):
                total_extra += 1
                continue

            pred_keys = set(args.keys())
            total_pred_args += len(pred_keys)

            allowed = allowed_args.get(name, set())

            total_extra += len(pred_keys - allowed)

        if total_pred_args == 0:
            rewards.append(1.0)
            continue

        extra_ratio = total_extra / total_pred_args

        reward = 1.0 - 2.0 * extra_ratio

        rewards.append(max(-1.0, min(1.0, reward)))

    return rewards


def argument_type_reward(
    prompts, completions, answers, **kwargs
):

    def json_type(value):
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, (int, float)):
            return "number"
        if isinstance(value, str):
            return "string"
        if isinstance(value, list):
            return "list"
        if isinstance(value, dict):
            return "object"
        return "other"

    rewards = []

    for comp, ans_raw in zip(completions, answers):
        text = _get_content(comp)
        predicted = _extract_json_objects(text)

        try:
            if isinstance(ans_raw, str):
                gt_calls = json.loads(ans_raw)
            else:
                gt_calls = list(ans_raw)
        except Exception:
            rewards.append(0.0)
            continue

        if not gt_calls:
            rewards.append(1.0)
            continue

        if not predicted:
            rewards.append(0.0)
            continue

        scores = []

        for gt in gt_calls:
            gt_name = gt.get("name")
            gt_args = gt.get("arguments", {})

            if not isinstance(gt_args, dict):
                gt_args = {}

            candidates = [
                p for p in predicted
                if p.get("name") == gt_name
                and isinstance(p.get("arguments"), dict)
            ]

            if not candidates:
                scores.append(0.0)
                continue

            best_score = 0.0

            for pred in candidates:
                pred_args = pred.get("arguments", {})

                if not gt_args:
                    best_score = max(best_score, 1.0)
                    continue

                comparable = [
                    key for key in gt_args
                    if key in pred_args
                ]

                if not comparable:
                    score = 0.0
                else:
                    correct_types = sum(
                        1
                        for key in comparable
                        if json_type(pred_args[key])
                        == json_type(gt_args[key])
                    )

                    score = correct_types / len(gt_args)

                best_score = max(best_score, score)

            scores.append(best_score)

        rewards.append(sum(scores) / len(scores))

    return rewards


def tool_order_reward(
    prompts, completions, answers, **kwargs
):
    
    rewards = []

    for comp, ans_raw in zip(completions, answers):
        text = _get_content(comp)
        predicted = _extract_json_objects(text)

        try:
            if isinstance(ans_raw, str):
                gt_calls = json.loads(ans_raw)
            else:
                gt_calls = list(ans_raw)
        except Exception:
            rewards.append(0.0)
            continue

        gt_names = [
            c.get("name")
            for c in gt_calls
            if isinstance(c, dict)
        ]

        pred_names = [
            p.get("name")
            for p in predicted
            if isinstance(p, dict)
            and isinstance(p.get("name"), str)
        ]

        if len(gt_names) <= 1:
            rewards.append(1.0)
            continue

        total_pairs = 0
        correct_pairs = 0

        for i in range(len(gt_names)):
            for j in range(i + 1, len(gt_names)):
                first = gt_names[i]
                second = gt_names[j]

                if first not in pred_names or second not in pred_names:
                    continue

                first_idx = pred_names.index(first)
                second_idx = pred_names.index(second)

                total_pairs += 1

                if first_idx < second_idx:
                    correct_pairs += 1

        if total_pairs == 0:
            rewards.append(0.0)
        else:
            rewards.append(correct_pairs / total_pairs)

    return rewards


def tool_necessity_reward(
    prompts, completions, answers, **kwargs
):

    rewards = []

    for comp, ans_raw in zip(completions, answers):
        text = _get_content(comp)
        predicted = _extract_json_objects(text)

        try:
            if isinstance(ans_raw, str):
                gt_calls = json.loads(ans_raw)
            else:
                gt_calls = list(ans_raw)
        except Exception:
            rewards.append(0.0)
            continue

        expected_tools = len(gt_calls) > 0
        predicted_tools = len(predicted) > 0

        if expected_tools and predicted_tools:
            rewards.append(1.0)

        elif expected_tools and not predicted_tools:
            rewards.append(-1.0)

        elif not expected_tools and not predicted_tools:
            rewards.append(1.0)

        else:
            rewards.append(-1.0)

    return rewards


def no_tool_reward(
    prompts, completions, answers, **kwargs
):

    rewards = []

    for comp, ans_raw in zip(completions, answers):
        text = _get_content(comp)
        predicted = _extract_json_objects(text)

        try:
            if isinstance(ans_raw, str):
                gt_calls = json.loads(ans_raw)
            else:
                gt_calls = list(ans_raw)
        except Exception:
            rewards.append(0.0)
            continue

        if len(gt_calls) == 0:
            rewards.append(1.0 if len(predicted) == 0 else -1.0)
        else:
            rewards.append(0.0)

    return rewards

def argument_value_reward(
    prompts, completions, answers, **kwargs
):

    rewards = []

    for comp, ans_raw in zip(completions, answers):
        text = _get_content(comp)
        predicted = _extract_json_objects(text)

        try:
            if isinstance(ans_raw, str):
                gt_calls = json.loads(ans_raw)
            else:
                gt_calls = list(ans_raw)
        except Exception:
            rewards.append(0.0)
            continue

        if not gt_calls:
            rewards.append(1.0)
            continue

        if not predicted:
            rewards.append(0.0)
            continue

        call_scores = []

        for gt in gt_calls:
            gt_name = gt.get("name")
            gt_args = gt.get("arguments", {})

            if not isinstance(gt_args, dict):
                gt_args = {}

            candidates = [
                p for p in predicted
                if p.get("name") == gt_name
                and isinstance(p.get("arguments"), dict)
            ]

            if not candidates:
                call_scores.append(0.0)
                continue

            best_score = 0.0

            for pred in candidates:
                pred_args = pred.get("arguments", {})

                if not gt_args:
                    score = 1.0
                else:
                    correct = 0

                    for key, expected_value in gt_args.items():
                        if key not in pred_args:
                            continue

                        if (
                            _normalize_value(pred_args[key])
                            == _normalize_value(expected_value)
                        ):
                            correct += 1

                    score = correct / len(gt_args)

                best_score = max(best_score, score)

            call_scores.append(best_score)

        rewards.append(sum(call_scores) / len(call_scores))

    return rewards