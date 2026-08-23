# SFT → GRPO for Tool Calling

A two-stage fine-tuning project investigating whether **Supervised Fine-Tuning (SFT)** followed by **Group Relative Policy Optimization (GRPO)** can improve tool-calling capabilities in **sub-billion-parameter language models**.

**Pipeline:**

```text
Base Model → SFT → GRPO
```

**Evaluated Models:**
* **`Qwen/Qwen2.5-0.5B-Instruct`** (0.5B parameters)
* **`HuggingFaceTB/SmolLM-135M-Instruct`** (135M parameters)

The project uses the **Salesforce xLAM-60k** tool-calling dataset and evaluates tool selection, argument generation, formatting, schema compliance, and exact tool-call correctness.

## Approach

### 1. Base Model

The pretrained instruction-tuned models are evaluated directly on the tool-calling task to establish a baseline.

### 2. SFT (Supervised Fine-Tuning)

Models are fine-tuned on gold-standard tool-calling examples to learn tool selection, argument generation, JSON formatting, and schema compliance across different LoRA parameter configurations (Rank $r=16, 32, 64$).

### 3. GRPO (Group Relative Policy Optimization)

GRPO optimizes task-specific rewards targeting correct tool selection, argument correctness, exact tool calls, and schema-compliant generation. While GRPO substantially improved most tool-calling metrics, hallucinated fields increased relative to SFT in the Qwen experiment.

### 4. Experimental Setup

| Model | SFT Data | GRPO Data | LoRA Rank | Eval Examples |
| :--- | :--- | :--- | :--- | :--- |
| Qwen2.5-0.5B | Full dataset | ~1,200 | 16 | 2,000 |
| Qwen2.5-0.5B | Full dataset | — | 32 | 2,000 |
| Qwen2.5-0.5B | Full dataset | — | 64 | 2,000 |
| SmolLM-135M | SFT dataset | GRPO dataset | — | 100 |

## Metrics

| Metric | Definition |
| :--- | :--- |
| **Correct Tool** | Whether the predicted tool matches the gold tool |
| **Exact Match** | Whether the complete generated output exactly matches the reference |
| **Tool-Call Accuracy** | Accuracy over correctly structured tool calls |
| **Exact Tool-Call Accuracy** | Whether tool name and all arguments exactly match |
| **Argument Field Accuracy** | Fraction of expected argument fields predicted correctly |
| **Parseable JSON** | Whether the output can be parsed as JSON |
| **Schema Validity** | Whether the parsed call conforms to the expected schema |

## Results

### 1. Qwen2.5-0.5B-Instruct + SFT (r=16) + GRPO Performance

Evaluation of Qwen2.5-0.5B-Instruct (Base) vs. Qwen2.5-0.5B-Instruct + SFT (r=16) vs. Qwen2.5-0.5B-Instruct + SFT (r=16) + GRPO:

*Note: GRPO was trained on approximately 1,200 examples due to compute constraints, while the SFT stage used the full training set. Therefore, the SFT and GRPO stages are not directly comparable in terms of training-data exposure.*

| Metric | Qwen2.5-0.5B-Instruct (Base) | Qwen2.5-0.5B-Instruct + SFT (r=16) | Qwen2.5-0.5B-Instruct + SFT (r=16) + GRPO |
| :--- | ---: | ---: | ---: |
| **Correct Tool** | 81.67% | 97.55% | **98.17%** |
| **Exact Match** | 44.50% | 77.00% | **81.17%** |
| **Tool-Call Accuracy** | 75.76% | 97.91% | **98.10%** |
| **Exact Tool-Call Accuracy** | 51.46% | 81.92% | **87.98%** |
| **Argument Field Accuracy** | 67.70% | 87.94% | **91.12%** |
| **Parseable JSON** | 86.83% | 99.40% | **99.67%** |
| **Correct Format** | 80.33% | 99.40% | **99.67%** |
| **Schema Validity** | 94.83% | 99.40% | **99.67%** |
| **Hallucinated Fields / Example (count)** | 0.508 | 0.0985 | 0.260 |
| **Avg. Completion Tokens** | 157.47 | 169.78 | **138.37** |

*Note: Because SFT already achieves near-perfect performance on formatting and schema-related metrics, GRPO provides only marginal gains on these metrics, while producing larger improvements in exact tool-call correctness and reduced average completion length (an 18.5% reduction, from 169.78 to 138.37 tokens).*

### 2. Qwen2.5-0.5B-Instruct SFT LoRA Rank Ablations

Comparison of the base model baseline and SFT models evaluated on the 2,000-example evaluation set across LoRA rank variations ($r=16, 32, 64$):

| Metric | Qwen2.5-0.5B-Instruct (Base) | SFT (r=16) | SFT (r=32) | SFT (r=64) |
| :--- | :---: | :---: | :---: | :---: |
| **Num Examples** | 2,000 | 2,000 | 2000 | 2000 |
| **Correct Tool** | 81.67% | 97.55% | 91.00% | **97.00%** |
| **Exact Match** | 44.50% | 77.00% | 78.00% | **82.00%** |
| **Tool-Call Accuracy** | 75.76% | 97.91% | 95.00% | **98.75%** |
| **Exact Tool-Call Accuracy** | 51.46% | 81.92% | 80.63% | **82.50%** |
| **Argument Field Accuracy** | 67.70% | 87.94% | 87.06% | **87.88%** |
| **Parseable JSON** | 86.83% | 99.40% | 99.00% | **100.00%** |
| **Correct Format** | 80.33% | 99.40% | 99.00% | **100.00%** |
| **Schema Validity** | 94.83% | 99.40% | 99.00% | **100.00%** |
| **Hallucinated Fields / Example (count)** | 0.5080 | 0.0985 | 0.1500 | **0.1300** |
| **Avg. Completion Tokens** | 157.47 | 169.78 | 171.64 | **167.24** |
| **Completion Rate** | 99.65% | 99.65% | 99.00% | **100.00%** |
| **Gold / Predicted Tool Calls** | 3,308 / 3,284 | 3,308 / 3,284 | 3,308 / 3,298 | **3,308 / 3,312** |

### 3. SmolLM-135M-Instruct Performance

Evaluated on **100 evaluation examples** comparing the base model baseline (`outputs-smolLM/base_model.json`), SFT (`outputs-smolLM/sft_eval_metrics.json`), and GRPO (`outputs-smolLM/grpo_eval_metrics.json`):

| Metric | SmolLM-135M-Instruct (Base) | SmolLM-135M-Instruct + SFT | SmolLM-135M-Instruct + SFT + GRPO |
| :--- | :---: | :---: | :---: |
| **Completion Rate** | 46.00% | 90.00% | **100.00%** |
| **Parseable JSON** | 0.00% | 64.00% | **97.00%** |
| **Correct Format** | 0.00% | 62.00% | **97.00%** |
| **Schema Validity** | 0.00% | 70.00% | **97.00%** |
| **Correct Tool** | 0.00% | 48.00% | **72.00%** |
| **Exact Match** | 0.00% | 21.00% | **37.00%** |
| **Tool-Call Accuracy** | 0.00% | 49.25% | **73.13%** |
| **Exact Tool-Call Accuracy** | 0.00% | 25.50% | **41.88%** |
| **Argument Field Accuracy** | 0.00% | 41.20% | **60.43%** |
| **Hallucinated Fields / Example (count)** | 0.0000 | 1.4200 | **0.7000** |
| **Avg. Completion Tokens** | 1024.00 | 245.50 | **151.88** |
| **Gold / Predicted Tool Calls** | 160 / 0 | 160 / 112 | **160 / 145** |

## Key Takeaways

* **Qwen2.5-0.5B-Instruct + SFT ($r=16$) + GRPO:** SFT + GRPO fine-tuning with LoRA rank $r=16$ significantly elevates performance compared to the base model. However, because SFT alone achieves up to 97.91% Tool-Call Accuracy, near-ceiling performance limited further gains in some evaluation scores during GRPO.

* **LoRA Rank Ablations ($r=16, 32, 64$):** Among the evaluated LoRA configurations, $r=64$ achieved the strongest overall SFT results, including the highest Exact Match, Tool-Call Accuracy, Parseable JSON, Correct Format, and Schema Validity, while maintaining a moderate average completion length of 167.24 tokens.

* **SmolLM-135M-Instruct Performance:** Although the base SmolLM model produced non-empty completions for 46% of examples, it produced zero parseable tool calls on the evaluation set. Consequently, the base model achieved 0% on Parseable JSON, Correct Format, Schema Validity, Correct Tool, Exact Match, Tool-Call Accuracy, Exact Tool-Call Accuracy, and Argument Field Accuracy. In observed failures, the base model frequently generated conversational or repetitive text rather than structured JSON. For example, when asked for A4 label sheet brands (`outputs-smolLM/base_model_predictions.json`), it outputs conversational text and repeatedly generates the same text until reaching the 1024-token generation limit. SFT converts SmolLM from a model that produces no valid tool calls into one capable of generating structured tool calls, while GRPO further improves both structural validity and semantic correctness. GRPO optimization significantly boosts SmolLM performance further, raising Parseable JSON to **97.00%**, Correct Tool selection to **72.00%**, and Exact Tool-Call Accuracy to **41.88%**.


