# SFT → GRPO for Tool Calling

A two-stage fine-tuning project investigating whether **Supervised Fine-Tuning (SFT)** followed by **Group Relative Policy Optimization (GRPO)** can improve tool-calling capabilities in a **sub-billion-parameter language model**.

**Pipeline:**

```text
Qwen2.5-0.5B-Instruct → SFT → GRPO
```

The base model is **`Qwen/Qwen2.5-0.5B-Instruct`**, a 0.5B-parameter instruction-tuned model.

The project uses the **Salesforce xLAM-60k** tool-calling dataset and evaluates tool selection, argument generation, formatting, schema compliance, and exact tool-call correctness.

## Approach

### 1. Base Model

The pretrained instruction-tuned Qwen model is evaluated directly on the tool-calling task to establish a baseline.

### 2. SFT

The base model is fine-tuned on gold-standard tool-calling examples to learn:

* Tool selection
* Argument generation
* JSON formatting
* Schema compliance

### 3. GRPO

GRPO starts **from the SFT checkpoint** and further optimizes tool-calling behavior using task-specific rewards targeting:

* Correct tool selection
* Correct arguments
* Exact tool calls
* Reduced hallucinated fields

## Results

Evaluation was performed on the same **600-example evaluation set**.

| Metric                        |   Base |    SFT | SFT + GRPO |
| ----------------------------- | -----: | -----: | ---------: |
| Correct Tool                  | 81.67% | 92.67% | **94.17%** |
| Exact Match                   | 44.50% | 60.67% | **64.17%** |
| Tool-Call Accuracy            | 75.76% | 95.61% | **96.10%** |
| Exact Tool-Call Accuracy      | 51.46% | 70.24% | **72.98%** |
| Argument Field Accuracy       | 67.70% | 79.98% | **83.12%** |
| Parseable JSON                | 86.83% | 99.67% | **99.67%** |
| Correct Format                | 80.33% | 99.67% | **99.67%** |
| Schema Validity               | 94.83% | 99.67% | **99.67%** |
| Hallucinated Fields / Example |  0.508 |  0.272 |  **0.260** |
| Avg. Completion Tokens        | 157.47 | 153.76 | **138.37** |

### Improvement Over Base

SFT provides the major performance improvement:

* **Correct Tool:** 81.67% → 92.67% (**+11.00 pp**)
* **Exact Match:** 44.50% → 60.67% (**+16.17 pp**)
* **Exact Tool-Call Accuracy:** 51.46% → 70.24% (**+18.78 pp**)
* **Argument Accuracy:** 67.70% → 79.98% (**+12.28 pp**)
* **Parseable JSON:** 86.83% → 99.67% (**+12.84 pp**)

GRPO then provides additional improvements on top of SFT:

* **Correct Tool:** +1.50 pp
* **Exact Match:** +3.50 pp
* **Exact Tool-Call Accuracy:** +2.73 pp
* **Argument Accuracy:** +3.14 pp
* **Hallucinated Fields:** −4.29%
* **Completion Length:** −10.00%

## Why Was GRPO's Improvement Smaller?

The smaller GRPO improvement is expected because **SFT had already transformed the base model into a strong tool-calling model**.

SFT increased correct tool selection from **81.67% to 92.67%** and exact tool-call accuracy from **51.46% to 70.24%**.

By the time GRPO started, the model already had:

* 99.67% schema validity
* 99.67% correct formatting
* 92.67% correct tool selection
* 95.61% tool-call accuracy

Therefore, GRPO had much less headroom. Its role was primarily to refine **harder semantic errors**, particularly argument correctness and exact tool-call matching.

## Key Takeaway

> **SFT teaches the 0.5B model how to perform tool calling; GRPO refines the remaining errors.**

The experiment demonstrates a clear progression:

```text
Base
  │
  │ SFT
  ▼
Large improvement
  │
  │ GRPO
  ▼
Smaller but targeted improvement
```

For this task, **SFT is responsible for the majority of the performance gain**, while **GRPO provides additional improvements in fine-grained tool-call correctness**.
