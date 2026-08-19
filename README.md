# SFT → GRPO for Tool Calling

A two-stage fine-tuning project investigating whether **Group Relative Policy Optimization (GRPO)** can improve tool-calling performance after **Supervised Fine-Tuning (SFT)**.

**Pipeline:**

```text
Base Model → SFT → GRPO
```

The project uses the **Salesforce xLAM-60k** dataset and evaluates tool selection, argument generation, formatting, and exact tool-call correctness.

## Approach

### 1. SFT

SFT trains the model on gold-standard tool-calling examples to learn:

* Tool selection
* Argument generation
* JSON formatting
* Schema compliance

### 2. GRPO

GRPO starts **from the trained SFT checkpoint** and further optimizes the model using tool-calling rewards, targeting:

* Correct tool selection
* Correct arguments
* Exact tool calls
* Reduced hallucinated fields

## Results

Evaluation was performed on **600 examples**.

| Metric                        |    SFT | SFT + GRPO |   Change |
| ----------------------------- | -----: | ---------: | -------: |
| Correct Tool                  | 92.67% | **94.17%** | +1.50 pp |
| Exact Match                   | 60.67% | **64.17%** | +3.50 pp |
| Tool-Call Accuracy            | 95.61% | **96.10%** | +0.49 pp |
| Exact Tool-Call Accuracy      | 70.24% | **72.98%** | +2.73 pp |
| Argument Field Accuracy       | 79.98% | **83.12%** | +3.14 pp |
| Schema Validity               | 99.67% | **99.67%** |        — |
| Hallucinated Fields / Example |  0.272 |  **0.260** |   −4.29% |
| Avg. Completion Tokens        | 153.76 | **138.37** |  −10.00% |

### Why GRPO Was Not a Huge Improvement

The SFT checkpoint was already very strong, achieving **99.67% schema validity** and **92.67% correct tool selection**. Therefore, GRPO had limited headroom to improve the basic tool-calling behavior.

Since GRPO was initialized from SFT, its role was primarily to optimize **remaining semantic errors**, such as incorrect arguments, tool selection mistakes, and hallucinated fields.

The largest gains were therefore in fine-grained metrics:

* **+3.50 pp** Exact Match
* **+3.14 pp** Argument Accuracy
* **+2.73 pp** Exact Tool-Call Accuracy

This suggests that GRPO works well as a **second-stage optimization method after SFT**, rather than as a replacement for SFT.

## Key Takeaway

> **SFT learns the tool-calling behavior; GRPO refines the remaining correctness errors.**

The experiment shows that GRPO can provide measurable improvements even when starting from a highly capable SFT checkpoint, although the gains are naturally smaller because the SFT model has already solved most of the task.
