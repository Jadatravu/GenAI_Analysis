# AnyJev CPU Agent Tool-Calling Benchmark

A CPU/12-GB-RAM tutorial for evaluating Nokia Applied Research AnyJev as a
typed tool-selection layer.

## Goal

Produce reproducible metrics suitable for a technical LinkedIn post:

- routing accuracy
- confidence
- confidence on correct vs incorrect decisions
- decision latency
- tool execution latency
- end-to-end latency
- CPU usage
- RAM usage
- per-tool statistics
- confusion matrix
- JSON and CSV results
- Markdown report for LinkedIn

## Hardware

Recommended:

- CPU-only
- 12 GB RAM
- Python 3.10+
- AnyJev L0
- Qwen3-0.6B

Do not claim GPU/7B performance from this benchmark.

## Install

python -m venv .venv

Windows:
.venv\Scripts\activate

Linux:
source .venv/bin/activate

pip install -r requirements.txt

## 1. Fast validation

Run:

python fake_benchmark.py

This validates the metric collection pipeline without downloading a model.

## 2. Real CPU benchmark

Run:

python benchmark.py

The first run downloads Qwen/Qwen3-0.6B.

For a longer benchmark:

python benchmark.py --repeat 3

For a different model:

python benchmark.py --model Qwen/Qwen3-0.6B

The benchmark uses batch size 1 and a single process.

## 3. Output

The benchmark creates:

results/
    benchmark_<timestamp>.csv
    benchmark_<timestamp>.json
    benchmark_<timestamp>.md

The Markdown file contains a LinkedIn-ready summary template.

## 4. Important methodology

The benchmark has a labeled dataset:

    user request -> expected tool

Accuracy is:

    correct predictions / total predictions

Do not report confidence as accuracy.

Latency is measured separately:

1. decision latency
2. tool latency
3. end-to-end latency

RAM is sampled using the current process RSS. CPU is sampled using psutil.

The benchmark does not measure a competing LLM baseline. If you want to claim
"X% faster than LLM tool calling", build and run the LLM baseline under the
same hardware, software, prompts, dataset and process conditions.

## 5. Suggested LinkedIn claims

Use only numbers actually produced by your run.

Good:

"On my CPU-only 12 GB RAM environment, AnyJev L0 with Qwen3-0.6B achieved
XX% tool-routing accuracy across NN labeled requests, with median decision
latency of YY ms."

Avoid:

"AnyJev is faster than all LLM agents."

Also disclose:

- CPU model
- RAM
- OS
- Python version
- model
- AnyJev level
- dataset size
- whether results are cold-start or warm
- number of repetitions

## 6. Suggested experiment design

Run at least 30-50 labeled requests for a meaningful demo.

Include ambiguous requests and "none" cases.

For stronger evidence, create 100+ requests and repeat each request 3-5 times.

## 7. Interpretation

Accuracy tells you whether the selected tool was correct.

Confidence tells you how strongly the decision favored the selected option.

Latency tells you how quickly the decision completed.

RAM/CPU tells you the local resource cost.

A good technical post should show all of these rather than reporting a single
headline number.
