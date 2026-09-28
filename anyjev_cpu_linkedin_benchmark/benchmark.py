from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

import psutil
from anyjev import Decider, Question
from anyjev.backends.hf import HFBackend

from config import MODEL_NAME, DEVICE, DTYPE, BATCH_SIZE, ANYJEV_LEVEL
from dataset import DATASET
from tools import TOOLS


OPTIONS = [
    "customer_lookup",
    "order_lookup",
    "calculator",
    "web_search",
    "none",
]


def build_decider(model):
    backend = HFBackend(
        model,
        device=DEVICE,
        dtype=DTYPE,
        batch_size=BATCH_SIZE,
    )
    return Decider(
        backend,
        level=ANYJEV_LEVEL,
        prior="batch",
    )


def build_question():
    return Question.choice(
        """Choose the single tool that should handle the user's request.

customer_lookup: customer information.
order_lookup: order information.
calculator: arithmetic.
web_search: external/current information.
none: no external tool is required.""",
        OPTIONS,
        name="tool_selection",
    )


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    index = int(round((len(values) - 1) * p))
    return values[index]


def benchmark(decider, repeat):
    question = build_question()
    process = psutil.Process(os.getpid())

    # Warm-up: excluded from reported benchmark latency.
    decider.decide(
        DATASET[0]["request"],
        [question],
        level=ANYJEV_LEVEL,
    )

    rows = []

    for r in range(1, repeat + 1):
        for item in DATASET:
            request = item["request"]
            expected = item["expected_tool"]

            cpu_before = process.cpu_percent(None)
            rss_before = process.memory_info().rss / (1024 * 1024)

            t0 = time.perf_counter()

            result = decider.decide(
                request,
                [question],
                level=ANYJEV_LEVEL,
            )

            decision_ms = (time.perf_counter() - t0) * 1000

            cpu_after = process.cpu_percent(None)
            rss_after = process.memory_info().rss / (1024 * 1024)

            decision = result["tool_selection"]
            predicted = decision.argmax
            confidence = float(decision.confidence)

            tool_ms = 0.0
            tool_result = None

            if predicted in TOOLS:
                t1 = time.perf_counter()
                tool_result = TOOLS[predicted](request)
                tool_ms = (time.perf_counter() - t1) * 1000

            correct = predicted == expected
            e2e_ms = decision_ms + tool_ms

            rows.append({
                "run": r,
                "id": item["id"],
                "request": request,
                "expected_tool": expected,
                "predicted_tool": predicted,
                "correct": correct,
                "confidence": confidence,
                "decision_latency_ms": decision_ms,
                "tool_latency_ms": tool_ms,
                "end_to_end_latency_ms": e2e_ms,
                "rss_before_mb": rss_before,
                "rss_after_mb": rss_after,
                "cpu_percent_sample": cpu_after,
                "tool_result": json.dumps(tool_result, default=str),
            })

            print(
                f"[{r}/{repeat}] "
                f"{item['id']:02d} "
                f"expected={expected:16s} "
                f"predicted={predicted:16s} "
                f"correct={str(correct):5s} "
                f"confidence={confidence:.3f} "
                f"latency={decision_ms:.1f}ms"
            )

    return rows


def summarize(rows):
    total = len(rows)
    correct = sum(row["correct"] for row in rows)

    decision_latencies = [r["decision_latency_ms"] for r in rows]
    e2e_latencies = [r["end_to_end_latency_ms"] for r in rows]
    confidences = [r["confidence"] for r in rows]

    correct_conf = [
        r["confidence"] for r in rows if r["correct"]
    ]
    incorrect_conf = [
        r["confidence"] for r in rows if not r["correct"]
    ]

    accuracy = correct / total if total else 0

    confusion = Counter(
        (r["expected_tool"], r["predicted_tool"])
        for r in rows
    )

    per_tool = defaultdict(list)
    for row in rows:
        per_tool[row["expected_tool"]].append(row)

    tool_summary = {}
    for tool, tool_rows in per_tool.items():
        tool_summary[tool] = {
            "count": len(tool_rows),
            "accuracy": sum(r["correct"] for r in tool_rows) / len(tool_rows),
            "avg_confidence": statistics.mean(
                r["confidence"] for r in tool_rows
            ),
            "median_decision_latency_ms": statistics.median(
                r["decision_latency_ms"] for r in tool_rows
            ),
        }

    return {
        "total_requests": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": accuracy,
        "avg_confidence": statistics.mean(confidences),
        "avg_confidence_correct": (
            statistics.mean(correct_conf) if correct_conf else None
        ),
        "avg_confidence_incorrect": (
            statistics.mean(incorrect_conf) if incorrect_conf else None
        ),
        "decision_latency_ms": {
            "mean": statistics.mean(decision_latencies),
            "median": statistics.median(decision_latencies),
            "p95": percentile(decision_latencies, 0.95),
        },
        "end_to_end_latency_ms": {
            "mean": statistics.mean(e2e_latencies),
            "median": statistics.median(e2e_latencies),
            "p95": percentile(e2e_latencies, 0.95),
        },
        "peak_rss_mb": max(r["rss_after_mb"] for r in rows),
        "avg_rss_mb": statistics.mean(r["rss_after_mb"] for r in rows),
        "avg_cpu_sample_percent": statistics.mean(
            r["cpu_percent_sample"] for r in rows
        ),
        "confusion_matrix": {
            f"{expected} -> {predicted}": count
            for (expected, predicted), count in confusion.items()
        },
        "per_expected_tool": tool_summary,
    }


def markdown_report(summary, metadata, rows):
    incorrect = [
        r for r in rows if not r["correct"]
    ]

    lines = [
        "# AnyJev CPU Agent Tool-Routing Benchmark",
        "",
        "## Environment",
        f"- Model: `{metadata['model']}`",
        f"- AnyJev level: `{metadata['anyjev_level']}`",
        f"- Device: `{metadata['device']}`",
        f"- dtype: `{metadata['dtype']}`",
        f"- Batch size: `{metadata['batch_size']}`",
        f"- OS: `{metadata['os']}`",
        f"- Python: `{metadata['python']}`",
        f"- CPU: `{metadata['cpu']}`",
        f"- RAM available at start: `{metadata['ram_available_mb']:.0f} MB`",
        f"- Dataset requests: `{metadata['dataset_size']}`",
        f"- Repetitions: `{metadata['repeat']}`",
        "",
        "## Results",
        f"- Accuracy: **{summary['accuracy'] * 100:.2f}%**",
        f"- Correct: `{summary['correct']}/{summary['total_requests']}`",
        f"- Average confidence: `{summary['avg_confidence']:.4f}`",
        f"- Confidence when correct: `{summary['avg_confidence_correct']}`",
        f"- Confidence when incorrect: `{summary['avg_confidence_incorrect']}`",
        f"- Mean decision latency: `{summary['decision_latency_ms']['mean']:.2f} ms`",
        f"- Median decision latency: `{summary['decision_latency_ms']['median']:.2f} ms`",
        f"- P95 decision latency: `{summary['decision_latency_ms']['p95']:.2f} ms`",
        f"- Mean end-to-end latency: `{summary['end_to_end_latency_ms']['mean']:.2f} ms`",
        f"- P95 end-to-end latency: `{summary['end_to_end_latency_ms']['p95']:.2f} ms`",
        f"- Peak RSS: `{summary['peak_rss_mb']:.2f} MB`",
        "",
        "## LinkedIn-ready draft",
        "",
        "I experimented with Nokia Applied Research's AnyJev as a typed "
        "decision layer for agent tool calling on a CPU-only 12 GB RAM environment.",
        "",
        f"Using `{metadata['model']}` with AnyJev `{metadata['anyjev_level']}`, "
        f"I evaluated {summary['total_requests']} labeled tool-routing requests.",
        "",
        f"Results from this run:",
        f"- Tool-routing accuracy: **{summary['accuracy'] * 100:.2f}%**",
        f"- Median decision latency: **{summary['decision_latency_ms']['median']:.2f} ms**",
        f"- P95 decision latency: **{summary['decision_latency_ms']['p95']:.2f} ms**",
        f"- Average confidence: **{summary['avg_confidence']:.3f}**",
        f"- Peak process RSS: **{summary['peak_rss_mb']:.0f} MB**",
        "",
        "The architecture separates decision from execution:",
        "",
        "User -> AnyJev -> typed tool decision -> Python/MCP tool -> result.",
        "",
        "The interesting part is not only accuracy. I also measured confidence, "
        "latency and local resource usage, and kept the tool executor outside "
        "the decision model.",
        "",
        "Next experiment: compare this against conventional LLM tool calling "
        "under the same hardware, dataset and prompts, and measure routing accuracy, "
        "latency, token usage and fallback rate.",
        "",
        "Important: these numbers are from one controlled local run and should "
        "not be generalized to other hardware or models.",
        "",
        "## Incorrect predictions",
    ]

    if not incorrect:
        lines.append("- None in this run.")
    else:
        for r in incorrect:
            lines.append(
                f"- `{r['id']}` expected `{r['expected_tool']}`, "
                f"predicted `{r['predicted_tool']}`, "
                f"confidence `{r['confidence']:.3f}`: {r['request']}"
            )

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    process = psutil.Process(os.getpid())
    vm = psutil.virtual_memory()

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "anyjev_level": ANYJEV_LEVEL,
        "device": DEVICE,
        "dtype": DTYPE,
        "batch_size": BATCH_SIZE,
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": platform.processor() or "unknown",
        "ram_available_mb": vm.available / (1024 * 1024),
        "dataset_size": len(DATASET),
        "repeat": args.repeat,
    }

    print("\nLoading model...\n")
    decider = build_decider(args.model)

    print("\nRunning benchmark...\n")
    rows = benchmark(decider, args.repeat)

    summary = summarize(rows)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = f"results/benchmark_{timestamp}.csv"
    json_path = f"results/benchmark_{timestamp}.json"
    md_path = f"results/benchmark_{timestamp}.md"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    output = {
        "metadata": metadata,
        "summary": summary,
        "rows": rows,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_report(summary, metadata, rows))

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Accuracy:             {summary['accuracy'] * 100:.2f}%")
    print(f"Correct:              {summary['correct']}/{summary['total_requests']}")
    print(f"Mean latency:         {summary['decision_latency_ms']['mean']:.2f} ms")
    print(f"Median latency:       {summary['decision_latency_ms']['median']:.2f} ms")
    print(f"P95 latency:          {summary['decision_latency_ms']['p95']:.2f} ms")
    print(f"Mean E2E latency:     {summary['end_to_end_latency_ms']['mean']:.2f} ms")
    print(f"Peak RSS:             {summary['peak_rss_mb']:.2f} MB")
    print(f"Avg confidence:       {summary['avg_confidence']:.4f}")

    print("\nGenerated:")
    print(csv_path)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
