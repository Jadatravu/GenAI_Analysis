from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime

import psutil
from anyjev import Decider, Question
from anyjev.backends.fake import FakeBackend

from dataset import DATASET
from tools import TOOLS


OPTIONS = [
    "customer_lookup",
    "order_lookup",
    "calculator",
    "web_search",
    "none",
]


def scoring(state, option):
    text = state.lower()

    if option == "calculator" and (
        "calculate" in text or
        "compute" in text or
        "*" in text or
        "+" in text or
        "/" in text or
        "-" in text
    ):
        return 5.0

    if option == "customer_lookup" and (
        "customer" in text or "client" in text
    ):
        return 5.0

    if option == "order_lookup" and (
        "order" in text or "shipment" in text
    ):
        return 5.0

    if option == "web_search" and (
        "search" in text or "latest" in text or
        "current" in text or "online" in text
    ):
        return 5.0

    if option == "none" and (
        text.strip() == "hello" or
        "explain" in text
    ):
        return 5.0

    return 0.0


def main():
    os.makedirs("results", exist_ok=True)

    backend = FakeBackend(
        scoring,
        position_bias=[0.0] * len(OPTIONS),
    )

    decider = Decider(
        backend,
        level="L0",
        prior="none",
    )

    question = Question.choice(
        "Which tool should handle the request?",
        OPTIONS,
        name="tool_selection",
    )

    process = psutil.Process(os.getpid())

    rows = []

    for item in DATASET:
        t0 = time.perf_counter()

        result = decider.decide(
            item["request"],
            [question],
            level="L0",
        )

        decision_ms = (time.perf_counter() - t0) * 1000

        decision = result["tool_selection"]

        tool_result = TOOLS[
            decision.argmax
        ](item["request"])

        rows.append({
            "id": item["id"],
            "request": item["request"],
            "expected_tool": item["expected_tool"],
            "predicted_tool": decision.argmax,
            "correct": decision.argmax == item["expected_tool"],
            "confidence": decision.confidence,
            "decision_latency_ms": decision_ms,
            "rss_mb": process.memory_info().rss / 1024 / 1024,
            "tool_result": json.dumps(tool_result, default=str),
        })

    accuracy = (
        sum(r["correct"] for r in rows)
        / len(rows)
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    path = (
        f"results/"
        f"fake_benchmark_{timestamp}.csv"
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("FAKE BENCHMARK")
    print("-" * 50)
    print(
        "Accuracy:",
        f"{accuracy * 100:.2f}%"
    )
    print(
        "Average confidence:",
        sum(
            r["confidence"]
            for r in rows
        ) / len(rows)
    )
    print(
        "Average latency:",
        sum(
            r["decision_latency_ms"]
            for r in rows
        ) / len(rows),
        "ms"
    )
    print(
        "Peak RSS:",
        max(
            r["rss_mb"]
            for r in rows
        ),
        "MB"
    )
    print("CSV:", path)


if __name__ == "__main__":
    main()
