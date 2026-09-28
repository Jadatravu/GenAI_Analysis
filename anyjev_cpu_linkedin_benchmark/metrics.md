# Metrics guide

## Accuracy

    accuracy = correct / total

Example:

    34 / 36 = 94.44%

## Confidence

Record both:

- average confidence overall
- average confidence for correct decisions
- average confidence for incorrect decisions

If incorrect predictions have high confidence, investigate calibration.

## Latency

Report:

- mean
- median
- P95

Median is useful for typical performance.
P95 shows the slower tail.

## Tool latency

Tool latency is measured separately from AnyJev decision latency.

This matters because:

    end_to_end = decision + tool

A slow database/API can dominate the end-to-end number.

## Memory

RSS is process resident memory.

Peak RSS during the benchmark is more useful than only reporting the
machine's total RAM.

## CPU

CPU measurements are sampled and can vary by OS and workload. Treat them
as approximate rather than a laboratory-grade power measurement.

## Reproducibility

For a stronger post, publish:

- hardware
- OS
- Python version
- model
- model version/revision if pinned
- AnyJev level
- dataset
- number of requests
- repetitions
- whether warm-up was excluded
- benchmark code

## What NOT to claim

Do not write:

"AnyJev is 3x faster than LLM agents"

unless you have actually benchmarked a baseline under identical conditions.

Instead:

"In this local benchmark, AnyJev L0 produced a median decision latency of X ms."

Then separately report the baseline if you run it.
