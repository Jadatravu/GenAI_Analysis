# LinkedIn Post Template — AnyJev Agent Tool Calling

Replace every `<...>` with values from your generated benchmark Markdown/JSON.

---

I experimented with **Nokia Applied Research's AnyJev** as a typed decision
layer for agent tool calling — running locally on a **CPU-only machine with
12 GB RAM**.

The objective was simple:

Can a small local model make a typed tool-selection decision, while keeping
the actual tool execution outside the model?

### Architecture

User request
↓
AnyJev typed decision
↓
Tool + confidence
↓
Python/MCP tool execution
↓
Tool result

### Test setup

- Model: `<MODEL>`
- AnyJev level: `<LEVEL>`
- Device: CPU
- RAM: 12 GB
- Dataset: `<N>` labeled requests
- Repetitions: `<N>`

### Results from my run

- Tool-routing accuracy: **<ACCURACY>%**
- Median decision latency: **<MEDIAN> ms**
- P95 decision latency: **<P95> ms**
- Average confidence: **<CONFIDENCE>**
- Peak process memory: **<RAM> MB**

I also captured individual request-level results rather than reporting only
a single aggregate number.

The key architectural idea I found interesting is the separation between:

**Decision ≠ Execution**

AnyJev determines *which* tool should be selected. The application remains
responsible for executing the tool and enforcing authorization.

### Next experiment

I want to compare this approach with conventional LLM tool calling under the
same:

- hardware
- prompts
- dataset
- number of requests

and measure:

- routing accuracy
- latency
- token usage
- RAM/CPU
- fallback rate
- wrong-tool rate

This is an early local experiment, so the numbers should not be generalized
to other hardware, models or workloads.

#GenAI #AI #AgenticAI #LLM #AnyJev #ToolCalling #AIEngineering #MLOps
