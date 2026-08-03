# Compute and Serving Environment

## Current hosted execution

The deadline Part 0, Part 1, and Part 2 panels were served through authenticated
InferenceHub-compatible chat endpoints. Provider-side accelerator type,
quantization, batching, and serving topology are not exposed to this artifact
and must not be inferred from a model label.

The client records the local Python/runtime platform, requested route, returned
model identity, completion ID, finish reason, token usage, supported controls,
and request/response hashes. Private provider payloads and credentials are not
released.

All hosted callers share a cross-process, provider-aware rate limiter. The
frozen policy bounds global and per-provider request starts and concurrency,
uses leases to recover from interrupted processes, and applies cooldowns after
throttling or transient service failures. Part 0 additionally serializes calls
within an upstream provider while allowing bounded progress across providers.
Part 2 bounds both trajectory and participant-level workers. These controls are
operational safeguards, not model hyperparameters.

## Frozen workload and availability accounting

- Part 0: 24 systems were targeted at 24 roots × 3 response-language
  conditions. Sixteen systems are included; eight are operationally
  unavailable. Fixed-judge batches contain at most eight visible responses.
- Part 1: 75 systems are reportable: 73 × n=96, one × n=12, and one × n=384.
  Three execution subjects are operationally unavailable and three additional
  registry targets were unavailable before execution.
- Part 2: 22 systems contribute 8 independent trajectories × 12 steps × up to
  5 active participants per step, for 176 included trajectories. Two systems
  are operationally unavailable.

Final compute totals must be calculated from the included rows and sanitized
availability bindings. A target-bound operationally unavailable system is
counted in frozen coverage but contributes no scored units; scheduled maxima
must not be reported as completed calls.

Retries are transport-only and bounded. Invalid structured output is retained
and is not regenerated. Durable reservation and append-only journal records
support exact resume after interruption without treating a duplicate response
as a new independent observation.

## Planning versus execution

The matched panel JSON retains intended settings of 48 Part 0 roots per
condition and 12 Part 2 trajectories. The deadline artifacts used explicit CLI
limits of 24 and eight. The sealed final aggregate coverage, rather than
scheduled maxima, is authoritative for compute accounting.

## Historical local environment

Earlier pilot runs used local Ollama models in an environment that reported an
NVIDIA H200 NVL, a 32 GiB cgroup memory limit, no swap, and a 10-CPU quota.
Those values provide historical context only. The April sidecars do not fully
capture immutable model digests or all decoding settings, and the local pilot
is not the compute environment for the current hosted panels.
