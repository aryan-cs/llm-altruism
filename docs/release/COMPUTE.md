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
  conditions, followed by fixed-judge batches of at most eight visible
  responses for included systems.
- Part 1: the execution roster assigns 75 systems × 96 roots, two slower
  systems × 12 roots, and one system × 384 roots. Three additional frozen
  registry targets were unavailable before execution and are not part of this
  78-target execution roster.
- Part 2: 24 systems were targeted at 8 independent trajectories × 12 steps ×
  up to 5 active participants per step.

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
condition and 12 Part 2 trajectories. The completed deadline artifacts used
explicit CLI limits of 24 and eight. Runtime manifests and final aggregate
coverage are authoritative for compute accounting.

## Historical local environment

Earlier pilot runs used local Ollama models in an environment that reported an
NVIDIA H200 NVL, a 32 GiB cgroup memory limit, no swap, and a 10-CPU quota.
Those values provide historical context only. The April sidecars do not fully
capture immutable model digests or all decoding settings, and the local pilot
is not the compute environment for the current hosted panels.
