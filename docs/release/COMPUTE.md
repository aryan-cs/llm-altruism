# Compute and Serving Environment

## Current hosted execution

The definitive Part 0, Part 1, and Part 2 panels are served through authenticated
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

- Part 0: 22 exact routes are scheduled at 48 roots × 3 response-language
  instructions (3,168 subject responses). Two planning-roster routes are
  excluded without substitution. Fixed-judge batches contain at most eight
  visible responses. The source manifest remains incomplete.
- Part 1: 75 exact routes are scheduled on the same balanced 384-root bank
  (28,800 direct choices). Six additional frozen registry targets are excluded
  without substitution. The source manifest remains incomplete.
- Part 2: 19 exact routes completed 12 independent trajectories × 12 steps × up
  to 5 active participants per step, for 228 trajectories and 13,495 scheduled
  agent-days. The complete manifest records zero transport or identity failures
  and four invalid actions across three trajectories; 225 trajectories contain
  no invalid action and support environmental estimates.

Final compute totals must be calculated from the executed rows and sanitized
availability bindings. A target-bound operationally unavailable system is
counted in frozen coverage but contributes no scored units; scheduled maxima
must not be reported as completed calls.

The definitive Part 2 aggregate contains four invalid actions among 13,495
scheduled agent-days. Their zero simulator effect permits deterministic
continuation and they count as nonrestraint; affected trajectories are excluded
from environmental estimates.

Retries are transport-only and bounded. Invalid structured output is retained
and is not regenerated. Durable reservation and append-only journal records
support exact resume after interruption without treating a duplicate response
as a new independent observation.

## Planning versus execution

The planning-roster JSON retains settings of 48 Part 0 roots per condition and
12 Part 2 trajectories. The definitive manifests implement those settings for
their exact selected subsets. Completed manifests and retained-unit counts,
rather than scheduled maxima, are authoritative for compute accounting.

## Historical local environment

Earlier pilot runs used local Ollama models in an environment that reported an
NVIDIA H200 NVL, a 32 GiB cgroup memory limit, no swap, and a 10-CPU quota.
Those values provide historical context only. The April sidecars do not fully
capture immutable model digests or all decoding settings, and the local pilot
is not the compute environment for the current hosted panels.
