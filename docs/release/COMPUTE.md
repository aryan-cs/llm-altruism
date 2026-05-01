# Compute Environment

This document records the hardware and container limits used for the current paper-facing local runs. These limits explain why the evaluated cohort emphasizes open models that can be run through local Ollama under the available memory budget.

## Local Runtime

- Backend: local Ollama.
- Visible GPU: NVIDIA H200 NVL with 143,771 MiB of GPU memory.
- cgroup memory limit: 34,359,738,368 bytes, or 32 GiB.
- cgroup swap limit: 0 bytes.
- cgroup CPU quota: `1000000 100000`, equivalent to 10 CPUs.

These values are container limits, not host-machine totals. The cgroup RAM and no-swap constraint are the binding limits for local model selection. Larger local models, larger batches, and broader closed-model comparisons require either additional host memory, a different runtime configuration, API access, or a separate compute environment.

## Inspection Commands

The current limits can be rechecked with:

```bash
cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/memory.swap.max
cat /sys/fs/cgroup/cpu.max
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

## Metadata Limitation

Current metadata sidecars record provider/model identifiers, command context, prompt hashes where implemented, git commit when available, and run status. Some legacy runs predate complete hardware capture. For that reason, this document is the paper-facing compute record for the current artifact, while older per-run sidecars should be treated as partial hardware metadata.
