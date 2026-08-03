# Part 1 Role-Conditioned Calibration v1

This is a separate exploratory calibration of the restored Part 1 design. It
does not modify the historical source-bound Part 1 runners, enlarge the matched
cross-axis panel, pool the existing 12-, 96-, and 384-root scopes, or authorize
a confirmatory or paper-facing claim by itself.

## Frozen design

The immutable configuration is
`experiments/part1/role_calibration_panel_v1.json` (SHA-256
`14012c3db18890dd970e826dcf59498eeba6b411fe0b5f04d2df52cc6c6e92b3`). The
runner refuses a substituted path or changed file bytes.

The schedule calls `build_role_schedule` on the deterministic structural draft
bank. Its frozen 96-root subset contains eight roots from every game-by-domain
cell. For each model and root, the schedule retains four counterbalanced draws
for each of three frames:

- `advice`;
- `observer_evaluation`;
- `prediction`.

Frames and models are separate estimands. No frame pooling, model pooling, or
cross-model ranking is permitted. The primary descriptive quantity first
computes the welfare-preserving rate among format-valid draws within each
scenario root, then averages those root rates within one model and one frame.
Format validity and coverage are reported separately.

The prespecified developer/capability-stratified sentinel panel is:

| Developer | Frozen study target | Sentinel stratum |
| --- | --- | --- |
| OpenAI | `openai/gpt-5.6-sol` | frontier reasoning |
| Anthropic | `anthropic/claude-opus-5` | frontier general |
| Google | `google/gemini-3.1-pro-preview` | frontier general |
| DeepSeek | `deepseek-ai/deepseek-v4-pro` | frontier reasoning |
| Qwen | `qwen/qwen3.6-35b-a3b` | midscale mixture of experts |
| NVIDIA | `nvidia/nemotron-3-super-v3` | large open reasoning |

These labels are sampling strata, not empirical quality judgments. The fixed
judge target `judge.nvidia-evals-nemotron-3-30b-a3b` is compatibility-checked
for disjoint identity and recorded as reserved, but the runner cannot dispatch
it. Subject responses are scored mechanically from the frozen X/Y mapping.

The complete schedule is 96 roots × 3 frames × 4 counterbalances × 6 models =
6,912 subject requests. There is no outcome-adaptive stopping or CLI option to
change the subjects, roots, frames, or blocks.

## Provider and evidence safeguards

`experiments.misc.inference_hub_part1_role_calibration_v1` accepts only routes
that passed the authenticated compatibility artifact. It verifies returned
model identity and obtains its client only from the versioned provider-safe v2
launcher. The launcher round-robins upstream providers and uses one shared,
cross-process limiter with exactly one in-flight request per provider. The
runner records unsupported controls and retries only eligible transport
failures.

Before every provider call, the runner fsyncs a credential-free request hash
and exact unit identity to a private append-only SHA-256 chain. Raw responses,
prompts, reasoning fields, and request routes are retained under mode-`0700`
directories and mode-`0600` files. A response is fsynced before its terminal
success ledger record. Resume validates every manifest checkpoint and chain,
recovers that narrow raw-first crash window without redispatch, and refuses
truncation, mutation, changed source/config bytes, or a changed route contract.

The `sanitized/` directory contains only:

- `summary.json`: per-model/per-frame root-weighted estimates and coverage; and
- `root_summaries.jsonl`: text-free per-model/per-frame/per-root counts.

Neither file includes prompts, responses, reasoning, exact request routes, or
credentials. Both retain the immutable config/schedule bindings and raw journal
tail hashes. Raw run directories remain excluded from the anonymous supplement.

## Execution

After creating current registry and compatibility evidence, a complete run is:

```bash
uv run python -m experiments.misc.inference_hub_part1_role_calibration_v1 \
  --registry agents/agent_config.registry.json \
  --compatibility /absolute/private/path/provider-compatibility.json \
  --output-dir data/private/inference_hub/part1-role-calibration-v1 \
  --max-workers 12 \
  --max-workers-per-provider 1
```

Use the identical command with `--resume` after an interruption. A completed
resume performs no subject redispatch. This repository includes the design and
runner for reproducibility; an execution becomes reportable only after its
complete private evidence and sanitized outputs pass the release audit.

## Tests

`tests/test_inference_hub_part1_role_calibration_v1.py` checks the immutable
configuration, complete 1,152-trial per-model schedule, exact 96-root stratified
coverage, all 6,912 fake-provider calls, concurrent six-subject execution,
judge non-dispatch, private permissions, every attempt-ledger chain link,
root-aware sanitized outputs, completed resume without redispatch, and tamper
refusal.
