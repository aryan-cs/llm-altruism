# Part 1 semantic-invalid repair

The module experiments.misc.inference_hub_part1_semantic_invalid_repair
creates a separate, auditable repair layer for malformed responses in the
completed definitive Part 1 campaign. It never edits the source journals,
replaces their labels, or changes the primary all-scheduled denominator.

## Fail-closed source contract

The runner accepts only a complete, self-hash-valid
inference_hub_part1_large_n_exploratory_panel manifest. Before creating an
output directory it:

- verifies the exact current provider-safe-v2 and main-accelerated launcher
  file digests;
- recomputes the main accelerated limiter policy hash and requires provider
  concurrency 2, provider request rate 1.5/second, global concurrency 12, and
  global request rate 8/second;
- validates every source raw journal private path, mode, complete checkpoint,
  file hash, and SHA-256 record chain;
- reconstructs the frozen Part 1 schedule and every eligible original request;
  and
- requires exact prompt hash and text, request hash, target, upstream provider,
  model, and route identity.

The source judge must remain reserved and non-dispatchable. The repair runner
contains no judge path.

## Eligibility and rounds

Only retained source units with format_valid set to false are eligible. This
includes malformed outputs whose original finish reason is content_filter or
length; valid primary records are never re-dispatched.

Repairs run in bounded, periodic semantic rounds. Each request uses the exact
original messages, prompt, route, model, and generation controls. If and only
if the compatibility-bound route supports seed, the runner derives a new,
deterministic seed from the original record hash and round number. Routes
without seed support continue to omit it.

The runner uses the source-bound main accelerated client. Its shared limiter
enforces the exact policy above. The --max-workers option controls only the
local task pool; it cannot bypass the provider-aware limiter.

## Crash safety and private evidence

Each logical round is assigned a deterministic attempt ID. The private,
mode-0600 SHA-256 journals record:

1. a durable reservation before dispatch;
2. the full raw provider response;
3. a completion after the raw response is fsynced.

On --resume, a raw response whose completion was interrupted is recovered
without redispatch. A reservation with no retained response is marked
indeterminate and is never dispatched again in the same round. A later round
uses its distinct deterministic attempt and, when supported, distinct seed.

Any response-model identity mismatch is retained privately for diagnosis, then
fails the run without emitting sanitized repair outcomes.

## Outputs

The private manifest, attempt ledger, and raw repair-response journal contain
the audit trail.

The sanitized repair_outcomes.json contains no raw response text. Each row is
keyed to the original target, trial, record hash, request hash, prompt hash, and
finish reason. It reports rounds reserved, rounds with retained responses, a
separate repair status, successful round, parsed X/Y action when available, and
the repaired response-text hash.

Every row explicitly records that the primary record and denominator were
unchanged. The manifest reports the original scheduled count, original invalid
count, separate repaired count, and separate unrepaired count. These data do
not authorize confirmatory or paper promotion.

For a completed source campaign:

    uv run python -m \
      experiments.misc.inference_hub_part1_semantic_invalid_repair \
      --source-manifest \
      data/private/inference_hub/provider-safe-v2-part1-large-n384-main75-v1/private/manifest.json \
      --output-dir \
      data/private/inference_hub/provider-safe-v2-part1-semantic-repair-v1 \
      --max-rounds 8 \
      --max-workers 8 \
      --round-interval-seconds 30

Resume only the same output and immutable source by adding --resume.

The runner fails if the source campaign is still running. A complete source
with zero invalids is a keyless no-op: it creates sealed empty outcomes without
initializing a network client.

## Verification

    UV_CACHE_DIR=/tmp/llm-altruism-uv-cache \
      uv run pytest -q \
      tests/test_inference_hub_part1_semantic_invalid_repair.py

The test suite includes the full 75-model by 384-trial source shape,
content_filter and length repairs, deterministic seeded and unseeded routes,
raw-fsync crash recovery without redispatch, a zero-invalid keyless no-op, and
failures for incomplete input, wrong policy, prompt drift, and response
identity mismatch.
