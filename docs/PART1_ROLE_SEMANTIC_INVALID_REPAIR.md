# Part 1 role-calibration semantic-invalid repair

The module
experiments.misc.inference_hub_part1_role_semantic_invalid_repair creates a
separate repair layer for the completed six-sentinel role-calibration
campaign. It never edits the source journals, replaces a primary response, or
changes the original 6 x 96 x 3 x 4 draw denominator.

## Source validation

Before creating output, the runner requires:

- a complete, self-hash-valid
  inference_hub_part1_role_calibration_private_v1 manifest;
- exact current hashes for provider-safe-v2, the exploratory accelerated
  launcher, the role-calibration runner, and the frozen panel configuration;
- the self-consistent accelerated policy with provider concurrency 3,
  provider rate 2 requests/second, global concurrency 12, and global rate
  8 requests/second;
- the immutable six sentinel IDs, 96 roots, three distinct frames, four
  generation blocks, and frozen schedule hash;
- complete, unchanged, private source journal checkpoints and SHA-256 record
  chains; and
- zero source transport or response-model identity failures.

For every eligible source row it reconstructs the original request and
requires exact target, upstream provider, model, route, prompt text and hash,
trial, root, frame, generation block, and request hash.

The source judge must remain reserved and non-dispatchable. The repair runner
has no judge path.

## Repair contract

Only first retained responses with format_valid set to false are eligible.
Valid role-calibration responses are never called again.

Repairs run in bounded periodic rounds using the exploratory accelerated
client. The exact original messages and generation controls are retained. A
new deterministic round seed is supplied only when the subject route supports
seed; otherwise seed remains omitted.

The local worker pool does not replace the provider-aware shared limiter.
Frames and models remain separate throughout the repair output.

## Evidence and resume safety

Private, mode-0600 SHA-256 journals use the following write order:

1. reserve a deterministic logical attempt before dispatch;
2. retain and fsync the complete raw provider response;
3. record completion.

On resume, raw evidence interrupted before its completion record is recovered
without redispatch. A reservation with no raw response is marked
indeterminate and is not dispatched again in that round. Response-model
identity mismatch is retained privately and then blocks sanitized output,
including on a later resume.

The sanitized repair_outcomes.json contains no raw response text. Every row is
keyed to the original target, trial, root, frame, generation block, record
hash, request hash, and prompt hash. It reports the original finish reason,
rounds reserved, retained-response rounds, separate repair status, successful
round, parsed X/Y action, and response-text hash. The original counterbalance
remains bound, so a successful text-free row also reports a separate repaired
welfare-preserving indicator.

A unit that remains malformed after the last round is explicitly retained as
unrepaired_after_bounded_rounds. It is never imputed or dropped. Every output
states that primary records and denominators were unchanged, frames were not
pooled, models were not pooled, the judge was not dispatched, and promotion is
not permitted.

## Run

    uv run python -m \
      experiments.misc.inference_hub_part1_role_semantic_invalid_repair \
      --source-manifest \
      data/private/inference_hub/provider-safe-part1-role-calibration-v1/private/manifest.json \
      --output-dir \
      data/private/inference_hub/provider-safe-part1-role-repair-v1 \
      --max-rounds 3 \
      --max-workers 6 \
      --round-interval-seconds 30

Add --resume only for the exact same output and immutable source.

A source campaign that is still running fails before output creation. A
complete source with zero invalids is a keyless no-op and emits sealed empty
outcomes without initializing a network client.

## Verification

    UV_CACHE_DIR=/tmp/llm-altruism-uv-cache \
      uv run pytest -q \
      tests/test_inference_hub_part1_role_semantic_invalid_repair.py

The tests use the full 6 x 96 x 3 x 4 schedule and cover content_filter,
length, deterministic seeded and unseeded routes, raw-fsync crash recovery,
zero-invalid no-op, incomplete input, prompt drift, identity mismatch, wrong
policy, wrong source binding, and invalid persistence after the final round.
