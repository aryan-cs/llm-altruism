# Isolated Confirmatory Campaign

`experiments/confirmatory_campaign.py` is the strict orchestration layer for
the final three-part campaign. It does not import or delegate planning to the
legacy `experiments/campaign.py`. The native Part 0 and Part 1 confirmatory
runners remain the sole producers of their results; Part 2 continues to use its
production entrypoint and native artifact verifier.

## Mandatory frozen inputs

Planning is fail-closed and does not make provider calls. Before returning a
plan it requires all of the following:

1. A unique ordered list of registry cohorts. With no `--cohort` flags the
   exact union is `current_sota` followed by `historical`. Duplicate target IDs
   or exact routes across cohorts are errors, not silently deduplicated.
2. Every subject target and the selected extractor and Part 0 judge must be an
   exact `provider=inference_hub` registry route with `verification_status`
   `verified`. The registry's complete verification bundle, routing-roster
   hash, and freshness policy must agree across cohorts. When the initial
   variance-pilot freeze is created, every route is passed through
   `require_fresh_route_verification`; stale evidence stops that initial plan.
3. An all-target evidence bundle produced by
   `experiments.misc.inference_hub_discovery verify-cohorts`, supplied with its
   exact file SHA-256. The campaign validates the bundle's internal hash,
   registry version/hash, routing-roster hash, cohort order and membership,
   target order/count, exact routes, endpoint, returned model identities,
   structured-output smoke controls, and equality with every registry target's
   retained verification evidence.
4. A human-approved Part 0 registry and Part 1 384-root bank, each supplied with
   an exact file SHA-256. Their production loaders are run during initial
   planning, manifest creation, and resume validation; there is no draft or
   approval bypass.
5. A clean worktree and exact 40-character commit. The manifest freezes the
   interpreter, all campaign/runner/grading/provider sources, prompt assets,
   model registry, analysis verifier, `pyproject.toml`, and `uv.lock`. The
   dependency lock has a separate digest. Resume refuses any drift.

The extractor and judge target IDs must themselves be members of the requested
cohort union, so the same all-target endpoint evidence covers every model role.

## Exact job matrix

The initial variance-pilot plan contains, for every subject target, one Part 0
sacrificial smoke and production job, one Part 1 sacrificial smoke and
production job, one production-shaped Part 2 smoke, and eight Part 2 variance
pilot jobs. The linked baseline-production continuation contains only its Part
2 smoke and the selected common number of Part 2 baseline jobs. It never calls
Part 0 or Part 1 a second time.

Part 0 and Part 1 use deterministic private output directories below their
respective `data/private/*_confirmatory` roots. Their fresh and resume argv
vectors are both frozen. The Part 2 smoke uses four agents, three days, and a
deliberately small reserve, exercising multi-day state feedback and day-specific
seeds while permitting collapse and attrition when agents overuse. Part 2 uses
the native timestamped artifact directory,
but each job freezes one exact subject target, generation seed, environment
seed, configuration, extractor route, and grading protocol. The executor takes
a metadata snapshot and accepts exactly one changed Part 2 artifact after a
successful subprocess. It then runs the native strict artifact verifier.

All subprocesses receive an argv list with `shell=False`; commands are never
rendered into a shell string. Per-job logs and the campaign manifest are private
and durable. The manifest has an immutable plan hash plus a hash over its live
status, attempts, and artifacts, and every update is an atomic fsync-backed
replacement.

## Part 2 variance pilot

`--part2-stage variance-pilot` is the default. It schedules exactly eight
common generation/environment seed pairs for every target. These seed pairs
are identical across the full target union; no model gets an extra or missing
replicate. For the default 30-target union the complete matrix is:

| Stage | Jobs |
| --- | ---: |
| Part 0 smoke | 30 |
| Part 1 smoke | 30 |
| Part 2 smoke | 30 |
| Part 0 production | 30 |
| Part 1 production | 30 |
| Part 2 variance pilot | 240 |
| **Total** | **390** |

This mode refuses a variance-selection artifact because the selection must be
made only after the identity-masked pilot workflow is complete.

## Part 2 baseline production gate

`--part2-stage baseline-production` requires the exact completed complete-union
pilot manifest and the sealed output from
`analysis.part2_confirmatory select-variance`, each with its exact file
SHA-256. The selector output has this exact outer schema:

```json
{
  "schema_version": 1,
  "artifact_type": "part2_identity_masked_variance_selection",
  "private_input_sha256": "<64 lowercase hex>",
  "pilot_campaign_manifest_sha256": "<exact pilot manifest file SHA-256>",
  "selection": {
    "schema_version": 1,
    "selection_rule": "smallest_n_with_t95_half_width_at_most_0.05_capped_20_40",
    "identity_masked": true,
    "selected_common_run_count": 20,
    "...": "the remaining exact native selector fields"
  },
  "artifact_sha256": "<native canonical payload SHA-256>"
}
```

`selected_common_run_count` must be an integer from 20 through 40. It is the
only source of the scientific Part 2 replicate count; there is no command-line
sample-size override. The campaign recomputes the selector seal, requires its
pilot lineage to match the supplied completed campaign, and revalidates every
pilot artifact before planning or resume. At `n=20`, the default 30-target
continuation contains 30 Part 2 smoke jobs and 600 baseline jobs, 630 total.

The initial pilot creation is the route-freshness boundary for this two-stage
chain. The baseline continuation may begin after the 168-hour wall-clock
window, but only with the pilot's exact cohort, registry, routing roster, role
routes, and endpoint-evidence bytes. Elapsed time cannot invalidate an
immutable campaign, while every route or panel substitution still fails.

After the pilot manifest is complete, derive and select the common run count
from native artifacts only:

```bash
python -m analysis.part2_confirmatory build-variance-input \
  --pilot-campaign /absolute/private/pilot/manifest.json \
  --pilot-campaign-sha256 <exact-file-sha256> \
  --output /absolute/private/pilot/identity-masked-variance-input.json

python -m analysis.part2_confirmatory select-variance \
  --input /absolute/private/pilot/identity-masked-variance-input.json \
  --pilot-campaign /absolute/private/pilot/manifest.json \
  --pilot-campaign-sha256 <exact-file-sha256> \
  --output /absolute/private/pilot/variance-selection.json
```

The baseline campaign then receives the same pilot path/hash plus
`--variance-selection` and `--variance-selection-sha256`. It also receives the
same cohort, role, registry, approved-input, and endpoint-evidence arguments as
the pilot; any mismatch is rejected.

## Smoke gates and exclusions

Every scientific job stores the ID of its matching same-target, same-experiment
smoke. A frozen SHA-256 scheduling seed block-randomizes target order and part
order within target; within each target-part block, the matching smoke always
precedes its scientific jobs. The executor checks the referenced status
immediately before a scientific job. A missing, failed, or incomplete smoke
sets the scientific job to `blocked_smoke`; failure for one target does not
authorize that target's production run and does not substitute another
target's smoke.

Part 0 additionally receives that exact same-target smoke directory through
its native `--completed-smoke-dir` production gate. The runner revalidates the
smoke plan, routes, seeds, full result set, attempt chain, metadata, and
exclusion marker before freezing or resuming the production plan.
Part 1 applies the analogous native `--completed-smoke-directory` gate.

The native Part 0/1 smoke runners write and validate their own hash-bound
analysis-exclusion markers. Part 2 smoke CSVs receive the exact exclusion
marker understood by `analysis.validation`, binding the CSV SHA-256 and reason
`sacrificial_campaign_smoke`. Production Part 0/1 artifacts refuse smoke
markers. Completed jobs are reverified on campaign resume rather than trusted
from manifest status alone.

## Dry run

`--dry-run` performs the complete planning validation: cohort union, route
freshness, all-target evidence, approved registries, optional variance gate,
clean Git/source/dependency freeze, commands, job counts, and hashes. It makes
no campaign directory, manifest, log, result file, or provider request. The
validated summary is printed to stdout.

Example pilot dry run:

```bash
python -m experiments.confirmatory_campaign \
  --campaign-id confirmatory-pilot-v1 \
  --extractor-target-id google.google-gemma-3-27b-it \
  --judge-target-id google.google-gemma-3-27b-it \
  --part0-registry /absolute/private/part0-registry.json \
  --part0-registry-sha256 <sha256> \
  --part1-bank /absolute/private/part1-bank.json \
  --part1-bank-sha256 <sha256> \
  --endpoint-evidence /absolute/private/all-target-evidence.json \
  --endpoint-evidence-sha256 <sha256> \
  --part2-stage variance-pilot \
  --dry-run
```

Remove `--dry-run` to create and execute the fresh private manifest. Resume an
existing campaign using only:

```bash
python -m experiments.confirmatory_campaign \
  --campaign-id confirmatory-pilot-v1 \
  --resume
```

Resume revalidates all pinned files, current cohort membership, verification
identity, routing roster, commit, source/dependency bundles, immutable plan,
live manifest hash, job states, and every already completed artifact before
continuing.

## Target shards

Repeatable `--target-id` flags can create disjoint execution shards while the
endpoint evidence still attests the full frozen cohort union. A single shard is
never a publication-complete panel: the final confirmatory data lock requires
`target_selection.mode=complete_union` until an exact master-shard merge
validator is implemented. One target's default pilot shard has 13 jobs.
