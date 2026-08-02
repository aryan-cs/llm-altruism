# Part 1 Confirmatory Bank Implementation

The confirmatory implementation is isolated in
`experiments/part1/confirmatory_design.py`; it does not reuse or alter the
exploratory Part 1 runner or its crime-framed scenarios. The module defines the
scenario-root schema, deterministic rendering, block schedule, response parser,
and production gates needed to implement the frozen Part 1 design.

## Design represented in code

`build_draft_bank()` deterministically creates 384 self-directed scenario
roots: 32 roots in every cell of two games by six neutral domains. The six
domains are shared workspaces, scientific facilities, civic infrastructure,
education resources, healthcare operations, and digital services. A root has a
unique `root_id`, a content-derived `semantic_cluster_id`, a cell-local design
index, semantic facets, objective action definitions, all four objective payoff
and consequence outcomes, an explicit welfare mapping, an ordinal payoff
ordering, a content hash, mapping-review states, and three human-review slots.

The internal action IDs describe the objective mapping and never appear in a
model prompt. Model-visible choices are only `X` and `Y`. Rendering applies one
of these four counterbalances after the root has been defined:

| Counterbalance | Welfare-preserving label | Its displayed position |
| --- | --- | --- |
| `CB_X_FIRST` | X | first |
| `CB_Y_FIRST` | Y | first |
| `CB_X_SECOND` | X | second |
| `CB_Y_SECOND` | Y | second |

Thus payoff semantics and operational consequences cannot be accidentally
changed by label or option order. The renderer includes the objective outcome
table, labels both actions neutrally, and asks for an exact final `X` or `Y`
line.

`build_primary_schedule()` creates 384 planned calls per model, exactly one for
each approved root. The sole primary block is block 0, and the counterbalance
for cell-local root index `i` is `i mod 4`. Within every game-domain cell, each
of the four counterbalances therefore has exactly eight roots.
Generation settings and independently derived SHA-256 seeds are stored per
trial. Requested provider and exact model route are explicit placeholders until
selected, while returned model, request ID, and finish reason remain empty
until a real response exists. Every prompt and referenced root is hash-bound.

## Secondary roles and demand-cue control

The design module retains utilities for constructing advice,
observer-evaluation, prediction, and demand-cue-control schedules. These are
not part of the confirmatory production runner. Confirmatory production contains
only the 384 `self_direct` roots and records a secondary trial count of zero.

The historical instruction about a rational participant maximizing immediate
personal payoff is not included in those frames. It has a separate builder,
frame ID, schedule, and `positive_demand_cue_control=True` field. Validation
rejects either the legacy wording without the positive-control label or that
label without the legacy wording.

## Fail-closed validation and parsing

The bank validator defaults to production mode. It refuses incorrect size,
duplicate IDs or semantic clusters, duplicate structured semantic facets,
exact or near-copy semantic text, excessive wording-family reuse, cells other
than the frozen 2-by-6 design, unbalanced cell indices, crime or morally loaded
confounds, payoff reversals, incomplete outcome matrices, inconsistent welfare
mappings, stale content hashes, unknown review states, deterministic-draft
authorship, incomplete mapping review, and anything short of three distinct and
unanimous human approvals bound to the current content hash.

The schedule validators independently reconstruct every prompt, prompt hash,
root binding, seed, render/counterbalance pairing, and Latin-square assignment.
Production validation also refuses unresolved requested model provenance. A
planned trial must keep returned provenance empty; the execution record fills
those fields only after the request.

`parse_exact_final_token()` requires an explicitly non-truncated completion, a
recognized normal finish reason, and a final line equal byte-for-byte to `X` or
`Y` after outer response whitespace is removed. Empty, punctuated, fenced,
otherwise malformed, unknown-finish, and truncated responses are `INVALID`.
`retry_decision()` permits retries only for transport failures. A malformed or
truncated semantic response is retained without regeneration.

`freeze_primary_plan()` is the only constructor for an executable frozen plan.
It reruns the production bank and schedule validators, then binds both complete
manifests with SHA-256. `iter_execution_trials()` rechecks the manifest hashes
before yielding a trial.

## Current draft limitation

No human content review was available during implementation. Accordingly, all
three reviewer slots and both mapping-review states are `NOT_REVIEWED`, and the
authorship method is `deterministic_structural_draft`. The generator creates
materially different structured roots from unique game, domain, actor,
operational period, resource, capacity, and decision-mechanism combinations;
the validators detect duplicated facets, copied or near-copied semantic text,
and overused wording families. This is useful for end-to-end design and
adversarial testing, but automatic diversity screens cannot establish semantic
independence or content validity.

The deterministic bank therefore cannot be presented as the independently
authored, human-approved confirmatory stimulus bank. Both production scheduling
and freezing deliberately refuse it. Replacing it requires genuinely
independently authored roots followed by three independent reviewers who
unanimously approve moral neutrality, welfare mapping, payoff ordering, and
material distinctness against the exact stored content hash. Approval fields
must record the real reviewer identity and timestamp; they must never be filled
synthetically to make a test pass.

The adversarial coverage is in
`tests/test_part1_confirmatory_design.py`. It exercises the complete 384-root
draft and both complete schedules, counterbalance balance, prompt/seed/hash
tampering, semantic duplication and template reuse, payoff reversal, prohibited
content, review refusal, role/control separation, exact parsing, and the
transport-only retry rule.

## Production execution runner

`experiments/part1/confirmatory_runner.py` is the only confirmatory execution
path. It does not import the exploratory `part_1.py` runner. Its input is a
UTF-8 JSON object with exactly `schema_version` and `roots`; `roots` contains
the 384 serialized `ScenarioRoot` records. The operator must supply the exact
SHA-256 of that file. The loader reconstructs every nested dataclass and reruns
`validate_bank(..., production=True)`. Consequently a deterministic draft,
missing or duplicated reviewer identity, non-unanimous decision, unchecked
review dimension, stale reviewed-content hash, missing timestamp, incomplete
mapping review, or any content/hash/design violation stops before a route is
called. The runner never manufactures reviewer fields or offers an override.

Freezing resolves the subject to an exact model-registry entry whose
verification status and evidence are current. It requires a clean,
40-character Git commit and binds the runner, design, shared execution code,
provider adapter, model registry, prompt assets, `pyproject.toml`, and `uv.lock`
by SHA-256. The plan contains exactly 384 self-directed trials and no neutral
role trials. It reconstructs and validates the complete primary schedule on
every fresh execution and resume. Each subject call has its own deterministically
derived generation seed.

For each trial the subject generates normally with its frozen per-trial
settings. The raw provider-visible assistant response is passed directly to the
existing `parse_exact_final_token()` function with the subject's explicit
finish and truncation evidence. Hidden reasoning and raw provider bodies are
never parser inputs. Exact terminal `X` and `Y` are scored; every other
semantic terminal form is durably retained as `INVALID`. Truncation, empty
content, and malformed terminal choices are retained as `INVALID`. None
triggers a semantic retry. Only classified transport/gateway failures may retry, at
most three times, on the same frozen route with the identical per-call seed.

Plans, visible responses, results, attempts, and metadata are restricted to
`data/private/part1_confirmatory/`. Directories are mode `0700` and files mode
`0600`; fresh runs refuse overwrites. Result and attempt JSONL files are
independently hash-chained. Prompts in the attempt log are redacted to their
SHA-256; raw responses, parsed provider bodies, and hidden reasoning are not
stored there. Exact resume verifies permissions, the persisted plan, metadata
hash, result and attempt hashes, schedule-prefix identity, model identities,
request hashes, response-audit hashes, parser output, and stage coverage.
Crashes after a semantic response but before its result commit fail closed
rather than replaying that response.

The CLI suitable for campaign orchestration is:

```bash
python -m experiments.part1.confirmatory_runner \
  --mode production \
  --bank /absolute/private/path/part1-bank.json \
  --bank-sha256 <64-lowercase-hex> \
  --subject-provider inference_hub \
  --subject-model <exact-verified-route> \
  --output-directory data/private/part1_confirmatory/<model-run>
```

Add `--resume` only when all four existing private artifacts are present and
unchanged. The focused runner coverage is in
`tests/test_part1_confirmatory_runner.py`; it exercises the full 384-row
schedule plus direct subject-response parsing, transport-only retries, exact
identity, malformed terminal answers, truncation,
private permissions, hash chains, completed resume, and tamper refusal.

## Sacrificial full-path smoke

Before a production campaign, use `--mode sacrificial-smoke` with the same
approved bank, exact routes, seed bases, and private-output requirements. This
is a separately frozen execution mode, not a production plan truncated by the
operator. It reconstructs the full 384-row production schedule and then
selects 12 trials without inspecting model output: exactly one self-directed
root from every game-domain cell. A fixed assignment covers each of the four
X/Y-label and displayed-position counterbalances exactly three times.
Purpose-bound SHA-256 ranking deterministically chooses the root within each
eligible cell.

Every smoke trial executes the real subject request and exact X/Y terminal
parser. Thus a successful smoke requires exactly 12 provider calls. Trial rows are explicitly
bound to `execution_mode=sacrificial_smoke` and
`analysis_eligible=false`. The plan records the deterministic selection method,
exact expected count and strata, complete call path, and a fail-closed analysis
eligibility object. Rewriting that mode as production, removing or adding a
row, changing a counterbalance, or changing either eligibility field fails
exact plan reconstruction even if an attacker recomputes the outer plan hash.

After all 12 results and final metadata are durably committed, the runner
writes `part1_confirmatory_analysis_exclude.json`. This mode-`0600` marker binds
the plan, selected schedule, approved-bank bytes, result artifact, result
count, and metadata by SHA-256 and includes its own canonical payload hash. A
completed smoke cannot resume if the marker is missing, stale, malformed, or
too permissive. Conversely, a production run refuses any smoke marker in its
output directory. These bidirectional checks prevent smoke results from being
accepted by the production runner even if filenames are copied or plan fields
are edited.

Example:

```bash
python -m experiments.part1.confirmatory_runner \
  --mode sacrificial-smoke \
  --bank /absolute/private/path/part1-bank.json \
  --bank-sha256 <64-lowercase-hex> \
  --subject-provider inference_hub \
  --subject-model <exact-verified-route> \
  --output-directory data/private/part1_confirmatory/<model-smoke>
```

The runner tests execute the entire 12-trial smoke lifecycle with 12 provider
responses and verify balance, call counts, parser output, private permissions,
marker hashes, zero-call completed resume, marker tampering, missing markers,
plan and schedule tampering, attempted smoke-to-production relabeling, and the
production-side marker prohibition.
