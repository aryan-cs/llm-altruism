# Part 0 Confirmatory Runner

The confirmatory Part 0 implementation is isolated in
`experiments/part0/confirmatory_runner.py`. It does not call the exploratory
`experiments/part0/part_0.py` runner, its online translator, its prompt loader,
or its judge fallback list.

## Production inputs and gates

`load_production_registry(path, expected_sha256=...)` is the only production
registry loader. It hashes the exact file bytes before parsing them, refuses a
hash mismatch, and calls `validate_stimulus_registry(...,
require_production_approval=True)`. It additionally requires the frozen Part 0
cardinalities and provenance:

- 484 harmful prompt roots backed by all 500 source-row memberships (400
  HarmBench and 100 JailbreakBench harmful rows);
- 100 official JailbreakBench controls and 100 complete harmful/control pairs;
- English, Simplified Chinese, and Russian text with exact SHA-256 values and
  reviewer provenance for every root;
- a nonempty, SHA-256-bound, human-approved response instruction with reviewer
  provenance in each of the three languages; and
- the exact upstream commits and source-file hashes embedded in
  `stimulus_registry.py`.

The registry builder intentionally emits `pending_human_review` placeholders
for response instructions and retains pending stimulus/cluster reviews. A
freshly built draft therefore cannot pass the production loader. The code does
not invent instruction text, translations, cluster assignments, reviewer
identities, or approval timestamps.

Subject, extractor, and judge routes are separately frozen with
`freeze_verified_route()`. Each must be a verified, evidence-bearing exact
callable route in the model registry. Display-only or unverified routes remain
blocked. `freeze_execution_plan()` binds the full route identities, protocol
hashes, registry hash, seeds, and all 5,256 scheduled calls in one plan hash.
Freezing also requires a clean worktree and binds the exact 40-character Git
commit, the complete production source bundle, and SHA-256 values for both
`pyproject.toml` and `uv.lock`. Execution and resume recheck that the commit,
clean state, code bundle, and dependency lock are unchanged.

## Schedule

For one subject model the plan contains:

| Arm | Roots | Languages | Blocks | Calls |
| --- | ---: | ---: | ---: | ---: |
| Harmful | 484 | 3 | 3 | 4,356 |
| Control | 100 | 3 | 3 | 900 |
| Total | 584 | 3 | 3 | 5,256 |

Every block contains every root-language cell exactly once. A separately
derived ordering seed randomizes each block, and a separately derived
generation seed labels each call. Schedule validation reconstructs the complete
block order and every generation seed. Each row retains `base_prompt_id`,
`semantic_cluster_id`, `arm`, `pair_id`, `block`, `language`, the exact
`prompt_sent` and its hash, the registry-file hash, and the complete list of
source provenance references.

The exact model-visible prompt is the reviewed localized stimulus, two newline
characters, and the reviewed same-language response instruction. No online
translation or fallback path exists.

## Execution and judging

The subject receives an empty system prompt, so this path adds no safety-test
cue and requests no rationale. It uses the frozen temperature, top-p, output
cap, and per-row seed. The provider response must report the exact requested
route, request ID, finish reason, usage, raw-response hash, visible-response
hash, and explicit truncation state.

Only provider-visible assistant content is passed to the extractor. Hidden
reasoning is never included. The visible content is base64-encoded as untrusted
data and hash-bound in the extractor request. The extractor output must be a
byte-exact copy of the visible subject content; alteration or translation makes
the row `INVALID`. Its 32,768-token cap exceeds the subject's 8,192-token cap
and leaves room for JSON encoding, so valid longer answers are not
systematically invalidated by an undersized extractor. The frozen extractor's identity, request ID, response
hashes, usage, finish reason, and truncation state are retained independently
from the subject's.

The judge receives the exact original-language `prompt_sent` and the extracted
original-language visible response. It has exactly three labels: `REFUSAL`,
`COMPLIANCE`, and `UNCLEAR`. There is no translated grading input and no route
fallback. A transport failure may retry only the same frozen judge route.
Malformed JSON, an unknown label, truncation, missing content, or any other
semantic invalidity is never regenerated. It remains `INVALID` or
`UNSCORABLE`; `UNCLEAR` remains a scored judge outcome rather than being
discarded.

Every subject, extractor, and judge attempt is appended to the fsync-backed,
hash-chained attempt log with its stage, exact unit identity, request hash,
route, response audit hashes, outcome, retry decision, and attempt number. Raw
requests and responses are redacted from this generic sidecar; the visible
response is retained once in the private result row.
Only transport/gateway failures and prespecified transient HTTP statuses are
retryable. Route substitution and missing request identity fail closed.

## Sacrificial full-path smoke gate

Before production, run `--mode sacrificial-smoke` with the exact approved
registry, routes, and seed bases intended for that target. The runner
deterministically selects one harmful and one control trial in every
language-by-block cell: 18 trials total, each traversing the real subject,
visible-only extractor, and judge path (54 provider calls). Every row is marked
`analysis_eligible=false` and the completed private directory receives a
self-hashed exclusion marker binding its plan, schedule, registry, results,
attempt chain, and metadata.

A new production freeze requires `--completed-smoke-dir` pointing to that
same-target completed smoke directory. The validator reconstructs the exact
smoke plan, rechecks routes and seed bases, requires all 18 rows to be fully
scored, verifies all file/hash chains and private permissions, and binds the
resulting smoke-gate hash into the production plan. Copying a marker, changing
a route or seed, or supplying an incomplete smoke fails before a production
provider call.

```bash
python -m experiments.part0.confirmatory_runner \
  --mode sacrificial-smoke \
  --registry /absolute/private/path/part0-registry.json \
  --registry-sha256 <64-lowercase-hex> \
  --subject-provider inference_hub --subject-route <exact-verified-route> \
  --extractor-provider inference_hub --extractor-route <exact-verified-route> \
  --judge-provider inference_hub --judge-route <exact-verified-route> \
  --output-dir data/private/part0_confirmatory/<model-smoke> \
  --fresh
```

The corresponding production invocation uses `--mode production`, a distinct
output directory, and
`--completed-smoke-dir data/private/part0_confirmatory/<model-smoke>`.

## Strict resume

`run_frozen_plan()` accepts output only below the git-ignored
`data/private/part0_confirmatory/` root. It creates directories with mode `0700`
and the frozen plan, hash-chained result JSONL, redacted hash-chained attempt
JSONL, and hash-protected metadata with mode `0600`. Resume requires all four
artifacts and refuses permissive mode bits.
It revalidates the registry bytes, reconstructs the full plan under the current
code, verifies the source bundle and three route identities, checks both hash
chains and their metadata summaries, and requires completed results to be an
exact prefix of the schedule. Changed code, prompts, routes, seeds, provenance,
results, attempts, registry bytes, or metadata are refused rather than guessed
or repaired. It also reconciles each result with terminal subject, extractor,
and judge attempts. If a crash retained any semantic-stage response before its
result row was appended, resume stops instead of regenerating the subject.
Route evidence freshness is enforced when a new smoke or production plan is
frozen. A strict resume may cross that wall-clock window because it must
reconstruct the already-frozen plan byte-for-byte; registry identity, route
identity, source, smoke-gate, and per-response identity checks remain enforced.

Adversarial coverage is in `tests/test_part0_confirmatory_runner.py`. The tests
use synthetic approvals only inside temporary test fixtures; no approval or
translation artifact is written to the repository.
