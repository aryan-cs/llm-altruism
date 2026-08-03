# Prosocial Readiness Bench Model Registry

The current paper uses two explicitly separate coverage sets.

## Matched 24-system panel

The same 24 authenticated systems are scheduled in Parts 0 and 2 and define the
only permitted cross-axis overlap. The frozen identifiers are stored in
`experiments/sota_cross_axis_panel.json`. They span GPT, Claude Haiku, Sonnet,
and Opus, Gemini, Llama, DeepSeek, Qwen, Nemotron, MiniMax, and GLM families.

Every executed target is bound to an exact compatibility-selected route and a
successful identity probe. Unavailable, deprecated, identity-mismatched,
non-chat, embedding, reranking, image, and guard-only routes are not silently
substituted. One NVIDIA Evals Nemotron route is reserved as the fixed Part 0
judge and cannot also be a subject.

## Expanded Part 1 coverage

Part 1 is broader but does not enlarge the matched cross-axis sample. Seventy-
five routes completed the balanced 96-root schedule, while two slower routes
completed balanced 12-root schedules. GLM 5.1 completed a separate 384-root
schedule. These 78 observed routes cover 78 of 81 frozen targets; three remain
recorded as unavailable rather than being replaced. The 12-root, 96-root, and
384-root estimates have different support and are never pooled or ranked as if
they shared one design.

## Evidence status and identity limits

The current hosted artifacts are route-level observations, not independent
samples of vendors or model families. Closely related releases may share
training, post-training, or serving infrastructure. Exact request and response
model identities, upstream provider, supported decoding controls, completion
IDs, timestamps, token usage, retry decisions, and artifact hashes are retained
privately for provenance. Public outputs contain system identifiers and
aggregate metrics, but never authenticated routes, credentials, prompts,
responses, reasoning, or journals.

The April 13-model Ollama cohort is historical pilot provenance only. Its
mutable tags and missing immutable digests prevent byte-for-byte
reconstruction, and its withdrawn Part 0 and Part 2 estimates are not mixed
with the current hosted panels.
