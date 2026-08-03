# Prosocial Readiness Bench Model Registry

The current paper uses two explicitly separate coverage sets.

## Matched 24-system panel

The same 24 authenticated systems form the frozen Parts 0 and 2 roster and
define the only permitted cross-axis overlap. The frozen identifiers are stored in
`experiments/sota_cross_axis_panel.json`. They span GPT, Claude Haiku, Sonnet,
and Opus, Gemini, Llama, DeepSeek, Qwen, Nemotron, MiniMax, and GLM families.

Every included target is bound to an exact compatibility-selected route and a
successful identity probe. Unavailable, deprecated, identity-mismatched,
non-chat, embedding, reranking, image, and guard-only routes are not silently
substituted. One NVIDIA Evals Nemotron route is reserved as the fixed Part 0
judge and cannot also be a subject.

Part 0 reports 16 systems and marks eight unavailable:
`anthropic/claude-haiku-4-5`, `anthropic/claude-opus-4-5`,
`anthropic/claude-opus-4-6`, `anthropic/claude-sonnet-4-5`,
`minimaxai/minimax-m2.7`, `openai/gpt-5`, `openai/gpt-5.2`, and
`openai/gpt-5.4`. Part 2 reports 22 systems and marks
`anthropic/claude-opus-4-6` and `minimaxai/minimax-m2.7` unavailable.

## Expanded Part 1 coverage

Part 1 is broader but does not enlarge the matched cross-axis sample. Seventy-
three routes report the balanced 96-root schedule, Qwen3.5 397B reports the
balanced 12-root schedule, and GLM 5.1 reports the 384-root schedule. Thus 75 of
81 frozen targets are reportable. `anthropic/claude-opus-4-5`,
`minimaxai/minimax-m2.7`, and `minimaxai/minimax-m3` have target-bound
operational failures; `moonshotai/kimi-k2.5`, `moonshotai/kimi-k2.6`, and
`zai-org/glm-5.2` were unavailable before execution. No unavailable target is
substituted. The n=12, n=96, and n=384 estimates have different support and are
never pooled or ranked as if they shared one design.

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
