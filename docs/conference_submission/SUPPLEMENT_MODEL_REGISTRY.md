# Prosocial Readiness Bench Model Registry

The current paper uses separate phase-specific primary rosters. A shared
planning roster does not imply that an unavailable route was observed.

## Planning roster and primary overlap

`experiments/sota_cross_axis_panel.json` stores a 24-system planning roster
spanning GPT, Claude Haiku, Sonnet, and Opus, Gemini, Llama, DeepSeek, Qwen,
Nemotron, MiniMax, and GLM families. The definitive Part 0 manifest selects 22
exact routes; the definitive Part 2 composition selects 23. Their 22-route
overlap is the only current three-primary-phase overlap.

Every included target is bound to an exact compatibility-selected route and a
successful identity probe. Unavailable, deprecated, identity-mismatched,
non-chat, embedding, reranking, image, and guard-only routes are not silently
substituted. One NVIDIA Evals Nemotron route is reserved as the fixed Part 0
judge and cannot also be a subject.

Part 0 schedules 48 archived request roots under three response-language
instructions for each selected route (144 responses per route). Its planning-
roster exclusions are `anthropic/claude-opus-4-5` and
`minimaxai/minimax-m2.7`. Its source manifest remains incomplete; all scheduled
units are terminal only under the explicit all-scheduled operational-invalid
policy, and the missing human-validation gate remains disclosed.

Part 2 is complete at 12 common-seed trajectories for each of 23 exact routes
(276 total), composed in order from the 21-route main, Nemotron Ultra
singleton, and DeepSeek V4 Flash singleton source/overlay pairs. All 276
trajectories are operationally eligible after 57 whole-trajectory repairs; 198
zero-invalid trajectories support environmental estimates. The other 78 are
invalid-bearing; 18 routes have an environmental summary and five zero-eligible
routes are reported as NE.
`anthropic/claude-opus-4-5` is the sole excluded planning-roster target and is
not substituted.

The fixed Part 2 contract uses 50 agents, 100 days, initial capacity 2,500,
OPTION_B private gain 2, reserve cost 2, unanimous group benefit/penalty 5, and
collapse death rate 0.2. Its 1,206,808 scheduled living agent-days contain
1,180,046 valid actions, 26,762 genuine semantic `INVALID` actions, 969,640
restraint actions, and 210,406 overuse actions.

## Expanded Part 1 coverage

Part 1 is broader but does not enlarge the 22-route current primary overlap. Its
definitive manifest selects 75 exact routes, each on the same balanced 384-root
direct self-choice bank. `anthropic/claude-opus-4-5`,
`minimaxai/minimax-m2.7`, `minimaxai/minimax-m3`,
`moonshotai/kimi-k2.5`, `moonshotai/kimi-k2.6`, and `zai-org/glm-5.2` are
frozen registry exclusions. No unavailable target is substituted. The source
manifest remains incomplete and is accepted only with its complete exact-source
operational overlay, which resolves three transport-null units without
regenerating malformed semantic responses.

Six role-calibration sentinels use three distinct frames over 96 roots with
four counterbalances per frame. Four pinned local open-weight controls use the
384-root Part 1 bank but remain a separate execution-scale panel rather than
hosted-route substitutes.

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
