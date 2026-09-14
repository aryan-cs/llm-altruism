# Model Registry

## Frozen planning roster and definitive subsets

`experiments/sota_cross_axis_panel.json` preserves the 24-system planning
roster below. These are stable study target IDs, not provider request routes.
Each definitive private manifest binds its selected subset to the exact
authenticated route that passed request-shape and response-identity checks.
Planning-roster membership alone never creates an observation.

| Developer family | Study target ID |
| --- | --- |
| OpenAI | `openai/gpt-3.5-turbo` |
| OpenAI | `openai/gpt-4o` |
| OpenAI | `openai/gpt-4.1` |
| OpenAI | `openai/gpt-5` |
| OpenAI | `openai/gpt-5.2` |
| OpenAI | `openai/gpt-5.4` |
| OpenAI | `openai/gpt-oss-20b` |
| Anthropic | `anthropic/claude-haiku-4-5` |
| Anthropic | `anthropic/claude-sonnet-4-5` |
| Anthropic | `anthropic/claude-sonnet-4-6` |
| Anthropic | `anthropic/claude-opus-4-5` |
| Anthropic | `anthropic/claude-opus-4-6` |
| Google | `google/gemini-2.5-flash` |
| Google | `google/gemini-2.5-pro` |
| Google | `google/gemini-3.1-pro-preview` |
| Google | `google/gemini-3.5-flash` |
| Meta | `meta/llama-3.3-70b-instruct` |
| DeepSeek | `deepseek-ai/deepseek-v4-flash` |
| Qwen | `qwen/qwen3.5-35b-a3b` |
| Qwen | `qwen/qwen3.6-27b` |
| NVIDIA | `nvidia/nemotron-3-super-v3` |
| NVIDIA | `nvidia/nemotron-3-ultra` |
| MiniMax | `minimaxai/minimax-m2.7` |
| Z.ai | `zai-org/glm-5.1` |

The definitive Part 0 manifest selects 22 exact routes and schedules 48 roots
under each of three response-language instructions (144 responses per route).
`anthropic/claude-opus-4-5` and `minimaxai/minimax-m2.7` are explicit
planning-roster exclusions. The manifest remains incomplete; reportable model
coverage is accepted only through the explicit all-scheduled
operational-invalid terminal policy, while missing human judge validation
remains disclosed.

The definitive Part 2 composition contains 23 exact routes and is complete at
12 independent common-seed trajectories per route (276 total). Its ordered
evidence consists of the 21-route main source with the complete cascading v4
overlay, the Nemotron Ultra singleton source/overlay pair, and the DeepSeek V4
Flash singleton source/overlay pair. All 276 trajectories are operationally
eligible after 57 whole-trajectory repairs, and 198 zero-invalid trajectories
support environmental estimates. The other 78 trajectories are invalid-bearing;
18 routes have an environmental summary and five zero-eligible routes are
reported as NE. `anthropic/claude-opus-4-5` is the sole planning-roster
exclusion and receives no substitute.

The fixed Part 2 environment has 50 agents, 100 days, initial capacity 2,500,
OPTION_B private gain 2, reserve cost 2, unanimous group benefit/penalty 5, and
collapse death rate 0.2. Across 1,206,808 scheduled living agent-days it records
1,180,046 valid actions, 26,762 genuine semantic `INVALID` actions, 969,640
restraint actions, and 210,406 overuse actions.

## Part 1 expansion

Part 1 has a wider frozen registry. Its definitive primary manifest selects 75
exact routes, each on the same balanced 384-root direct self-choice bank. The
six registry exclusions are `anthropic/claude-opus-4-5`,
`minimaxai/minimax-m2.7`, `minimaxai/minimax-m3`,
`moonshotai/kimi-k2.5`, `moonshotai/kimi-k2.6`, and `zai-org/glm-5.2`. No
excluded target is substituted. The primary manifest remains incomplete, so
the 75-route evidence is accepted only together with its complete exact-source
operational overlay, which resolves three transport-null units.

First-attempt malformed outputs remain in every route's 384-unit primary
denominator. Periodic semantic-invalid repair uses the exact same route and
frozen work ID, but its artifact is isolated and cannot overwrite the primary
response, change that denominator, or enter the primary rate. The wider Part 1
registry does not enlarge the 22-route overlap shared by all three current
primary phases.

## Fixed judge

`judge.nvidia-evals-nemotron-3-30b-a3b` is the sole Part 0 judge target. It is
not one of the 22 Part 0 subjects. Route selection rejects target-ID, exact-route, and
upstream-model overlap with every subject. Judge batches contain only the
request and visible subject response, never subject identity or hidden
reasoning.

## Route eligibility and reporting

A route is executable only if the authenticated model catalog, compatibility
probe, and response identity agree. The runner records unsupported decoding
controls rather than silently changing one model's request. Transport failures
may be retried under the common rate limit; malformed semantic output is
retained as invalid or unclear.

Public reporting may include study target ID, developer family, model label,
scope, coverage, and aggregate metrics. Exact authenticated routes, credentials,
completion payloads, prompts, responses, and journals remain private. A study
target is one served system at one collection time, not an independent draw
from its developer or family.

## Historical local cohort

The April 13-model Ollama cohort spans GPT-OSS, Llama 2, Qwen 2.5, and Qwen 3.5
variants. It is historical pilot provenance only. Immutable model digests and
several decoding details were not retained, so those tags cannot reconstruct
the April model bytes. Current hosted estimates are not pooled with the legacy
cohort, and withdrawn legacy Part 0 or Part 2 rates are not current evidence.
