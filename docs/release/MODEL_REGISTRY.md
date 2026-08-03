# Model Registry

## Executed matched panel

Parts 0 and 2 use the same frozen 24-system panel. These are stable study target
IDs, not provider request routes. The private compatibility artifact binds each
ID to the exact authenticated route that passed the request-shape and response-
identity probe.

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

The matched panel is defined in `experiments/sota_cross_axis_panel.json`. The
file also records intended limits of 48 Part 0 roots per response-language
condition and 12 Part 2 trajectories. Deadline execution used lower explicit
CLI limits: 24 roots and eight trajectories. Final public metadata derives
included and operationally unavailable counts from sanitized manifest bindings
rather than treating planning defaults or scheduled systems as completed work.

Part 0 includes 16 systems. Its eight unavailable IDs are
`anthropic/claude-haiku-4-5`, `anthropic/claude-opus-4-5`,
`anthropic/claude-opus-4-6`, `anthropic/claude-sonnet-4-5`,
`minimaxai/minimax-m2.7`, `openai/gpt-5`, `openai/gpt-5.2`, and
`openai/gpt-5.4`. Part 2 executes 22 systems and 176 trajectories. Twenty
systems and 159 fully valid trajectories support estimates; 17 trajectories
are protocol-invalid, and two all-invalid executed systems are non-estimable.
The two unavailable IDs are `anthropic/claude-opus-4-6` and
`minimaxai/minimax-m2.7`.

## Part 1 expansion

Part 1 has a wider frozen execution roster. The sealed result contains 75
reportable targets: 73 at balanced n=96, `qwen/qwen3-5-397b-a17b` at balanced
n=12, and `zai-org/glm-5.1` at n=384. The three operationally unavailable
execution subjects are `anthropic/claude-opus-4-5`,
`minimaxai/minimax-m2.7`, and `minimaxai/minimax-m3`. The three additional
pre-execution unavailable targets are `moonshotai/kimi-k2.5`,
`moonshotai/kimi-k2.6`, and `zai-org/glm-5.2`; they were never manifest
subjects. Together these account for all 81 frozen Part 1 targets.

Two complete n=96 DeepSeek repair artifacts reproduce the same frozen study
IDs and schedules as their failed primary slices. They are exact evidence
replacements, not model substitutions. Every included Part 1 row is released
with explicit `scope` and `root_count`
fields. Operationally unavailable execution-roster targets are retained only
as sanitized, target-bound provenance and receive no estimate. The 12-root,
96-root, and 384-root estimates are not pooled. The larger Part 1 target set
does not change the frozen 24-system matched cross-axis cohort.

## Fixed judge

`judge.nvidia-evals-nemotron-3-30b-a3b` is the sole Part 0 judge target. It is
not one of the 24 subjects. Route selection rejects target-ID, exact-route, and
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
