# Model Registry

The current local/open-weight pilot cohort includes:

| Family | Model label | Style | Parameter/quantization note |
| --- | --- | --- | --- |
| GPT-OSS | `ollama/gpt-oss:20b` | Standard | 20B tag; quantization not recorded |
| GPT-OSS | `ollama/gpt-oss-safeguard:20b` | Safeguard | 20B tag; quantization not recorded |
| GPT-OSS | `ollama/gurubot/gpt-oss-derestricted:20b` | Unrestricted | 20B tag; quantization not recorded |
| Llama | `ollama/llama2` | Standard | tag default; quantization not recorded |
| Llama | `ollama/llama2-uncensored` | Unrestricted | tag default; quantization not recorded |
| Qwen 2.5 | `ollama/qwen2.5:7b` | Standard | 7B tag; quantization not recorded |
| Qwen 2.5 | `ollama/huihui_ai/qwen2.5-abliterate:7b` | Unrestricted | 7B tag; quantization not recorded |
| Qwen 2.5 | `ollama/qwen2.5:7b-instruct` | Instruct | 7B instruct tag; quantization not recorded |
| Qwen 2.5 | `ollama/huihui_ai/qwen2.5-abliterate:7b-instruct` | Instruct, unrestricted | 7B instruct tag; quantization not recorded |
| Qwen 3.5 | `ollama/qwen3.5` | Standard | tag default; quantization not recorded |
| Qwen 3.5 | `ollama/aratan/qwen3.5-uncensored:9b` | Unrestricted | 9B tag; quantization not recorded |
| Qwen 3.5 | `ollama/sorc/qwen3.5-instruct` | Instruct | tag default; quantization not recorded |
| Qwen 3.5 | `ollama/sorc/qwen3.5-instruct-uncensored` | Instruct, unrestricted | tag default; quantization not recorded |

Submission analyses should report model family, safeguard/unrestricted status, instruct/base status, provider, and exact local/API model identifier. API models should be added here only after their exact provider IDs, run dates, and metadata are available.

Pilot tag provenance is limited by what the April runs retained. Stable source
pages were rechecked on 2026-08-01 for
[`gpt-oss-derestricted`](https://ollama.com/gurubot/gpt-oss-derestricted),
[`qwen2.5-abliterate`](https://ollama.com/huihui_ai/qwen2.5-abliterate),
[`qwen3.5-instruct`](https://ollama.com/sorc/qwen3.5-instruct), and
[`llama2-uncensored`](https://ollama.com/library/llama2-uncensored). The exact
`aratan/qwen3.5-uncensored:9b` model page was no longer available at its expected
URL. These pages document mutable tags, not the immutable blobs used in April;
because the original Ollama digests were not recorded, the pilot variants cannot
be reconstructed byte-for-byte. Confirmatory runs must capture digests before
promotion.

## Exact-Version Confirmatory Registry

The machine-readable registry is `agents/agent_config.registry.json`, schema 1,
registry version `2026-08-01.2`, SHA-256
`b73ecad3109c3a287ffc77903645bf97a1607bc2a8f65b2cbd89fec258ba569f`.
These entries are catalog-display-only placeholders for planned confirmatory
targets. Every entry is marked `verification_status=unverified` and
`route_source=catalog_display_only`; preflight and campaign execution reject
them. Catalog labels are not callable model IDs. Each route must be replaced by
the exact backend-namespaced ID reported by InferenceHub Developer Tools before
promotion. Because the benchmark adapter never sends an unverified route, the
first approved smoke record must be captured through the separate route-review
workflow and added as verification evidence. Only then may the entry be marked
verified and used by the benchmark adapter or included in any result table.

A verified entry may use only the authoritative route source
`inference_hub_models_api`. It must include a
`verification_evidence` object with an ISO-8601 UTC `verified_at_utc`, a
lowercase SHA-256 `discovery_sha256`, and a `smoke_test` record containing its
UTC completion time, request ID, exact response model, and response-body
SHA-256. The smoke response model must equal the registered route. Unverified
entries may use only `catalog_display_only` and must not carry verification
evidence. Even after schema validation, the production adapter checks that the
requested InferenceHub route is registered and verified before credential use,
then rejects a missing or mismatched provider response-model identity.

### Current-SOTA cohort

| Upstream family | Unverified catalog label |
| --- | --- |
| OpenAI | `gpt-5.6-sol` |
| Anthropic | `claude-fable-5` |
| Anthropic | `claude-opus-5` |
| Anthropic | `claude-sonnet-5` |
| Anthropic | `claude-haiku-4-5-20251001` |
| Google | `gemini-3.1-pro-preview` |
| Google | `gemini-3.6-flash` |
| Google | `google/gemma-4-31b-it` |
| NVIDIA | `nvidia/nemotron-3-ultra-550b-a55b` |
| DeepSeek | `deepseek-ai/deepseek-v4-pro` |
| Qwen | `qwen/qwen3-next-80b-a3b-thinking` |
| Moonshot AI | `moonshotai/kimi-k2-thinking` |
| Z.ai | `z-ai/glm-5.2` |
| Mistral | `mistralai/mistral-nemotron` |

### Historical comparison cohort

| Upstream family | Unverified catalog label |
| --- | --- |
| OpenAI | `gpt-3.5-turbo-0125` |
| OpenAI | `gpt-4.1-2025-04-14` |
| OpenAI | `gpt-5-2025-08-07` |
| Google | `gemini-2.5-pro` |
| Google | `google/gemma-3-27b-it` |
| OpenAI | `openai/gpt-oss-120b` |

## Promotion Gate

For every target, preserve the registry version and hash, Developer
Tools-derived route source and verification status, provider
requested and response model IDs, model-identity match status, endpoint profile,
run time, supported decoding controls,
request/retry log, prompt/config hashes, and output hashes. A missing credential,
catalog-only label, 404/deprecated route, schema failure, unsupported control,
incomplete phase, or validation failure keeps that target out of the reported
cohort. The campaign manifest records failures rather than replacing a route or
silently changing a model version.
