# Model Registry

The current local/open-weight pilot cohort includes:

| Family | Model label | Style |
| --- | --- | --- |
| GPT-OSS | `ollama/gpt-oss:20b` | Standard |
| GPT-OSS | `ollama/gpt-oss-safeguard:20b` | Safeguard |
| GPT-OSS | `ollama/gurubot/gpt-oss-derestricted:20b` | Unrestricted |
| Llama | `ollama/llama2` | Standard |
| Llama | `ollama/llama2-uncensored` | Unrestricted |
| Qwen 2.5 | `ollama/qwen2.5:7b` | Standard |
| Qwen 2.5 | `ollama/huihui_ai/qwen2.5-abliterate:7b` | Unrestricted |
| Qwen 2.5 | `ollama/qwen2.5:7b-instruct` | Instruct |
| Qwen 2.5 | `ollama/huihui_ai/qwen2.5-abliterate:7b-instruct` | Instruct, unrestricted |
| Qwen 3.5 | `ollama/qwen3.5` | Standard |
| Qwen 3.5 | `ollama/aratan/qwen3.5-uncensored:9b` | Unrestricted |
| Qwen 3.5 | `ollama/sorc/qwen3.5-instruct` | Instruct |
| Qwen 3.5 | `ollama/sorc/qwen3.5-instruct-uncensored` | Instruct, unrestricted |

Submission analyses should report model family, safeguard/unrestricted status, instruct/base status, provider, and exact local/API model identifier. API models should be added here only after their exact provider IDs, run dates, and metadata are available.

