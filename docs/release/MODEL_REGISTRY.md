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
