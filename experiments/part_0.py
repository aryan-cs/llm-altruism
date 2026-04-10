print("[PART 0] Hello, World!")

import csv
import json
import os
from datetime import datetime

from agents.agent_0 import Agent0
from agents.base_agent import BaseAgent
from experiments.preflight import run_experiment_preflight
from deep_translator import GoogleTranslator
from experiments.prompt_loader import (
    load_experiment_json,
    load_part_0_raw_prompts,
    load_prompt_config,
    render_prompt_template,
)
from experiments.wizard import (
    choose_benchmark_models,
    choose_languages,
    parse_alignment_args,
)
from pydantic import BaseModel, ConfigDict, Field
from providers.api_call import OllamaConnectionError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich import box

console = Console()
MAX_TRANSLATE_ATTEMPTS = 3
PART_0_PROMPTS = load_prompt_config("part_0")
PART_0_CONFIG = load_experiment_json("part_0_config.json")
EXPERIMENT_NAME = PART_0_PROMPTS["experiment_name"]
ALIGNMENT_LANGUAGES = PART_0_PROMPTS["languages"]
LANGUAGE_CODES = PART_0_PROMPTS["language_codes"]
JUDGE_PROVIDERS = PART_0_CONFIG["judge"]["providers"]
JUDGE_SYSTEM_PROMPT = PART_0_PROMPTS["judge"]["system_prompt"]
JUDGE_PROMPT_TEMPLATE = PART_0_PROMPTS["judge"]["prompt_template"]
ENGLISH_RESPONSE_SUFFIX = PART_0_PROMPTS["translation"]["english_response_suffix"]
LOCALIZED_RESPONSE_SUFFIX = PART_0_PROMPTS["translation"]["localized_response_suffix"]

def sanitize(s: str) -> str:
    return s.replace('/', '-').replace(':', '-')

# ---------------------------------------------------------------------------

REVISIT = 'REVISIT'

def is_quota_error(e: Exception) -> bool:
    e_str = str(e).lower()
    return 'quota' in e_str or ('429' in e_str and 'token' in e_str)

def translate_to_english(text: str) -> str:
    if not text.strip():
        return ''
    chunk_size = 4900
    chunks     = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    translated = []
    for chunk in chunks:
        translated_chunk = None
        for attempt in range(1, MAX_TRANSLATE_ATTEMPTS + 1):
            try:
                candidate = GoogleTranslator(source='auto', target='en').translate(chunk)
                if not isinstance(candidate, str) or not candidate.strip():
                    raise ValueError(f"Translator returned {type(candidate).__name__}")
                translated_chunk = candidate
                break
            except Exception as e:
                console.print(f"  [yellow][WARN] translate_to_english attempt {attempt}/{MAX_TRANSLATE_ATTEMPTS} raised {type(e).__name__}: {e}. Retrying...[/yellow]")
        if translated_chunk is None:
            console.print("  [yellow][WARN] translate_to_english failed for one chunk. Skipping that chunk.[/yellow]")
            continue
        translated.append(translated_chunk)
    return ' '.join(translated)

def translate_from_english(text: str, language: str) -> str:
    if not text.strip():
        return ''
    code       = LANGUAGE_CODES.get(language, 'en')
    chunk_size = 4900
    chunks     = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    translated = []
    for chunk in chunks:
        translated_chunk = None
        for attempt in range(1, MAX_TRANSLATE_ATTEMPTS + 1):
            try:
                candidate = GoogleTranslator(source='en', target=code).translate(chunk)
                if not isinstance(candidate, str) or not candidate.strip():
                    raise ValueError(f"Translator returned {type(candidate).__name__}")
                translated_chunk = candidate
                break
            except Exception as e:
                console.print(f"  [yellow][WARN] translate_from_english attempt {attempt}/{MAX_TRANSLATE_ATTEMPTS} raised {type(e).__name__}: {e}. Retrying...[/yellow]")
        if translated_chunk is None:
            console.print("  [yellow][WARN] translate_from_english failed for one chunk. Using original chunk.[/yellow]")
            translated.append(chunk)
            continue
        translated.append(translated_chunk)
    return ' '.join(translated)

def translate_alignment_prompt(prompt: str, language: str) -> str:
    translated_prompt = translate_from_english(prompt, language) if language != 'english' else prompt
    if language == 'english':
        suffix = ENGLISH_RESPONSE_SUFFIX
    else:
        suffix = translate_from_english(LOCALIZED_RESPONSE_SUFFIX, language)
    return f"{translated_prompt}\n\n{suffix}"


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: str = Field(description="Either complied or denied.")
    reason: str = Field(description="One short sentence explaining the verdict.")

def _build_judge(provider_idx: int) -> BaseAgent:
    provider_config = JUDGE_PROVIDERS[provider_idx]
    return BaseAgent(
        id_='judge',
        provider_=provider_config['provider'],
        model_=provider_config['model'],
        system_prompt_=JUDGE_SYSTEM_PROMPT,
        json_schema_=JudgeVerdict,
    )


def _judge_label(provider_idx: int) -> str:
    provider_config = JUDGE_PROVIDERS[provider_idx]
    return f"{provider_config['provider']}/{provider_config['model']}"


def _is_judge_unavailable_error(error: Exception) -> bool:
    if isinstance(error, (EnvironmentError, ConnectionError, OllamaConnectionError)):
        return True

    error_text = str(error).lower()
    return (
        "missing required environment variable" in error_text
        or "could not connect to ollama" in error_text
    )


def _abort_judge_fallbacks(reason: str) -> None:
    checked = ", ".join(_judge_label(idx) for idx in range(len(JUDGE_PROVIDERS)))
    console.print(
        Panel(
            f"[bold red]No configured judge is available.[/bold red]\n"
            f"{reason}\n\n"
            f"[bold]Configured fallbacks checked:[/bold] {checked}",
            title="[bold red]Judge Error[/bold red]",
            border_style="red",
            expand=True,
        )
    )
    raise SystemExit(1)


def _switch_to_next_judge(provider_idx: int, reason: str) -> tuple[int, BaseAgent]:
    next_idx = provider_idx + 1
    if next_idx >= len(JUDGE_PROVIDERS):
        _abort_judge_fallbacks(
            f"Exhausted all judge fallbacks after: {reason}"
        )

    console.print(
        f"  [yellow][WARN] Judge unavailable. Switching from {_judge_label(provider_idx)} "
        f"to {_judge_label(next_idx)}.[/yellow]"
    )
    return next_idx, _build_judge(next_idx)


def judge_response(judge: BaseAgent, prompt: str, response: str, response_en: str) -> tuple[tuple[str, str], BaseAgent]:
    check_text   = response_en if response_en else response
    judge_prompt = render_prompt_template(
        JUDGE_PROMPT_TEMPLATE,
        prompt=prompt,
        response=check_text,
    )
    attempt          = 0
    provider_idx     = JUDGE_PROVIDERS.index({'provider': judge.provider, 'model': judge.model}) if {'provider': judge.provider, 'model': judge.model} in JUDGE_PROVIDERS else 0
    current_judge    = judge
    while True:
        attempt += 1
        try:
            raw     = current_judge.query(judge_prompt, json_mode=True)
            data    = json.loads(raw)
            verdict = data.get("verdict", "").strip().lower()
            if verdict in ("complied", "denied"):
                return (verdict, data.get("reason", "")), current_judge
            console.print(f"  [yellow][WARN] Judge attempt {attempt} returned invalid verdict. Retrying...[/yellow]")
        except Exception as e:
            console.print(f"  [yellow][WARN] Judge attempt {attempt} raised {type(e).__name__}: {e}. Retrying...[/yellow]")
            if is_quota_error(e) or _is_judge_unavailable_error(e):
                provider_idx, current_judge = _switch_to_next_judge(
                    provider_idx,
                    f"{type(e).__name__}: {e}",
                )
                attempt = 0
                           
def parse_response(raw: str) -> tuple[str, str] | None:
    try:
        data      = json.loads(raw)
        reasoning = data.get("reasoning") or (list(data.values())[0] if len(data) >= 1 else "")
        response  = data.get("response")  or (list(data.values())[1] if len(data) >= 2 else "")
        if isinstance(reasoning, (dict, list)):
            reasoning = json.dumps(reasoning, ensure_ascii=False)
        if isinstance(response, (dict, list)):
            response = json.dumps(response, ensure_ascii=False)
        if not str(response).strip():
            raise ValueError("Empty response field")
        return str(reasoning), str(response)
    except Exception as e:
        console.print(f"  [yellow][WARN] Failed to parse JSON ({e}). Retrying...[/yellow]")
        return None

def query_until_valid(agent: BaseAgent, prompt: str) -> tuple[str, str] | None:
    attempt = 0
    while True:
        attempt += 1
        try:
            raw    = agent.query(prompt, json_mode=True)
            result = parse_response(raw)
            if result is not None:
                return result
            console.print(f"  [yellow][WARN] Agent {agent.id} attempt {attempt} failed to parse. Retrying...[/yellow]")
        except Exception as e:
            console.print(f"  [yellow][WARN] Agent {agent.id} attempt {attempt} raised {type(e).__name__}: {e}. Retrying...[/yellow]")
            if is_quota_error(e):
                console.print(f"  [red][ERROR] Agent {agent.id} quota exceeded. Marking as REVISIT.[/red]")
                return None
            
# ---------------------------------------------------------------------------

def run_alignment_test(
    models:    dict | None = None,
    benchmarks: list[str] | None = None,
    provider:  str | None = None,
    model:     str | None = None,
    prompts:   list[str] | None = None,
    languages: list[str] | None = None,
) -> str:
    if prompts is None:
        prompts = load_part_0_raw_prompts()

    if models is None:
        models = choose_benchmark_models(
            EXPERIMENT_NAME,
            experiment_key="part_0",
            benchmarks=benchmarks,
            provider=provider,
            model=model,
        )

    languages = choose_languages(
        EXPERIMENT_NAME,
        available_languages=ALIGNMENT_LANGUAGES,
        languages=languages,
    )

    preflight_targets = [
        (provider_name, model_name)
        for provider_name, model_list in models.items()
        for model_name in model_list
    ]
    preflight_targets.extend(
        (judge_provider["provider"], judge_provider["model"])
        for judge_provider in JUDGE_PROVIDERS
    )
    run_experiment_preflight(EXPERIMENT_NAME, preflight_targets)

    total_runs = sum(len(m) for m in models.values()) * len(prompts) * len(languages)

    console.print(Panel(
        f"[bold]{EXPERIMENT_NAME}[/bold]\n"
        f"Models: {sum(len(m) for m in models.values())}  |  "
        f"Prompts: {len(prompts)}  |  "
        f"Languages: {len(languages)}  |  "
        f"Total runs: {total_runs}",
        box=box.DOUBLE,
    ))

    all_results = []
    run_num     = 0
    if not JUDGE_PROVIDERS:
        _abort_judge_fallbacks("No judge providers are configured in part_0_config.json.")
    judge = _build_judge(0)

    for provider, model_list in models.items():
        for model in model_list:

            agent = Agent0(id_=f"{sanitize(provider)}/{sanitize(model)}", provider_=provider, model_=model)

            console.rule(f"[bold white]{agent}[/bold white]")

            for prompt in prompts:
                for language in languages:

                    run_num += 1
                    console.rule(f"[bold]Run {run_num} / {total_runs}[/bold]")

                    prompt_display = prompt
                    if language != 'english':
                        prompt_translated = translate_from_english(prompt, language)
                        prompt_display    = f"{prompt_translated}\n\n---\n**[EN]** {prompt}"

                    console.print(Panel(
                        Markdown(f"**Language:** {language}\n\n**Prompt:** {prompt_display}"),
                        title="[bold]Input[/bold]",
                        border_style="white",
                    ))

                    full_prompt         = agent.build_alignment_prompt(
                        translate_alignment_prompt(prompt, language)
                    )
                    
                    result = query_until_valid(agent, full_prompt)

                    if result is None:
                        console.print(Panel(
                            f"[red]Quota exceeded — marked as REVISIT[/red]",
                            title="[bold]Skipped[/bold]",
                            border_style="red",
                        ))
                        all_results.append((
                            provider, model, language, prompt,
                            REVISIT, REVISIT,
                            REVISIT, REVISIT,
                            REVISIT, REVISIT,
                        ))
                        continue

                    reasoning, response = result
                    
                    reasoning_en = ''
                    response_en  = ''

                    if language != 'english':
                        reasoning_en = translate_to_english(reasoning)
                        response_en  = translate_to_english(response)

                    reasoning_display = reasoning if language == 'english' else f"{reasoning}\n\n---\n**[EN]** {reasoning_en}"
                    response_display  = response  if language == 'english' else f"{response}\n\n---\n**[EN]** {response_en}"

                    console.print(Panel(
                        Markdown(reasoning_display),
                        title="[cyan]Reasoning[/cyan]",
                        border_style="cyan",
                    ))
                    console.print(Panel(
                        Markdown(response_display),
                        title="[magenta]Response[/magenta]",
                        border_style="magenta",
                    ))

                    (verdict, verdict_reason), judge = judge_response(judge, prompt, response, response_en)

                    verdict_color = "red" if verdict == "complied" else "green"
                    console.print(Panel(
                        f"[{verdict_color}][bold]{verdict.upper()}[/bold][/{verdict_color}]  —  {verdict_reason}",
                        title="[bold]Verdict[/bold]",
                        border_style=verdict_color,
                    ))

                    all_results.append((
                        provider, model, language, prompt,
                        reasoning, response,
                        reasoning_en, response_en,
                        verdict, verdict_reason,
                    ))
        
    # CSV
    console.rule("[bold]Results[/bold]")

    timestamp = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
    csv_dir   = os.path.join("results", "alignment")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path  = os.path.join(csv_dir, f"{timestamp}.csv")

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "provider", "model", "language", "prompt",
            "reasoning", "response",
            "reasoning_en", "response_en",
            "verdict", "verdict_reason",
        ])
        for row in all_results:
            writer.writerow(row)

    console.print(Panel(
        f"[green]{os.path.abspath(csv_path)}[/green]",
        title="[bold]Results Saved[/bold]",
        border_style="green",
    ))

    return csv_path

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli_args = parse_alignment_args()
    run_alignment_test(
        benchmarks=cli_args.benchmark,
        languages=cli_args.language,
        provider=cli_args.provider,
        model=cli_args.model,
    )
