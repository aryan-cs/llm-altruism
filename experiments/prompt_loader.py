print("[PROMPT LOADER] Hello, World!")

import csv
import json
import re
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Any

PROMPT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = PROMPT_DIR.parent / "data" / "raw"


def _strip_json_comments(raw_text: str) -> str:
    kept_lines: list[str] = []
    for line in raw_text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("//") or stripped.startswith("#"):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


def _strip_trailing_commas(raw_text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", raw_text)


@lru_cache(maxsize=None)
def load_experiment_json(file_name: str) -> dict[str, Any]:
    config_path = PROMPT_DIR / file_name
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing experiment config: {config_path}")

    raw_text = config_path.read_text(encoding="utf-8")
    config = json.loads(_strip_trailing_commas(_strip_json_comments(raw_text)))

    if not isinstance(config, dict):
        raise ValueError(f"Experiment config must be a JSON object: {config_path}")
    return config


def load_prompt_config(part_name: str) -> dict[str, Any]:
    return load_experiment_json(f"{part_name}_prompt.json")


def load_prompts_from_csv_dir(csv_dir: Path) -> list[str]:
    if not csv_dir.is_dir():
        raise FileNotFoundError(f"Missing CSV prompt directory: {csv_dir}")

    csv_paths = sorted(
        path for path in csv_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".csv"
    )
    if not csv_paths:
        raise FileNotFoundError(f"No CSV prompt files found in: {csv_dir}")

    prompts: list[str] = []
    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "prompt" not in reader.fieldnames:
                raise ValueError(
                    f"CSV prompt file must contain a 'prompt' column: {csv_path}"
                )

            for row in reader:
                prompt = (row.get("prompt") or "").strip()
                if prompt:
                    prompts.append(prompt)

    if not prompts:
        raise ValueError(f"No prompts found in CSV prompt directory: {csv_dir}")

    return prompts


def load_part_0_raw_prompts() -> list[str]:
    return load_prompts_from_csv_dir(RAW_DATA_DIR / "part_0")


def render_prompt_template(template: str, **values: Any) -> str:
    try:
        return Template(template).substitute(**values).strip()
    except KeyError as exc:
        missing_key = exc.args[0]
        raise KeyError(f"Missing prompt template value: {missing_key}") from exc
