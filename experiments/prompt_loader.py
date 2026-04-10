print("[PROMPT LOADER] Hello, World!")

import json
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Any

PROMPT_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def load_experiment_json(file_name: str) -> dict[str, Any]:
    config_path = PROMPT_DIR / file_name
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing experiment config: {config_path}")

    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Experiment config must be a JSON object: {config_path}")
    return config


def load_prompt_config(part_name: str) -> dict[str, Any]:
    return load_experiment_json(f"{part_name}_prompt.json")


def render_prompt_template(template: str, **values: Any) -> str:
    try:
        return Template(template).substitute(**values).strip()
    except KeyError as exc:
        missing_key = exc.args[0]
        raise KeyError(f"Missing prompt template value: {missing_key}") from exc
