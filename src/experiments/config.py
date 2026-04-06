"""Pydantic config models and YAML loading for experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from src.agents.memory import MemoryMode


class ModelSpec(BaseModel):
    """Model/provider selection for one agent slot."""

    model: str
    provider: str | None = None

    @classmethod
    def from_value(cls, value: Any) -> "ModelSpec":
        """Coerce a string or mapping into a ModelSpec."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(model=value)
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError(f"Unsupported model specification: {value!r}")


class PopulationSpec(BaseModel):
    """Population entry for society simulations."""

    model: str
    count: int = 1
    provider: str | None = None

    @classmethod
    def from_value(cls, value: Any) -> "PopulationSpec":
        """Coerce a string or mapping into a population spec."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(model=value, count=1)
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError(f"Unsupported population specification: {value!r}")


class HistoryConfig(BaseModel):
    """Memory settings for iterated experiments."""

    mode: Literal["none", "full", "windowed", "summarized"] = "full"
    window_size: int | None = 10

    def to_memory_mode(self) -> MemoryMode:
        """Convert to the Agent memory enum."""
        return MemoryMode(self.mode)


class PromptVariantConfig(BaseModel):
    """Prompt files to compose for one experiment condition."""

    name: str
    system_prompt: str
    framing: str | None = None
    persona: str | None = None


class ParameterConfig(BaseModel):
    """Search/grid parameters shared across runners."""

    temperature: list[float] = Field(default_factory=lambda: [0.7])
    payoff_visibility: bool = True
    max_tokens: int = 512
    budget_usd: float | None = None
    seed: int = 42
    concurrency: int = 8

    @field_validator("temperature", mode="before")
    @classmethod
    def _normalize_temperatures(cls, value: Any) -> list[float]:
        if isinstance(value, (int, float)):
            return [float(value)]
        if isinstance(value, list):
            return [float(item) for item in value]
        raise TypeError("temperature must be a number or list of numbers")


class WorldConfigModel(BaseModel):
    """Serializable config mirror for the simulation world."""

    initial_public_resources: int = 40
    max_public_resources: int = 60
    regeneration_rate: float = 0.15
    initial_agent_resources: int = 6
    gather_amount: int = 3
    steal_amount: int = 2
    survival_cost: int = 1
    reproduction_threshold: int = 16
    offspring_start_resources: int = 5
    max_agents: int = 100


class SocietyConfig(BaseModel):
    """Feature flags for the society runner."""

    allow_steal: bool = True
    allow_private_messages: bool = True
    allow_unmonitored_agents: bool = False
    unmonitored_fraction: float = 0.0
    trade_offer_ttl: int = 3


class ReputationConfigModel(BaseModel):
    """Serializable config mirror for the reputation system."""

    enabled: bool = True
    decay: float = 1.0
    anonymous_ratings: bool = False
    min_rating: int = 1
    max_rating: int = 5


class ExperimentSettings(BaseModel):
    """Main experiment configuration used by all runners."""

    name: str
    part: Literal[1, 2, 3]
    game: str | None = None
    game_options: dict[str, Any] = Field(default_factory=dict)
    rounds: int = 5
    repetitions: int = 1
    pairings: list[tuple[ModelSpec, ModelSpec]] = Field(default_factory=list)
    agents: list[PopulationSpec] = Field(default_factory=list)
    history: HistoryConfig = Field(default_factory=HistoryConfig)
    prompt_variants: list[PromptVariantConfig] = Field(default_factory=list)
    parameters: ParameterConfig = Field(default_factory=ParameterConfig)
    world: WorldConfigModel | None = None
    society: SocietyConfig | None = None
    reputation: ReputationConfigModel | None = None

    @field_validator("pairings", mode="before")
    @classmethod
    def _coerce_pairings(cls, value: Any) -> list[tuple[ModelSpec, ModelSpec]]:
        if value is None:
            return []
        result: list[tuple[ModelSpec, ModelSpec]] = []
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("each pairing must contain exactly two model specs")
            result.append((ModelSpec.from_value(item[0]), ModelSpec.from_value(item[1])))
        return result

    @field_validator("agents", mode="before")
    @classmethod
    def _coerce_agents(cls, value: Any) -> list[PopulationSpec]:
        if value is None:
            return []
        return [PopulationSpec.from_value(item) for item in value]

    @model_validator(mode="after")
    def _validate_shape(self) -> "ExperimentSettings":
        if self.part == 1:
            if not self.game:
                raise ValueError("part 1 experiments require a game")
            if not self.pairings:
                raise ValueError("part 1 experiments require model pairings")
        else:
            if not self.agents:
                raise ValueError("part 2/3 experiments require an agent population")
            if self.world is None:
                self.world = WorldConfigModel()
            if self.society is None:
                self.society = SocietyConfig()

        if self.part == 3 and self.reputation is None:
            self.reputation = ReputationConfigModel()

        return self


class ExperimentConfig(BaseModel):
    """Top-level YAML document model."""

    experiment: ExperimentSettings


def load_experiment_config(path: str | Path) -> ExperimentSettings:
    """Load and validate an experiment YAML file."""
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if "experiment" not in payload:
        payload = {"experiment": payload}

    loaded = ExperimentConfig.model_validate(payload)
    return loaded.experiment
