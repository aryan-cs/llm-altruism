"""
Cost tracking for LLM API calls during experiments.

Tracks token usage and costs across multiple models and providers.
"""

from typing import Optional

# Pricing information per model (input cost per 1k tokens, output cost per 1k tokens) in USD
MODEL_PRICING = {
    # OpenAI
    "gpt-4o": (0.005, 0.015),
    "gpt-4o-mini": (0.00015, 0.0006),
    "o1": (0.015, 0.06),
    "o3-mini": (0.002, 0.008),
    # Anthropic
    "claude-3-5-sonnet": (0.003, 0.015),
    "claude-3-5-sonnet-20241022": (0.003, 0.015),
    "claude-3-5-haiku": (0.00080, 0.004),
    "claude-3-5-haiku-20241022": (0.00080, 0.004),
    "claude-sonnet-4": (0.003, 0.015),
    "claude-sonnet-4-20250514": (0.003, 0.015),
    "claude-opus-4": (0.015, 0.06),
    "claude-opus-4-20250514": (0.015, 0.06),
    # Google
    "gemini-2.0-flash": (0.000075, 0.0003),
    "gemini-2.5-pro": (0.001, 0.004),
    # xAI
    "grok-3": (0.002, 0.008),
    # Cerebras
    "llama-3.3-70b": (0.000625, 0.001),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimate the cost of an LLM call.

    Args:
        model: The model name (e.g., "gpt-4o")
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Estimated cost in USD

    Raises:
        ValueError: If the model is not found in pricing table
    """
    if model not in MODEL_PRICING:
        raise ValueError(
            f"Unknown model: {model}. Available models: {list(MODEL_PRICING.keys())}"
        )

    input_cost_per_1k, output_cost_per_1k = MODEL_PRICING[model]
    input_cost = (input_tokens / 1000) * input_cost_per_1k
    output_cost = (output_tokens / 1000) * output_cost_per_1k
    return input_cost + output_cost


class CostTracker:
    """Track and accumulate costs across an experiment."""

    def __init__(self):
        """Initialize cost tracker with empty state."""
        self.costs_by_model: dict[str, dict[str, int | float]] = {}
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._total_cost = 0.0

    def add(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Add a cost for an LLM call.

        Args:
            model: The model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            The cost for this specific call in USD

        Raises:
            ValueError: If the model is not in the pricing table
        """
        cost = estimate_cost(model, input_tokens, output_tokens)

        # Initialize model entry if not present
        if model not in self.costs_by_model:
            self.costs_by_model[model] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": 0.0,
                "call_count": 0,
            }

        # Update model tracking
        self.costs_by_model[model]["input_tokens"] += input_tokens
        self.costs_by_model[model]["output_tokens"] += output_tokens
        self.costs_by_model[model]["cost"] += cost
        self.costs_by_model[model]["call_count"] += 1

        # Update totals
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self._total_cost += cost

        return cost

    def total(self) -> float:
        """
        Get the total accumulated cost.

        Returns:
            Total cost in USD
        """
        return self._total_cost

    def summary(self) -> dict:
        """
        Get a detailed summary of costs by model.

        Returns:
            Dictionary with:
                - total_cost: total across all models
                - total_input_tokens: total input tokens
                - total_output_tokens: total output tokens
                - by_model: dict of model -> {input_tokens, output_tokens, cost, call_count}
        """
        return {
            "total_cost": self._total_cost,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "by_model": self.costs_by_model.copy(),
        }

    def check_budget(self, budget: float) -> bool:
        """
        Check if the current cost is within budget.

        Args:
            budget: Maximum allowed cost in USD

        Returns:
            True if current cost is under budget, False otherwise
        """
        return self._total_cost <= budget

    def get_budget_remaining(self, budget: float) -> float:
        """
        Get remaining budget.

        Args:
            budget: Maximum allowed cost in USD

        Returns:
            Remaining budget in USD (can be negative if over budget)
        """
        return budget - self._total_cost

    def reset(self) -> None:
        """Reset the cost tracker to empty state."""
        self.costs_by_model.clear()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._total_cost = 0.0
