"""
Structured JSON logging for experiments.

Provides ExperimentLogger for recording trials, rounds, and aggregated results
as JSON-formatted logs for later analysis.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional


class ExperimentLogger:
    """Structured JSON logger for experiment data."""

    def __init__(self, results_dir: str = "results"):
        """
        Initialize the logger.

        Args:
            results_dir: Directory to write results to (defaults to 'results')
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_id: Optional[str] = None
        self.log_file: Optional[Path] = None
        self.log_data: dict[str, Any] = {}

    def start_experiment(self, experiment_id: str, config: dict) -> None:
        """
        Initialize a new experiment log.

        Creates a log file and writes the experiment metadata.

        Args:
            experiment_id: Unique identifier for this experiment
            config: Configuration dictionary for the experiment
        """
        self.experiment_id = experiment_id
        self.log_file = self.results_dir / f"{experiment_id}.jsonl"

        # Initialize log data structure
        self.log_data = {
            "experiment_id": experiment_id,
            "started_at": self._now(),
            "config": config,
            "trials": [],
        }

        # Write initial metadata
        self._write_line(
            {
                "type": "experiment_start",
                "experiment_id": experiment_id,
                "timestamp": self._now(),
                "config": config,
            }
        )

    def log_round(self, trial_id: int, round_num: int, data: dict) -> None:
        """
        Log data for a single round.

        Args:
            trial_id: The trial ID
            round_num: The round number
            data: Round data (actions, payoffs, etc.)
        """
        if not self.log_file:
            raise RuntimeError("Experiment not started. Call start_experiment first.")

        line = {
            "type": "round",
            "trial_id": trial_id,
            "round_num": round_num,
            "timestamp": self._now(),
            "data": data,
        }
        self._write_line(line)

    def log_trial_summary(self, trial_id: int, summary: dict) -> None:
        """
        Log a summary for a completed trial.

        Args:
            trial_id: The trial ID
            summary: Summary data (total payoff, final state, etc.)
        """
        if not self.log_file:
            raise RuntimeError("Experiment not started. Call start_experiment first.")

        line = {
            "type": "trial_summary",
            "trial_id": trial_id,
            "timestamp": self._now(),
            "summary": summary,
        }
        self._write_line(line)

    def finalize(self, total_cost: float, total_duration: float) -> None:
        """
        Finalize the experiment log with summary information.

        Args:
            total_cost: Total cost for the experiment in USD
            total_duration: Total duration in seconds
        """
        if not self.log_file:
            raise RuntimeError("Experiment not started. Call start_experiment first.")

        line = {
            "type": "experiment_summary",
            "experiment_id": self.experiment_id,
            "timestamp": self._now(),
            "total_cost_usd": total_cost,
            "total_duration_seconds": total_duration,
        }
        self._write_line(line)

    @staticmethod
    def _now() -> str:
        """Return an ISO timestamp in UTC."""
        return datetime.now(UTC).isoformat()

    def _write_line(self, data: dict) -> None:
        """
        Write a single line to the log file in JSONL format.

        Args:
            data: Dictionary to write as JSON
        """
        if not self.log_file:
            return

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")

    @staticmethod
    def read_log(log_file: Path) -> list[dict]:
        """
        Read a JSONL log file and return all lines.

        Args:
            log_file: Path to the log file

        Returns:
            List of parsed JSON objects (one per line)
        """
        lines = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        lines.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Skip malformed lines
                        pass
        return lines

    @staticmethod
    def read_experiment(experiment_id: str, results_dir: str = "results") -> list[dict]:
        """
        Read a complete experiment log by ID.

        Args:
            experiment_id: The experiment ID
            results_dir: Directory containing results

        Returns:
            List of parsed JSON objects from the log
        """
        log_file = Path(results_dir) / f"{experiment_id}.jsonl"
        if not log_file.exists():
            return []
        return ExperimentLogger.read_log(log_file)
