from __future__ import annotations

import unittest

from experiments.misc.inference_hub_provider_safe_v2 import provider_round_robin


class ProviderRoundRobinTests(unittest.TestCase):
    def test_stably_interleaves_upstream_providers(self) -> None:
        subjects = [
            {"target_id": "openai-1", "upstream_provider": "openai"},
            {"target_id": "openai-2", "upstream_provider": "openai"},
            {"target_id": "anthropic-1", "upstream_provider": "anthropic"},
            {"target_id": "anthropic-2", "upstream_provider": "anthropic"},
            {"target_id": "google-1", "upstream_provider": "google"},
        ]

        ordered = provider_round_robin(subjects)

        self.assertEqual(
            [row["target_id"] for row in ordered],
            ["openai-1", "anthropic-1", "google-1", "openai-2", "anthropic-2"],
        )
        self.assertEqual(
            sorted(row["target_id"] for row in ordered),
            sorted(row["target_id"] for row in subjects),
        )

    def test_missing_provider_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "upstream provider"):
            provider_round_robin([{"target_id": "broken"}])


if __name__ == "__main__":
    unittest.main()
