from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.misc import inference_hub_provider_safe as provider_safe


class ProviderSafeLauncherTests(unittest.TestCase):
    def test_client_enforces_one_in_flight_request_per_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            environment = {
                "NVIDIA_API_KEY": "test-key-not-secret",
                "INFERENCE_HUB_BASE_URL": "https://inference-api.nvidia.com/v1",
                "TMPDIR": temporary,
            }
            with patch.dict(os.environ, environment, clear=False):
                client = provider_safe._provider_safe_client(17.0)

        contract = client.rate_limit_contract
        self.assertEqual(contract["provider_concurrency"], 1)
        self.assertEqual(contract["global_concurrency"], 16)
        self.assertEqual(contract["provider_requests_per_second"], 1.0)
        self.assertEqual(client.timeout_seconds, 17.0)

    def test_launcher_source_is_bound_to_new_evidence(self) -> None:
        class FakeModule:
            _SOURCE_PATHS = (Path("existing.py"),)

        provider_safe._bind_source(FakeModule)
        self.assertIn(Path(provider_safe.__file__).resolve(), FakeModule._SOURCE_PATHS)
        provider_safe._bind_source(FakeModule)
        self.assertEqual(
            FakeModule._SOURCE_PATHS.count(Path(provider_safe.__file__).resolve()), 1
        )

    def test_missing_subcommand_fails_closed(self) -> None:
        self.assertEqual(provider_safe.cli([]), 2)


if __name__ == "__main__":
    unittest.main()
