from __future__ import annotations

import unittest
from unittest.mock import patch

from experiments.misc import inference_hub_compatibility as compatibility
from experiments.misc import inference_hub_compatibility_provider_safe as wrapper


class CompatibilityProviderSafeTests(unittest.TestCase):
    def test_wrapper_binds_provider_safe_factory_and_forwards_arguments(self) -> None:
        with patch.object(compatibility, "cli", return_value=7) as delegated:
            result = wrapper.cli(["--help"])

        self.assertEqual(result, 7)
        delegated.assert_called_once_with(["--help"])
        self.assertIs(
            compatibility._client_from_environment,
            wrapper._provider_safe_client,
        )


if __name__ == "__main__":
    unittest.main()
