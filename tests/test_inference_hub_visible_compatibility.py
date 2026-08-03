from __future__ import annotations

import unittest
from unittest.mock import patch

from experiments.misc import inference_hub_compatibility as compatibility
from experiments.misc import inference_hub_visible_compatibility as wrapper


class VisibleCompatibilityTests(unittest.TestCase):
    def test_profiles_exclude_structured_response(self) -> None:
        self.assertTrue(wrapper.VISIBLE_CONTENT_PROFILE_ORDER)
        self.assertNotIn(
            "structured_response",
            {
                control
                for profile in wrapper.VISIBLE_CONTENT_PROFILE_ORDER
                for control in profile
            },
        )

    def test_wrapper_binds_policy_and_forwards_arguments(self) -> None:
        original_order = compatibility.EXECUTION_PROFILE_ORDER
        original_factory = compatibility._client_from_environment
        try:
            with patch.object(compatibility, "cli", return_value=5) as delegated:
                result = wrapper.cli(["--help"])
            self.assertEqual(result, 5)
            delegated.assert_called_once_with(["--help"])
            self.assertEqual(
                compatibility.EXECUTION_PROFILE_ORDER,
                wrapper.VISIBLE_CONTENT_PROFILE_ORDER,
            )
            self.assertIs(
                compatibility._client_from_environment,
                wrapper._provider_safe_client,
            )
        finally:
            compatibility.EXECUTION_PROFILE_ORDER = original_order
            compatibility._client_from_environment = original_factory


if __name__ == "__main__":
    unittest.main()
