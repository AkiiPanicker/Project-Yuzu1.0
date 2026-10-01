"""Tests for the model configuration."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from numpy_gpt import ModelConfig  # noqa: E402


class ModelConfigTests(unittest.TestCase):
    def test_default_shape_and_parameter_count(self) -> None:
        config = ModelConfig()
        self.assertEqual(config.head_dim, 64)
        self.assertEqual(config.parameter_count(), 3_279_104)

    def test_json_configuration_matches_defaults(self) -> None:
        config = ModelConfig.from_json(PROJECT_ROOT / "configs" / "tiny.json")
        self.assertEqual(config, ModelConfig())

    def test_invalid_head_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            ModelConfig(d_model=256, n_heads=3)


if __name__ == "__main__":
    unittest.main()

