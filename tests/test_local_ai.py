import unittest
from unittest.mock import patch

from app.local_ai import _find_model, LocalAIError, VISION_MODEL_PREFIXES, REASONING_MODEL_PREFIXES


class LocalAIDiscoveryTests(unittest.TestCase):
    def test_finds_tagged_vision_model(self):
        self.assertEqual(_find_model(["phi4-mini:latest", "qwen2.5vl:3b"], VISION_MODEL_PREFIXES), "qwen2.5vl:3b")

    def test_finds_reasoner(self):
        self.assertEqual(_find_model(["phi4-mini:latest"], REASONING_MODEL_PREFIXES), "phi4-mini:latest")

    def test_missing_model_is_none(self):
        self.assertIsNone(_find_model(["llama3.2:latest"], VISION_MODEL_PREFIXES))


if __name__ == "__main__":
    unittest.main()
