"""Small offline tests; these do not download models or call paid services."""
import unittest
from unittest.mock import patch

from app import local_backend


class LocalBackendTests(unittest.TestCase):
    def test_phi_fallback_preserves_original_prompt_when_ollama_unavailable(self):
        prompt = "A compact music player with USB-C and two physical buttons."
        with patch.object(local_backend, "OLLAMA_URL", "http://127.0.0.1:1"):
            refined, note = local_backend.refine_prompt_with_ollama(prompt)
        self.assertEqual(refined, prompt)
        self.assertIn("structured prompt", note)

    def test_empty_prompt_skips_reasoning(self):
        refined, note = local_backend.refine_prompt_with_ollama("  ")
        self.assertEqual(refined, "")
        self.assertIn("empty", note.lower())

    def test_unknown_mode_is_rejected_before_local_worker_request(self):
        with self.assertRaisesRegex(ValueError, "Unsupported local generation mode"):
            local_backend.generate_local_image("test prompt", "invalid-mode")


if __name__ == "__main__":
    unittest.main()
