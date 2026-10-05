import unittest


class TestTokenizerFimAndSpecialTokens(unittest.TestCase):
    """
    Test suite for FIM (Fill-In-the-Middle) and special token constants for DeepSeek-Coder V1.
    """
    def setUp(self):
        self.fim_begin = "<｜fim begin｜>"
        self.fim_hole = "<｜fim hole｜>"
        self.fim_end = "<｜fim end｜>"
        self.eot_token = "<|EOT|>"

    def test_fim_token_constants(self):
        self.assertEqual(self.fim_begin, "<｜fim begin｜>")
        self.assertEqual(self.fim_hole, "<｜fim hole｜>")
        self.assertEqual(self.fim_end, "<｜fim end｜>")
        self.assertEqual(self.eot_token, "<|EOT|>")

    def test_fim_prompt_construction(self):
        prefix = "def calculate_sum(a, b):\n"
        suffix = "\n    return result"
        fim_prompt = f"{self.fim_begin}{prefix}{self.fim_hole}{suffix}{self.fim_end}"
        self.assertIn(self.fim_begin, fim_prompt)
        self.assertIn(self.fim_hole, fim_prompt)
        self.assertIn(self.fim_end, fim_prompt)

if __name__ == "__main__":
    unittest.main()
