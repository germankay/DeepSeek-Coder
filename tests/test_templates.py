import unittest

from finetune.finetune_deepseekcoder import EOT_TOKEN, build_instruction_prompt


class TestTemplates(unittest.TestCase):
    """
    Test suite for instruction and chat prompt formatting templates.
    """
    def test_instruction_prompt_structure(self):
        instruction = "Write a python function to check if a number is prime."
        prompt = build_instruction_prompt(instruction)

        self.assertIn("DeepSeek Coder model", prompt)
        self.assertIn("### Instruction:\n" + instruction, prompt)
        self.assertIn("### Response:", prompt)

    def test_eot_token_format(self):
        self.assertEqual(EOT_TOKEN, "<|EOT|>")
        output_with_eot = f"def is_prime(n):\n    return True\n{EOT_TOKEN}"
        self.assertTrue(output_with_eot.endswith("<|EOT|>"))

if __name__ == "__main__":
    unittest.main()
