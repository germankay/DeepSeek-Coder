import unittest

from finetune.finetune_deepseekcoder import DEFAULT_MODEL_NAME


class TestNonRegressionV1(unittest.TestCase):
    """
    Non-regression test suite ensuring model strict restriction to DeepSeek-Coder V1.
    """
    def test_default_model_is_v1(self):
        self.assertIn("deepseek-coder", DEFAULT_MODEL_NAME)
        self.assertNotIn("v2", DEFAULT_MODEL_NAME.lower())
        self.assertNotIn("v3", DEFAULT_MODEL_NAME.lower())
        self.assertNotIn("r1", DEFAULT_MODEL_NAME.lower())

    def test_prohibit_v2_v3_r1_names(self):
        prohibited_names = [
            "deepseek-ai/DeepSeek-Coder-V2-Instruct",
            "deepseek-ai/DeepSeek-V3",
            "deepseek-ai/DeepSeek-R1"
        ]
        for name in prohibited_names:
            is_prohibited = "v2" in name.lower() or "v3" in name.lower() or "r1" in name.lower()
            self.assertTrue(is_prohibited, f"Model name {name} must be detected as prohibited.")

if __name__ == "__main__":
    unittest.main()
