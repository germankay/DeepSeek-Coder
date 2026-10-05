import json
import os
import tempfile
import unittest

from finetune.finetune_cybercode import (
    anonymize_code_and_text,
    clean_and_anonymize_dataset,
)


class TestFinetuneCyberCode(unittest.TestCase):
    """
    Test suite for data anonymization and fine-tuning dataset preprocessing.
    """
    def test_anonymize_ip_address(self):
        text = "Connect to database server at 192.168.1.100 on port 5432."
        anonymized = anonymize_code_and_text(text)
        self.assertNotIn("192.168.1.100", anonymized)
        self.assertIn("[ANONYMIZED_IP]", anonymized)

    def test_anonymize_api_key_and_bearer(self):
        text = 'api_key = "sk-proj-99998888777766665555"\nheaders = {"Authorization": "Bearer secret_token_123"}'
        anonymized = anonymize_code_and_text(text)
        self.assertNotIn("sk-proj-99998888777766665555", anonymized)
        self.assertNotIn("secret_token_123", anonymized)
        self.assertIn("[ANONYMIZED_SECRET]", anonymized)

    def test_clean_and_anonymize_dataset_file(self):
        sample_dataset = [
            {
                "instruction": "Fix bug on server 10.0.0.5",
                "input": "api_key = 'abcdef1234567890'",
                "output": "Used environment variable instead."
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, "raw.json")
            output_file = os.path.join(tmpdir, "cleaned.json")
            with open(input_file, "w") as f:
                json.dump(sample_dataset, f)

            clean_and_anonymize_dataset(input_file, output_file)
            self.assertTrue(os.path.exists(output_file))

            with open(output_file, "r") as f:
                cleaned_data = json.load(f)

            self.assertNotIn("10.0.0.5", cleaned_data[0]["instruction"])
            self.assertNotIn("abcdef1234567890", cleaned_data[0]["input"])

if __name__ == "__main__":
    unittest.main()
