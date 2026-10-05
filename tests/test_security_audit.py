import unittest

from serve.security_audit import SecurityAuditor


class TestSecurityAudit(unittest.TestCase):
    """
    Test suite for security vulnerability detection module.
    """
    def setUp(self):
        self.auditor = SecurityAuditor()

    def test_sql_injection_detection(self):
        vulnerable_code = "query = \"SELECT * FROM users WHERE username = '\" + user_input + \"'\""
        report = self.auditor.analyze(vulnerable_code, language="python")

        self.assertLess(report.security_score, 100)
        vuln_types = [v.vulnerability_type for v in report.vulnerabilities]
        self.assertIn("SQL Injection", vuln_types)

    def test_hardcoded_secret_detection(self):
        vulnerable_code = "api_key = \"sk-1234567890abcdef12345678\""
        report = self.auditor.analyze(vulnerable_code, language="python")

        vuln_types = [v.vulnerability_type for v in report.vulnerabilities]
        self.assertIn("Hardcoded Secret", vuln_types)

    def test_command_injection_detection(self):
        vulnerable_code = "import os\nos.system('rm -rf ' + user_path)"
        report = self.auditor.analyze(vulnerable_code, language="python")

        vuln_types = [v.vulnerability_type for v in report.vulnerabilities]
        self.assertIn("Code/Command Injection", vuln_types)

    def test_clean_code_high_score(self):
        clean_code = "def add(a: int, b: int) -> int:\n    return a + b"
        report = self.auditor.analyze(clean_code, language="python")

        self.assertEqual(report.security_score, 100)
        self.assertEqual(len(report.vulnerabilities), 0)

if __name__ == "__main__":
    unittest.main()
