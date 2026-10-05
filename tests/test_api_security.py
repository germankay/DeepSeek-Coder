import unittest

from fastapi.testclient import TestClient

from serve.api_server import API_KEY, app


class TestApiSecurity(unittest.TestCase):
    """
    Test suite for FastAPI endpoints, authentication, and rate limiting.
    """
    def setUp(self):
        self.client = TestClient(app)
        self.auth_header = {"Authorization": f"Bearer {API_KEY}"}

    def test_healthz_unauthenticated(self):
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_models_unauthorized_missing_token(self):
        response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 401)

    def test_models_forbidden_invalid_token(self):
        response = self.client.get("/v1/models", headers={"Authorization": "Bearer wrong-token"})
        self.assertEqual(response.status_code, 403)

    def test_models_authorized(self):
        response = self.client.get("/v1/models", headers=self.auth_header)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["object"], "list")

    def test_security_audit_endpoint(self):
        payload = {
            "code": "query = 'SELECT * FROM users WHERE id = ' + user_id",
            "langage": "python"
        }
        response = self.client.post("/v1/security/audit", json=payload, headers=self.auth_header)
        self.assertEqual(response.status_code, 200)
        res_json = response.json()
        self.assertIn("security_score", res_json)
        self.assertIn("vulnerabilities", res_json)

    def test_code_review_endpoint(self):
        payload = {
            "diff": "eval(user_input)",
            "langage": "python"
        }
        response = self.client.post("/v1/code/review", json=payload, headers=self.auth_header)
        self.assertEqual(response.status_code, 200)
        res_json = response.json()
        self.assertEqual(res_json["status"], "completed")

    def test_code_generate_endpoint(self):
        payload = {
            "task": "authenticate user via JWT",
            "langage": "python"
        }
        response = self.client.post("/v1/code/generate", json=payload, headers=self.auth_header)
        self.assertEqual(response.status_code, 200)
        res_json = response.json()
        self.assertIn("generated_code", res_json)

if __name__ == "__main__":
    unittest.main()
