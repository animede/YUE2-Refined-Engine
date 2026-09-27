import os
import tempfile
import unittest
from pathlib import Path


_OUTPUT_DIR = tempfile.TemporaryDirectory()
os.environ["YUE2_INIT_AT_STARTUP"] = "0"
os.environ["YUE2_API_KEY"] = "test-secret"
os.environ["YUE2_OUTPUT_DIR"] = _OUTPUT_DIR.name

from fastapi.testclient import TestClient

import yue2_api_server as server


class ApiSecurityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_context = TestClient(server.app)
        cls.client = cls.client_context.__enter__()
        cls.auth = {"Authorization": "Bearer test-secret"}

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)
        _OUTPUT_DIR.cleanup()

    def test_health_remains_available_for_local_readiness_checks(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["status"], "ok")

    def test_task_api_requires_bearer_token(self):
        denied = self.client.post("/query_result", json={"task_id_list": "[]"})
        self.assertEqual(denied.status_code, 401)

        allowed = self.client.post(
            "/query_result",
            json={"task_id_list": "[]"},
            headers=self.auth,
        )
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(allowed.json()["data"], [])

    def test_audio_download_requires_token_and_stays_inside_output_root(self):
        artifact = Path(_OUTPUT_DIR.name) / "job-id" / "audio.flac"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"test-audio")

        denied = self.client.get("/v1/audio", params={"path": "job-id/audio.flac"})
        self.assertEqual(denied.status_code, 401)

        allowed = self.client.get(
            "/v1/audio",
            params={"path": "job-id/audio.flac"},
            headers=self.auth,
        )
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(allowed.content, b"test-audio")

        traversal = self.client.get(
            "/v1/audio",
            params={"path": "../../etc/passwd"},
            headers=self.auth,
        )
        self.assertEqual(traversal.status_code, 403)

    def test_generated_audio_url_does_not_expose_absolute_server_path(self):
        path = server.OUTPUT_ROOT / "job-id" / "audio.flac"
        url = server._audio_url(path)
        self.assertEqual(url, "/v1/audio?path=job-id/audio.flac")
        self.assertNotIn(str(server.OUTPUT_ROOT), url)


if __name__ == "__main__":
    unittest.main()
