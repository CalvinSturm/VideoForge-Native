import os
import sys
import unittest


sys.path.insert(0, os.path.join(os.getcwd(), "python"))

from ipc_protocol import IpcError, RequestEnvelope, ResponseEnvelope


class IpcProtocolTest(unittest.TestCase):
    def test_request_roundtrip_minimal(self):
        raw = {
            "version": 1,
            "request_id": "7",
            "job_id": "job-1",
            "kind": "load_model",
            "payload": {"model_name": "RCAN_x4"},
        }
        req = RequestEnvelope.from_json(raw)
        self.assertEqual(req.version, 1)
        self.assertEqual(req.request_id, "7")
        self.assertEqual(req.job_id, "job-1")
        self.assertEqual(req.kind, "load_model")
        self.assertEqual(req.payload["model_name"], "RCAN_x4")

    def test_missing_required_fields_rejected(self):
        raw = {
            "version": 1,
            "request_id": "7",
            "kind": "load_model",
            "payload": {"model_name": "RCAN_x4"},
        }
        with self.assertRaises(ValueError):
            RequestEnvelope.from_json(raw)

    def test_unknown_fields_preserved(self):
        raw = {
            "version": 1,
            "request_id": "7",
            "job_id": "job-1",
            "kind": "load_model",
            "payload": {"model_name": "RCAN_x4"},
            "future_field": True,
            "another_future_field": {"nested": "x"},
        }
        req = RequestEnvelope.from_json(raw)
        self.assertIn("future_field", req.extra)
        self.assertIn("another_future_field", req.extra)

        resp = ResponseEnvelope.status_response(
            status="MODEL_LOADED",
            req=req,
            extra=req.extra,
        )
        emitted = resp.to_json()
        self.assertTrue(emitted["future_field"])
        self.assertEqual(emitted["another_future_field"]["nested"], "x")

    def test_response_shape_matches_expected(self):
        req = RequestEnvelope.from_json(
            {
                "version": 1,
                "request_id": "9",
                "job_id": "job-2",
                "kind": "shutdown",
                "payload": {},
            }
        )
        resp = ResponseEnvelope(
            version=1,
            request_id=req.request_id,
            job_id=req.job_id,
            kind="error",
            status="error",
            error=IpcError(code="MODEL_NOT_FOUND", message="missing"),
            extra={"scale": 4},
        )
        emitted = resp.to_json()
        for key in ("version", "request_id", "job_id", "kind", "status", "error"):
            self.assertIn(key, emitted)
        self.assertEqual(emitted["version"], 1)
        self.assertIsInstance(emitted["request_id"], str)
        self.assertIsInstance(emitted["job_id"], str)
        self.assertEqual(emitted["kind"], "error")
        self.assertEqual(emitted["status"], "error")
        self.assertEqual(emitted["error"]["code"], "MODEL_NOT_FOUND")
        self.assertEqual(emitted["error"]["message"], "missing")
        self.assertEqual(emitted["scale"], 4)


if __name__ == "__main__":
    unittest.main()

