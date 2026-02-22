import json
import os
import subprocess
import sys
import tempfile
import unittest


class BenchScriptTest(unittest.TestCase):
    def test_bench_emits_perf_sample_v1_jsonl(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            out_path = tmp.name

        proc = subprocess.run(
            [
                sys.executable,
                "python/bench/frame_loop_bench.py",
                "--iterations",
                "3",
                "--warmup",
                "0",
                "--width",
                "8",
                "--height",
                "8",
                "--device",
                "cpu",
                "--out",
                out_path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)

        try:
            with open(out_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]

            self.assertEqual(len(lines), 3)
            for line in lines:
                obj = json.loads(line)
                self.assertEqual(obj["schema_version"], "videoforge.perf_sample.v1")
                self.assertIn("ts_utc", obj)
                self.assertIn("iteration", obj)
                self.assertIn("config", obj)
                self.assertIn("durations_ms", obj)
                for key in ("cpu_to_tensor", "to_device", "mock_infer", "postprocess", "total"):
                    self.assertIn(key, obj["durations_ms"])
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)


if __name__ == "__main__":
    unittest.main()
