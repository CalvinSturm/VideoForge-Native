import subprocess
import sys
import unittest


class WorkerHelpTest(unittest.TestCase):
    def test_help_works_without_zenoh(self):
        proc = subprocess.run(
            [sys.executable, "python/shm_worker.py", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
        for flag in (
            "--use-typed-ipc",
            "--use-events",
            "--prealloc-tensors",
            "--deterministic",
            "--log-level",
        ):
            self.assertIn(flag, proc.stdout)


if __name__ == "__main__":
    unittest.main()
