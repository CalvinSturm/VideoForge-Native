import os
import subprocess
import sys
import unittest


class LoggingSetupTest(unittest.TestCase):
    def test_setup_logging_levels(self):
        sys.path.insert(0, os.path.join(os.getcwd(), "python"))
        from logging_setup import setup_logging

        logger = setup_logging("debug")
        self.assertEqual(logger.level, 10)  # logging.DEBUG

        logger = setup_logging(None)
        self.assertEqual(logger.level, 20)  # logging.INFO

    def test_import_runtime_modules_without_zenoh(self):
        code = (
            "import os,sys; "
            "sys.path.insert(0, os.path.join(os.getcwd(),'python')); "
            "import logging_setup, shm_worker, model_manager, research_layer, blender_engine, arch_wrappers"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)


if __name__ == "__main__":
    unittest.main()

