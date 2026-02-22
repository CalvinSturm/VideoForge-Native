import logging
import os
import sys
import unittest

import torch


sys.path.insert(0, os.path.join(os.getcwd(), "python"))

from shm_worker import PreallocBuffers, enforce_deterministic_mode


class PreallocTest(unittest.TestCase):
    def test_prealloc_reuses_buffers_when_shape_same(self):
        pool = PreallocBuffers(logging.getLogger("videoforge.test"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32

        pool.ensure(16, 16, 2, dtype, device)
        in_id_1 = id(pool.input_gpu)
        out_id_1 = id(pool.output_gpu)

        pool.ensure(16, 16, 2, dtype, device)
        in_id_2 = id(pool.input_gpu)
        out_id_2 = id(pool.output_gpu)

        self.assertEqual(in_id_1, in_id_2)
        self.assertEqual(out_id_1, out_id_2)

    def test_prealloc_reallocates_on_shape_change(self):
        pool = PreallocBuffers(logging.getLogger("videoforge.test"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32

        pool.ensure(16, 16, 2, dtype, device)
        in_id_1 = id(pool.input_gpu)
        out_id_1 = id(pool.output_gpu)

        pool.ensure(17, 16, 2, dtype, device)
        in_id_2 = id(pool.input_gpu)
        out_id_2 = id(pool.output_gpu)

        self.assertNotEqual(in_id_1, in_id_2)
        self.assertNotEqual(out_id_1, out_id_2)

    def test_deterministic_guardrails_do_not_crash(self):
        logger = logging.getLogger("videoforge.test")
        enforce_deterministic_mode(logger, enabled=True)
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cudnn.benchmark)


if __name__ == "__main__":
    unittest.main()

