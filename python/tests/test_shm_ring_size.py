import unittest

from shm_worker import resolve_effective_ring_size, validate_shm_ring_override


class ShmRingSizeTests(unittest.TestCase):
    def test_resolution_precedence_payload_over_cli_over_default(self) -> None:
        self.assertEqual(
            resolve_effective_ring_size({"shm_ring_size": 8, "ring_size": 6}, 6, 6),
            8,
        )
        self.assertEqual(resolve_effective_ring_size({"ring_size": 6}, 8, 6), 8)
        self.assertEqual(resolve_effective_ring_size({}, None, 6), 6)

    def test_override_requires_v2(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_shm_ring_override(8, shm_proto_v2=False)
        self.assertIn("SHM_RING_SIZE_REQUIRES_V2", str(ctx.exception))

    def test_default_ring_size_allowed_without_v2(self) -> None:
        validate_shm_ring_override(6, shm_proto_v2=False)

    def test_invalid_ring_size_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_shm_ring_override(9, shm_proto_v2=True)
        self.assertIn("SHM_RING_SIZE_INVALID", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
