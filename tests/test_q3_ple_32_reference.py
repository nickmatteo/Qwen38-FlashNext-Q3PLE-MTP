import hashlib
import math
import random
import unittest

from scripts.q3_ple_32_reference import (
    BLOCK_BYTES,
    BLOCK_VALUES,
    ROW_BYTES,
    ROW_VALUES,
    dequantize_block,
    encode_table,
    get_rows,
    pack_codes,
    quantize_block,
    quantize_row,
    unpack_codes,
)


class Q3Ple32ReferenceTests(unittest.TestCase):
    def test_layout_constants(self):
        self.assertEqual(BLOCK_VALUES, 32)
        self.assertEqual(BLOCK_BYTES, 14)
        self.assertEqual(ROW_VALUES, 160)
        self.assertEqual(ROW_BYTES, 70)

    def test_known_code_vectors_and_cross_byte_boundaries(self):
        self.assertEqual(pack_codes([0] * 32), bytes(12))
        self.assertEqual(pack_codes([7] * 32), bytes([0xFF] * 12))
        codes = list(range(8)) * 4
        packed = pack_codes(codes)
        self.assertEqual(packed[:3], bytes.fromhex("88c6fa"))
        self.assertEqual(unpack_codes(packed), codes)

    def test_zero_block_has_zero_scale_and_zero_quants(self):
        block = quantize_block([0.0] * 32)
        self.assertEqual(len(block), 14)
        self.assertEqual(block, bytes(2) + pack_codes([4] * 32))
        self.assertEqual(dequantize_block(block), [0.0] * 32)

    def test_extrema_and_monotonic_are_deterministic(self):
        values = [(index - 16) / 4 for index in range(32)]
        first = quantize_block(values, refinement_passes=2)
        second = quantize_block(values, refinement_passes=2)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 14)
        decoded = dequantize_block(first)
        self.assertTrue(all(math.isfinite(value) for value in decoded))

    def test_non_finite_is_rejected(self):
        for bad in (math.nan, math.inf, -math.inf):
            values = [0.0] * 32
            values[7] = bad
            with self.assertRaises(ValueError):
                quantize_block(values)

    def test_bf16_scale_preserves_tiny_nonzero_values_that_f16_loses(self):
        values = [1.0e-9 * (index - 16) for index in range(32)]
        f16_decoded = dequantize_block(quantize_block(values, scale_dtype="f16"), "f16")
        bf16_decoded = dequantize_block(quantize_block(values, scale_dtype="bf16"), "bf16")
        self.assertTrue(all(value == 0.0 for value in f16_decoded))
        self.assertTrue(any(value != 0.0 for value in bf16_decoded))

    def test_tiny_table_get_rows_repeated_out_of_order_and_boundary(self):
        rows = [
            [((row + 1) * (column - 80)) / 97.0 for column in range(160)]
            for row in range(4)
        ]
        table = encode_table(rows, refinement_passes=1)
        self.assertEqual(len(table), 4 * 70)
        selected = get_rows(table, [3, 0, 3, 1])
        self.assertEqual(selected[0], selected[2])
        self.assertEqual(selected[0], get_rows(table, [3])[0])
        self.assertEqual(selected[1], get_rows(table, [0])[0])
        with self.assertRaises(IndexError):
            get_rows(table, [4])

    def test_random_fixture_hashes_pin_bf16_default_and_f16_legacy(self):
        rng = random.Random(380032)
        rows = [[rng.gauss(0.0, 0.75) for _ in range(160)] for _ in range(5)]
        default_table = encode_table(rows, refinement_passes=1)
        bf16_table = encode_table(rows, refinement_passes=1, scale_dtype="bf16")
        f16_table = encode_table(rows, refinement_passes=1, scale_dtype="f16")

        self.assertEqual(default_table, bf16_table)
        self.assertEqual(
            hashlib.sha256(bf16_table).hexdigest(),
            "8e6d9648947b129665336d6fbe13b0b632486a37a9d43122812588e360e56ada",
        )
        self.assertEqual(
            hashlib.sha256(f16_table).hexdigest(),
            "b1629b835c4dc6000a27abfb82c3948b90c57b9fa9cb830b54639937aeaf0d8d",
        )

    def test_refinement_does_not_break_shape(self):
        values = [math.sin(index / 3.0) * 2.5 for index in range(160)]
        for passes in (0, 1, 2):
            self.assertEqual(len(quantize_row(values, passes)), 70)


if __name__ == "__main__":
    unittest.main()
