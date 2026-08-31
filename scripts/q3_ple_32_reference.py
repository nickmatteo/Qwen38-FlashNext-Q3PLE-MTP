"""Executable reference codec for the proposed PLE-only Q3_PLE_32 format."""

from __future__ import annotations

import math
import struct
from collections.abc import Iterable, Sequence

BLOCK_VALUES = 32
BLOCK_BYTES = 14
ROW_VALUES = 160
BLOCKS_PER_ROW = ROW_VALUES // BLOCK_VALUES
ROW_BYTES = BLOCKS_PER_ROW * BLOCK_BYTES


def _f16(value: float) -> float:
    """Round a Python float to IEEE binary16 and return the rounded value."""
    return struct.unpack("<e", struct.pack("<e", value))[0]


def _bf16_bits(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return (rounded >> 16) & 0xFFFF


def _bf16_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def _store_scale(value: float, scale_dtype: str) -> tuple[bytes, float]:
    if scale_dtype == "f16":
        stored = _f16(value)
        return struct.pack("<e", stored), stored
    if scale_dtype == "bf16":
        bits = _bf16_bits(value)
        if value > 0.0 and bits == 0:
            bits = 1
        return struct.pack("<H", bits), _bf16_from_bits(bits)
    raise ValueError(f"unsupported scale dtype: {scale_dtype}")


def _load_scale(payload: bytes, scale_dtype: str) -> float:
    if scale_dtype == "f16":
        return struct.unpack("<e", payload)[0]
    if scale_dtype == "bf16":
        return _bf16_from_bits(struct.unpack("<H", payload)[0])
    raise ValueError(f"unsupported scale dtype: {scale_dtype}")


def pack_codes(codes: Sequence[int]) -> bytes:
    if len(codes) != BLOCK_VALUES:
        raise ValueError(f"expected {BLOCK_VALUES} codes, got {len(codes)}")
    packed = 0
    for index, code in enumerate(codes):
        if not 0 <= code <= 7:
            raise ValueError(f"code {index} is outside 0..7: {code}")
        packed |= code << (3 * index)
    return packed.to_bytes(12, "little")


def unpack_codes(payload: bytes) -> list[int]:
    if len(payload) != 12:
        raise ValueError(f"expected 12 code bytes, got {len(payload)}")
    packed = int.from_bytes(payload, "little")
    return [(packed >> (3 * index)) & 7 for index in range(BLOCK_VALUES)]


def _round_ties_even(value: float) -> int:
    return int(round(value))


def _codes_for_scale(values: Sequence[float], scale: float) -> list[int]:
    if scale == 0.0:
        return [4] * BLOCK_VALUES
    return [max(-4, min(3, _round_ties_even(value / scale))) + 4 for value in values]


def quantize_block(
    values: Sequence[float], refinement_passes: int = 1, scale_dtype: str = "bf16"
) -> bytes:
    if len(values) != BLOCK_VALUES:
        raise ValueError(f"expected {BLOCK_VALUES} values, got {len(values)}")
    if refinement_passes < 0:
        raise ValueError("refinement_passes must be non-negative")
    source = [float(value) for value in values]
    if not all(math.isfinite(value) for value in source):
        raise ValueError("Q3_PLE_32 cannot encode non-finite values")

    minimum = min(source)
    maximum = max(source)
    scale = max(-minimum / 4.0, maximum / 3.0)
    if scale == 0.0:
        return _store_scale(0.0, scale_dtype)[0] + pack_codes([4] * BLOCK_VALUES)

    codes = _codes_for_scale(source, scale)
    for _ in range(refinement_passes):
        quants = [code - 4 for code in codes]
        denominator = sum(quant * quant for quant in quants)
        if denominator == 0:
            break
        refined = sum(value * quant for value, quant in zip(source, quants)) / denominator
        if refined <= 0.0 or not math.isfinite(refined):
            break
        new_codes = _codes_for_scale(source, refined)
        scale = refined
        if new_codes == codes:
            codes = new_codes
            break
        codes = new_codes

    scale_bytes, stored_scale = _store_scale(scale, scale_dtype)
    if not math.isfinite(stored_scale):
        raise ValueError("stored scale is not finite")
    codes = _codes_for_scale(source, stored_scale)
    block = scale_bytes + pack_codes(codes)
    assert len(block) == BLOCK_BYTES
    return block


def dequantize_block(block: bytes, scale_dtype: str = "bf16") -> list[float]:
    if len(block) != BLOCK_BYTES:
        raise ValueError(f"expected {BLOCK_BYTES} block bytes, got {len(block)}")
    scale = _load_scale(block[:2], scale_dtype)
    return [scale * (code - 4) for code in unpack_codes(block[2:])]


def quantize_row(
    values: Sequence[float], refinement_passes: int = 1, scale_dtype: str = "bf16"
) -> bytes:
    if len(values) != ROW_VALUES:
        raise ValueError(f"expected {ROW_VALUES} row values, got {len(values)}")
    return b"".join(
        quantize_block(values[offset : offset + BLOCK_VALUES], refinement_passes, scale_dtype)
        for offset in range(0, ROW_VALUES, BLOCK_VALUES)
    )


def dequantize_row(row: bytes, scale_dtype: str = "bf16") -> list[float]:
    if len(row) != ROW_BYTES:
        raise ValueError(f"expected {ROW_BYTES} row bytes, got {len(row)}")
    values: list[float] = []
    for offset in range(0, ROW_BYTES, BLOCK_BYTES):
        values.extend(dequantize_block(row[offset : offset + BLOCK_BYTES], scale_dtype))
    return values


def encode_table(
    rows: Iterable[Sequence[float]], refinement_passes: int = 1, scale_dtype: str = "bf16"
) -> bytes:
    return b"".join(quantize_row(row, refinement_passes, scale_dtype) for row in rows)


def get_rows(
    table: bytes, row_indices: Sequence[int], scale_dtype: str = "bf16"
) -> list[list[float]]:
    if len(table) % ROW_BYTES:
        raise ValueError("table payload is not an integral number of rows")
    row_count = len(table) // ROW_BYTES
    output: list[list[float]] = []
    for index in row_indices:
        if not 0 <= index < row_count:
            raise IndexError(f"row index {index} outside 0..{row_count - 1}")
        offset = index * ROW_BYTES
        output.append(dequantize_row(table[offset : offset + ROW_BYTES], scale_dtype))
    return output
