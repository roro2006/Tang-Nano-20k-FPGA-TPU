"""Minimal PC-side transport for the Tang Nano attention packet protocol."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

from q8_attention import pack_frame, unpack_frame


def read_matrix(path: Path, rows: int, cols: int) -> list[list[int]]:
    values = [int(line.strip(), 16) for line in path.read_text().splitlines() if line.strip()]
    values = [value - 0x10000 if value & 0x8000 else value for value in values]
    if len(values) != rows * cols:
        raise ValueError(f"{path} contains {len(values)} values; expected {rows * cols}")
    return [values[row * cols:(row + 1) * cols] for row in range(rows)]


def run_once(port: str, baud: int, payload: bytes, timeout: float) -> bytes:
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("install pyserial to use the hardware host: python3 -m pip install pyserial") from exc
    with serial.Serial(port, baudrate=baud, timeout=timeout) as link:
        link.write(pack_frame(payload))
        link.flush()
        header = link.read(4)
        if len(header) != 4:
            raise TimeoutError("timed out waiting for FPGA response header")
        length = struct.unpack("<H", header[2:])[0]
        body = link.read(length + 1)
        if len(body) != length + 1:
            raise TimeoutError("timed out waiting for FPGA response payload")
        return unpack_frame(header + body)


def encode_q8_matrix(matrix: list[list[int]]) -> bytes:
    return b"".join(struct.pack("<h", value) for row in matrix for value in row)


def decode_q8_matrix(payload: bytes, rows: int, cols: int) -> list[list[int]]:
    if len(payload) != rows * cols * 2:
        raise ValueError("response does not contain the expected matrix")
    values = struct.unpack("<" + "h" * (rows * cols), payload)
    return [list(values[row * cols:(row + 1) * cols]) for row in range(rows)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("port")
    parser.add_argument("payload", type=Path, help="raw payload file")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=2.0)
    args = parser.parse_args()
    response = run_once(args.port, args.baud, args.payload.read_bytes(), args.timeout)
    print(f"received {len(response)} bytes")


if __name__ == "__main__":
    main()
