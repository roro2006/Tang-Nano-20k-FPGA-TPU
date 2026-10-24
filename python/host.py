"""PC-side transport for the Tang Nano attention packet protocol.

Request payload is ``op, arg, data``; the response payload is ``op, status,
data``. See rtl/tpu_top.sv for the authoritative command table.
"""

from __future__ import annotations

import argparse
import random
import struct
import time
from pathlib import Path
from typing import Sequence

from q8_attention import attention_q8_8, pack_frame, unpack_frame

OP_ECHO, OP_LOAD, OP_ATTN, OP_INFO, OP_ERROR = 0x01, 0x02, 0x03, 0x04, 0xEE
MATRIX_NAMES = ("q", "k", "v", "o")
TOKENS, WIDTH = 16, 64
# Must match BAUD in rtl/tang_nano_20k_top.sv.
DEFAULT_BAUD = 1_000_000


def read_matrix(path: Path, rows: int, cols: int) -> list[list[int]]:
    values = [int(line.strip(), 16) for line in path.read_text().splitlines() if line.strip()]
    values = [value - 0x10000 if value & 0x8000 else value for value in values]
    if len(values) != rows * cols:
        raise ValueError(f"{path} contains {len(values)} values; expected {rows * cols}")
    return [values[row * cols:(row + 1) * cols] for row in range(rows)]


def encode_q8_matrix(matrix: Sequence[Sequence[int]]) -> bytes:
    return b"".join(struct.pack("<h", value) for row in matrix for value in row)


def decode_q8_matrix(payload: bytes, rows: int, cols: int) -> list[list[int]]:
    if len(payload) != rows * cols * 2:
        raise ValueError("response does not contain the expected matrix")
    values = struct.unpack("<" + "h" * (rows * cols), payload)
    return [list(values[row * cols:(row + 1) * cols]) for row in range(rows)]


def request(op: int, arg: int = 0, data: bytes = b"") -> bytes:
    return pack_frame(bytes([op, arg]) + data)


def load_request(layer: int, name: str, matrix: Sequence[Sequence[int]]) -> bytes:
    if len(matrix) != WIDTH or any(len(row) != WIDTH for row in matrix):
        raise ValueError("attention weights must be 64x64")
    return request(OP_LOAD, layer * 4 + MATRIX_NAMES.index(name), encode_q8_matrix(matrix))


def attention_request(layer: int, x: Sequence[Sequence[int]]) -> bytes:
    if len(x) != TOKENS or any(len(row) != WIDTH for row in x):
        raise ValueError("attention input must be 16x64")
    return request(OP_ATTN, layer, encode_q8_matrix(x))


def response(op: int, status: int = 0, data: bytes = b"") -> bytes:
    return pack_frame(bytes([op, status]) + data)


class TpuLink:
    """Synchronous request/response client for the FPGA."""

    def __init__(self, port: str, baud: int = DEFAULT_BAUD, timeout: float = 5.0):
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("install pyserial to use the hardware host: python -m pip install pyserial") from exc
        self.link = serial.Serial(port, baudrate=baud, timeout=timeout)
        # The USB bridge can still hold bytes from whatever ran before; let
        # them arrive, then discard them so the first header is ours.
        time.sleep(0.2)
        self.link.reset_input_buffer()

    def close(self) -> None:
        self.link.close()

    def __enter__(self) -> "TpuLink":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def transact(self, frame: bytes) -> tuple[int, int, bytes]:
        self.link.write(frame)
        self.link.flush()
        header = self.link.read(4)
        if len(header) != 4 or header[:2] != b"\xA5\x5A":
            raise TimeoutError(f"bad or missing response header: {header.hex()}")
        length = struct.unpack("<H", header[2:])[0]
        body = self.link.read(length + 1)
        if len(body) != length + 1:
            raise TimeoutError("timed out waiting for FPGA response payload")
        payload = unpack_frame(header + body)
        if len(payload) < 2:
            raise ValueError("response payload is too short")
        return payload[0], payload[1], payload[2:]

    def _checked(self, frame: bytes, op: int) -> bytes:
        got_op, status, data = self.transact(frame)
        if got_op == OP_ERROR:
            raise IOError(f"FPGA reported a framing error (status {status})")
        if got_op != op or status != 0:
            raise IOError(f"command {op:#04x} failed: op={got_op:#04x} status={status}")
        return data

    def info(self) -> bytes:
        return self._checked(request(OP_INFO), OP_INFO)

    def echo(self, data: bytes) -> bytes:
        return self._checked(request(OP_ECHO, 0, data), OP_ECHO)

    def load_weights(self, layer: int, weights: dict[str, Sequence[Sequence[int]]]) -> None:
        for name in MATRIX_NAMES:
            self._checked(load_request(layer, name, weights[name]), OP_LOAD)

    def attention(self, layer: int, x: Sequence[Sequence[int]]) -> list[list[int]]:
        data = self._checked(attention_request(layer, x), OP_ATTN)
        return decode_q8_matrix(data, TOKENS, WIDTH)


def run_once(port: str, baud: int, payload: bytes, timeout: float) -> bytes:
    """Send one raw payload (op, arg, data) and return the raw response payload."""
    with TpuLink(port, baud, timeout) as tpu:
        op, status, data = tpu.transact(pack_frame(payload))
        return bytes([op, status]) + data


def random_matrix(rng: random.Random, rows: int, cols: int, span: int) -> list[list[int]]:
    return [[rng.randint(-span, span) for _ in range(cols)] for _ in range(rows)]


def random_layer(rng: random.Random, span: int = 64) -> dict[str, list[list[int]]]:
    return {name: random_matrix(rng, WIDTH, WIDTH, span) for name in MATRIX_NAMES}


def selftest(tpu: TpuLink, seed: int, runs: int) -> bool:
    rng = random.Random(seed)
    info = tpu.info()
    print(f"info: {info[:3].decode(errors='replace')} v{info[3] if len(info) > 3 else '?'}")
    probe = bytes(rng.randrange(256) for _ in range(256))
    if tpu.echo(probe) != probe:
        print("echo: MISMATCH")
        return False
    print("echo: ok (256 bytes)")
    layers = [random_layer(rng), random_layer(rng)]
    start = time.perf_counter()
    for layer, weights in enumerate(layers):
        tpu.load_weights(layer, weights)
    print(f"weights: loaded 2 layers in {time.perf_counter() - start:.1f} s")
    ok = True
    for run in range(runs):
        layer = run % 2
        x = random_matrix(rng, TOKENS, WIDTH, 512)
        w = layers[layer]
        expected = attention_q8_8(x, w["q"], w["k"], w["v"], w["o"])
        start = time.perf_counter()
        got = tpu.attention(layer, x)
        elapsed = time.perf_counter() - start
        bad = sum(g != e for gr, er in zip(got, expected) for g, e in zip(gr, er))
        print(f"attention run {run} layer {layer}: {'ok' if bad == 0 else f'{bad} mismatches'} "
              f"({elapsed * 1000:.0f} ms round trip)")
        ok = ok and bad == 0
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", help="UART port, e.g. COM4 or /dev/ttyUSB1")
    parser.add_argument("command", choices=("info", "selftest", "raw"))
    parser.add_argument("payload", type=Path, nargs="?", help="raw payload file for 'raw'")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--runs", type=int, default=4)
    args = parser.parse_args()
    with TpuLink(args.port, args.baud, args.timeout) as tpu:
        if args.command == "info":
            print(tpu.info())
        elif args.command == "selftest":
            raise SystemExit(0 if selftest(tpu, args.seed, args.runs) else 1)
        else:
            if args.payload is None:
                parser.error("raw needs a payload file")
            op, status, data = tpu.transact(pack_frame(args.payload.read_bytes()))
            print(f"op={op:#04x} status={status} {len(data)} data bytes")


if __name__ == "__main__":
    main()
