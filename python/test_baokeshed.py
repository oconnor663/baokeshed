import baokeshed
import binascii
from duct import cmd
from pathlib import Path
from threading import Lock

_BAOKESHED_RUST_MUTEX = Lock()
_BAOKESHED_RUST_PATH = None

CASES = [
    0,
    1,
    baokeshed.BLOCK_LEN - 1,
    baokeshed.BLOCK_LEN,
    baokeshed.BLOCK_LEN + 1,
    baokeshed.CHUNK_LEN - 1,
    baokeshed.CHUNK_LEN,
    baokeshed.CHUNK_LEN + 1,
    2 * baokeshed.CHUNK_LEN - 1,
    2 * baokeshed.CHUNK_LEN,
    2 * baokeshed.CHUNK_LEN + 1,
    3 * baokeshed.CHUNK_LEN,
    4 * baokeshed.CHUNK_LEN,
    5 * baokeshed.CHUNK_LEN,
]


def baokeshed_path():
    global _BAOKESHED_RUST_MUTEX, _BAOKESHED_RUST_PATH
    with _BAOKESHED_RUST_MUTEX:
        if _BAOKESHED_RUST_PATH is None:
            bin_root = Path(__file__).parent / "../baokeshed_bin"
            cmd("cargo", "build", "--quiet").dir(bin_root).run()
            _BAOKESHED_RUST_PATH = bin_root / "target/debug/baokeshed"
        return _BAOKESHED_RUST_PATH


# Matches test.rs::paint_test_input().
def make_test_input(length):
    buf = bytearray()
    counter = 1
    while len(buf) < length:
        counter_bytes = counter.to_bytes(4, "little")
        take = min(4, length - len(buf))
        buf.extend(counter_bytes[:take])
        counter += 1
    return buf


def to_hex(b):
    return binascii.hexlify(b).decode("utf-8")


def test_default_functions():
    key_bytes = make_test_input(baokeshed.KEY_LEN)
    key_hex = to_hex(key_bytes)
    for case in CASES:
        print("case:", case)
        input_bytes = make_test_input(case)

        # The default hash function.
        rust_output = cmd(baokeshed_path()).stdin_bytes(input_bytes).read()
        python_output = to_hex(baokeshed.hash(input_bytes))
        assert rust_output == python_output

        # The keyed hash function.
        rust_output = cmd(baokeshed_path(), "--key",
                          key_hex).stdin_bytes(input_bytes).read()
        python_output = to_hex(baokeshed.keyed_hash(input_bytes, key_bytes))
        assert rust_output == python_output

        # The KDF.
        rust_output = cmd(baokeshed_path(), "--derive-key",
                          key_hex).stdin_bytes(input_bytes).read()
        python_output = to_hex(baokeshed.derive_key(key_bytes, input_bytes))
        assert rust_output == python_output
