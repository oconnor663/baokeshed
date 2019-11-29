import baokeshed
import binascii
import hashlib
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


def test_compare_to_rust():
    key_bytes = make_test_input(baokeshed.KEY_LEN)
    key_hex = to_hex(key_bytes)
    # Use a length that's not a multiple of 4.
    output_len = 303
    for case in CASES:
        print("case:", case)
        input_bytes = make_test_input(case)

        # The default hash function.
        rust_output = cmd(baokeshed_path(), "--length",
                          str(output_len)).stdin_bytes(input_bytes).read()
        python_output = to_hex(baokeshed.hash(input_bytes, output_len))
        assert rust_output == python_output

        # The keyed hash function.
        rust_output = cmd(baokeshed_path(), "--length", str(output_len),
                          "--key", key_hex).stdin_bytes(input_bytes).read()
        python_output = to_hex(
            baokeshed.keyed_hash(key_bytes, input_bytes, output_len))
        assert rust_output == python_output

        # The KDF.
        rust_output = cmd(baokeshed_path(), "--length", str(output_len),
                          "--derive-key",
                          key_hex).stdin_bytes(input_bytes).read()
        python_output = to_hex(
            baokeshed.derive_key(key_bytes, input_bytes, output_len))
        assert rust_output == python_output


def test_round_fn_compatible_with_blake2s():
    input_bytes = b"hello world"
    hello_world_hash = \
        "9aec6806794561107e594b1f6a8a6b0c92a0cba9acf5e5e93cca06f781813b0b"
    assert hello_world_hash == hashlib.blake2s(input_bytes).hexdigest()
    block_len = len(input_bytes)
    input_buffer = bytearray(64)
    input_buffer[0:block_len] = input_bytes
    input_words = baokeshed.words_from_bytes(input_buffer)
    schedules = baokeshed.MSG_SCHEDULE + [
        [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
        [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
        [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    ]

    cv = [
        # hash_length = 32, fanout = 1, max_depth = 1
        baokeshed.IV[0] ^ 32 ^ (1 << 16) ^ (1 << 24),
        baokeshed.IV[1],
        baokeshed.IV[2],
        baokeshed.IV[3],
        baokeshed.IV[4],
        baokeshed.IV[5],
        baokeshed.IV[6],
        baokeshed.IV[7],
    ]

    state = cv + [
        baokeshed.IV[0],
        baokeshed.IV[1],
        baokeshed.IV[2],
        baokeshed.IV[3],
        # total bytes compressed = block_len (no set bits in the second word)
        baokeshed.IV[4] ^ block_len,
        baokeshed.IV[5],
        # last block flag
        baokeshed.IV[6] ^ (2**32 - 1),
        baokeshed.IV[7],
    ]

    for round_number in range(10):
        baokeshed.round(state, input_words, schedules[round_number])

    output_words = [state[i] ^ state[i + 8] ^ cv[i] for i in range(8)]
    output_bytes = baokeshed.bytes_from_words(output_words)
    assert hello_world_hash == to_hex(output_bytes)
