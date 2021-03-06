#include "baokeshed64_impl.h"
#include <string.h>

INLINE void g(uint64_t *state, size_t a, size_t b, size_t c, size_t d,
              uint64_t x, uint64_t y) {
  state[a] = state[a] + state[b] + x;
  state[d] = rotr64(state[d] ^ state[a], 32);
  state[c] = state[c] + state[d];
  state[b] = rotr64(state[b] ^ state[c], 24);
  state[a] = state[a] + state[b] + y;
  state[d] = rotr64(state[d] ^ state[a], 16);
  state[c] = state[c] + state[d];
  state[b] = rotr64(state[b] ^ state[c], 63);
}

INLINE void round_fn(uint64_t *state, const uint64_t *msg, size_t round) {
  // Select the message schedule based on the round.
  const uint8_t *schedule = MSG_SCHEDULE[round];

  // Mix the columns.
  g(state, 0, 4, 8, 12, msg[schedule[0]], msg[schedule[1]]);
  g(state, 1, 5, 9, 13, msg[schedule[2]], msg[schedule[3]]);
  g(state, 2, 6, 10, 14, msg[schedule[4]], msg[schedule[5]]);
  g(state, 3, 7, 11, 15, msg[schedule[6]], msg[schedule[7]]);

  // Mix the rows.
  g(state, 0, 5, 10, 15, msg[schedule[8]], msg[schedule[9]]);
  g(state, 1, 6, 11, 12, msg[schedule[10]], msg[schedule[11]]);
  g(state, 2, 7, 8, 13, msg[schedule[12]], msg[schedule[13]]);
  g(state, 3, 4, 9, 14, msg[schedule[14]], msg[schedule[15]]);
}

void baokeshed64_compress_portable(uint64_t state[4],
                                   const uint8_t block[BLOCK_LEN],
                                   uint8_t block_len, uint64_t offset,
                                   uint8_t flags) {
  uint64_t block_words[16];
  load_msg_words(block, block_words); // This handles big-endianness.

  uint64_t full_state[16] = {
      state[0],        state[1], state[2], state[3], IV[4],
      IV[5],           IV[6],    IV[7],    IV[0],    IV[1],
      IV[2],           IV[3],    offset,   IV[5],    (uint64_t)block_len,
      (uint64_t)flags,
  };

  round_fn(full_state, block_words, 0);
  round_fn(full_state, block_words, 1);
  round_fn(full_state, block_words, 2);
  round_fn(full_state, block_words, 3);
  round_fn(full_state, block_words, 4);
  round_fn(full_state, block_words, 5);
  round_fn(full_state, block_words, 6);
  round_fn(full_state, block_words, 7);

  state[0] = full_state[0] ^ full_state[4];
  state[1] = full_state[1] ^ full_state[5];
  state[2] = full_state[2] ^ full_state[6];
  state[3] = full_state[3] ^ full_state[7];
}

INLINE void hash_one_portable(const uint8_t *input, size_t blocks,
                              const uint64_t key_words[4], uint64_t offset,
                              uint8_t flags, uint8_t flags_start,
                              uint8_t flags_end, uint8_t out[OUT_LEN]) {
  uint64_t state[4];
  memcpy(state, key_words, KEY_LEN);
  uint8_t block_flags = flags | flags_start;
  while (blocks > 0) {
    if (blocks == 1) {
      block_flags |= flags_end;
    }
    baokeshed64_compress_portable(state, input, BLOCK_LEN, offset, block_flags);
    input = &input[BLOCK_LEN];
    blocks -= 1;
    block_flags = flags;
  }
  write_state_bytes(state, out); // This handles big-endianness.
}

void baokeshed64_hash_many_portable(const uint8_t *const *inputs,
                                    size_t num_inputs, size_t blocks,
                                    const uint64_t key_words[4],
                                    uint64_t offset,
                                    const uint64_t offset_deltas[2],
                                    uint8_t flags, uint8_t flags_start,
                                    uint8_t flags_end, uint8_t *out) {
  while (num_inputs > 0) {
    hash_one_portable(inputs[0], blocks, key_words, offset, flags, flags_start,
                      flags_end, out);
    inputs += 1;
    num_inputs -= 1;
    offset += offset_deltas[1];
    out = &out[OUT_LEN];
  }
}
