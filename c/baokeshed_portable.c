#include "baokeshed_impl.h"

INLINE void g(uint32_t *state, size_t a, size_t b, size_t c, size_t d,
              uint32_t x, uint32_t y) {
  state[a] = state[a] + state[b] + x;
  state[d] = rotr32(state[d] ^ state[a], 16);
  state[c] = state[c] + state[d];
  state[b] = rotr32(state[b] ^ state[c], 12);
  state[a] = state[a] + state[b] + y;
  state[d] = rotr32(state[d] ^ state[a], 8);
  state[c] = state[c] + state[d];
  state[b] = rotr32(state[b] ^ state[c], 7);
}

INLINE void round_fn(uint32_t *state, const uint32_t *msg, size_t round) {
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

void compress(uint32_t state[8], const uint8_t block[BLOCK_LEN],
              uint8_t block_len, uint64_t offset, uint8_t internal_flags,
              uint32_t context) {
  uint32_t block_words[16];
  load_msg_words(block, block_words);

  uint32_t full_state[16] = {
      state[0],
      state[1],
      state[2],
      state[3],
      state[4],
      state[5],
      state[6],
      state[7],
      IV[0],
      IV[1],
      IV[2],
      IV[3],
      IV[4] ^ offset_low(offset),
      IV[5] ^ offset_high(offset),
      IV[6] ^ block_flags(block_len, internal_flags),
      IV[7] ^ context,
  };

  round_fn(&full_state[0], &block_words[0], 0);
  round_fn(&full_state[0], &block_words[0], 1);
  round_fn(&full_state[0], &block_words[0], 2);
  round_fn(&full_state[0], &block_words[0], 3);
  round_fn(&full_state[0], &block_words[0], 4);
  round_fn(&full_state[0], &block_words[0], 5);
  round_fn(&full_state[0], &block_words[0], 6);

  state[0] = full_state[0] ^ full_state[8];
  state[1] = full_state[1] ^ full_state[9];
  state[2] = full_state[2] ^ full_state[10];
  state[3] = full_state[3] ^ full_state[11];
  state[4] = full_state[4] ^ full_state[12];
  state[5] = full_state[5] ^ full_state[13];
  state[6] = full_state[6] ^ full_state[14];
  state[7] = full_state[7] ^ full_state[15];
}