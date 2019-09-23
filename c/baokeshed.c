// NB: This is only for benchmarking. The guy who wrote this file hasn't
// touched C since college. Please don't use this code in production.

#include <stddef.h>
#include <stdint.h>

static const uint32_t IV[8] = {0x6A09E667UL, 0xBB67AE85UL, 0x3C6EF372UL,
                               0xA54FF53AUL, 0x510E527FUL, 0x9B05688CUL,
                               0x1F83D9ABUL, 0x5BE0CD19UL};

static const uint8_t MSG_SCHEDULE[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
};

static inline uint32_t load32(const void *src) {
  const uint8_t *p = (const uint8_t *)src;
  return ((uint32_t)(p[0]) << 0) | ((uint32_t)(p[1]) << 8) |
         ((uint32_t)(p[2]) << 16) | ((uint32_t)(p[3]) << 24);
}

static inline void store32(void *dst, uint32_t w) {
  uint8_t *p = (uint8_t *)dst;
  p[0] = (uint8_t)(w >> 0);
  p[1] = (uint8_t)(w >> 8);
  p[2] = (uint8_t)(w >> 16);
  p[3] = (uint8_t)(w >> 24);
}

static inline uint32_t rotr32(uint32_t w, uint32_t c) {
  return (w >> c) | (w << (32 - c));
}

static inline void g(uint32_t *state, size_t a, size_t b, size_t c, size_t d,
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

static inline void round_fn(uint32_t *state, const uint32_t *msg,
                            size_t round) {
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

static inline uint32_t offset_low(uint64_t offset) { return (uint32_t)offset; }

static inline uint32_t offset_high(uint64_t offset) {
  return (uint32_t)(offset >> 32);
}

static inline uint32_t block_flags(uint8_t block_len, uint8_t internal_flags) {
  // The lower bits of the block flags word are the block length. The higher
  // bits are the internal flags (ROOT etc.). The middle bits are currently
  // unused.
  return ((uint32_t)block_len) | (((uint32_t)internal_flags) << 24);
}

void compress(uint32_t *state, const uint8_t *block, uint8_t block_len,
              uint64_t offset, uint8_t internal_flags, uint32_t context) {
  uint32_t block_words[16];
  block_words[0] = load32(block + 4 * 0);
  block_words[1] = load32(block + 4 * 1);
  block_words[2] = load32(block + 4 * 2);
  block_words[3] = load32(block + 4 * 3);
  block_words[4] = load32(block + 4 * 4);
  block_words[5] = load32(block + 4 * 5);
  block_words[6] = load32(block + 4 * 6);
  block_words[7] = load32(block + 4 * 7);
  block_words[8] = load32(block + 4 * 8);
  block_words[9] = load32(block + 4 * 9);
  block_words[10] = load32(block + 4 * 10);
  block_words[11] = load32(block + 4 * 11);
  block_words[12] = load32(block + 4 * 12);
  block_words[13] = load32(block + 4 * 13);
  block_words[14] = load32(block + 4 * 14);
  block_words[15] = load32(block + 4 * 15);

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
