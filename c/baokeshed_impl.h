#pragma once

#include <stddef.h>
#include <stdint.h>

// size constants
#define BLOCK_LEN 64
#define CHUNK_LEN 4096
#define KEY_LEN 32
#define OUT_LEN 32
#define MAX_DEPTH 52

// internal flags
#define ROOT 128
#define PARENT 64
#define CHUNK_END 32
#define CHUNK_START 16

// This isn't portable. Not production code!
#define INLINE __attribute__((always_inline)) static inline

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

INLINE uint32_t load32(const void *src) {
  const uint8_t *p = (const uint8_t *)src;
  return ((uint32_t)(p[0]) << 0) | ((uint32_t)(p[1]) << 8) |
         ((uint32_t)(p[2]) << 16) | ((uint32_t)(p[3]) << 24);
}

INLINE void store32(void *dst, uint32_t w) {
  uint8_t *p = (uint8_t *)dst;
  p[0] = (uint8_t)(w >> 0);
  p[1] = (uint8_t)(w >> 8);
  p[2] = (uint8_t)(w >> 16);
  p[3] = (uint8_t)(w >> 24);
}

INLINE uint32_t rotr32(uint32_t w, uint32_t c) {
  return (w >> c) | (w << (32 - c));
}

// Count the number of 1 bits.
INLINE uint8_t popcnt(uint64_t x) {
  uint8_t count = 0;
  while (x > 0) {
    count += ((uint8_t)x) & 1;
    x >>= 1;
  }
  return count;
}

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

INLINE uint32_t offset_low(uint64_t offset) { return (uint32_t)offset; }

INLINE uint32_t offset_high(uint64_t offset) {
  return (uint32_t)(offset >> 32);
}

INLINE uint32_t block_flags(uint8_t block_len, uint8_t internal_flags) {
  // The lower bits of the block flags word are the block length. The higher
  // bits are the internal flags (ROOT etc.). The middle bits are currently
  // unused.
  return ((uint32_t)block_len) | (((uint32_t)internal_flags) << 24);
}

INLINE void load_msg_words(const uint8_t block[BLOCK_LEN], uint32_t words[8]) {
  words[0] = load32(block + 4 * 0);
  words[1] = load32(block + 4 * 1);
  words[2] = load32(block + 4 * 2);
  words[3] = load32(block + 4 * 3);
  words[4] = load32(block + 4 * 4);
  words[5] = load32(block + 4 * 5);
  words[6] = load32(block + 4 * 6);
  words[7] = load32(block + 4 * 7);
  words[8] = load32(block + 4 * 8);
  words[9] = load32(block + 4 * 9);
  words[10] = load32(block + 4 * 10);
  words[11] = load32(block + 4 * 11);
  words[12] = load32(block + 4 * 12);
  words[13] = load32(block + 4 * 13);
  words[14] = load32(block + 4 * 14);
  words[15] = load32(block + 4 * 15);
}

INLINE void load_key_words(const uint8_t key[KEY_LEN], uint32_t words[8]) {
  words[0] = load32(key + 4 * 0);
  words[1] = load32(key + 4 * 1);
  words[2] = load32(key + 4 * 2);
  words[3] = load32(key + 4 * 3);
  words[4] = load32(key + 4 * 4);
  words[5] = load32(key + 4 * 5);
  words[6] = load32(key + 4 * 6);
  words[7] = load32(key + 4 * 7);
}

INLINE void write_state_bytes(const uint32_t state[8], uint8_t out[OUT_LEN]) {
  store32(&out[4 * 0], state[0]);
  store32(&out[4 * 1], state[1]);
  store32(&out[4 * 2], state[2]);
  store32(&out[4 * 3], state[3]);
  store32(&out[4 * 4], state[4]);
  store32(&out[4 * 5], state[5]);
  store32(&out[4 * 6], state[6]);
  store32(&out[4 * 7], state[7]);
}

INLINE void init_iv(const uint32_t key_words[8], uint32_t state[8]) {
  state[0] = IV[0] ^ key_words[0];
  state[1] = IV[1] ^ key_words[1];
  state[2] = IV[2] ^ key_words[2];
  state[3] = IV[3] ^ key_words[3];
  state[4] = IV[4] ^ key_words[4];
  state[5] = IV[5] ^ key_words[5];
  state[6] = IV[6] ^ key_words[6];
  state[7] = IV[7] ^ key_words[7];
}
