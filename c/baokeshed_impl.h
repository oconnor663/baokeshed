#pragma once

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if __POPCNT__
#include <nmmintrin.h>
#endif

// size constants
#define BLOCK_LEN 64
#define CHUNK_LEN 2048
#define KEY_LEN 32
#define OUT_LEN 32
#define MAX_DEPTH 53
#define MAX_SIMD_DEGREE 16

// internal flags
#define CHUNK_START 1
#define CHUNK_END 2
#define PARENT 4
#define ROOT 8
#define KEYED_HASH 16
#define DERIVE_KEY 32

// This C implementation tries to support recent versions of GCC, Clang, and
// MSVC.
#if defined(_MSC_VER)
#define INLINE __forceinline static
#else
#define INLINE __attribute__((always_inline)) static inline
#endif

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

static const uint64_t CHUNK_OFFSET_DELTAS[17] = {
    CHUNK_LEN * 0,  CHUNK_LEN * 1,  CHUNK_LEN * 2,  CHUNK_LEN * 3,
    CHUNK_LEN * 4,  CHUNK_LEN * 5,  CHUNK_LEN * 6,  CHUNK_LEN * 7,
    CHUNK_LEN * 8,  CHUNK_LEN * 9,  CHUNK_LEN * 10, CHUNK_LEN * 11,
    CHUNK_LEN * 12, CHUNK_LEN * 13, CHUNK_LEN * 14, CHUNK_LEN * 15,
    CHUNK_LEN * 16,
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
#if __POPCNT__
  return (uint8_t)_mm_popcnt_u64(x);
#else
  uint8_t count = 0;
  while (x > 0) {
    count += ((uint8_t)x) & 1;
    x >>= 1;
  }
  return count;
#endif
}

INLINE uint32_t offset_low(uint64_t offset) { return (uint32_t)offset; }

INLINE uint32_t offset_high(uint64_t offset) {
  return (uint32_t)(offset >> 32);
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

// Declarations for implementation-specific functions.
void compress_portable(uint32_t state[8], const uint8_t block[BLOCK_LEN],
                       uint8_t block_len, uint64_t offset, uint8_t flags);
void compress_sse41(uint32_t state[8], const uint8_t block[BLOCK_LEN],
                    uint8_t block_len, uint64_t offset, uint8_t flags);
void compress_avx512(uint32_t state[8], const uint8_t block[BLOCK_LEN],
                     uint8_t block_len, uint64_t offset, uint8_t flags);
void hash_many_portable(const uint8_t *const *inputs, size_t num_inputs,
                        size_t blocks, const uint32_t key_words[8],
                        uint64_t offset, const uint64_t offset_deltas[2],
                        uint8_t flags, uint8_t flags_start, uint8_t flags_end,
                        uint8_t *out);
void hash_many_sse41(const uint8_t *const *inputs, size_t num_inputs,
                     size_t blocks, const uint32_t key_words[8],
                     uint64_t offset, const uint64_t offset_deltas[5],
                     uint8_t flags, uint8_t flags_start, uint8_t flags_end,
                     uint8_t *out);
void hash_many_avx2(const uint8_t *const *inputs, size_t num_inputs,
                    size_t blocks, const uint32_t key_words[8], uint64_t offset,
                    const uint64_t offset_deltas[9], uint8_t flags,
                    uint8_t flags_start, uint8_t flags_end, uint8_t *out);
void hash_many_avx512(const uint8_t *const *inputs, size_t num_inputs,
                      size_t blocks, const uint32_t key_words[8],
                      uint64_t offset, const uint64_t offset_deltas[17],
                      uint8_t flags, uint8_t flags_start, uint8_t flags_end,
                      uint8_t *out);
void hash_many_neon(const uint8_t *const *inputs, size_t num_inputs,
                    size_t blocks, const uint32_t key_words[8], uint64_t offset,
                    const uint64_t offset_deltas[17], uint8_t flags,
                    uint8_t flags_start, uint8_t flags_end, uint8_t *out);
