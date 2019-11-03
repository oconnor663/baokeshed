#include "baokeshed64_impl.h"

#if defined(__ARM_NEON)

#include <arm_neon.h>

// TODO: This is probably incorrect for big-endian ARM. How should that work?
INLINE uint64x2_t loadu_128(const uint8_t src[16]) {
  return vld1q_u64((const uint64_t *)src);
}

INLINE void storeu_128(uint64x2_t src, uint8_t dest[16]) {
  vst1q_u64((uint64_t *)dest, src);
}

INLINE uint64x2_t add_128(uint64x2_t a, uint64x2_t b) {
  return vaddq_u64(a, b);
}

INLINE uint64x2_t xor_128(uint64x2_t a, uint64x2_t b) {
  return veorq_u64(a, b);
}

INLINE uint64x2_t set1_128(uint64_t x) { return vld1q_dup_u64(&x); }

INLINE uint64x2_t set2(uint64_t a, uint64_t b) {
  uint64_t array[2] = {a, b};
  return vld1q_u64(array);
}

INLINE uint64x2_t rot32_128(uint64x2_t x) {
  return vorrq_u64(vshrq_n_u64(x, 32), vshlq_n_u64(x, 64 - 32));
}

INLINE uint64x2_t rot24_128(uint64x2_t x) {
  return vorrq_u64(vshrq_n_u64(x, 24), vshlq_n_u64(x, 64 - 24));
}

INLINE uint64x2_t rot16_128(uint64x2_t x) {
  return vorrq_u64(vshrq_n_u64(x, 16), vshlq_n_u64(x, 64 - 16));
}

INLINE uint64x2_t rot63_128(uint64x2_t x) {
  return vorrq_u64(vshrq_n_u64(x, 63), vshlq_n_u64(x, 64 - 63));
}

// TODO: compress_neon

// TODO: hash2_neon

/*
 * ----------------------------------------------------------------------------
 * hash2_neon
 * ----------------------------------------------------------------------------
 */

INLINE void round_fn2(uint64x2_t v[16], uint64x2_t m[16], size_t r) {
  v[0] = add_128(v[0], m[(size_t)MSG_SCHEDULE[r][0]]);
  v[1] = add_128(v[1], m[(size_t)MSG_SCHEDULE[r][2]]);
  v[2] = add_128(v[2], m[(size_t)MSG_SCHEDULE[r][4]]);
  v[3] = add_128(v[3], m[(size_t)MSG_SCHEDULE[r][6]]);
  v[0] = add_128(v[0], v[4]);
  v[1] = add_128(v[1], v[5]);
  v[2] = add_128(v[2], v[6]);
  v[3] = add_128(v[3], v[7]);
  v[12] = xor_128(v[12], v[0]);
  v[13] = xor_128(v[13], v[1]);
  v[14] = xor_128(v[14], v[2]);
  v[15] = xor_128(v[15], v[3]);
  v[12] = rot32_128(v[12]);
  v[13] = rot32_128(v[13]);
  v[14] = rot32_128(v[14]);
  v[15] = rot32_128(v[15]);
  v[8] = add_128(v[8], v[12]);
  v[9] = add_128(v[9], v[13]);
  v[10] = add_128(v[10], v[14]);
  v[11] = add_128(v[11], v[15]);
  v[4] = xor_128(v[4], v[8]);
  v[5] = xor_128(v[5], v[9]);
  v[6] = xor_128(v[6], v[10]);
  v[7] = xor_128(v[7], v[11]);
  v[4] = rot24_128(v[4]);
  v[5] = rot24_128(v[5]);
  v[6] = rot24_128(v[6]);
  v[7] = rot24_128(v[7]);
  v[0] = add_128(v[0], m[(size_t)MSG_SCHEDULE[r][1]]);
  v[1] = add_128(v[1], m[(size_t)MSG_SCHEDULE[r][3]]);
  v[2] = add_128(v[2], m[(size_t)MSG_SCHEDULE[r][5]]);
  v[3] = add_128(v[3], m[(size_t)MSG_SCHEDULE[r][7]]);
  v[0] = add_128(v[0], v[4]);
  v[1] = add_128(v[1], v[5]);
  v[2] = add_128(v[2], v[6]);
  v[3] = add_128(v[3], v[7]);
  v[12] = xor_128(v[12], v[0]);
  v[13] = xor_128(v[13], v[1]);
  v[14] = xor_128(v[14], v[2]);
  v[15] = xor_128(v[15], v[3]);
  v[12] = rot16_128(v[12]);
  v[13] = rot16_128(v[13]);
  v[14] = rot16_128(v[14]);
  v[15] = rot16_128(v[15]);
  v[8] = add_128(v[8], v[12]);
  v[9] = add_128(v[9], v[13]);
  v[10] = add_128(v[10], v[14]);
  v[11] = add_128(v[11], v[15]);
  v[4] = xor_128(v[4], v[8]);
  v[5] = xor_128(v[5], v[9]);
  v[6] = xor_128(v[6], v[10]);
  v[7] = xor_128(v[7], v[11]);
  v[4] = rot63_128(v[4]);
  v[5] = rot63_128(v[5]);
  v[6] = rot63_128(v[6]);
  v[7] = rot63_128(v[7]);

  v[0] = add_128(v[0], m[(size_t)MSG_SCHEDULE[r][8]]);
  v[1] = add_128(v[1], m[(size_t)MSG_SCHEDULE[r][10]]);
  v[2] = add_128(v[2], m[(size_t)MSG_SCHEDULE[r][12]]);
  v[3] = add_128(v[3], m[(size_t)MSG_SCHEDULE[r][14]]);
  v[0] = add_128(v[0], v[5]);
  v[1] = add_128(v[1], v[6]);
  v[2] = add_128(v[2], v[7]);
  v[3] = add_128(v[3], v[4]);
  v[15] = xor_128(v[15], v[0]);
  v[12] = xor_128(v[12], v[1]);
  v[13] = xor_128(v[13], v[2]);
  v[14] = xor_128(v[14], v[3]);
  v[15] = rot32_128(v[15]);
  v[12] = rot32_128(v[12]);
  v[13] = rot32_128(v[13]);
  v[14] = rot32_128(v[14]);
  v[10] = add_128(v[10], v[15]);
  v[11] = add_128(v[11], v[12]);
  v[8] = add_128(v[8], v[13]);
  v[9] = add_128(v[9], v[14]);
  v[5] = xor_128(v[5], v[10]);
  v[6] = xor_128(v[6], v[11]);
  v[7] = xor_128(v[7], v[8]);
  v[4] = xor_128(v[4], v[9]);
  v[5] = rot24_128(v[5]);
  v[6] = rot24_128(v[6]);
  v[7] = rot24_128(v[7]);
  v[4] = rot24_128(v[4]);
  v[0] = add_128(v[0], m[(size_t)MSG_SCHEDULE[r][9]]);
  v[1] = add_128(v[1], m[(size_t)MSG_SCHEDULE[r][11]]);
  v[2] = add_128(v[2], m[(size_t)MSG_SCHEDULE[r][13]]);
  v[3] = add_128(v[3], m[(size_t)MSG_SCHEDULE[r][15]]);
  v[0] = add_128(v[0], v[5]);
  v[1] = add_128(v[1], v[6]);
  v[2] = add_128(v[2], v[7]);
  v[3] = add_128(v[3], v[4]);
  v[15] = xor_128(v[15], v[0]);
  v[12] = xor_128(v[12], v[1]);
  v[13] = xor_128(v[13], v[2]);
  v[14] = xor_128(v[14], v[3]);
  v[15] = rot16_128(v[15]);
  v[12] = rot16_128(v[12]);
  v[13] = rot16_128(v[13]);
  v[14] = rot16_128(v[14]);
  v[10] = add_128(v[10], v[15]);
  v[11] = add_128(v[11], v[12]);
  v[8] = add_128(v[8], v[13]);
  v[9] = add_128(v[9], v[14]);
  v[5] = xor_128(v[5], v[10]);
  v[6] = xor_128(v[6], v[11]);
  v[7] = xor_128(v[7], v[8]);
  v[4] = xor_128(v[4], v[9]);
  v[5] = rot63_128(v[5]);
  v[6] = rot63_128(v[6]);
  v[7] = rot63_128(v[7]);
  v[4] = rot63_128(v[4]);
}

INLINE void transpose_vecs_128(uint64x2_t vecs[2]) {
  uint64_t row0[2];
  uint64_t row1[2];
  storeu_128(vecs[0], (uint8_t *)row0);
  storeu_128(vecs[1], (uint8_t *)row1);
  vecs[0] = set2(row0[0], row1[0]);
  vecs[1] = set2(row0[1], row1[1]);
}

INLINE void transpose_msg_vecs2(const uint8_t *const *inputs,
                                size_t block_offset, uint64x2_t out[16]) {
  out[0] = loadu_128(&inputs[0][block_offset + 0 * sizeof(uint64x2_t)]);
  out[1] = loadu_128(&inputs[1][block_offset + 0 * sizeof(uint64x2_t)]);
  out[2] = loadu_128(&inputs[0][block_offset + 1 * sizeof(uint64x2_t)]);
  out[3] = loadu_128(&inputs[1][block_offset + 1 * sizeof(uint64x2_t)]);
  out[4] = loadu_128(&inputs[0][block_offset + 2 * sizeof(uint64x2_t)]);
  out[5] = loadu_128(&inputs[1][block_offset + 2 * sizeof(uint64x2_t)]);
  out[6] = loadu_128(&inputs[0][block_offset + 3 * sizeof(uint64x2_t)]);
  out[7] = loadu_128(&inputs[1][block_offset + 3 * sizeof(uint64x2_t)]);
  out[8] = loadu_128(&inputs[0][block_offset + 4 * sizeof(uint64x2_t)]);
  out[9] = loadu_128(&inputs[1][block_offset + 4 * sizeof(uint64x2_t)]);
  out[10] = loadu_128(&inputs[0][block_offset + 5 * sizeof(uint64x2_t)]);
  out[11] = loadu_128(&inputs[1][block_offset + 5 * sizeof(uint64x2_t)]);
  out[12] = loadu_128(&inputs[0][block_offset + 6 * sizeof(uint64x2_t)]);
  out[13] = loadu_128(&inputs[1][block_offset + 6 * sizeof(uint64x2_t)]);
  out[14] = loadu_128(&inputs[0][block_offset + 7 * sizeof(uint64x2_t)]);
  out[15] = loadu_128(&inputs[1][block_offset + 7 * sizeof(uint64x2_t)]);
  transpose_vecs_128(&out[0]);
  transpose_vecs_128(&out[2]);
  transpose_vecs_128(&out[4]);
  transpose_vecs_128(&out[6]);
  transpose_vecs_128(&out[8]);
  transpose_vecs_128(&out[10]);
  transpose_vecs_128(&out[12]);
  transpose_vecs_128(&out[14]);
}

INLINE void load_offsets2(uint64_t offset, const uint64_t deltas[2],
                          uint64x2_t *out) {
  *out = set2(offset + deltas[0], offset + deltas[1]);
}

void baokeshed64_hash2_neon(const uint8_t *const *inputs, size_t blocks,
                            const uint64_t key_words[4], uint64_t offset,
                            const uint64_t offset_deltas[2], uint8_t flags,
                            uint8_t flags_start, uint8_t flags_end,
                            uint8_t *out) {
  uint64x2_t h_vecs[4] = {
      set1_128(key_words[0]),
      set1_128(key_words[1]),
      set1_128(key_words[2]),
      set1_128(key_words[3]),
  };
  uint64x2_t offset_vec;
  load_offsets2(offset, offset_deltas, &offset_vec);
  uint8_t block_flags = flags | flags_start;

  for (size_t block = 0; block < blocks; block++) {
    if (block + 1 == blocks) {
      block_flags |= flags_end;
    }
    uint64x2_t block_len_vec = set1_128(BLOCK_LEN);
    uint64x2_t block_flags_vec = set1_128(block_flags);
    uint64x2_t msg_vecs[16];
    transpose_msg_vecs2(inputs, block * BLOCK_LEN, msg_vecs);

    uint64x2_t v[16] = {
        h_vecs[0],
        h_vecs[1],
        h_vecs[2],
        h_vecs[3],
        set1_128(IV[4]),
        set1_128(IV[5]),
        set1_128(IV[6]),
        set1_128(IV[7]),
        set1_128(IV[0]),
        set1_128(IV[1]),
        set1_128(IV[2]),
        set1_128(IV[3]),
        xor_128(set1_128(IV[4]), offset_vec),
        set1_128(IV[5]),
        xor_128(set1_128(IV[6]), block_len_vec),
        xor_128(set1_128(IV[7]), block_flags_vec),
    };
    round_fn2(v, msg_vecs, 0);
    round_fn2(v, msg_vecs, 1);
    round_fn2(v, msg_vecs, 2);
    round_fn2(v, msg_vecs, 3);
    round_fn2(v, msg_vecs, 4);
    round_fn2(v, msg_vecs, 5);
    round_fn2(v, msg_vecs, 6);
    round_fn2(v, msg_vecs, 7);
    h_vecs[0] = xor_128(v[0], v[4]);
    h_vecs[1] = xor_128(v[1], v[5]);
    h_vecs[2] = xor_128(v[2], v[6]);
    h_vecs[3] = xor_128(v[3], v[7]);

    block_flags = flags;
  }

  transpose_vecs_128(&h_vecs[0]);
  transpose_vecs_128(&h_vecs[2]);
  // The first two vecs now contain the first half of each output, and the
  // second two vecs contain the second half of each output.
  storeu_128(h_vecs[0], &out[0 * sizeof(uint64x2_t)]);
  storeu_128(h_vecs[2], &out[1 * sizeof(uint64x2_t)]);
  storeu_128(h_vecs[1], &out[2 * sizeof(uint64x2_t)]);
  storeu_128(h_vecs[3], &out[3 * sizeof(uint64x2_t)]);
}

/*
 * ----------------------------------------------------------------------------
 * hash_many_neon
 * ----------------------------------------------------------------------------
 */

INLINE void hash_one_neon(const uint8_t *input, size_t blocks,
                          const uint64_t key_words[4], uint64_t offset,
                          uint8_t flags, uint8_t flags_start, uint8_t flags_end,
                          uint8_t out[OUT_LEN]) {
  uint64_t state[4];
  memcpy(state, key_words, KEY_LEN);
  uint8_t block_flags = flags | flags_start;
  while (blocks > 0) {
    if (blocks == 1) {
      block_flags |= flags_end;
    }
    // TODO: use compress_neon
    baokeshed64_compress_portable(state, input, BLOCK_LEN, offset, block_flags);
    input = &input[BLOCK_LEN];
    blocks -= 1;
    block_flags = flags;
  }
  memcpy(out, state, OUT_LEN);
}

void baokeshed64_hash_many_neon(const uint8_t *const *inputs, size_t num_inputs,
                                size_t blocks, const uint64_t key_words[4],
                                uint64_t offset,
                                const uint64_t offset_deltas[17], uint8_t flags,
                                uint8_t flags_start, uint8_t flags_end,
                                uint8_t *out) {
  while (num_inputs >= 2) {
    baokeshed64_hash2_neon(inputs, blocks, key_words, offset, offset_deltas,
                           flags, flags_start, flags_end, out);
    inputs += 2;
    num_inputs -= 2;
    offset += offset_deltas[2];
    out = &out[2 * OUT_LEN];
  }
  while (num_inputs > 0) {
    hash_one_neon(inputs[0], blocks, key_words, offset, flags, flags_start,
                  flags_end, out);
    inputs += 1;
    num_inputs -= 1;
    offset += offset_deltas[1];
    out = &out[OUT_LEN];
  }
}

#else // __ARM_NEON

// NEON is only enabled statically in the build, with --features=c_armv7neon.
// (Rust's dynamic feature detection for ARM is not yet stable.) So we don't
// need to provide any stubs here.

#endif
