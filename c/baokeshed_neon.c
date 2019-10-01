#include "baokeshed_impl.h"

#if defined(__ARM_NEON)

#include <arm_neon.h>

// TODO: This is probably incorrect for big-endian ARM. How should that work?
INLINE uint32x4_t loadu_128(const uint8_t src[16]) {
  return vld1q_u32((const uint32_t *)src);
}

INLINE void storeu_128(uint32x4_t src, uint8_t dest[16]) {
  vst1q_u32((uint32_t *)dest, src);
}

INLINE uint32x4_t add_128(uint32x4_t a, uint32x4_t b) {
  return vaddq_u32(a, b);
}

INLINE uint32x4_t xor_128(uint32x4_t a, uint32x4_t b) {
  return veorq_u32(a, b);
}

INLINE uint32x4_t set1_128(uint32_t x) { return vld1q_dup_u32(&x); }

INLINE uint32x4_t set4(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  uint32_t array[4] = {a, b, c, d};
  return vld1q_u32(array);
}

INLINE uint32x4_t rot16_128(uint32x4_t x) {
  return vorrq_u32(vshrq_n_u32(x, 16), vshlq_n_u32(x, 32 - 16));
}

INLINE uint32x4_t rot12_128(uint32x4_t x) {
  return vorrq_u32(vshrq_n_u32(x, 12), vshlq_n_u32(x, 32 - 12));
}

INLINE uint32x4_t rot8_128(uint32x4_t x) {
  return vorrq_u32(vshrq_n_u32(x, 8), vshlq_n_u32(x, 32 - 8));
}

INLINE uint32x4_t rot7_128(uint32x4_t x) {
  return vorrq_u32(vshrq_n_u32(x, 7), vshlq_n_u32(x, 32 - 7));
}

// TODO: compress_neon

// TODO: hash2_neon

/*
 * ----------------------------------------------------------------------------
 * hash4_neon
 * ----------------------------------------------------------------------------
 */

INLINE void round_fn4(uint32x4_t v[16], uint32x4_t m[16], size_t r) {
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
  v[4] = rot12_128(v[4]);
  v[5] = rot12_128(v[5]);
  v[6] = rot12_128(v[6]);
  v[7] = rot12_128(v[7]);
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
  v[12] = rot8_128(v[12]);
  v[13] = rot8_128(v[13]);
  v[14] = rot8_128(v[14]);
  v[15] = rot8_128(v[15]);
  v[8] = add_128(v[8], v[12]);
  v[9] = add_128(v[9], v[13]);
  v[10] = add_128(v[10], v[14]);
  v[11] = add_128(v[11], v[15]);
  v[4] = xor_128(v[4], v[8]);
  v[5] = xor_128(v[5], v[9]);
  v[6] = xor_128(v[6], v[10]);
  v[7] = xor_128(v[7], v[11]);
  v[4] = rot7_128(v[4]);
  v[5] = rot7_128(v[5]);
  v[6] = rot7_128(v[6]);
  v[7] = rot7_128(v[7]);

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
  v[5] = rot12_128(v[5]);
  v[6] = rot12_128(v[6]);
  v[7] = rot12_128(v[7]);
  v[4] = rot12_128(v[4]);
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
  v[15] = rot8_128(v[15]);
  v[12] = rot8_128(v[12]);
  v[13] = rot8_128(v[13]);
  v[14] = rot8_128(v[14]);
  v[10] = add_128(v[10], v[15]);
  v[11] = add_128(v[11], v[12]);
  v[8] = add_128(v[8], v[13]);
  v[9] = add_128(v[9], v[14]);
  v[5] = xor_128(v[5], v[10]);
  v[6] = xor_128(v[6], v[11]);
  v[7] = xor_128(v[7], v[8]);
  v[4] = xor_128(v[4], v[9]);
  v[5] = rot7_128(v[5]);
  v[6] = rot7_128(v[6]);
  v[7] = rot7_128(v[7]);
  v[4] = rot7_128(v[4]);
}

INLINE void transpose_vecs_128(uint32x4_t vecs[4]) {
  // Individually transpose the four 2x2 sub-matrices in each corner.
  uint32x4x2_t rows01 = vtrnq_u32(vecs[0], vecs[1]);
  uint32x4x2_t rows23 = vtrnq_u32(vecs[2], vecs[3]);

  // Swap the top-right and bottom-left 2x2s (which just got transposed).
  vecs[0] =
      vcombine_u32(vget_low_u32(rows01.val[0]), vget_low_u32(rows23.val[0]));
  vecs[1] =
      vcombine_u32(vget_low_u32(rows01.val[1]), vget_low_u32(rows23.val[1]));
  vecs[2] =
      vcombine_u32(vget_high_u32(rows01.val[0]), vget_high_u32(rows23.val[0]));
  vecs[3] =
      vcombine_u32(vget_high_u32(rows01.val[1]), vget_high_u32(rows23.val[1]));
}

INLINE void transpose_msg_vecs4(const uint8_t *const *inputs,
                                size_t block_offset, uint32x4_t out[16]) {
  out[0] = loadu_128(&inputs[0][block_offset + 0 * sizeof(uint32x4_t)]);
  out[1] = loadu_128(&inputs[1][block_offset + 0 * sizeof(uint32x4_t)]);
  out[2] = loadu_128(&inputs[2][block_offset + 0 * sizeof(uint32x4_t)]);
  out[3] = loadu_128(&inputs[3][block_offset + 0 * sizeof(uint32x4_t)]);
  out[4] = loadu_128(&inputs[0][block_offset + 1 * sizeof(uint32x4_t)]);
  out[5] = loadu_128(&inputs[1][block_offset + 1 * sizeof(uint32x4_t)]);
  out[6] = loadu_128(&inputs[2][block_offset + 1 * sizeof(uint32x4_t)]);
  out[7] = loadu_128(&inputs[3][block_offset + 1 * sizeof(uint32x4_t)]);
  out[8] = loadu_128(&inputs[0][block_offset + 2 * sizeof(uint32x4_t)]);
  out[9] = loadu_128(&inputs[1][block_offset + 2 * sizeof(uint32x4_t)]);
  out[10] = loadu_128(&inputs[2][block_offset + 2 * sizeof(uint32x4_t)]);
  out[11] = loadu_128(&inputs[3][block_offset + 2 * sizeof(uint32x4_t)]);
  out[12] = loadu_128(&inputs[0][block_offset + 3 * sizeof(uint32x4_t)]);
  out[13] = loadu_128(&inputs[1][block_offset + 3 * sizeof(uint32x4_t)]);
  out[14] = loadu_128(&inputs[2][block_offset + 3 * sizeof(uint32x4_t)]);
  out[15] = loadu_128(&inputs[3][block_offset + 3 * sizeof(uint32x4_t)]);
  transpose_vecs_128(&out[0]);
  transpose_vecs_128(&out[4]);
  transpose_vecs_128(&out[8]);
  transpose_vecs_128(&out[12]);
}

INLINE void load_offsets4(uint64_t offset, const uint64_t deltas[4],
                          uint32x4_t *out_lo, uint32x4_t *out_hi) {
  *out_lo =
      set4(offset_low(offset + deltas[0]), offset_low(offset + deltas[1]),
           offset_low(offset + deltas[2]), offset_low(offset + deltas[3]));
  *out_hi =
      set4(offset_high(offset + deltas[0]), offset_high(offset + deltas[1]),
           offset_high(offset + deltas[2]), offset_high(offset + deltas[3]));
}

void hash4_neon(const uint8_t *const *inputs, size_t blocks,
                const uint32_t key_words[8], uint64_t offset,
                const uint64_t offset_deltas[4], uint8_t internal_flags_start,
                uint8_t internal_flags_end, uint32_t context, uint8_t *out) {
  uint32x4_t h_vecs[8] = {
      xor_128(set1_128(IV[0]), set1_128(key_words[0])),
      xor_128(set1_128(IV[1]), set1_128(key_words[1])),
      xor_128(set1_128(IV[2]), set1_128(key_words[2])),
      xor_128(set1_128(IV[3]), set1_128(key_words[3])),
      xor_128(set1_128(IV[4]), set1_128(key_words[4])),
      xor_128(set1_128(IV[5]), set1_128(key_words[5])),
      xor_128(set1_128(IV[6]), set1_128(key_words[6])),
      xor_128(set1_128(IV[7]), set1_128(key_words[7])),
  };
  uint32x4_t offset_low_vec, offset_high_vec;
  load_offsets4(offset, offset_deltas, &offset_low_vec, &offset_high_vec);
  const uint32x4_t context_vec = set1_128(context);
  uint8_t internal_flags = internal_flags_start;

  for (size_t block = 0; block < blocks; block++) {
    if (block + 1 == blocks) {
      internal_flags |= internal_flags_end;
    }
    uint32x4_t block_flags_vec =
        set1_128(block_flags(BLOCK_LEN, internal_flags));
    uint32x4_t msg_vecs[16];
    transpose_msg_vecs4(inputs, block * BLOCK_LEN, msg_vecs);

    uint32x4_t v[16] = {
        h_vecs[0],
        h_vecs[1],
        h_vecs[2],
        h_vecs[3],
        h_vecs[4],
        h_vecs[5],
        h_vecs[6],
        h_vecs[7],
        set1_128(IV[0]),
        set1_128(IV[1]),
        set1_128(IV[2]),
        set1_128(IV[3]),
        xor_128(set1_128(IV[4]), offset_low_vec),
        xor_128(set1_128(IV[5]), offset_high_vec),
        xor_128(set1_128(IV[6]), block_flags_vec),
        xor_128(set1_128(IV[7]), context_vec),
    };
    round_fn4(v, msg_vecs, 0);
    round_fn4(v, msg_vecs, 1);
    round_fn4(v, msg_vecs, 2);
    round_fn4(v, msg_vecs, 3);
    round_fn4(v, msg_vecs, 4);
    round_fn4(v, msg_vecs, 5);
    round_fn4(v, msg_vecs, 6);
    h_vecs[0] = xor_128(v[0], v[8]);
    h_vecs[1] = xor_128(v[1], v[9]);
    h_vecs[2] = xor_128(v[2], v[10]);
    h_vecs[3] = xor_128(v[3], v[11]);
    h_vecs[4] = xor_128(v[4], v[12]);
    h_vecs[5] = xor_128(v[5], v[13]);
    h_vecs[6] = xor_128(v[6], v[14]);
    h_vecs[7] = xor_128(v[7], v[15]);

    internal_flags = 0;
  }

  transpose_vecs_128(&h_vecs[0]);
  transpose_vecs_128(&h_vecs[4]);
  // The first four vecs now contain the first half of each output, and the
  // second four vecs contain the second half of each output.
  storeu_128(h_vecs[0], &out[0 * sizeof(uint32x4_t)]);
  storeu_128(h_vecs[4], &out[1 * sizeof(uint32x4_t)]);
  storeu_128(h_vecs[1], &out[2 * sizeof(uint32x4_t)]);
  storeu_128(h_vecs[5], &out[3 * sizeof(uint32x4_t)]);
  storeu_128(h_vecs[2], &out[4 * sizeof(uint32x4_t)]);
  storeu_128(h_vecs[6], &out[5 * sizeof(uint32x4_t)]);
  storeu_128(h_vecs[3], &out[6 * sizeof(uint32x4_t)]);
  storeu_128(h_vecs[7], &out[7 * sizeof(uint32x4_t)]);
}

/*
 * ----------------------------------------------------------------------------
 * hash_many_neon
 * ----------------------------------------------------------------------------
 */

INLINE void hash_one_neon(const uint8_t *input, size_t blocks,
                          const uint32_t key_words[8], uint64_t offset,
                          uint8_t internal_flags_start,
                          uint8_t internal_flags_end, uint32_t context,
                          uint8_t out[OUT_LEN]) {
  uint32_t state[8];
  init_iv(key_words, state);
  uint8_t flags = internal_flags_start;
  while (blocks > 0) {
    if (blocks == 1) {
      flags |= internal_flags_end;
    }
    // TODO: use compress_neon
    compress_portable(state, input, BLOCK_LEN, offset, flags, context);
    input = &input[BLOCK_LEN];
    blocks -= 1;
    flags = 0;
  }
  memcpy(out, state, OUT_LEN);
}

void hash_many_neon(const uint8_t *const *inputs, size_t num_inputs,
                    size_t blocks, const uint32_t key_words[8], uint64_t offset,
                    const uint64_t offset_deltas[17],
                    uint8_t internal_flags_start, uint8_t internal_flags_end,
                    uint32_t context, uint8_t *out) {
  while (num_inputs >= 4) {
    hash4_neon(inputs, blocks, key_words, offset, offset_deltas,
               internal_flags_start, internal_flags_end, context, out);
    inputs += 4;
    num_inputs -= 4;
    offset += offset_deltas[4];
    out = &out[4 * OUT_LEN];
  }
  while (num_inputs > 0) {
    hash_one_neon(inputs[0], blocks, key_words, offset, internal_flags_start,
                  internal_flags_end, context, out);
    inputs += 1;
    num_inputs -= 1;
    offset += offset_deltas[1];
    out = &out[OUT_LEN];
  }
}

#else // __ARM_NEON

// NEON is only enabled statically in the build, with --features=c_neon.
// (Rust's dynamic feature detection for ARM is not yet stable.) So we don't
// need to provide any stubs here.

#endif