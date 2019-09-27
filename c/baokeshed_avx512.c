#include "baokeshed_impl.h"
#include <assert.h>
#include <immintrin.h>
#include <stdbool.h>
#include <string.h>

#if defined(__AVX512F__) && defined(__AVX512VL__)

INLINE __m128i loadu_128(const uint8_t src[16]) {
  return _mm_loadu_si128((const __m128i *)src);
}

INLINE __m256i loadu_256(const uint8_t src[32]) {
  return _mm256_loadu_si256((const __m256i *)src);
}

INLINE __m512i loadu_512(const uint8_t src[64]) {
  return _mm512_loadu_si512((const __m512i *)src);
}

INLINE void storeu_128(__m128i src, uint8_t dest[16]) {
  return _mm_storeu_si128((__m128i *)dest, src);
}

INLINE void storeu_256(__m256i src, uint8_t dest[16]) {
  return _mm256_storeu_si256((__m256i *)dest, src);
}

INLINE __m128i add_128(__m128i a, __m128i b) { return _mm_add_epi32(a, b); }

INLINE __m256i add_256(__m256i a, __m256i b) { return _mm256_add_epi32(a, b); }

INLINE __m512i add_512(__m512i a, __m512i b) { return _mm512_add_epi32(a, b); }

INLINE __m128i xor_128(__m128i a, __m128i b) { return _mm_xor_si128(a, b); }

INLINE __m256i xor_256(__m256i a, __m256i b) { return _mm256_xor_si256(a, b); }

INLINE __m512i xor_512(__m512i a, __m512i b) { return _mm512_xor_si512(a, b); }

INLINE __m128i set1_128(uint32_t x) { return _mm_set1_epi32(x); }

INLINE __m256i set1_256(uint32_t x) { return _mm256_set1_epi32(x); }

INLINE __m512i set1_512(uint32_t x) { return _mm512_set1_epi32(x); }

INLINE __m128i set4(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  return _mm_setr_epi32(a, b, c, d);
}

INLINE __m128i rot16_128(__m128i x) { return _mm_ror_epi32(x, 16); }

INLINE __m256i rot16_256(__m256i x) { return _mm256_ror_epi32(x, 16); }

INLINE __m512i rot16_512(__m512i x) { return _mm512_ror_epi32(x, 16); }

INLINE __m128i rot12_128(__m128i x) { return _mm_ror_epi32(x, 12); }

INLINE __m256i rot12_256(__m256i x) { return _mm256_ror_epi32(x, 12); }

INLINE __m512i rot12_512(__m512i x) { return _mm512_ror_epi32(x, 12); }

INLINE __m128i rot8_128(__m128i x) { return _mm_ror_epi32(x, 8); }

INLINE __m256i rot8_256(__m256i x) { return _mm256_ror_epi32(x, 8); }

INLINE __m512i rot8_512(__m512i x) { return _mm512_ror_epi32(x, 8); }

INLINE __m128i rot7_128(__m128i x) { return _mm_ror_epi32(x, 7); }

INLINE __m256i rot7_256(__m256i x) { return _mm256_ror_epi32(x, 7); }

INLINE __m512i rot7_512(__m512i x) { return _mm512_ror_epi32(x, 7); }

/*
 * ----------------------------------------------------------------------------
 * compress_avx512
 * ----------------------------------------------------------------------------
 */

INLINE void g1(__m128i *row1, __m128i *row2, __m128i *row3, __m128i *row4,
               __m128i m) {
  *row1 = add_128(add_128(*row1, m), *row2);
  *row4 = xor_128(*row4, *row1);
  *row4 = rot16_128(*row4);
  *row3 = add_128(*row3, *row4);
  *row2 = xor_128(*row2, *row3);
  *row2 = rot12_128(*row2);
}

INLINE void g2(__m128i *row1, __m128i *row2, __m128i *row3, __m128i *row4,
               __m128i m) {
  *row1 = add_128(add_128(*row1, m), *row2);
  *row4 = xor_128(*row4, *row1);
  *row4 = rot8_128(*row4);
  *row3 = add_128(*row3, *row4);
  *row2 = xor_128(*row2, *row3);
  *row2 = rot7_128(*row2);
}

// Note the optimization here of leaving row2 as the unrotated row, rather than
// row1. All the message loads below are adjusted to compensate for this. See
// discussion at https://github.com/sneves/blake2-avx2/pull/4
INLINE void diagonalize(__m128i *row1, __m128i *row3, __m128i *row4) {
  *row1 = _mm_shuffle_epi32(*row1, _MM_SHUFFLE(2, 1, 0, 3));
  *row4 = _mm_shuffle_epi32(*row4, _MM_SHUFFLE(1, 0, 3, 2));
  *row3 = _mm_shuffle_epi32(*row3, _MM_SHUFFLE(0, 3, 2, 1));
}

INLINE void undiagonalize(__m128i *row1, __m128i *row3, __m128i *row4) {
  *row1 = _mm_shuffle_epi32(*row1, _MM_SHUFFLE(0, 3, 2, 1));
  *row4 = _mm_shuffle_epi32(*row4, _MM_SHUFFLE(1, 0, 3, 2));
  *row3 = _mm_shuffle_epi32(*row3, _MM_SHUFFLE(2, 1, 0, 3));
}

// This is the same as MSG_SCHEDULE, with two changes:
// 1) The indexes are ordered according to how they're used in vector
//    operations. Rather than doing g1 for one column, then g2 for that column,
//    the vector operations do g1 for all columns, followed by g2 for all
//    columns. This 0/1/2/3/... in the first row becomes 0/2/4/8/...
// 2) Rows after the first are reinterpreted to load from the previous round's
//    permuted message words, rather than the original message words. This
//    means we don't need to keep the original message words in their own
//    register.
// These optimizations were originally done by Samuel Neves at
// https://git.zx2c4.com/WireGuard/tree/src/crypto/zinc/blake2s/blake2s-x86_64.S#n189
static uint32_t MSG_SCHEDULE_IN_PLACE[7][16] = {
    {0, 2, 4, 6, 1, 3, 5, 7, 14, 8, 10, 12, 15, 9, 11, 13},
    {8, 2, 13, 15, 10, 9, 12, 3, 6, 4, 0, 14, 5, 11, 1, 7},
    {11, 13, 8, 6, 5, 10, 14, 3, 2, 4, 12, 15, 1, 0, 7, 9},
    {11, 10, 7, 0, 8, 15, 1, 13, 3, 6, 2, 12, 4, 14, 9, 5},
    {4, 10, 9, 14, 15, 0, 11, 8, 1, 7, 3, 13, 2, 5, 6, 12},
    {2, 11, 4, 15, 14, 3, 10, 8, 13, 6, 5, 7, 0, 12, 1, 9},
    {4, 8, 15, 9, 14, 11, 13, 5, 3, 2, 1, 12, 6, 10, 7, 0},
};

void compress_avx512(uint32_t state[8], const uint8_t block[BLOCK_LEN],
                     uint8_t block_len, uint64_t offset, uint8_t internal_flags,
                     uint32_t context) {
  __m128i row1 = loadu_128((uint8_t *)&state[0]);
  __m128i row2 = loadu_128((uint8_t *)&state[4]);
  __m128i row3 = set4(IV[0], IV[1], IV[2], IV[3]);
  __m128i row4 =
      set4(IV[4] ^ offset_low(offset), IV[5] ^ offset_high(offset),
           IV[6] ^ block_flags(block_len, internal_flags), IV[7] ^ context);

  // Use two __m256i's instead of one __m512i to avoid paying a CPU frequency
  // penalty.
  __m256i m0 = loadu_256(&block[sizeof(__m256i) * 0]);
  __m256i m1 = loadu_256(&block[sizeof(__m256i) * 1]);

  for (size_t round=0 ; round < 7; round ++ ) {
      __m256i schedule0 = _mm256_loadu_si256((const __m256i*)&MSG_SCHEDULE_IN_PLACE[round][0]);
      __m256i schedule1 = _mm256_loadu_si256((const __m256i*)&MSG_SCHEDULE_IN_PLACE[round][8]);
      __m256i new_m0 = _mm256_permutex2var_epi32(m0, schedule0, m1);
      __m256i new_m1 = _mm256_permutex2var_epi32(m0, schedule1, m1);
      m0 = new_m0;
      m1 = new_m1;
      g1(&row1, &row2, &row3, &row4, _mm256_extracti128_si256(m0, 0));
      g2(&row1, &row2, &row3, &row4, _mm256_extracti128_si256(m0, 1));
      diagonalize(&row1, &row3, &row4);
      g1(&row1, &row2, &row3, &row4, _mm256_extracti128_si256(m1, 0));
      g2(&row1, &row2, &row3, &row4, _mm256_extracti128_si256(m1, 1));
      undiagonalize(&row1, &row3, &row4);
  }

  storeu_128(xor_128(row1, row3), (uint8_t *)&state[0]);
  storeu_128(xor_128(row2, row4), (uint8_t *)&state[4]);
}

/*
 * ----------------------------------------------------------------------------
 * hash4_avx512
 * ----------------------------------------------------------------------------
 */

INLINE void round_fn4(__m128i v[16], __m128i m[16], size_t r) {
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

INLINE void transpose_vecs_128(__m128i vecs[4]) {
  // Interleave 32-bit lates. The low unpack is lanes 00/11 and the high is
  // 22/33. Note that this doesn't split the vector into two lanes, as the
  // AVX2 counterparts do.
  __m128i ab_01 = _mm_unpacklo_epi32(vecs[0], vecs[1]);
  __m128i ab_23 = _mm_unpackhi_epi32(vecs[0], vecs[1]);
  __m128i cd_01 = _mm_unpacklo_epi32(vecs[2], vecs[3]);
  __m128i cd_23 = _mm_unpackhi_epi32(vecs[2], vecs[3]);

  // Interleave 64-bit lanes.
  __m128i abcd_0 = _mm_unpacklo_epi64(ab_01, cd_01);
  __m128i abcd_1 = _mm_unpackhi_epi64(ab_01, cd_01);
  __m128i abcd_2 = _mm_unpacklo_epi64(ab_23, cd_23);
  __m128i abcd_3 = _mm_unpackhi_epi64(ab_23, cd_23);

  vecs[0] = abcd_0;
  vecs[1] = abcd_1;
  vecs[2] = abcd_2;
  vecs[3] = abcd_3;
}

INLINE void transpose_msg_vecs4(const uint8_t *const *inputs,
                                size_t block_offset, __m128i out[16]) {
  out[0] = loadu_128(&inputs[0][block_offset + 0 * sizeof(__m128i)]);
  out[1] = loadu_128(&inputs[1][block_offset + 0 * sizeof(__m128i)]);
  out[2] = loadu_128(&inputs[2][block_offset + 0 * sizeof(__m128i)]);
  out[3] = loadu_128(&inputs[3][block_offset + 0 * sizeof(__m128i)]);
  out[4] = loadu_128(&inputs[0][block_offset + 1 * sizeof(__m128i)]);
  out[5] = loadu_128(&inputs[1][block_offset + 1 * sizeof(__m128i)]);
  out[6] = loadu_128(&inputs[2][block_offset + 1 * sizeof(__m128i)]);
  out[7] = loadu_128(&inputs[3][block_offset + 1 * sizeof(__m128i)]);
  out[8] = loadu_128(&inputs[0][block_offset + 2 * sizeof(__m128i)]);
  out[9] = loadu_128(&inputs[1][block_offset + 2 * sizeof(__m128i)]);
  out[10] = loadu_128(&inputs[2][block_offset + 2 * sizeof(__m128i)]);
  out[11] = loadu_128(&inputs[3][block_offset + 2 * sizeof(__m128i)]);
  out[12] = loadu_128(&inputs[0][block_offset + 3 * sizeof(__m128i)]);
  out[13] = loadu_128(&inputs[1][block_offset + 3 * sizeof(__m128i)]);
  out[14] = loadu_128(&inputs[2][block_offset + 3 * sizeof(__m128i)]);
  out[15] = loadu_128(&inputs[3][block_offset + 3 * sizeof(__m128i)]);
  transpose_vecs_128(&out[0]);
  transpose_vecs_128(&out[4]);
  transpose_vecs_128(&out[8]);
  transpose_vecs_128(&out[12]);
}

INLINE void load_offsets4(uint64_t offset, const uint64_t deltas[4],
                          __m128i *out_lo, __m128i *out_hi) {
  __m256i a = _mm256_add_epi64(_mm256_set1_epi64x(offset),
                               _mm256_loadu_si256((const __m256i *)deltas));
  *out_lo = _mm256_cvtepi64_epi32(a);
  *out_hi = _mm256_cvtepi64_epi32(_mm256_srli_epi64(a, 32));
}

void hash4_avx512(const uint8_t *const *inputs, size_t blocks,
                  const uint32_t key_words[8], uint64_t offset,
                  const uint64_t offset_deltas[4], uint8_t internal_flags_start,
                  uint8_t internal_flags_end, uint32_t context, uint8_t *out) {
  __m128i h_vecs[8] = {
      xor_128(set1_128(IV[0]), set1_128(key_words[0])),
      xor_128(set1_128(IV[1]), set1_128(key_words[1])),
      xor_128(set1_128(IV[2]), set1_128(key_words[2])),
      xor_128(set1_128(IV[3]), set1_128(key_words[3])),
      xor_128(set1_128(IV[4]), set1_128(key_words[4])),
      xor_128(set1_128(IV[5]), set1_128(key_words[5])),
      xor_128(set1_128(IV[6]), set1_128(key_words[6])),
      xor_128(set1_128(IV[7]), set1_128(key_words[7])),
  };
  __m128i offset_low_vec, offset_high_vec;
  load_offsets4(offset, offset_deltas, &offset_low_vec, &offset_high_vec);
  const __m128i context_vec = set1_128(context);
  uint8_t internal_flags = internal_flags_start;

  for (size_t block = 0; block < blocks; block++) {
    if (block + 1 == blocks) {
      internal_flags |= internal_flags_end;
    }
    __m128i block_flags_vec = set1_128(block_flags(BLOCK_LEN, internal_flags));
    __m128i msg_vecs[16];
    transpose_msg_vecs4(inputs, block * BLOCK_LEN, msg_vecs);

    __m128i v[16] = {
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
  storeu_128(h_vecs[0], &out[0 * sizeof(__m128i)]);
  storeu_128(h_vecs[4], &out[1 * sizeof(__m128i)]);
  storeu_128(h_vecs[1], &out[2 * sizeof(__m128i)]);
  storeu_128(h_vecs[5], &out[3 * sizeof(__m128i)]);
  storeu_128(h_vecs[2], &out[4 * sizeof(__m128i)]);
  storeu_128(h_vecs[6], &out[5 * sizeof(__m128i)]);
  storeu_128(h_vecs[3], &out[6 * sizeof(__m128i)]);
  storeu_128(h_vecs[7], &out[7 * sizeof(__m128i)]);
}

/*
 * ----------------------------------------------------------------------------
 * hash8_avx512
 * ----------------------------------------------------------------------------
 */

INLINE void round_fn8(__m256i v[16], __m256i m[16], size_t r) {
  v[0] = add_256(v[0], m[(size_t)MSG_SCHEDULE[r][0]]);
  v[1] = add_256(v[1], m[(size_t)MSG_SCHEDULE[r][2]]);
  v[2] = add_256(v[2], m[(size_t)MSG_SCHEDULE[r][4]]);
  v[3] = add_256(v[3], m[(size_t)MSG_SCHEDULE[r][6]]);
  v[0] = add_256(v[0], v[4]);
  v[1] = add_256(v[1], v[5]);
  v[2] = add_256(v[2], v[6]);
  v[3] = add_256(v[3], v[7]);
  v[12] = xor_256(v[12], v[0]);
  v[13] = xor_256(v[13], v[1]);
  v[14] = xor_256(v[14], v[2]);
  v[15] = xor_256(v[15], v[3]);
  v[12] = rot16_256(v[12]);
  v[13] = rot16_256(v[13]);
  v[14] = rot16_256(v[14]);
  v[15] = rot16_256(v[15]);
  v[8] = add_256(v[8], v[12]);
  v[9] = add_256(v[9], v[13]);
  v[10] = add_256(v[10], v[14]);
  v[11] = add_256(v[11], v[15]);
  v[4] = xor_256(v[4], v[8]);
  v[5] = xor_256(v[5], v[9]);
  v[6] = xor_256(v[6], v[10]);
  v[7] = xor_256(v[7], v[11]);
  v[4] = rot12_256(v[4]);
  v[5] = rot12_256(v[5]);
  v[6] = rot12_256(v[6]);
  v[7] = rot12_256(v[7]);
  v[0] = add_256(v[0], m[(size_t)MSG_SCHEDULE[r][1]]);
  v[1] = add_256(v[1], m[(size_t)MSG_SCHEDULE[r][3]]);
  v[2] = add_256(v[2], m[(size_t)MSG_SCHEDULE[r][5]]);
  v[3] = add_256(v[3], m[(size_t)MSG_SCHEDULE[r][7]]);
  v[0] = add_256(v[0], v[4]);
  v[1] = add_256(v[1], v[5]);
  v[2] = add_256(v[2], v[6]);
  v[3] = add_256(v[3], v[7]);
  v[12] = xor_256(v[12], v[0]);
  v[13] = xor_256(v[13], v[1]);
  v[14] = xor_256(v[14], v[2]);
  v[15] = xor_256(v[15], v[3]);
  v[12] = rot8_256(v[12]);
  v[13] = rot8_256(v[13]);
  v[14] = rot8_256(v[14]);
  v[15] = rot8_256(v[15]);
  v[8] = add_256(v[8], v[12]);
  v[9] = add_256(v[9], v[13]);
  v[10] = add_256(v[10], v[14]);
  v[11] = add_256(v[11], v[15]);
  v[4] = xor_256(v[4], v[8]);
  v[5] = xor_256(v[5], v[9]);
  v[6] = xor_256(v[6], v[10]);
  v[7] = xor_256(v[7], v[11]);
  v[4] = rot7_256(v[4]);
  v[5] = rot7_256(v[5]);
  v[6] = rot7_256(v[6]);
  v[7] = rot7_256(v[7]);

  v[0] = add_256(v[0], m[(size_t)MSG_SCHEDULE[r][8]]);
  v[1] = add_256(v[1], m[(size_t)MSG_SCHEDULE[r][10]]);
  v[2] = add_256(v[2], m[(size_t)MSG_SCHEDULE[r][12]]);
  v[3] = add_256(v[3], m[(size_t)MSG_SCHEDULE[r][14]]);
  v[0] = add_256(v[0], v[5]);
  v[1] = add_256(v[1], v[6]);
  v[2] = add_256(v[2], v[7]);
  v[3] = add_256(v[3], v[4]);
  v[15] = xor_256(v[15], v[0]);
  v[12] = xor_256(v[12], v[1]);
  v[13] = xor_256(v[13], v[2]);
  v[14] = xor_256(v[14], v[3]);
  v[15] = rot16_256(v[15]);
  v[12] = rot16_256(v[12]);
  v[13] = rot16_256(v[13]);
  v[14] = rot16_256(v[14]);
  v[10] = add_256(v[10], v[15]);
  v[11] = add_256(v[11], v[12]);
  v[8] = add_256(v[8], v[13]);
  v[9] = add_256(v[9], v[14]);
  v[5] = xor_256(v[5], v[10]);
  v[6] = xor_256(v[6], v[11]);
  v[7] = xor_256(v[7], v[8]);
  v[4] = xor_256(v[4], v[9]);
  v[5] = rot12_256(v[5]);
  v[6] = rot12_256(v[6]);
  v[7] = rot12_256(v[7]);
  v[4] = rot12_256(v[4]);
  v[0] = add_256(v[0], m[(size_t)MSG_SCHEDULE[r][9]]);
  v[1] = add_256(v[1], m[(size_t)MSG_SCHEDULE[r][11]]);
  v[2] = add_256(v[2], m[(size_t)MSG_SCHEDULE[r][13]]);
  v[3] = add_256(v[3], m[(size_t)MSG_SCHEDULE[r][15]]);
  v[0] = add_256(v[0], v[5]);
  v[1] = add_256(v[1], v[6]);
  v[2] = add_256(v[2], v[7]);
  v[3] = add_256(v[3], v[4]);
  v[15] = xor_256(v[15], v[0]);
  v[12] = xor_256(v[12], v[1]);
  v[13] = xor_256(v[13], v[2]);
  v[14] = xor_256(v[14], v[3]);
  v[15] = rot8_256(v[15]);
  v[12] = rot8_256(v[12]);
  v[13] = rot8_256(v[13]);
  v[14] = rot8_256(v[14]);
  v[10] = add_256(v[10], v[15]);
  v[11] = add_256(v[11], v[12]);
  v[8] = add_256(v[8], v[13]);
  v[9] = add_256(v[9], v[14]);
  v[5] = xor_256(v[5], v[10]);
  v[6] = xor_256(v[6], v[11]);
  v[7] = xor_256(v[7], v[8]);
  v[4] = xor_256(v[4], v[9]);
  v[5] = rot7_256(v[5]);
  v[6] = rot7_256(v[6]);
  v[7] = rot7_256(v[7]);
  v[4] = rot7_256(v[4]);
}

INLINE void transpose_vecs_256(__m256i vecs[8]) {
  // Interleave 32-bit lanes. The low unpack is lanes 00/11/44/55, and the high
  // is 22/33/66/77.
  __m256i ab_0145 = _mm256_unpacklo_epi32(vecs[0], vecs[1]);
  __m256i ab_2367 = _mm256_unpackhi_epi32(vecs[0], vecs[1]);
  __m256i cd_0145 = _mm256_unpacklo_epi32(vecs[2], vecs[3]);
  __m256i cd_2367 = _mm256_unpackhi_epi32(vecs[2], vecs[3]);
  __m256i ef_0145 = _mm256_unpacklo_epi32(vecs[4], vecs[5]);
  __m256i ef_2367 = _mm256_unpackhi_epi32(vecs[4], vecs[5]);
  __m256i gh_0145 = _mm256_unpacklo_epi32(vecs[6], vecs[7]);
  __m256i gh_2367 = _mm256_unpackhi_epi32(vecs[6], vecs[7]);

  // Interleave 64-bit lates. The low unpack is lanes 00/22 and the high is
  // 11/33.
  __m256i abcd_04 = _mm256_unpacklo_epi64(ab_0145, cd_0145);
  __m256i abcd_15 = _mm256_unpackhi_epi64(ab_0145, cd_0145);
  __m256i abcd_26 = _mm256_unpacklo_epi64(ab_2367, cd_2367);
  __m256i abcd_37 = _mm256_unpackhi_epi64(ab_2367, cd_2367);
  __m256i efgh_04 = _mm256_unpacklo_epi64(ef_0145, gh_0145);
  __m256i efgh_15 = _mm256_unpackhi_epi64(ef_0145, gh_0145);
  __m256i efgh_26 = _mm256_unpacklo_epi64(ef_2367, gh_2367);
  __m256i efgh_37 = _mm256_unpackhi_epi64(ef_2367, gh_2367);

  // Interleave 128-bit lanes.
  vecs[0] = _mm256_permute2x128_si256(abcd_04, efgh_04, 0x20);
  vecs[1] = _mm256_permute2x128_si256(abcd_15, efgh_15, 0x20);
  vecs[2] = _mm256_permute2x128_si256(abcd_26, efgh_26, 0x20);
  vecs[3] = _mm256_permute2x128_si256(abcd_37, efgh_37, 0x20);
  vecs[4] = _mm256_permute2x128_si256(abcd_04, efgh_04, 0x31);
  vecs[5] = _mm256_permute2x128_si256(abcd_15, efgh_15, 0x31);
  vecs[6] = _mm256_permute2x128_si256(abcd_26, efgh_26, 0x31);
  vecs[7] = _mm256_permute2x128_si256(abcd_37, efgh_37, 0x31);
}

INLINE void transpose_msg_vecs8(const uint8_t *const *inputs,
                                size_t block_offset, __m256i out[16]) {
  out[0] = loadu_256(&inputs[0][block_offset + 0 * sizeof(__m256i)]);
  out[1] = loadu_256(&inputs[1][block_offset + 0 * sizeof(__m256i)]);
  out[2] = loadu_256(&inputs[2][block_offset + 0 * sizeof(__m256i)]);
  out[3] = loadu_256(&inputs[3][block_offset + 0 * sizeof(__m256i)]);
  out[4] = loadu_256(&inputs[4][block_offset + 0 * sizeof(__m256i)]);
  out[5] = loadu_256(&inputs[5][block_offset + 0 * sizeof(__m256i)]);
  out[6] = loadu_256(&inputs[6][block_offset + 0 * sizeof(__m256i)]);
  out[7] = loadu_256(&inputs[7][block_offset + 0 * sizeof(__m256i)]);
  out[8] = loadu_256(&inputs[0][block_offset + 1 * sizeof(__m256i)]);
  out[9] = loadu_256(&inputs[1][block_offset + 1 * sizeof(__m256i)]);
  out[10] = loadu_256(&inputs[2][block_offset + 1 * sizeof(__m256i)]);
  out[11] = loadu_256(&inputs[3][block_offset + 1 * sizeof(__m256i)]);
  out[12] = loadu_256(&inputs[4][block_offset + 1 * sizeof(__m256i)]);
  out[13] = loadu_256(&inputs[5][block_offset + 1 * sizeof(__m256i)]);
  out[14] = loadu_256(&inputs[6][block_offset + 1 * sizeof(__m256i)]);
  out[15] = loadu_256(&inputs[7][block_offset + 1 * sizeof(__m256i)]);
  transpose_vecs_256(&out[0]);
  transpose_vecs_256(&out[8]);
}

INLINE void load_offsets8(uint64_t offset, const uint64_t deltas[8],
                          __m256i *out_lo, __m256i *out_hi) {
  __m512i a = _mm512_add_epi64(_mm512_set1_epi64(offset),
                               _mm512_loadu_si512((const __m512i *)deltas));
  *out_lo = _mm512_cvtepi64_epi32(a);
  *out_hi = _mm512_cvtepi64_epi32(_mm512_srli_epi64(a, 32));
}

void hash8_avx512(const uint8_t *const *inputs, size_t blocks,
                  const uint32_t key_words[8], uint64_t offset,
                  const uint64_t offset_deltas[8], uint8_t internal_flags_start,
                  uint8_t internal_flags_end, uint32_t context, uint8_t *out) {
  __m256i h_vecs[8] = {
      xor_256(set1_256(IV[0]), set1_256(key_words[0])),
      xor_256(set1_256(IV[1]), set1_256(key_words[1])),
      xor_256(set1_256(IV[2]), set1_256(key_words[2])),
      xor_256(set1_256(IV[3]), set1_256(key_words[3])),
      xor_256(set1_256(IV[4]), set1_256(key_words[4])),
      xor_256(set1_256(IV[5]), set1_256(key_words[5])),
      xor_256(set1_256(IV[6]), set1_256(key_words[6])),
      xor_256(set1_256(IV[7]), set1_256(key_words[7])),
  };
  __m256i offset_low_vec, offset_high_vec;
  load_offsets8(offset, offset_deltas, &offset_low_vec, &offset_high_vec);
  const __m256i context_vec = set1_256(context);
  uint8_t internal_flags = internal_flags_start;

  for (size_t block = 0; block < blocks; block++) {
    if (block + 1 == blocks) {
      internal_flags |= internal_flags_end;
    }
    __m256i block_flags_vec = set1_256(block_flags(BLOCK_LEN, internal_flags));
    __m256i msg_vecs[16];
    transpose_msg_vecs8(inputs, block * BLOCK_LEN, msg_vecs);

    __m256i v[16] = {
        h_vecs[0],
        h_vecs[1],
        h_vecs[2],
        h_vecs[3],
        h_vecs[4],
        h_vecs[5],
        h_vecs[6],
        h_vecs[7],
        set1_256(IV[0]),
        set1_256(IV[1]),
        set1_256(IV[2]),
        set1_256(IV[3]),
        xor_256(set1_256(IV[4]), offset_low_vec),
        xor_256(set1_256(IV[5]), offset_high_vec),
        xor_256(set1_256(IV[6]), block_flags_vec),
        xor_256(set1_256(IV[7]), context_vec),
    };
    round_fn8(v, msg_vecs, 0);
    round_fn8(v, msg_vecs, 1);
    round_fn8(v, msg_vecs, 2);
    round_fn8(v, msg_vecs, 3);
    round_fn8(v, msg_vecs, 4);
    round_fn8(v, msg_vecs, 5);
    round_fn8(v, msg_vecs, 6);
    h_vecs[0] = xor_256(v[0], v[8]);
    h_vecs[1] = xor_256(v[1], v[9]);
    h_vecs[2] = xor_256(v[2], v[10]);
    h_vecs[3] = xor_256(v[3], v[11]);
    h_vecs[4] = xor_256(v[4], v[12]);
    h_vecs[5] = xor_256(v[5], v[13]);
    h_vecs[6] = xor_256(v[6], v[14]);
    h_vecs[7] = xor_256(v[7], v[15]);

    internal_flags = 0;
  }

  transpose_vecs_256(h_vecs);
  storeu_256(h_vecs[0], &out[0 * sizeof(__m256i)]);
  storeu_256(h_vecs[1], &out[1 * sizeof(__m256i)]);
  storeu_256(h_vecs[2], &out[2 * sizeof(__m256i)]);
  storeu_256(h_vecs[3], &out[3 * sizeof(__m256i)]);
  storeu_256(h_vecs[4], &out[4 * sizeof(__m256i)]);
  storeu_256(h_vecs[5], &out[5 * sizeof(__m256i)]);
  storeu_256(h_vecs[6], &out[6 * sizeof(__m256i)]);
  storeu_256(h_vecs[7], &out[7 * sizeof(__m256i)]);
}

/*
 * ----------------------------------------------------------------------------
 * hash16_avx512
 * ----------------------------------------------------------------------------
 */

INLINE void round_fn16(__m512i v[16], __m512i m[16], size_t r) {
  v[0] = add_512(v[0], m[(size_t)MSG_SCHEDULE[r][0]]);
  v[1] = add_512(v[1], m[(size_t)MSG_SCHEDULE[r][2]]);
  v[2] = add_512(v[2], m[(size_t)MSG_SCHEDULE[r][4]]);
  v[3] = add_512(v[3], m[(size_t)MSG_SCHEDULE[r][6]]);
  v[0] = add_512(v[0], v[4]);
  v[1] = add_512(v[1], v[5]);
  v[2] = add_512(v[2], v[6]);
  v[3] = add_512(v[3], v[7]);
  v[12] = xor_512(v[12], v[0]);
  v[13] = xor_512(v[13], v[1]);
  v[14] = xor_512(v[14], v[2]);
  v[15] = xor_512(v[15], v[3]);
  v[12] = rot16_512(v[12]);
  v[13] = rot16_512(v[13]);
  v[14] = rot16_512(v[14]);
  v[15] = rot16_512(v[15]);
  v[8] = add_512(v[8], v[12]);
  v[9] = add_512(v[9], v[13]);
  v[10] = add_512(v[10], v[14]);
  v[11] = add_512(v[11], v[15]);
  v[4] = xor_512(v[4], v[8]);
  v[5] = xor_512(v[5], v[9]);
  v[6] = xor_512(v[6], v[10]);
  v[7] = xor_512(v[7], v[11]);
  v[4] = rot12_512(v[4]);
  v[5] = rot12_512(v[5]);
  v[6] = rot12_512(v[6]);
  v[7] = rot12_512(v[7]);
  v[0] = add_512(v[0], m[(size_t)MSG_SCHEDULE[r][1]]);
  v[1] = add_512(v[1], m[(size_t)MSG_SCHEDULE[r][3]]);
  v[2] = add_512(v[2], m[(size_t)MSG_SCHEDULE[r][5]]);
  v[3] = add_512(v[3], m[(size_t)MSG_SCHEDULE[r][7]]);
  v[0] = add_512(v[0], v[4]);
  v[1] = add_512(v[1], v[5]);
  v[2] = add_512(v[2], v[6]);
  v[3] = add_512(v[3], v[7]);
  v[12] = xor_512(v[12], v[0]);
  v[13] = xor_512(v[13], v[1]);
  v[14] = xor_512(v[14], v[2]);
  v[15] = xor_512(v[15], v[3]);
  v[12] = rot8_512(v[12]);
  v[13] = rot8_512(v[13]);
  v[14] = rot8_512(v[14]);
  v[15] = rot8_512(v[15]);
  v[8] = add_512(v[8], v[12]);
  v[9] = add_512(v[9], v[13]);
  v[10] = add_512(v[10], v[14]);
  v[11] = add_512(v[11], v[15]);
  v[4] = xor_512(v[4], v[8]);
  v[5] = xor_512(v[5], v[9]);
  v[6] = xor_512(v[6], v[10]);
  v[7] = xor_512(v[7], v[11]);
  v[4] = rot7_512(v[4]);
  v[5] = rot7_512(v[5]);
  v[6] = rot7_512(v[6]);
  v[7] = rot7_512(v[7]);

  v[0] = add_512(v[0], m[(size_t)MSG_SCHEDULE[r][8]]);
  v[1] = add_512(v[1], m[(size_t)MSG_SCHEDULE[r][10]]);
  v[2] = add_512(v[2], m[(size_t)MSG_SCHEDULE[r][12]]);
  v[3] = add_512(v[3], m[(size_t)MSG_SCHEDULE[r][14]]);
  v[0] = add_512(v[0], v[5]);
  v[1] = add_512(v[1], v[6]);
  v[2] = add_512(v[2], v[7]);
  v[3] = add_512(v[3], v[4]);
  v[15] = xor_512(v[15], v[0]);
  v[12] = xor_512(v[12], v[1]);
  v[13] = xor_512(v[13], v[2]);
  v[14] = xor_512(v[14], v[3]);
  v[15] = rot16_512(v[15]);
  v[12] = rot16_512(v[12]);
  v[13] = rot16_512(v[13]);
  v[14] = rot16_512(v[14]);
  v[10] = add_512(v[10], v[15]);
  v[11] = add_512(v[11], v[12]);
  v[8] = add_512(v[8], v[13]);
  v[9] = add_512(v[9], v[14]);
  v[5] = xor_512(v[5], v[10]);
  v[6] = xor_512(v[6], v[11]);
  v[7] = xor_512(v[7], v[8]);
  v[4] = xor_512(v[4], v[9]);
  v[5] = rot12_512(v[5]);
  v[6] = rot12_512(v[6]);
  v[7] = rot12_512(v[7]);
  v[4] = rot12_512(v[4]);
  v[0] = add_512(v[0], m[(size_t)MSG_SCHEDULE[r][9]]);
  v[1] = add_512(v[1], m[(size_t)MSG_SCHEDULE[r][11]]);
  v[2] = add_512(v[2], m[(size_t)MSG_SCHEDULE[r][13]]);
  v[3] = add_512(v[3], m[(size_t)MSG_SCHEDULE[r][15]]);
  v[0] = add_512(v[0], v[5]);
  v[1] = add_512(v[1], v[6]);
  v[2] = add_512(v[2], v[7]);
  v[3] = add_512(v[3], v[4]);
  v[15] = xor_512(v[15], v[0]);
  v[12] = xor_512(v[12], v[1]);
  v[13] = xor_512(v[13], v[2]);
  v[14] = xor_512(v[14], v[3]);
  v[15] = rot8_512(v[15]);
  v[12] = rot8_512(v[12]);
  v[13] = rot8_512(v[13]);
  v[14] = rot8_512(v[14]);
  v[10] = add_512(v[10], v[15]);
  v[11] = add_512(v[11], v[12]);
  v[8] = add_512(v[8], v[13]);
  v[9] = add_512(v[9], v[14]);
  v[5] = xor_512(v[5], v[10]);
  v[6] = xor_512(v[6], v[11]);
  v[7] = xor_512(v[7], v[8]);
  v[4] = xor_512(v[4], v[9]);
  v[5] = rot7_512(v[5]);
  v[6] = rot7_512(v[6]);
  v[7] = rot7_512(v[7]);
  v[4] = rot7_512(v[4]);
}

// 0b10001000, or lanes a0/a2/b0/b2 in little-endian order
#define LO_IMM8 0x88

INLINE __m512i unpack_lo_128(__m512i a, __m512i b) {
  return _mm512_shuffle_i32x4(a, b, LO_IMM8);
}

// 0b11011101, or lanes a1/a3/b1/b3 in little-endian order
#define HI_IMM8 0xdd

INLINE __m512i unpack_hi_128(__m512i a, __m512i b) {
  return _mm512_shuffle_i32x4(a, b, HI_IMM8);
}

INLINE void transpose_vecs_512(__m512i vecs[16]) {
  // Interleave 32-bit lanes. The _0 unpack is lanes
  // 0/0/1/1/4/4/5/5/8/8/9/9/12/12/13/13, and the _2 unpack is lanes
  // 2/2/3/3/6/6/7/7/10/10/11/11/14/14/15/15.
  __m512i ab_0 = _mm512_unpacklo_epi32(vecs[0], vecs[1]);
  __m512i ab_2 = _mm512_unpackhi_epi32(vecs[0], vecs[1]);
  __m512i cd_0 = _mm512_unpacklo_epi32(vecs[2], vecs[3]);
  __m512i cd_2 = _mm512_unpackhi_epi32(vecs[2], vecs[3]);
  __m512i ef_0 = _mm512_unpacklo_epi32(vecs[4], vecs[5]);
  __m512i ef_2 = _mm512_unpackhi_epi32(vecs[4], vecs[5]);
  __m512i gh_0 = _mm512_unpacklo_epi32(vecs[6], vecs[7]);
  __m512i gh_2 = _mm512_unpackhi_epi32(vecs[6], vecs[7]);
  __m512i ij_0 = _mm512_unpacklo_epi32(vecs[8], vecs[9]);
  __m512i ij_2 = _mm512_unpackhi_epi32(vecs[8], vecs[9]);
  __m512i kl_0 = _mm512_unpacklo_epi32(vecs[10], vecs[11]);
  __m512i kl_2 = _mm512_unpackhi_epi32(vecs[10], vecs[11]);
  __m512i mn_0 = _mm512_unpacklo_epi32(vecs[12], vecs[13]);
  __m512i mn_2 = _mm512_unpackhi_epi32(vecs[12], vecs[13]);
  __m512i op_0 = _mm512_unpacklo_epi32(vecs[14], vecs[15]);
  __m512i op_2 = _mm512_unpackhi_epi32(vecs[14], vecs[15]);

  // Interleave 64-bit lates. The _0 unpack is lanes
  // 0/0/0/0/4/4/4/4/8/8/8/8/12/12/12/12, the _1 unpack is lanes
  // 1/1/1/1/5/5/5/5/9/9/9/9/13/13/13/13, the _2 unpack is lanes
  // 2/2/2/2/6/6/6/6/10/10/10/10/14/14/14/14, and the _3 unpack is lanes
  // 3/3/3/3/7/7/7/7/11/11/11/11/15/15/15/15.
  __m512i abcd_0 = _mm512_unpacklo_epi64(ab_0, cd_0);
  __m512i abcd_1 = _mm512_unpackhi_epi64(ab_0, cd_0);
  __m512i abcd_2 = _mm512_unpacklo_epi64(ab_2, cd_2);
  __m512i abcd_3 = _mm512_unpackhi_epi64(ab_2, cd_2);
  __m512i efgh_0 = _mm512_unpacklo_epi64(ef_0, gh_0);
  __m512i efgh_1 = _mm512_unpackhi_epi64(ef_0, gh_0);
  __m512i efgh_2 = _mm512_unpacklo_epi64(ef_2, gh_2);
  __m512i efgh_3 = _mm512_unpackhi_epi64(ef_2, gh_2);
  __m512i ijkl_0 = _mm512_unpacklo_epi64(ij_0, kl_0);
  __m512i ijkl_1 = _mm512_unpackhi_epi64(ij_0, kl_0);
  __m512i ijkl_2 = _mm512_unpacklo_epi64(ij_2, kl_2);
  __m512i ijkl_3 = _mm512_unpackhi_epi64(ij_2, kl_2);
  __m512i mnop_0 = _mm512_unpacklo_epi64(mn_0, op_0);
  __m512i mnop_1 = _mm512_unpackhi_epi64(mn_0, op_0);
  __m512i mnop_2 = _mm512_unpacklo_epi64(mn_2, op_2);
  __m512i mnop_3 = _mm512_unpackhi_epi64(mn_2, op_2);

  // Interleave 128-bit lanes. The _0 unpack is
  // 0/0/0/0/8/8/8/8/0/0/0/0/8/8/8/8, the _1 unpack is
  // 1/1/1/1/9/9/9/9/1/1/1/1/9/9/9/9, and so on.
  __m512i abcdefgh_0 = unpack_lo_128(abcd_0, efgh_0);
  __m512i abcdefgh_1 = unpack_lo_128(abcd_1, efgh_1);
  __m512i abcdefgh_2 = unpack_lo_128(abcd_2, efgh_2);
  __m512i abcdefgh_3 = unpack_lo_128(abcd_3, efgh_3);
  __m512i abcdefgh_4 = unpack_hi_128(abcd_0, efgh_0);
  __m512i abcdefgh_5 = unpack_hi_128(abcd_1, efgh_1);
  __m512i abcdefgh_6 = unpack_hi_128(abcd_2, efgh_2);
  __m512i abcdefgh_7 = unpack_hi_128(abcd_3, efgh_3);
  __m512i ijklmnop_0 = unpack_lo_128(ijkl_0, mnop_0);
  __m512i ijklmnop_1 = unpack_lo_128(ijkl_1, mnop_1);
  __m512i ijklmnop_2 = unpack_lo_128(ijkl_2, mnop_2);
  __m512i ijklmnop_3 = unpack_lo_128(ijkl_3, mnop_3);
  __m512i ijklmnop_4 = unpack_hi_128(ijkl_0, mnop_0);
  __m512i ijklmnop_5 = unpack_hi_128(ijkl_1, mnop_1);
  __m512i ijklmnop_6 = unpack_hi_128(ijkl_2, mnop_2);
  __m512i ijklmnop_7 = unpack_hi_128(ijkl_3, mnop_3);

  // Interleave 128-bit lanes again for the final outputs.
  vecs[0] = unpack_lo_128(abcdefgh_0, ijklmnop_0);
  vecs[1] = unpack_lo_128(abcdefgh_1, ijklmnop_1);
  vecs[2] = unpack_lo_128(abcdefgh_2, ijklmnop_2);
  vecs[3] = unpack_lo_128(abcdefgh_3, ijklmnop_3);
  vecs[4] = unpack_lo_128(abcdefgh_4, ijklmnop_4);
  vecs[5] = unpack_lo_128(abcdefgh_5, ijklmnop_5);
  vecs[6] = unpack_lo_128(abcdefgh_6, ijklmnop_6);
  vecs[7] = unpack_lo_128(abcdefgh_7, ijklmnop_7);
  vecs[8] = unpack_hi_128(abcdefgh_0, ijklmnop_0);
  vecs[9] = unpack_hi_128(abcdefgh_1, ijklmnop_1);
  vecs[10] = unpack_hi_128(abcdefgh_2, ijklmnop_2);
  vecs[11] = unpack_hi_128(abcdefgh_3, ijklmnop_3);
  vecs[12] = unpack_hi_128(abcdefgh_4, ijklmnop_4);
  vecs[13] = unpack_hi_128(abcdefgh_5, ijklmnop_5);
  vecs[14] = unpack_hi_128(abcdefgh_6, ijklmnop_6);
  vecs[15] = unpack_hi_128(abcdefgh_7, ijklmnop_7);
}

INLINE void transpose_msg_vecs16(const uint8_t *const *inputs,
                                 size_t block_offset, __m512i out[16]) {
  out[0] = loadu_512(&inputs[0][block_offset]);
  out[1] = loadu_512(&inputs[1][block_offset]);
  out[2] = loadu_512(&inputs[2][block_offset]);
  out[3] = loadu_512(&inputs[3][block_offset]);
  out[4] = loadu_512(&inputs[4][block_offset]);
  out[5] = loadu_512(&inputs[5][block_offset]);
  out[6] = loadu_512(&inputs[6][block_offset]);
  out[7] = loadu_512(&inputs[7][block_offset]);
  out[8] = loadu_512(&inputs[8][block_offset]);
  out[9] = loadu_512(&inputs[9][block_offset]);
  out[10] = loadu_512(&inputs[10][block_offset]);
  out[11] = loadu_512(&inputs[11][block_offset]);
  out[12] = loadu_512(&inputs[12][block_offset]);
  out[13] = loadu_512(&inputs[13][block_offset]);
  out[14] = loadu_512(&inputs[14][block_offset]);
  out[15] = loadu_512(&inputs[15][block_offset]);
  transpose_vecs_512(out);
}

INLINE void load_offsets16(uint64_t offset, const uint64_t deltas[16],
                           __m512i *out_lo, __m512i *out_hi) {
  __m512i a = _mm512_add_epi64(_mm512_set1_epi64(offset),
                               _mm512_loadu_si512((const __m512i *)&deltas[0]));
  __m512i b = _mm512_add_epi64(_mm512_set1_epi64(offset),
                               _mm512_loadu_si512((const __m512i *)&deltas[8]));
  __m256i a_lo = _mm512_cvtepi64_epi32(a);
  __m256i b_lo = _mm512_cvtepi64_epi32(b);
  __m256i a_hi = _mm512_cvtepi64_epi32(_mm512_srli_epi64(a, 32));
  __m256i b_hi = _mm512_cvtepi64_epi32(_mm512_srli_epi64(b, 32));
  // Note that _mm512_inserti32x8 requires AVX512DQ
  *out_lo = _mm512_inserti64x4(_mm512_castsi256_si512(a_lo), b_lo, 1);
  *out_hi = _mm512_inserti64x4(_mm512_castsi256_si512(a_hi), b_hi, 1);
}

void hash16_avx512(const uint8_t *const *inputs, size_t blocks,
                   const uint32_t key_words[8], uint64_t offset,
                   const uint64_t offset_deltas[16],
                   uint8_t internal_flags_start, uint8_t internal_flags_end,
                   uint32_t context, uint8_t *out) {
  __m512i h_vecs[8] = {
      xor_512(set1_512(IV[0]), set1_512(key_words[0])),
      xor_512(set1_512(IV[1]), set1_512(key_words[1])),
      xor_512(set1_512(IV[2]), set1_512(key_words[2])),
      xor_512(set1_512(IV[3]), set1_512(key_words[3])),
      xor_512(set1_512(IV[4]), set1_512(key_words[4])),
      xor_512(set1_512(IV[5]), set1_512(key_words[5])),
      xor_512(set1_512(IV[6]), set1_512(key_words[6])),
      xor_512(set1_512(IV[7]), set1_512(key_words[7])),
  };
  __m512i offset_low_vec, offset_high_vec;
  load_offsets16(offset, offset_deltas, &offset_low_vec, &offset_high_vec);
  const __m512i context_vec = set1_512(context);
  uint8_t internal_flags = internal_flags_start;

  for (size_t block = 0; block < blocks; block++) {
    if (block + 1 == blocks) {
      internal_flags |= internal_flags_end;
    }
    __m512i block_flags_vec = set1_512(block_flags(BLOCK_LEN, internal_flags));
    __m512i msg_vecs[16];
    transpose_msg_vecs16(inputs, block * BLOCK_LEN, msg_vecs);

    __m512i v[16] = {
        h_vecs[0],
        h_vecs[1],
        h_vecs[2],
        h_vecs[3],
        h_vecs[4],
        h_vecs[5],
        h_vecs[6],
        h_vecs[7],
        set1_512(IV[0]),
        set1_512(IV[1]),
        set1_512(IV[2]),
        set1_512(IV[3]),
        xor_512(set1_512(IV[4]), offset_low_vec),
        xor_512(set1_512(IV[5]), offset_high_vec),
        xor_512(set1_512(IV[6]), block_flags_vec),
        xor_512(set1_512(IV[7]), context_vec),
    };
    round_fn16(v, msg_vecs, 0);
    round_fn16(v, msg_vecs, 1);
    round_fn16(v, msg_vecs, 2);
    round_fn16(v, msg_vecs, 3);
    round_fn16(v, msg_vecs, 4);
    round_fn16(v, msg_vecs, 5);
    round_fn16(v, msg_vecs, 6);
    h_vecs[0] = xor_512(v[0], v[8]);
    h_vecs[1] = xor_512(v[1], v[9]);
    h_vecs[2] = xor_512(v[2], v[10]);
    h_vecs[3] = xor_512(v[3], v[11]);
    h_vecs[4] = xor_512(v[4], v[12]);
    h_vecs[5] = xor_512(v[5], v[13]);
    h_vecs[6] = xor_512(v[6], v[14]);
    h_vecs[7] = xor_512(v[7], v[15]);

    internal_flags = 0;
  }

  // transpose_vecs_512 operates on a 16x16 matrix of words, but we only have 8
  // state vectors. Pad the matrix with zeros. After transposition, store the
  // lower half of each vector.
  __m512i padded[16] = {
      h_vecs[0],   h_vecs[1],   h_vecs[2],   h_vecs[3],
      h_vecs[4],   h_vecs[5],   h_vecs[6],   h_vecs[7],
      set1_512(0), set1_512(0), set1_512(0), set1_512(0),
      set1_512(0), set1_512(0), set1_512(0), set1_512(0),
  };
  transpose_vecs_512(padded);
  storeu_256(_mm512_castsi512_si256(padded[0]), &out[0 * sizeof(__m256i)]);
  storeu_256(_mm512_castsi512_si256(padded[1]), &out[1 * sizeof(__m256i)]);
  storeu_256(_mm512_castsi512_si256(padded[2]), &out[2 * sizeof(__m256i)]);
  storeu_256(_mm512_castsi512_si256(padded[3]), &out[3 * sizeof(__m256i)]);
  storeu_256(_mm512_castsi512_si256(padded[4]), &out[4 * sizeof(__m256i)]);
  storeu_256(_mm512_castsi512_si256(padded[5]), &out[5 * sizeof(__m256i)]);
  storeu_256(_mm512_castsi512_si256(padded[6]), &out[6 * sizeof(__m256i)]);
  storeu_256(_mm512_castsi512_si256(padded[7]), &out[7 * sizeof(__m256i)]);
  storeu_256(_mm512_castsi512_si256(padded[8]), &out[8 * sizeof(__m256i)]);
  storeu_256(_mm512_castsi512_si256(padded[9]), &out[9 * sizeof(__m256i)]);
  storeu_256(_mm512_castsi512_si256(padded[10]), &out[10 * sizeof(__m256i)]);
  storeu_256(_mm512_castsi512_si256(padded[11]), &out[11 * sizeof(__m256i)]);
  storeu_256(_mm512_castsi512_si256(padded[12]), &out[12 * sizeof(__m256i)]);
  storeu_256(_mm512_castsi512_si256(padded[13]), &out[13 * sizeof(__m256i)]);
  storeu_256(_mm512_castsi512_si256(padded[14]), &out[14 * sizeof(__m256i)]);
  storeu_256(_mm512_castsi512_si256(padded[15]), &out[15 * sizeof(__m256i)]);
}

/*
 * ----------------------------------------------------------------------------
 * hash_many_avx512
 * ----------------------------------------------------------------------------
 */

INLINE void hash_one_avx512(const uint8_t *input, size_t blocks,
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
    compress_avx512(state, input, BLOCK_LEN, offset, flags, context);
    input = &input[BLOCK_LEN];
    blocks -= 1;
    flags = 0;
  }
  memcpy(out, state, OUT_LEN);
}

void hash_many_avx512(const uint8_t *const *inputs, size_t num_inputs,
                      size_t blocks, const uint32_t key_words[8],
                      uint64_t offset, const uint64_t offset_deltas[17],
                      uint8_t internal_flags_start, uint8_t internal_flags_end,
                      uint32_t context, uint8_t *out) {
  while (num_inputs >= 16) {
    hash16_avx512(inputs, blocks, key_words, offset, offset_deltas,
                  internal_flags_start, internal_flags_end, context, out);
    inputs += 16;
    num_inputs -= 16;
    offset += offset_deltas[16];
    out = &out[16 * OUT_LEN];
  }
  while (num_inputs >= 8) {
    hash8_avx512(inputs, blocks, key_words, offset, offset_deltas,
                 internal_flags_start, internal_flags_end, context, out);
    inputs += 8;
    num_inputs -= 8;
    offset += offset_deltas[8];
    out = &out[8 * OUT_LEN];
  }
  while (num_inputs >= 4) {
    hash4_avx512(inputs, blocks, key_words, offset, offset_deltas,
                 internal_flags_start, internal_flags_end, context, out);
    inputs += 4;
    num_inputs -= 4;
    offset += offset_deltas[4];
    out = &out[4 * OUT_LEN];
  }
  while (num_inputs > 0) {
    hash_one_avx512(inputs[0], blocks, key_words, offset, internal_flags_start,
                    internal_flags_end, context, out);
    inputs += 1;
    num_inputs -= 1;
    offset += offset_deltas[1];
    out = &out[OUT_LEN];
  }
}

#else // __AVX512F__ or __AVX512VL__ not defined

// When AVX512 isn't enabled in the build (e.g. with -march=native, depending
// on the platform), other C code doesn't call into this file at all. But the
// Rust test framework links against these functions unconditionally, and then
// does runtime feature detection to decide whether to run tests. So we need to
// provide empty stubs in the not-supported case, to avoid breaking the build.

void compress_avx512(uint32_t state[8], const uint8_t block[BLOCK_LEN],
                     uint8_t block_len, uint64_t offset, uint8_t internal_flags,
                     uint32_t context) {
  // Suppress unused parameter warnings.
  (void)state;
  (void)block;
  (void)block_len;
  (void)offset;
  (void)internal_flags;
  (void)context;
  assert(false);
}

void hash_many_avx512(const uint8_t *const *inputs, size_t num_inputs,
                      size_t blocks, const uint32_t key_words[8],
                      uint64_t offset, const uint64_t offset_deltas[16],
                      uint8_t internal_flags_start, uint8_t internal_flags_end,
                      uint32_t context, uint8_t *out) {
  // Suppress unused parameter warnings.
  (void)inputs;
  (void)num_inputs;
  (void)blocks;
  (void)key_words;
  (void)offset;
  (void)offset_deltas;
  (void)internal_flags_start;
  (void)internal_flags_end;
  (void)context;
  (void)out;
  assert(false);
}

#endif
