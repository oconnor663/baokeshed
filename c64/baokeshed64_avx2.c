#include "baokeshed64_impl.h"

#include <immintrin.h>

#define DEGREE 4

INLINE __m256i loadu(const uint8_t src[16]) {
  return _mm256_loadu_si256((const __m256i *)src);
}

INLINE void storeu(__m256i src, uint8_t dest[16]) {
  _mm256_storeu_si256((__m256i *)dest, src);
}

INLINE __m256i addv(__m256i a, __m256i b) { return _mm256_add_epi64(a, b); }

// Note that clang-format doesn't like the name "xor" for some reason.
INLINE __m256i xorv(__m256i a, __m256i b) { return _mm256_xor_si256(a, b); }

INLINE __m256i set1(uint64_t x) { return _mm256_set1_epi64x((int64_t)x); }

INLINE __m256i set4(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
  return _mm256_setr_epi64x((int64_t)a, (int64_t)b, (int64_t)c, (int64_t)d);
}

INLINE __m256i rot32(__m256i x) {
  return _mm256_shuffle_epi32(x, _MM_SHUFFLE(2, 3, 0, 1));
}

INLINE __m256i rot24(__m256i x) {
  return _mm256_shuffle_epi8(x, _mm256_setr_epi8(3, 4, 5, 6, 7, 0, 1, 2, 11, 12,
                                                 13, 14, 15, 8, 9, 10, 3, 4, 5,
                                                 6, 7, 0, 1, 2, 11, 12, 13, 14,
                                                 15, 8, 9, 10));
}

INLINE __m256i rot16(__m256i x) {
  return _mm256_shuffle_epi8(x, _mm256_setr_epi8(2, 3, 4, 5, 6, 7, 0, 1, 10, 11,
                                                 12, 13, 14, 15, 8, 9, 2, 3, 4,
                                                 5, 6, 7, 0, 1, 10, 11, 12, 13,
                                                 14, 15, 8, 9));
}

INLINE __m256i rot63(__m256i x) {
  return _mm256_or_si256(_mm256_srli_epi64(x, 63), _mm256_add_epi64(x, x));
}

INLINE void round_fn(__m256i v[16], __m256i m[16], size_t r) {
  v[0] = addv(v[0], m[(size_t)MSG_SCHEDULE[r][0]]);
  v[1] = addv(v[1], m[(size_t)MSG_SCHEDULE[r][2]]);
  v[2] = addv(v[2], m[(size_t)MSG_SCHEDULE[r][4]]);
  v[3] = addv(v[3], m[(size_t)MSG_SCHEDULE[r][6]]);
  v[0] = addv(v[0], v[4]);
  v[1] = addv(v[1], v[5]);
  v[2] = addv(v[2], v[6]);
  v[3] = addv(v[3], v[7]);
  v[12] = xorv(v[12], v[0]);
  v[13] = xorv(v[13], v[1]);
  v[14] = xorv(v[14], v[2]);
  v[15] = xorv(v[15], v[3]);
  v[12] = rot32(v[12]);
  v[13] = rot32(v[13]);
  v[14] = rot32(v[14]);
  v[15] = rot32(v[15]);
  v[8] = addv(v[8], v[12]);
  v[9] = addv(v[9], v[13]);
  v[10] = addv(v[10], v[14]);
  v[11] = addv(v[11], v[15]);
  v[4] = xorv(v[4], v[8]);
  v[5] = xorv(v[5], v[9]);
  v[6] = xorv(v[6], v[10]);
  v[7] = xorv(v[7], v[11]);
  v[4] = rot24(v[4]);
  v[5] = rot24(v[5]);
  v[6] = rot24(v[6]);
  v[7] = rot24(v[7]);
  v[0] = addv(v[0], m[(size_t)MSG_SCHEDULE[r][1]]);
  v[1] = addv(v[1], m[(size_t)MSG_SCHEDULE[r][3]]);
  v[2] = addv(v[2], m[(size_t)MSG_SCHEDULE[r][5]]);
  v[3] = addv(v[3], m[(size_t)MSG_SCHEDULE[r][7]]);
  v[0] = addv(v[0], v[4]);
  v[1] = addv(v[1], v[5]);
  v[2] = addv(v[2], v[6]);
  v[3] = addv(v[3], v[7]);
  v[12] = xorv(v[12], v[0]);
  v[13] = xorv(v[13], v[1]);
  v[14] = xorv(v[14], v[2]);
  v[15] = xorv(v[15], v[3]);
  v[12] = rot16(v[12]);
  v[13] = rot16(v[13]);
  v[14] = rot16(v[14]);
  v[15] = rot16(v[15]);
  v[8] = addv(v[8], v[12]);
  v[9] = addv(v[9], v[13]);
  v[10] = addv(v[10], v[14]);
  v[11] = addv(v[11], v[15]);
  v[4] = xorv(v[4], v[8]);
  v[5] = xorv(v[5], v[9]);
  v[6] = xorv(v[6], v[10]);
  v[7] = xorv(v[7], v[11]);
  v[4] = rot63(v[4]);
  v[5] = rot63(v[5]);
  v[6] = rot63(v[6]);
  v[7] = rot63(v[7]);

  v[0] = addv(v[0], m[(size_t)MSG_SCHEDULE[r][8]]);
  v[1] = addv(v[1], m[(size_t)MSG_SCHEDULE[r][10]]);
  v[2] = addv(v[2], m[(size_t)MSG_SCHEDULE[r][12]]);
  v[3] = addv(v[3], m[(size_t)MSG_SCHEDULE[r][14]]);
  v[0] = addv(v[0], v[5]);
  v[1] = addv(v[1], v[6]);
  v[2] = addv(v[2], v[7]);
  v[3] = addv(v[3], v[4]);
  v[15] = xorv(v[15], v[0]);
  v[12] = xorv(v[12], v[1]);
  v[13] = xorv(v[13], v[2]);
  v[14] = xorv(v[14], v[3]);
  v[15] = rot32(v[15]);
  v[12] = rot32(v[12]);
  v[13] = rot32(v[13]);
  v[14] = rot32(v[14]);
  v[10] = addv(v[10], v[15]);
  v[11] = addv(v[11], v[12]);
  v[8] = addv(v[8], v[13]);
  v[9] = addv(v[9], v[14]);
  v[5] = xorv(v[5], v[10]);
  v[6] = xorv(v[6], v[11]);
  v[7] = xorv(v[7], v[8]);
  v[4] = xorv(v[4], v[9]);
  v[5] = rot24(v[5]);
  v[6] = rot24(v[6]);
  v[7] = rot24(v[7]);
  v[4] = rot24(v[4]);
  v[0] = addv(v[0], m[(size_t)MSG_SCHEDULE[r][9]]);
  v[1] = addv(v[1], m[(size_t)MSG_SCHEDULE[r][11]]);
  v[2] = addv(v[2], m[(size_t)MSG_SCHEDULE[r][13]]);
  v[3] = addv(v[3], m[(size_t)MSG_SCHEDULE[r][15]]);
  v[0] = addv(v[0], v[5]);
  v[1] = addv(v[1], v[6]);
  v[2] = addv(v[2], v[7]);
  v[3] = addv(v[3], v[4]);
  v[15] = xorv(v[15], v[0]);
  v[12] = xorv(v[12], v[1]);
  v[13] = xorv(v[13], v[2]);
  v[14] = xorv(v[14], v[3]);
  v[15] = rot16(v[15]);
  v[12] = rot16(v[12]);
  v[13] = rot16(v[13]);
  v[14] = rot16(v[14]);
  v[10] = addv(v[10], v[15]);
  v[11] = addv(v[11], v[12]);
  v[8] = addv(v[8], v[13]);
  v[9] = addv(v[9], v[14]);
  v[5] = xorv(v[5], v[10]);
  v[6] = xorv(v[6], v[11]);
  v[7] = xorv(v[7], v[8]);
  v[4] = xorv(v[4], v[9]);
  v[5] = rot63(v[5]);
  v[6] = rot63(v[6]);
  v[7] = rot63(v[7]);
  v[4] = rot63(v[4]);
}

INLINE void transpose_vecs(__m256i vecs[DEGREE]) {
  // Interleave 64-bit lates. The low unpack is lanes 00/22 and the high is
  // 11/33.
  __m256i ab_02 = _mm256_unpacklo_epi64(vecs[0], vecs[1]);
  __m256i ab_13 = _mm256_unpackhi_epi64(vecs[0], vecs[1]);
  __m256i cd_02 = _mm256_unpacklo_epi64(vecs[2], vecs[3]);
  __m256i cd_13 = _mm256_unpackhi_epi64(vecs[2], vecs[3]);

  // Interleave 128-bit lanes.
  vecs[0] = _mm256_permute2x128_si256(ab_02, cd_02, 0x20);
  vecs[2] = _mm256_permute2x128_si256(ab_02, cd_02, 0x31);
  vecs[1] = _mm256_permute2x128_si256(ab_13, cd_13, 0x20);
  vecs[3] = _mm256_permute2x128_si256(ab_13, cd_13, 0x31);
}

INLINE void transpose_msg_vecs(const uint8_t *const *inputs,
                               size_t block_offset, __m256i out[16]) {
  out[0] = loadu(&inputs[0][block_offset + 0 * sizeof(__m256i)]);
  out[1] = loadu(&inputs[1][block_offset + 0 * sizeof(__m256i)]);
  out[2] = loadu(&inputs[2][block_offset + 0 * sizeof(__m256i)]);
  out[3] = loadu(&inputs[3][block_offset + 0 * sizeof(__m256i)]);
  out[4] = loadu(&inputs[0][block_offset + 1 * sizeof(__m256i)]);
  out[5] = loadu(&inputs[1][block_offset + 1 * sizeof(__m256i)]);
  out[6] = loadu(&inputs[2][block_offset + 1 * sizeof(__m256i)]);
  out[7] = loadu(&inputs[3][block_offset + 1 * sizeof(__m256i)]);
  out[8] = loadu(&inputs[0][block_offset + 2 * sizeof(__m256i)]);
  out[9] = loadu(&inputs[1][block_offset + 2 * sizeof(__m256i)]);
  out[10] = loadu(&inputs[2][block_offset + 2 * sizeof(__m256i)]);
  out[11] = loadu(&inputs[3][block_offset + 2 * sizeof(__m256i)]);
  out[12] = loadu(&inputs[0][block_offset + 3 * sizeof(__m256i)]);
  out[13] = loadu(&inputs[1][block_offset + 3 * sizeof(__m256i)]);
  out[14] = loadu(&inputs[2][block_offset + 3 * sizeof(__m256i)]);
  out[15] = loadu(&inputs[3][block_offset + 3 * sizeof(__m256i)]);
  transpose_vecs(&out[0]);
  transpose_vecs(&out[4]);
  transpose_vecs(&out[8]);
  transpose_vecs(&out[12]);
}

INLINE void load_offsets(uint64_t offset, const uint64_t offset_deltas[4],
                         __m256i *out) {
  *out = set4(offset + offset_deltas[0], offset + offset_deltas[1],
              offset + offset_deltas[2], offset + offset_deltas[3]);
}

void baokeshed64_hash4_avx2(const uint8_t *const *inputs, size_t blocks,
                            const uint64_t key_words[4], uint64_t offset,
                            const uint64_t offset_deltas[4], uint8_t flags,
                            uint8_t flags_start, uint8_t flags_end,
                            uint8_t *out) {
  __m256i h_vecs[4] = {
      set1(key_words[0]),
      set1(key_words[1]),
      set1(key_words[2]),
      set1(key_words[3]),
  };
  __m256i offset_vec;
  load_offsets(offset, offset_deltas, &offset_vec);
  uint8_t block_flags = flags | flags_start;

  for (size_t block = 0; block < blocks; block++) {
    if (block + 1 == blocks) {
      block_flags |= flags_end;
    }
    __m256i block_len_vec = set1(BLOCK_LEN);
    __m256i block_flags_vec = set1(block_flags);
    __m256i msg_vecs[16];
    transpose_msg_vecs(inputs, block * BLOCK_LEN, msg_vecs);

    __m256i v[16] = {
        h_vecs[0],
        h_vecs[1],
        h_vecs[2],
        h_vecs[3],
        set1(IV[4]),
        set1(IV[5]),
        set1(IV[6]),
        set1(IV[7]),
        set1(IV[0]),
        set1(IV[1]),
        set1(IV[2]),
        set1(IV[3]),
        xorv(set1(IV[4]), offset_vec),
        set1(IV[5]),
        xorv(set1(IV[6]), block_len_vec),
        xorv(set1(IV[7]), block_flags_vec),
    };
    round_fn(v, msg_vecs, 0);
    round_fn(v, msg_vecs, 1);
    round_fn(v, msg_vecs, 2);
    round_fn(v, msg_vecs, 3);
    round_fn(v, msg_vecs, 4);
    round_fn(v, msg_vecs, 5);
    round_fn(v, msg_vecs, 6);
    round_fn(v, msg_vecs, 7);
    h_vecs[0] = xorv(v[0], v[4]);
    h_vecs[1] = xorv(v[1], v[5]);
    h_vecs[2] = xorv(v[2], v[6]);
    h_vecs[3] = xorv(v[3], v[7]);

    block_flags = flags;
  }

  transpose_vecs(h_vecs);
  storeu(h_vecs[0], &out[0 * sizeof(__m256i)]);
  storeu(h_vecs[1], &out[1 * sizeof(__m256i)]);
  storeu(h_vecs[2], &out[2 * sizeof(__m256i)]);
  storeu(h_vecs[3], &out[3 * sizeof(__m256i)]);
}

void baokeshed64_hash_many_avx2(const uint8_t *const *inputs, size_t num_inputs,
                                size_t blocks, const uint64_t key_words[4],
                                uint64_t offset,
                                const uint64_t offset_deltas[5], uint8_t flags,
                                uint8_t flags_start, uint8_t flags_end,
                                uint8_t *out) {
  while (num_inputs >= DEGREE) {
    baokeshed64_hash4_avx2(inputs, blocks, key_words, offset, offset_deltas,
                           flags, flags_start, flags_end, out);
    inputs += DEGREE;
    num_inputs -= DEGREE;
    offset += offset_deltas[DEGREE];
    out = &out[DEGREE * OUT_LEN];
  }
  baokeshed64_hash_many_sse41(inputs, num_inputs, blocks, key_words, offset,
                              offset_deltas, flags, flags_start, flags_end,
                              out);
}
