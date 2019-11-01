#include "baokeshed_impl.h"

#ifdef __SSE4_1__

#include <immintrin.h>

#define DEGREE 4

INLINE __m128i loadu(const uint8_t src[16]) {
  return _mm_loadu_si128((const __m128i *)src);
}

INLINE void storeu(__m128i src, uint8_t dest[16]) {
  _mm_storeu_si128((__m128i *)dest, src);
}

INLINE __m128i addv(__m128i a, __m128i b) { return _mm_add_epi32(a, b); }

// Note that clang-format doesn't like the name "xor" for some reason.
INLINE __m128i xorv(__m128i a, __m128i b) { return _mm_xor_si128(a, b); }

INLINE __m128i set1(uint32_t x) { return _mm_set1_epi32((int32_t)x); }

INLINE __m128i set4(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  return _mm_setr_epi32((int32_t)a, (int32_t)b, (int32_t)c, (int32_t)d);
}

INLINE __m128i rot16(__m128i x) {
  return _mm_shuffle_epi8(
      x, _mm_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2));
}

INLINE __m128i rot12(__m128i x) {
  return xorv(_mm_srli_epi32(x, 12), _mm_slli_epi32(x, 32 - 12));
}

INLINE __m128i rot8(__m128i x) {
  return _mm_shuffle_epi8(
      x, _mm_set_epi8(12, 15, 14, 13, 8, 11, 10, 9, 4, 7, 6, 5, 0, 3, 2, 1));
}

INLINE __m128i rot7(__m128i x) {
  return xorv(_mm_srli_epi32(x, 7), _mm_slli_epi32(x, 32 - 7));
}

INLINE void g1(__m128i *row1, __m128i *row2, __m128i *row3, __m128i *row4,
               __m128i m) {
  *row1 = addv(addv(*row1, m), *row2);
  *row4 = xorv(*row4, *row1);
  *row4 = rot16(*row4);
  *row3 = addv(*row3, *row4);
  *row2 = xorv(*row2, *row3);
  *row2 = rot12(*row2);
}

INLINE void g2(__m128i *row1, __m128i *row2, __m128i *row3, __m128i *row4,
               __m128i m) {
  *row1 = addv(addv(*row1, m), *row2);
  *row4 = xorv(*row4, *row1);
  *row4 = rot8(*row4);
  *row3 = addv(*row3, *row4);
  *row2 = xorv(*row2, *row3);
  *row2 = rot7(*row2);
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

void compress_sse41(uint32_t state[8], const uint8_t block[BLOCK_LEN],
                    uint8_t block_len, uint64_t offset, uint8_t flags) {
  __m128i row1 = loadu((uint8_t *)&state[0]);
  __m128i row2 = loadu((uint8_t *)&state[4]);
  __m128i row3 = set4(IV[0], IV[1], IV[2], IV[3]);
  __m128i row4 = set4(IV[4] ^ offset_low(offset), IV[5] ^ offset_high(offset),
                      IV[6] ^ (uint32_t)block_len, IV[7] ^ (uint32_t)flags);

  __m128i m0 = loadu(&block[sizeof(__m128i) * 0]);
  __m128i m1 = loadu(&block[sizeof(__m128i) * 1]);
  __m128i m2 = loadu(&block[sizeof(__m128i) * 2]);
  __m128i m3 = loadu(&block[sizeof(__m128i) * 3]);

  __m128i buf, t0, t1, t2;

  // round 1
  buf = _mm_castps_si128(_mm_shuffle_ps(
      _mm_castsi128_ps(m0), _mm_castsi128_ps(m1), _MM_SHUFFLE(2, 0, 2, 0)));
  g1(&row1, &row2, &row3, &row4, buf);
  buf = _mm_castps_si128(_mm_shuffle_ps(
      _mm_castsi128_ps(m0), _mm_castsi128_ps(m1), _MM_SHUFFLE(3, 1, 3, 1)));
  g2(&row1, &row2, &row3, &row4, buf);
  diagonalize(&row1, &row3, &row4);
  t0 = _mm_shuffle_epi32(m2, _MM_SHUFFLE(3, 2, 0, 1));
  t1 = _mm_shuffle_epi32(m3, _MM_SHUFFLE(0, 1, 3, 2));
  buf = _mm_blend_epi16(t0, t1, 0xC3);
  g1(&row1, &row2, &row3, &row4, buf);
  t0 = _mm_blend_epi16(t0, t1, 0x3C);
  buf = _mm_shuffle_epi32(t0, _MM_SHUFFLE(2, 3, 0, 1));
  g2(&row1, &row2, &row3, &row4, buf);
  undiagonalize(&row1, &row3, &row4);

  // round 2
  t0 = _mm_blend_epi16(m1, m2, 0x0C);
  t1 = _mm_slli_si128(m3, 4);
  t2 = _mm_blend_epi16(t0, t1, 0xF0);
  buf = _mm_shuffle_epi32(t2, _MM_SHUFFLE(2, 1, 0, 3));
  g1(&row1, &row2, &row3, &row4, buf);
  t0 = _mm_shuffle_epi32(m2, _MM_SHUFFLE(0, 0, 2, 0));
  t1 = _mm_blend_epi16(m1, m3, 0xC0);
  t2 = _mm_blend_epi16(t0, t1, 0xF0);
  buf = _mm_shuffle_epi32(t2, _MM_SHUFFLE(2, 3, 0, 1));
  g2(&row1, &row2, &row3, &row4, buf);
  diagonalize(&row1, &row3, &row4);
  t0 = _mm_slli_si128(m1, 4);
  t1 = _mm_blend_epi16(m2, t0, 0x30);
  t2 = _mm_blend_epi16(m0, t1, 0xF0);
  buf = _mm_shuffle_epi32(t2, _MM_SHUFFLE(3, 0, 1, 2));
  g1(&row1, &row2, &row3, &row4, buf);
  t0 = _mm_unpackhi_epi32(m0, m1);
  t1 = _mm_slli_si128(m3, 4);
  t2 = _mm_blend_epi16(t0, t1, 0x0C);
  buf = _mm_shuffle_epi32(t2, _MM_SHUFFLE(3, 0, 1, 2));
  g2(&row1, &row2, &row3, &row4, buf);
  undiagonalize(&row1, &row3, &row4);

  // round 3
  t0 = _mm_unpackhi_epi32(m2, m3);
  t1 = _mm_blend_epi16(m3, m1, 0x0C);
  t2 = _mm_blend_epi16(t0, t1, 0x0F);
  buf = _mm_shuffle_epi32(t2, _MM_SHUFFLE(3, 1, 0, 2));
  g1(&row1, &row2, &row3, &row4, buf);
  t0 = _mm_unpacklo_epi32(m2, m0);
  t1 = _mm_blend_epi16(t0, m0, 0xF0);
  t2 = _mm_slli_si128(m3, 8);
  buf = _mm_blend_epi16(t1, t2, 0xC0);
  g2(&row1, &row2, &row3, &row4, buf);
  diagonalize(&row1, &row3, &row4);
  t0 = _mm_blend_epi16(m0, m2, 0x3C);
  t1 = _mm_srli_si128(m1, 12);
  t2 = _mm_blend_epi16(t0, t1, 0x03);
  buf = _mm_shuffle_epi32(t2, _MM_SHUFFLE(0, 3, 2, 1));
  g1(&row1, &row2, &row3, &row4, buf);
  t0 = _mm_slli_si128(m3, 4);
  t1 = _mm_blend_epi16(m0, m1, 0x33);
  t2 = _mm_blend_epi16(t1, t0, 0xC0);
  buf = _mm_shuffle_epi32(t2, _MM_SHUFFLE(1, 2, 3, 0));
  g2(&row1, &row2, &row3, &row4, buf);
  undiagonalize(&row1, &row3, &row4);

  // round 4
  t0 = _mm_unpackhi_epi32(m0, m1);
  t1 = _mm_unpackhi_epi32(t0, m2);
  t2 = _mm_blend_epi16(t1, m3, 0x0C);
  buf = _mm_shuffle_epi32(t2, _MM_SHUFFLE(3, 1, 0, 2));
  g1(&row1, &row2, &row3, &row4, buf);
  t0 = _mm_slli_si128(m2, 8);
  t1 = _mm_blend_epi16(m3, m0, 0x0C);
  t2 = _mm_blend_epi16(t1, t0, 0xC0);
  buf = _mm_shuffle_epi32(t2, _MM_SHUFFLE(2, 0, 1, 3));
  g2(&row1, &row2, &row3, &row4, buf);
  diagonalize(&row1, &row3, &row4);
  t0 = _mm_blend_epi16(m0, m1, 0x0F);
  t1 = _mm_blend_epi16(t0, m3, 0xC0);
  buf = _mm_shuffle_epi32(t1, _MM_SHUFFLE(0, 1, 2, 3));
  g1(&row1, &row2, &row3, &row4, buf);
  t0 = _mm_alignr_epi8(m0, m1, 4);
  buf = _mm_blend_epi16(t0, m2, 0x33);
  g2(&row1, &row2, &row3, &row4, buf);
  undiagonalize(&row1, &row3, &row4);

  // round 5
  t0 = _mm_unpacklo_epi64(m1, m2);
  t1 = _mm_unpackhi_epi64(m0, m2);
  t2 = _mm_blend_epi16(t0, t1, 0x33);
  buf = _mm_shuffle_epi32(t2, _MM_SHUFFLE(2, 0, 1, 3));
  g1(&row1, &row2, &row3, &row4, buf);
  t0 = _mm_unpackhi_epi64(m1, m3);
  t1 = _mm_unpacklo_epi64(m0, m1);
  buf = _mm_blend_epi16(t0, t1, 0x33);
  g2(&row1, &row2, &row3, &row4, buf);
  diagonalize(&row1, &row3, &row4);
  t0 = _mm_unpackhi_epi64(m3, m1);
  t1 = _mm_unpackhi_epi64(m2, m0);
  t2 = _mm_blend_epi16(t1, t0, 0x33);
  buf = _mm_shuffle_epi32(t2, _MM_SHUFFLE(2, 1, 0, 3));
  g1(&row1, &row2, &row3, &row4, buf);
  t0 = _mm_blend_epi16(m0, m2, 0x03);
  t1 = _mm_slli_si128(t0, 8);
  t2 = _mm_blend_epi16(t1, m3, 0x0F);
  buf = _mm_shuffle_epi32(t2, _MM_SHUFFLE(2, 0, 3, 1));
  g2(&row1, &row2, &row3, &row4, buf);
  undiagonalize(&row1, &row3, &row4);

  // round 6
  t0 = _mm_unpackhi_epi32(m0, m1);
  t1 = _mm_unpacklo_epi32(m0, m2);
  buf = _mm_unpacklo_epi64(t0, t1);
  g1(&row1, &row2, &row3, &row4, buf);
  t0 = _mm_srli_si128(m2, 4);
  t1 = _mm_blend_epi16(m0, m3, 0x03);
  buf = _mm_blend_epi16(t1, t0, 0x3C);
  g2(&row1, &row2, &row3, &row4, buf);
  diagonalize(&row1, &row3, &row4);
  t0 = _mm_blend_epi16(m1, m0, 0x0C);
  t1 = _mm_srli_si128(m3, 4);
  t2 = _mm_blend_epi16(t0, t1, 0x30);
  buf = _mm_shuffle_epi32(t2, _MM_SHUFFLE(2, 3, 0, 1));
  g1(&row1, &row2, &row3, &row4, buf);
  t0 = _mm_unpacklo_epi64(m2, m1);
  t1 = _mm_shuffle_epi32(m3, _MM_SHUFFLE(2, 0, 1, 0));
  t2 = _mm_srli_si128(t0, 4);
  buf = _mm_blend_epi16(t1, t2, 0x33);
  g2(&row1, &row2, &row3, &row4, buf);
  undiagonalize(&row1, &row3, &row4);

  // round 7
  t0 = _mm_slli_si128(m1, 12);
  t1 = _mm_blend_epi16(m0, m3, 0x33);
  buf = _mm_blend_epi16(t1, t0, 0xC0);
  g1(&row1, &row2, &row3, &row4, buf);
  t0 = _mm_blend_epi16(m3, m2, 0x30);
  t1 = _mm_srli_si128(m1, 4);
  t2 = _mm_blend_epi16(t0, t1, 0x03);
  buf = _mm_shuffle_epi32(t2, _MM_SHUFFLE(2, 1, 3, 0));
  g2(&row1, &row2, &row3, &row4, buf);
  diagonalize(&row1, &row3, &row4);
  t0 = _mm_unpacklo_epi64(m0, m2);
  t1 = _mm_srli_si128(m1, 4);
  buf =
      _mm_shuffle_epi32(_mm_blend_epi16(t0, t1, 0x0C), _MM_SHUFFLE(3, 1, 0, 2));
  g1(&row1, &row2, &row3, &row4, buf);
  t0 = _mm_unpackhi_epi32(m1, m2);
  t1 = _mm_unpackhi_epi64(m0, t0);
  buf = _mm_shuffle_epi32(t1, _MM_SHUFFLE(0, 1, 2, 3));
  g2(&row1, &row2, &row3, &row4, buf);
  undiagonalize(&row1, &row3, &row4);

  storeu(xorv(row1, row3), (uint8_t *)&state[0]);
  storeu(xorv(row2, row4), (uint8_t *)&state[4]);
}

INLINE void round_fn(__m128i v[16], __m128i m[16], size_t r) {
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
  v[4] = rot12(v[4]);
  v[5] = rot12(v[5]);
  v[6] = rot12(v[6]);
  v[7] = rot12(v[7]);
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
  v[12] = rot8(v[12]);
  v[13] = rot8(v[13]);
  v[14] = rot8(v[14]);
  v[15] = rot8(v[15]);
  v[8] = addv(v[8], v[12]);
  v[9] = addv(v[9], v[13]);
  v[10] = addv(v[10], v[14]);
  v[11] = addv(v[11], v[15]);
  v[4] = xorv(v[4], v[8]);
  v[5] = xorv(v[5], v[9]);
  v[6] = xorv(v[6], v[10]);
  v[7] = xorv(v[7], v[11]);
  v[4] = rot7(v[4]);
  v[5] = rot7(v[5]);
  v[6] = rot7(v[6]);
  v[7] = rot7(v[7]);

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
  v[5] = rot12(v[5]);
  v[6] = rot12(v[6]);
  v[7] = rot12(v[7]);
  v[4] = rot12(v[4]);
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
  v[15] = rot8(v[15]);
  v[12] = rot8(v[12]);
  v[13] = rot8(v[13]);
  v[14] = rot8(v[14]);
  v[10] = addv(v[10], v[15]);
  v[11] = addv(v[11], v[12]);
  v[8] = addv(v[8], v[13]);
  v[9] = addv(v[9], v[14]);
  v[5] = xorv(v[5], v[10]);
  v[6] = xorv(v[6], v[11]);
  v[7] = xorv(v[7], v[8]);
  v[4] = xorv(v[4], v[9]);
  v[5] = rot7(v[5]);
  v[6] = rot7(v[6]);
  v[7] = rot7(v[7]);
  v[4] = rot7(v[4]);
}

INLINE void transpose_vecs(__m128i vecs[DEGREE]) {
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

INLINE void transpose_msg_vecs(const uint8_t *const *inputs,
                               size_t block_offset, __m128i out[16]) {
  out[0] = loadu(&inputs[0][block_offset + 0 * sizeof(__m128i)]);
  out[1] = loadu(&inputs[1][block_offset + 0 * sizeof(__m128i)]);
  out[2] = loadu(&inputs[2][block_offset + 0 * sizeof(__m128i)]);
  out[3] = loadu(&inputs[3][block_offset + 0 * sizeof(__m128i)]);
  out[4] = loadu(&inputs[0][block_offset + 1 * sizeof(__m128i)]);
  out[5] = loadu(&inputs[1][block_offset + 1 * sizeof(__m128i)]);
  out[6] = loadu(&inputs[2][block_offset + 1 * sizeof(__m128i)]);
  out[7] = loadu(&inputs[3][block_offset + 1 * sizeof(__m128i)]);
  out[8] = loadu(&inputs[0][block_offset + 2 * sizeof(__m128i)]);
  out[9] = loadu(&inputs[1][block_offset + 2 * sizeof(__m128i)]);
  out[10] = loadu(&inputs[2][block_offset + 2 * sizeof(__m128i)]);
  out[11] = loadu(&inputs[3][block_offset + 2 * sizeof(__m128i)]);
  out[12] = loadu(&inputs[0][block_offset + 3 * sizeof(__m128i)]);
  out[13] = loadu(&inputs[1][block_offset + 3 * sizeof(__m128i)]);
  out[14] = loadu(&inputs[2][block_offset + 3 * sizeof(__m128i)]);
  out[15] = loadu(&inputs[3][block_offset + 3 * sizeof(__m128i)]);
  transpose_vecs(&out[0]);
  transpose_vecs(&out[4]);
  transpose_vecs(&out[8]);
  transpose_vecs(&out[12]);
}

INLINE void load_offsets(uint64_t offset, const uint64_t offset_deltas[4],
                         __m128i *out_low, __m128i *out_high) {
  *out_low = set4(offset_low(offset + offset_deltas[0]),
                  offset_low(offset + offset_deltas[1]),
                  offset_low(offset + offset_deltas[2]),
                  offset_low(offset + offset_deltas[3]));
  *out_high = set4(offset_high(offset + offset_deltas[0]),
                   offset_high(offset + offset_deltas[1]),
                   offset_high(offset + offset_deltas[2]),
                   offset_high(offset + offset_deltas[3]));
}

void hash4_sse41(const uint8_t *const *inputs, size_t blocks,
                 const uint32_t key_words[8], uint64_t offset,
                 const uint64_t offset_deltas[4], uint8_t flags,
                 uint8_t flags_start, uint8_t flags_end, uint8_t *out) {
  __m128i h_vecs[8] = {
      set1(key_words[0]), set1(key_words[1]), set1(key_words[2]),
      set1(key_words[3]), set1(key_words[4]), set1(key_words[5]),
      set1(key_words[6]), set1(key_words[7]),
  };
  __m128i offset_low_vec, offset_high_vec;
  load_offsets(offset, offset_deltas, &offset_low_vec, &offset_high_vec);
  uint8_t block_flags = flags | flags_start;

  for (size_t block = 0; block < blocks; block++) {
    if (block + 1 == blocks) {
      block_flags |= flags_end;
    }
    __m128i block_len_vec = set1(BLOCK_LEN);
    __m128i block_flags_vec = set1(block_flags);
    __m128i msg_vecs[16];
    transpose_msg_vecs(inputs, block * BLOCK_LEN, msg_vecs);

    __m128i v[16] = {
        h_vecs[0],
        h_vecs[1],
        h_vecs[2],
        h_vecs[3],
        h_vecs[4],
        h_vecs[5],
        h_vecs[6],
        h_vecs[7],
        set1(IV[0]),
        set1(IV[1]),
        set1(IV[2]),
        set1(IV[3]),
        xorv(set1(IV[4]), offset_low_vec),
        xorv(set1(IV[5]), offset_high_vec),
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
    h_vecs[0] = xorv(v[0], v[8]);
    h_vecs[1] = xorv(v[1], v[9]);
    h_vecs[2] = xorv(v[2], v[10]);
    h_vecs[3] = xorv(v[3], v[11]);
    h_vecs[4] = xorv(v[4], v[12]);
    h_vecs[5] = xorv(v[5], v[13]);
    h_vecs[6] = xorv(v[6], v[14]);
    h_vecs[7] = xorv(v[7], v[15]);

    block_flags = flags;
  }

  transpose_vecs(&h_vecs[0]);
  transpose_vecs(&h_vecs[4]);
  // The first four vecs now contain the first half of each output, and the
  // second four vecs contain the second half of each output.
  storeu(h_vecs[0], &out[0 * sizeof(__m128i)]);
  storeu(h_vecs[4], &out[1 * sizeof(__m128i)]);
  storeu(h_vecs[1], &out[2 * sizeof(__m128i)]);
  storeu(h_vecs[5], &out[3 * sizeof(__m128i)]);
  storeu(h_vecs[2], &out[4 * sizeof(__m128i)]);
  storeu(h_vecs[6], &out[5 * sizeof(__m128i)]);
  storeu(h_vecs[3], &out[6 * sizeof(__m128i)]);
  storeu(h_vecs[7], &out[7 * sizeof(__m128i)]);
}

INLINE void hash_one_sse41(const uint8_t *input, size_t blocks,
                           const uint32_t key_words[8], uint64_t offset,
                           uint8_t flags, uint8_t flags_start,
                           uint8_t flags_end, uint8_t out[OUT_LEN]) {
  uint32_t state[8];
  memcpy(state, key_words, KEY_LEN);
  uint8_t block_flags = flags | flags_start;
  while (blocks > 0) {
    if (blocks == 1) {
      block_flags |= flags_end;
    }
    compress_sse41(state, input, BLOCK_LEN, offset, block_flags);
    input = &input[BLOCK_LEN];
    blocks -= 1;
    block_flags = flags;
  }
  memcpy(out, state, OUT_LEN);
}

void hash_many_sse41(const uint8_t *const *inputs, size_t num_inputs,
                     size_t blocks, const uint32_t key_words[8],
                     uint64_t offset, const uint64_t offset_deltas[5],
                     uint8_t flags, uint8_t flags_start, uint8_t flags_end,
                     uint8_t *out) {
  while (num_inputs >= DEGREE) {
    hash4_sse41(inputs, blocks, key_words, offset, offset_deltas, flags,
                flags_start, flags_end, out);
    inputs += DEGREE;
    num_inputs -= DEGREE;
    offset += offset_deltas[DEGREE];
    out = &out[DEGREE * OUT_LEN];
  }
  while (num_inputs > 0) {
    hash_one_sse41(inputs[0], blocks, key_words, offset, flags, flags_start,
                   flags_end, out);
    inputs += 1;
    num_inputs -= 1;
    offset += offset_deltas[1];
    out = &out[OUT_LEN];
  }
}

#else // __SSE4_1__ not defined

// When SSE4.1 isn't enabled in the build (e.g. with -march=native, depending
// on the platform), other C code doesn't call into this file at all. But the
// Rust test framework links against these functions unconditionally, and then
// does runtime feature detection to decide whether to run tests. So we need to
// provide empty stubs in the not-supported case, to avoid breaking the build.

void compress_sse41(uint32_t state[8], const uint8_t block[BLOCK_LEN],
                    uint8_t block_len, uint64_t offset, uint8_t flags) {
  // Suppress unused parameter warnings.
  (void)state;
  (void)block;
  (void)block_len;
  (void)offset;
  (void)flags;
  assert(false);
}

void hash_many_sse41(const uint8_t *const *inputs, size_t num_inputs,
                     size_t blocks, const uint32_t key_words[8],
                     uint64_t offset, const uint64_t offset_deltas[5],
                     uint8_t flags, uint8_t flags_start, uint8_t flags_end,
                     uint8_t *out) {
  // Suppress unused parameter warnings.
  (void)inputs;
  (void)num_inputs;
  (void)blocks;
  (void)key_words;
  (void)offset;
  (void)offset_deltas;
  (void)flags;
  (void)flags_start;
  (void)flags_end;
  (void)out;
  assert(false);
}

#endif
