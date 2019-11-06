#include "baokeshed64_impl.h"

#include <immintrin.h>

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
  _mm_storeu_si128((__m128i *)dest, src);
}

INLINE void storeu_256(__m256i src, uint8_t dest[16]) {
  _mm256_storeu_si256((__m256i *)dest, src);
}

INLINE __m128i add_128(__m128i a, __m128i b) { return _mm_add_epi64(a, b); }

INLINE __m256i add_256(__m256i a, __m256i b) { return _mm256_add_epi64(a, b); }

INLINE __m512i add_512(__m512i a, __m512i b) { return _mm512_add_epi64(a, b); }

INLINE __m128i xor_128(__m128i a, __m128i b) { return _mm_xor_si128(a, b); }

INLINE __m256i xor_256(__m256i a, __m256i b) { return _mm256_xor_si256(a, b); }

INLINE __m512i xor_512(__m512i a, __m512i b) { return _mm512_xor_si512(a, b); }

INLINE __m128i set1_128(uint64_t x) { return _mm_set1_epi64x((int64_t)x); }

INLINE __m256i set1_256(uint64_t x) { return _mm256_set1_epi64x((int64_t)x); }

INLINE __m512i set1_512(uint64_t x) { return _mm512_set1_epi64((int64_t)x); }

INLINE __m128i set2(uint64_t a, uint64_t b) {
  return _mm_set_epi64x((int64_t)b,
                        (int64_t)a); // set, not setr, note argument order
}

INLINE __m256i set4(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
  return _mm256_setr_epi64x((int64_t)a, (int64_t)b, (int64_t)c, (int64_t)d);
}

INLINE __m128i rot32_128(__m128i x) { return _mm_ror_epi64(x, 32); }

INLINE __m256i rot32_256(__m256i x) { return _mm256_ror_epi64(x, 32); }

INLINE __m512i rot32_512(__m512i x) { return _mm512_ror_epi64(x, 32); }

INLINE __m128i rot24_128(__m128i x) { return _mm_ror_epi64(x, 24); }

INLINE __m256i rot24_256(__m256i x) { return _mm256_ror_epi64(x, 24); }

INLINE __m512i rot24_512(__m512i x) { return _mm512_ror_epi64(x, 24); }

INLINE __m128i rot16_128(__m128i x) { return _mm_ror_epi64(x, 16); }

INLINE __m256i rot16_256(__m256i x) { return _mm256_ror_epi64(x, 16); }

INLINE __m512i rot16_512(__m512i x) { return _mm512_ror_epi64(x, 16); }

INLINE __m128i rot63_128(__m128i x) { return _mm_ror_epi64(x, 63); }

INLINE __m256i rot63_256(__m256i x) { return _mm256_ror_epi64(x, 63); }

INLINE __m512i rot63_512(__m512i x) { return _mm512_ror_epi64(x, 63); }

/*
 * ----------------------------------------------------------------------------
 * compress_avx512
 * ----------------------------------------------------------------------------
 */

INLINE void g1(__m256i *a, __m256i *b, __m256i *c, __m256i *d, __m256i *m) {
  *a = add_256(*a, *m);
  *a = add_256(*a, *b);
  *d = xor_256(*d, *a);
  *d = rot32_256(*d);
  *c = add_256(*c, *d);
  *b = xor_256(*b, *c);
  *b = rot24_256(*b);
}

INLINE void g2(__m256i *a, __m256i *b, __m256i *c, __m256i *d, __m256i *m) {
  *a = add_256(*a, *m);
  *a = add_256(*a, *b);
  *d = xor_256(*d, *a);
  *d = rot16_256(*d);
  *c = add_256(*c, *d);
  *b = xor_256(*b, *c);
  *b = rot63_256(*b);
}

// Note the optimization here of leaving b as the unrotated row, rather than a.
// All the message loads below are adjusted to compensate for this. See
// discussion at https://github.com/sneves/blake2-avx2/pull/4
INLINE void diagonalize(__m256i *a, __m256i *_b, __m256i *c, __m256i *d) {
  (void)_b; // silence the unused parameter warning
  *a = _mm256_permute4x64_epi64(*a, _MM_SHUFFLE(2, 1, 0, 3));
  *d = _mm256_permute4x64_epi64(*d, _MM_SHUFFLE(1, 0, 3, 2));
  *c = _mm256_permute4x64_epi64(*c, _MM_SHUFFLE(0, 3, 2, 1));
}

INLINE void undiagonalize(__m256i *a, __m256i *_b, __m256i *c, __m256i *d) {
  (void)_b; // silence the unused parameter warning
  *a = _mm256_permute4x64_epi64(*a, _MM_SHUFFLE(0, 3, 2, 1));
  *d = _mm256_permute4x64_epi64(*d, _MM_SHUFFLE(1, 0, 3, 2));
  *c = _mm256_permute4x64_epi64(*c, _MM_SHUFFLE(2, 1, 0, 3));
}

void baokeshed64_compress_avx512(uint64_t state[4],
                                 const uint8_t block[BLOCK_LEN],
                                 uint8_t block_len, uint64_t offset,
                                 uint8_t flags) {
  __m256i a = loadu_256((uint8_t *)state);
  __m256i b = loadu_256((uint8_t *)&IV[4]);
  __m256i c = loadu_256((uint8_t *)&IV[0]);
  __m256i flags_vec = set4(offset, 0, (uint64_t)block_len, (uint64_t)flags);
  __m256i d = xor_256(loadu_256((uint8_t *)&IV[4]), flags_vec);

  __m256i m0 = _mm256_broadcastsi128_si256(
      _mm_loadu_si128((__m128i const *)&block[0 * 16]));
  __m256i m1 = _mm256_broadcastsi128_si256(
      _mm_loadu_si128((__m128i const *)&block[1 * 16]));
  __m256i m2 = _mm256_broadcastsi128_si256(
      _mm_loadu_si128((__m128i const *)&block[2 * 16]));
  __m256i m3 = _mm256_broadcastsi128_si256(
      _mm_loadu_si128((__m128i const *)&block[3 * 16]));
  __m256i m4 = _mm256_broadcastsi128_si256(
      _mm_loadu_si128((__m128i const *)&block[4 * 16]));
  __m256i m5 = _mm256_broadcastsi128_si256(
      _mm_loadu_si128((__m128i const *)&block[5 * 16]));
  __m256i m6 = _mm256_broadcastsi128_si256(
      _mm_loadu_si128((__m128i const *)&block[6 * 16]));
  __m256i m7 = _mm256_broadcastsi128_si256(
      _mm_loadu_si128((__m128i const *)&block[7 * 16]));

  __m256i t0;
  __m256i t1;
  __m256i b0;

  // round 1
  t0 = _mm256_unpacklo_epi64(m0, m1);
  t1 = _mm256_unpacklo_epi64(m2, m3);
  b0 = _mm256_blend_epi32(t0, t1, 0xF0);
  g1(&a, &b, &c, &d, &b0);
  t0 = _mm256_unpackhi_epi64(m0, m1);
  t1 = _mm256_unpackhi_epi64(m2, m3);
  b0 = _mm256_blend_epi32(t0, t1, 0xF0);
  g2(&a, &b, &c, &d, &b0);
  diagonalize(&a, &b, &c, &d);
  t0 = _mm256_unpacklo_epi64(m7, m4);
  t1 = _mm256_unpacklo_epi64(m5, m6);
  b0 = _mm256_blend_epi32(t0, t1, 0xF0);
  g1(&a, &b, &c, &d, &b0);
  t0 = _mm256_unpackhi_epi64(m7, m4);
  t1 = _mm256_unpackhi_epi64(m5, m6);
  b0 = _mm256_blend_epi32(t0, t1, 0xF0);
  g2(&a, &b, &c, &d, &b0);
  undiagonalize(&a, &b, &c, &d);

  // round 2
  t0 = _mm256_unpacklo_epi64(m7, m2);
  t1 = _mm256_unpackhi_epi64(m4, m6);
  b0 = _mm256_blend_epi32(t0, t1, 0xF0);
  g1(&a, &b, &c, &d, &b0);
  t0 = _mm256_unpacklo_epi64(m5, m4);
  t1 = _mm256_alignr_epi8(m3, m7, 8);
  b0 = _mm256_blend_epi32(t0, t1, 0xF0);
  g2(&a, &b, &c, &d, &b0);
  diagonalize(&a, &b, &c, &d);
  t0 = _mm256_unpackhi_epi64(m2, m0);
  t1 = _mm256_blend_epi32(m5, m0, 0x33);
  b0 = _mm256_blend_epi32(t0, t1, 0xF0);
  g1(&a, &b, &c, &d, &b0);
  t0 = _mm256_alignr_epi8(m6, m1, 8);
  t1 = _mm256_blend_epi32(m3, m1, 0x33);
  b0 = _mm256_blend_epi32(t0, t1, 0xF0);
  g2(&a, &b, &c, &d, &b0);
  undiagonalize(&a, &b, &c, &d);

  // round 3
  t0 = _mm256_alignr_epi8(m6, m5, 8);
  t1 = _mm256_unpackhi_epi64(m2, m7);
  b0 = _mm256_blend_epi32(t0, t1, 0xF0);
  g1(&a, &b, &c, &d, &b0);
  t0 = _mm256_unpacklo_epi64(m4, m0);
  t1 = _mm256_blend_epi32(m6, m1, 0x33);
  b0 = _mm256_blend_epi32(t0, t1, 0xF0);
  g2(&a, &b, &c, &d, &b0);
  diagonalize(&a, &b, &c, &d);
  t0 = _mm256_alignr_epi8(m5, m4, 8);
  t1 = _mm256_unpackhi_epi64(m1, m3);
  b0 = _mm256_blend_epi32(t0, t1, 0xF0);
  g1(&a, &b, &c, &d, &b0);
  t0 = _mm256_unpacklo_epi64(m2, m7);
  t1 = _mm256_blend_epi32(m0, m3, 0x33);
  b0 = _mm256_blend_epi32(t0, t1, 0xF0);
  g2(&a, &b, &c, &d, &b0);
  undiagonalize(&a, &b, &c, &d);

  // round 4
  t0 = _mm256_unpackhi_epi64(m3, m1);
  t1 = _mm256_unpackhi_epi64(m6, m5);
  b0 = _mm256_blend_epi32(t0, t1, 0xF0);
  g1(&a, &b, &c, &d, &b0);
  t0 = _mm256_unpackhi_epi64(m4, m0);
  t1 = _mm256_unpacklo_epi64(m6, m7);
  b0 = _mm256_blend_epi32(t0, t1, 0xF0);
  g2(&a, &b, &c, &d, &b0);
  diagonalize(&a, &b, &c, &d);
  t0 = _mm256_alignr_epi8(m1, m7, 8);
  t1 = _mm256_shuffle_epi32(m2, _MM_SHUFFLE(1, 0, 3, 2));
  b0 = _mm256_blend_epi32(t0, t1, 0xF0);
  g1(&a, &b, &c, &d, &b0);
  t0 = _mm256_unpacklo_epi64(m4, m3);
  t1 = _mm256_unpacklo_epi64(m5, m0);
  b0 = _mm256_blend_epi32(t0, t1, 0xF0);
  g2(&a, &b, &c, &d, &b0);
  undiagonalize(&a, &b, &c, &d);

  // round 5
  t0 = _mm256_unpackhi_epi64(m4, m2);
  t1 = _mm256_unpacklo_epi64(m1, m5);
  b0 = _mm256_blend_epi32(t0, t1, 0xF0);
  g1(&a, &b, &c, &d, &b0);
  t0 = _mm256_blend_epi32(m3, m0, 0x33);
  t1 = _mm256_blend_epi32(m7, m2, 0x33);
  b0 = _mm256_blend_epi32(t0, t1, 0xF0);
  g2(&a, &b, &c, &d, &b0);
  diagonalize(&a, &b, &c, &d);
  t0 = _mm256_alignr_epi8(m7, m1, 8);
  t1 = _mm256_alignr_epi8(m3, m5, 8);
  b0 = _mm256_blend_epi32(t0, t1, 0xF0);
  g1(&a, &b, &c, &d, &b0);
  t0 = _mm256_unpackhi_epi64(m6, m0);
  t1 = _mm256_unpacklo_epi64(m6, m4);
  b0 = _mm256_blend_epi32(t0, t1, 0xF0);
  g2(&a, &b, &c, &d, &b0);
  undiagonalize(&a, &b, &c, &d);

  // round 6
  t0 = _mm256_unpacklo_epi64(m1, m3);
  t1 = _mm256_unpacklo_epi64(m0, m4);
  b0 = _mm256_blend_epi32(t0, t1, 0xF0);
  g1(&a, &b, &c, &d, &b0);
  t0 = _mm256_unpacklo_epi64(m6, m5);
  t1 = _mm256_unpackhi_epi64(m5, m1);
  b0 = _mm256_blend_epi32(t0, t1, 0xF0);
  g2(&a, &b, &c, &d, &b0);
  diagonalize(&a, &b, &c, &d);
  t0 = _mm256_alignr_epi8(m2, m0, 8);
  t1 = _mm256_unpackhi_epi64(m3, m7);
  b0 = _mm256_blend_epi32(t0, t1, 0xF0);
  g1(&a, &b, &c, &d, &b0);
  t0 = _mm256_unpackhi_epi64(m4, m6);
  t1 = _mm256_alignr_epi8(m7, m2, 8);
  b0 = _mm256_blend_epi32(t0, t1, 0xF0);
  g2(&a, &b, &c, &d, &b0);
  undiagonalize(&a, &b, &c, &d);

  // round 7
  t0 = _mm256_blend_epi32(m0, m6, 0x33);
  t1 = _mm256_unpacklo_epi64(m7, m2);
  b0 = _mm256_blend_epi32(t0, t1, 0xF0);
  g1(&a, &b, &c, &d, &b0);
  t0 = _mm256_unpackhi_epi64(m2, m7);
  t1 = _mm256_alignr_epi8(m5, m6, 8);
  b0 = _mm256_blend_epi32(t0, t1, 0xF0);
  g2(&a, &b, &c, &d, &b0);
  diagonalize(&a, &b, &c, &d);
  t0 = _mm256_unpacklo_epi64(m4, m0);
  t1 = _mm256_blend_epi32(m4, m3, 0x33);
  b0 = _mm256_blend_epi32(t0, t1, 0xF0);
  g1(&a, &b, &c, &d, &b0);
  t0 = _mm256_unpackhi_epi64(m5, m3);
  t1 = _mm256_shuffle_epi32(m1, _MM_SHUFFLE(1, 0, 3, 2));
  b0 = _mm256_blend_epi32(t0, t1, 0xF0);
  g2(&a, &b, &c, &d, &b0);
  undiagonalize(&a, &b, &c, &d);

  // round 8
  t0 = _mm256_unpackhi_epi64(m6, m3);
  t1 = _mm256_blend_epi32(m1, m6, 0x33);
  b0 = _mm256_blend_epi32(t0, t1, 0xF0);
  g1(&a, &b, &c, &d, &b0);
  t0 = _mm256_alignr_epi8(m7, m5, 8);
  t1 = _mm256_unpackhi_epi64(m0, m4);
  b0 = _mm256_blend_epi32(t0, t1, 0xF0);
  g2(&a, &b, &c, &d, &b0);
  diagonalize(&a, &b, &c, &d);
  t0 = _mm256_blend_epi32(m2, m1, 0x33);
  t1 = _mm256_alignr_epi8(m4, m7, 8);
  b0 = _mm256_blend_epi32(t0, t1, 0xF0);
  g1(&a, &b, &c, &d, &b0);
  t0 = _mm256_unpacklo_epi64(m5, m0);
  t1 = _mm256_unpacklo_epi64(m2, m3);
  b0 = _mm256_blend_epi32(t0, t1, 0xF0);
  g2(&a, &b, &c, &d, &b0);
  undiagonalize(&a, &b, &c, &d);

  storeu_256(xor_256(a, b), (uint8_t *)state);
}

/*
 * ----------------------------------------------------------------------------
 * baokeshed64_hash2_avx512
 * ----------------------------------------------------------------------------
 */

INLINE void round_fn2(__m128i v[16], __m128i m[16], size_t r) {
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

INLINE void transpose_vecs_128(__m128i vecs[2]) {
  uint64_t vec0[2];
  uint64_t vec1[2];
  storeu_128(vecs[0], (uint8_t *)vec0);
  storeu_128(vecs[1], (uint8_t *)vec1);
  vecs[0] = set2(vec0[0], vec1[0]);
  vecs[1] = set2(vec0[1], vec1[1]);
}

INLINE void transpose_msg_vecs2(const uint8_t *const *inputs,
                                size_t block_offset, __m128i out[16]) {
  out[0] = loadu_128(&inputs[0][block_offset + 0 * sizeof(__m128i)]);
  out[1] = loadu_128(&inputs[1][block_offset + 0 * sizeof(__m128i)]);
  out[2] = loadu_128(&inputs[0][block_offset + 1 * sizeof(__m128i)]);
  out[3] = loadu_128(&inputs[1][block_offset + 1 * sizeof(__m128i)]);
  out[4] = loadu_128(&inputs[0][block_offset + 2 * sizeof(__m128i)]);
  out[5] = loadu_128(&inputs[1][block_offset + 2 * sizeof(__m128i)]);
  out[6] = loadu_128(&inputs[0][block_offset + 3 * sizeof(__m128i)]);
  out[7] = loadu_128(&inputs[1][block_offset + 3 * sizeof(__m128i)]);
  out[8] = loadu_128(&inputs[0][block_offset + 4 * sizeof(__m128i)]);
  out[9] = loadu_128(&inputs[1][block_offset + 4 * sizeof(__m128i)]);
  out[10] = loadu_128(&inputs[0][block_offset + 5 * sizeof(__m128i)]);
  out[11] = loadu_128(&inputs[1][block_offset + 5 * sizeof(__m128i)]);
  out[12] = loadu_128(&inputs[0][block_offset + 6 * sizeof(__m128i)]);
  out[13] = loadu_128(&inputs[1][block_offset + 6 * sizeof(__m128i)]);
  out[14] = loadu_128(&inputs[0][block_offset + 7 * sizeof(__m128i)]);
  out[15] = loadu_128(&inputs[1][block_offset + 7 * sizeof(__m128i)]);
  transpose_vecs_128(&out[0]);
  transpose_vecs_128(&out[2]);
  transpose_vecs_128(&out[4]);
  transpose_vecs_128(&out[6]);
  transpose_vecs_128(&out[8]);
  transpose_vecs_128(&out[10]);
  transpose_vecs_128(&out[12]);
  transpose_vecs_128(&out[14]);
}

INLINE void load_offsets2(uint64_t offset, const uint64_t offset_deltas[2],
                          __m128i *out) {
  *out = add_128(set1_128(offset), loadu_128((uint8_t*)offset_deltas));
}

void baokeshed64_hash2_avx512(const uint8_t *const *inputs, size_t blocks,
                              const uint64_t key_words[4], uint64_t offset,
                              const uint64_t offset_deltas[2], uint8_t flags,
                              uint8_t flags_start, uint8_t flags_end,
                              uint8_t *out) {
  __m128i h_vecs[4] = {
      set1_128(key_words[0]),
      set1_128(key_words[1]),
      set1_128(key_words[2]),
      set1_128(key_words[3]),
  };
  __m128i offset_vec;
  load_offsets2(offset, offset_deltas, &offset_vec);
  uint8_t block_flags = flags | flags_start;

  for (size_t block = 0; block < blocks; block++) {
    if (block + 1 == blocks) {
      block_flags |= flags_end;
    }
    __m128i block_len_vec = set1_128(BLOCK_LEN);
    __m128i block_flags_vec = set1_128(block_flags);
    __m128i msg_vecs[16];
    transpose_msg_vecs2(inputs, block * BLOCK_LEN, msg_vecs);

    __m128i v[16] = {
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
  // The first four vecs now contain the first half of each output, and the
  // second four vecs contain the second half of each output.
  storeu_128(h_vecs[0], &out[0 * sizeof(__m128i)]);
  storeu_128(h_vecs[2], &out[1 * sizeof(__m128i)]);
  storeu_128(h_vecs[1], &out[2 * sizeof(__m128i)]);
  storeu_128(h_vecs[3], &out[3 * sizeof(__m128i)]);
}

/*
 * ----------------------------------------------------------------------------
 * baokeshed64_hash4_avx512
 * ----------------------------------------------------------------------------
 */

INLINE void round_fn4(__m256i v[16], __m256i m[16], size_t r) {
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
  v[12] = rot32_256(v[12]);
  v[13] = rot32_256(v[13]);
  v[14] = rot32_256(v[14]);
  v[15] = rot32_256(v[15]);
  v[8] = add_256(v[8], v[12]);
  v[9] = add_256(v[9], v[13]);
  v[10] = add_256(v[10], v[14]);
  v[11] = add_256(v[11], v[15]);
  v[4] = xor_256(v[4], v[8]);
  v[5] = xor_256(v[5], v[9]);
  v[6] = xor_256(v[6], v[10]);
  v[7] = xor_256(v[7], v[11]);
  v[4] = rot24_256(v[4]);
  v[5] = rot24_256(v[5]);
  v[6] = rot24_256(v[6]);
  v[7] = rot24_256(v[7]);
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
  v[4] = rot63_256(v[4]);
  v[5] = rot63_256(v[5]);
  v[6] = rot63_256(v[6]);
  v[7] = rot63_256(v[7]);

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
  v[15] = rot32_256(v[15]);
  v[12] = rot32_256(v[12]);
  v[13] = rot32_256(v[13]);
  v[14] = rot32_256(v[14]);
  v[10] = add_256(v[10], v[15]);
  v[11] = add_256(v[11], v[12]);
  v[8] = add_256(v[8], v[13]);
  v[9] = add_256(v[9], v[14]);
  v[5] = xor_256(v[5], v[10]);
  v[6] = xor_256(v[6], v[11]);
  v[7] = xor_256(v[7], v[8]);
  v[4] = xor_256(v[4], v[9]);
  v[5] = rot24_256(v[5]);
  v[6] = rot24_256(v[6]);
  v[7] = rot24_256(v[7]);
  v[4] = rot24_256(v[4]);
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
  v[5] = rot63_256(v[5]);
  v[6] = rot63_256(v[6]);
  v[7] = rot63_256(v[7]);
  v[4] = rot63_256(v[4]);
}

INLINE void transpose_vecs_256(__m256i vecs[4]) {
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

INLINE void transpose_msg_vecs4(const uint8_t *const *inputs,
                                size_t block_offset, __m256i out[16]) {
  out[0] = loadu_256(&inputs[0][block_offset + 0 * sizeof(__m256i)]);
  out[1] = loadu_256(&inputs[1][block_offset + 0 * sizeof(__m256i)]);
  out[2] = loadu_256(&inputs[2][block_offset + 0 * sizeof(__m256i)]);
  out[3] = loadu_256(&inputs[3][block_offset + 0 * sizeof(__m256i)]);
  out[4] = loadu_256(&inputs[0][block_offset + 1 * sizeof(__m256i)]);
  out[5] = loadu_256(&inputs[1][block_offset + 1 * sizeof(__m256i)]);
  out[6] = loadu_256(&inputs[2][block_offset + 1 * sizeof(__m256i)]);
  out[7] = loadu_256(&inputs[3][block_offset + 1 * sizeof(__m256i)]);
  out[8] = loadu_256(&inputs[0][block_offset + 2 * sizeof(__m256i)]);
  out[9] = loadu_256(&inputs[1][block_offset + 2 * sizeof(__m256i)]);
  out[10] = loadu_256(&inputs[2][block_offset + 2 * sizeof(__m256i)]);
  out[11] = loadu_256(&inputs[3][block_offset + 2 * sizeof(__m256i)]);
  out[12] = loadu_256(&inputs[0][block_offset + 3 * sizeof(__m256i)]);
  out[13] = loadu_256(&inputs[1][block_offset + 3 * sizeof(__m256i)]);
  out[14] = loadu_256(&inputs[2][block_offset + 3 * sizeof(__m256i)]);
  out[15] = loadu_256(&inputs[3][block_offset + 3 * sizeof(__m256i)]);
  transpose_vecs_256(&out[0]);
  transpose_vecs_256(&out[4]);
  transpose_vecs_256(&out[8]);
  transpose_vecs_256(&out[12]);
}

INLINE void load_offsets4(uint64_t offset, const uint64_t offset_deltas[4],
                         __m256i *out) {
  *out = add_256(set1_256(offset), loadu_256((uint8_t*)offset_deltas));
}

void baokeshed64_hash4_avx512(const uint8_t *const *inputs, size_t blocks,
                              const uint64_t key_words[4], uint64_t offset,
                              const uint64_t offset_deltas[4], uint8_t flags,
                              uint8_t flags_start, uint8_t flags_end,
                              uint8_t *out) {
  __m256i h_vecs[4] = {
      set1_256(key_words[0]),
      set1_256(key_words[1]),
      set1_256(key_words[2]),
      set1_256(key_words[3]),
  };
  __m256i offset_vec;
  load_offsets4(offset, offset_deltas, &offset_vec);
  uint8_t block_flags = flags | flags_start;

  for (size_t block = 0; block < blocks; block++) {
    if (block + 1 == blocks) {
      block_flags |= flags_end;
    }
    __m256i block_len_vec = set1_256(BLOCK_LEN);
    __m256i block_flags_vec = set1_256(block_flags);
    __m256i msg_vecs[16];
    transpose_msg_vecs4(inputs, block * BLOCK_LEN, msg_vecs);

    __m256i v[16] = {
        h_vecs[0],
        h_vecs[1],
        h_vecs[2],
        h_vecs[3],
        set1_256(IV[4]),
        set1_256(IV[5]),
        set1_256(IV[6]),
        set1_256(IV[7]),
        set1_256(IV[0]),
        set1_256(IV[1]),
        set1_256(IV[2]),
        set1_256(IV[3]),
        xor_256(set1_256(IV[4]), offset_vec),
        set1_256(IV[5]),
        xor_256(set1_256(IV[6]), block_len_vec),
        xor_256(set1_256(IV[7]), block_flags_vec),
    };
    round_fn4(v, msg_vecs, 0);
    round_fn4(v, msg_vecs, 1);
    round_fn4(v, msg_vecs, 2);
    round_fn4(v, msg_vecs, 3);
    round_fn4(v, msg_vecs, 4);
    round_fn4(v, msg_vecs, 5);
    round_fn4(v, msg_vecs, 6);
    round_fn4(v, msg_vecs, 7);
    h_vecs[0] = xor_256(v[0], v[4]);
    h_vecs[1] = xor_256(v[1], v[5]);
    h_vecs[2] = xor_256(v[2], v[6]);
    h_vecs[3] = xor_256(v[3], v[7]);

    block_flags = flags;
  }

  transpose_vecs_256(h_vecs);
  storeu_256(h_vecs[0], &out[0 * sizeof(__m256i)]);
  storeu_256(h_vecs[1], &out[1 * sizeof(__m256i)]);
  storeu_256(h_vecs[2], &out[2 * sizeof(__m256i)]);
  storeu_256(h_vecs[3], &out[3 * sizeof(__m256i)]);
}

/*
 * ----------------------------------------------------------------------------
 * baokeshed64_hash8_avx512
 * ----------------------------------------------------------------------------
 */

INLINE void round_fn8(__m512i v[16], __m512i m[16], size_t r) {
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
  v[12] = rot32_512(v[12]);
  v[13] = rot32_512(v[13]);
  v[14] = rot32_512(v[14]);
  v[15] = rot32_512(v[15]);
  v[8] = add_512(v[8], v[12]);
  v[9] = add_512(v[9], v[13]);
  v[10] = add_512(v[10], v[14]);
  v[11] = add_512(v[11], v[15]);
  v[4] = xor_512(v[4], v[8]);
  v[5] = xor_512(v[5], v[9]);
  v[6] = xor_512(v[6], v[10]);
  v[7] = xor_512(v[7], v[11]);
  v[4] = rot24_512(v[4]);
  v[5] = rot24_512(v[5]);
  v[6] = rot24_512(v[6]);
  v[7] = rot24_512(v[7]);
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
  v[4] = rot63_512(v[4]);
  v[5] = rot63_512(v[5]);
  v[6] = rot63_512(v[6]);
  v[7] = rot63_512(v[7]);

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
  v[15] = rot32_512(v[15]);
  v[12] = rot32_512(v[12]);
  v[13] = rot32_512(v[13]);
  v[14] = rot32_512(v[14]);
  v[10] = add_512(v[10], v[15]);
  v[11] = add_512(v[11], v[12]);
  v[8] = add_512(v[8], v[13]);
  v[9] = add_512(v[9], v[14]);
  v[5] = xor_512(v[5], v[10]);
  v[6] = xor_512(v[6], v[11]);
  v[7] = xor_512(v[7], v[8]);
  v[4] = xor_512(v[4], v[9]);
  v[5] = rot24_512(v[5]);
  v[6] = rot24_512(v[6]);
  v[7] = rot24_512(v[7]);
  v[4] = rot24_512(v[4]);
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
  v[5] = rot63_512(v[5]);
  v[6] = rot63_512(v[6]);
  v[7] = rot63_512(v[7]);
  v[4] = rot63_512(v[4]);
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
  // Interleave 64-bit lates. The _0 unpack is lanes 0/0/2/2/4/4/6/6, and the
  // _1 unpack is 1/1/3/3/5/5/7/7.
  __m512i ab_0 = _mm512_unpacklo_epi64(vecs[0], vecs[1]);
  __m512i ab_1 = _mm512_unpackhi_epi64(vecs[0], vecs[1]);
  __m512i cd_0 = _mm512_unpacklo_epi64(vecs[2], vecs[3]);
  __m512i cd_1 = _mm512_unpackhi_epi64(vecs[2], vecs[3]);
  __m512i ef_0 = _mm512_unpacklo_epi64(vecs[4], vecs[5]);
  __m512i ef_1 = _mm512_unpackhi_epi64(vecs[4], vecs[5]);
  __m512i gh_0 = _mm512_unpacklo_epi64(vecs[6], vecs[7]);
  __m512i gh_1 = _mm512_unpackhi_epi64(vecs[6], vecs[7]);

  // Interleave 128-bit lanes. The _0 unpack is 0/0/4/4/0/0/4/4, the _1 unpack
  // is 1/1/5/5/1/1/5/5, the _2 unpack is 2/2/6/6/2/2/6/6, and the _3 unpack is
  // 3/3/7/7/3/3/7/7.
  __m512i abcd_0 = unpack_lo_128(ab_0, cd_0);
  __m512i abcd_1 = unpack_lo_128(ab_1, cd_1);
  __m512i abcd_2 = unpack_hi_128(ab_0, cd_0);
  __m512i abcd_3 = unpack_hi_128(ab_1, cd_1);
  __m512i efgh_0 = unpack_lo_128(ef_0, gh_0);
  __m512i efgh_1 = unpack_lo_128(ef_1, gh_1);
  __m512i efgh_2 = unpack_hi_128(ef_0, gh_0);
  __m512i efgh_3 = unpack_hi_128(ef_1, gh_1);

  // Interleave 128-bit lanes again for the final outputs.
  vecs[0] = unpack_lo_128(abcd_0, efgh_0);
  vecs[1] = unpack_lo_128(abcd_1, efgh_1);
  vecs[2] = unpack_lo_128(abcd_2, efgh_2);
  vecs[3] = unpack_lo_128(abcd_3, efgh_3);
  vecs[4] = unpack_hi_128(abcd_0, efgh_0);
  vecs[5] = unpack_hi_128(abcd_1, efgh_1);
  vecs[6] = unpack_hi_128(abcd_2, efgh_2);
  vecs[7] = unpack_hi_128(abcd_3, efgh_3);
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

INLINE void load_offsets8(uint64_t offset, const uint64_t deltas[16], __m512i *out) {
  *out = add_512(set1_512(offset), loadu_512((uint8_t*)offset_deltas));
}

void baokeshed64_hash16_avx512(const uint8_t *const *inputs, size_t blocks,
                   const uint64_t key_words[4], uint64_t offset,
                   const uint64_t offset_deltas[16], uint8_t flags,
                   uint8_t flags_start, uint8_t flags_end, uint8_t *out) {
  __m512i h_vecs[4] = {
      set1_512(key_words[0]), set1_512(key_words[1]), set1_512(key_words[2]),
      set1_512(key_words[3]),
  };
  __m512i offset_vec;
  load_offsets8(offset, offset_deltas, &offset_vec);
  uint8_t block_flags = flags | flags_start;

  for (size_t block = 0; block < blocks; block++) {
    if (block + 1 == blocks) {
      block_flags |= flags_end;
    }
    __m512i block_len_vec = set1_512(BLOCK_LEN);
    __m512i block_flags_vec = set1_512(block_flags);
    __m512i msg_vecs[16];
    transpose_msg_vecs16(inputs, block * BLOCK_LEN, msg_vecs);

    __m512i v[16] = {
        h_vecs[0],
        h_vecs[1],
        h_vecs[2],
        h_vecs[3],
        set1_512(IV[4]),
        set1_512(IV[5]),
        set1_512(IV[6]),
        set1_512(IV[7]),
        set1_512(IV[0]),
        set1_512(IV[1]),
        set1_512(IV[2]),
        set1_512(IV[3]),
        xor_512(set1_512(IV[4]), offset_vec),
        set1_512(IV[5]),
        xor_512(set1_512(IV[6]), block_len_vec),
        xor_512(set1_512(IV[7]), block_flags_vec),
    };
    round_fn8(v, msg_vecs, 0);
    round_fn8(v, msg_vecs, 1);
    round_fn8(v, msg_vecs, 2);
    round_fn8(v, msg_vecs, 3);
    round_fn8(v, msg_vecs, 4);
    round_fn8(v, msg_vecs, 5);
    round_fn8(v, msg_vecs, 6);
    round_fn8(v, msg_vecs, 7);
    h_vecs[0] = xor_512(v[0], v[4]);
    h_vecs[1] = xor_512(v[1], v[5]);
    h_vecs[2] = xor_512(v[2], v[6]);
    h_vecs[3] = xor_512(v[3], v[7]);

    block_flags = flags;
  }

  // transpose_vecs_512 operates on a 8x8 matrix of words, but we only have 4
  // state vectors. Pad the matrix with zeros. After transposition, store the
  // lower half of each vector.
  __m512i padded[8] = {
      h_vecs[0],   h_vecs[1],   h_vecs[2],   h_vecs[3],
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
}

/*
 * ----------------------------------------------------------------------------
 * baokeshed64_hash_many_avx512
 * ----------------------------------------------------------------------------
 */

INLINE void hash_one_avx512(const uint8_t *input, size_t blocks,
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
    baokeshed64_compress_avx512(state, input, BLOCK_LEN, offset, block_flags);
    input = &input[BLOCK_LEN];
    blocks -= 1;
    block_flags = flags;
  }
  memcpy(out, state, OUT_LEN);
}

void hash_many_avx512(const uint8_t *const *inputs, size_t num_inputs,
                      size_t blocks, const uint32_t key_words[8],
                      uint64_t offset, const uint64_t offset_deltas[9],
                      uint8_t flags, uint8_t flags_start, uint8_t flags_end,
                      uint8_t *out) {
  while (num_inputs >= 8) {
    hash8_avx512(inputs, blocks, key_words, offset, offset_deltas, flags,
                  flags_start, flags_end, out);
    inputs += 8;
    num_inputs -= 8;
    offset += offset_deltas[8];
    out = &out[8 * OUT_LEN];
  }
  while (num_inputs >= 4) {
    hash8_avx512(inputs, blocks, key_words, offset, offset_deltas, flags,
                 flags_start, flags_end, out);
    inputs += 4;
    num_inputs -= 4;
    offset += offset_deltas[4];
    out = &out[4 * OUT_LEN];
  }
  while (num_inputs >= 2) {
    hash4_avx512(inputs, blocks, key_words, offset, offset_deltas, flags,
                 flags_start, flags_end, out);
    inputs += 2;
    num_inputs -= 2;
    offset += offset_deltas[2];
    out = &out[2 * OUT_LEN];
  }
  while (num_inputs > 0) {
    hash_one_avx512(inputs[0], blocks, key_words, offset, flags, flags_start,
                    flags_end, out);
    inputs += 1;
    num_inputs -= 1;
    offset += offset_deltas[1];
    out = &out[OUT_LEN];
  }
}
