/* Private fused kernels, included by oblas_lite.c so tables and CPU
 * selection are shared with the single-row implementations. */
#ifndef OBLAS_FUSED_H
#define OBLAS_FUSED_H

/* Keep coefficient dispatch inside the row batch, especially for short rows. */
#if defined(__GNUC__) || defined(__clang__)
#define OBL_FUSED_INLINE static inline __attribute__((always_inline))
#else
#define OBL_FUSED_INLINE static inline
#endif

static inline uint8_t obl_fused_mul(uint8_t u, uint8_t b) {
#if defined(OBLAS_TINY)
  return gf2_8_mul(u, b);
#else
  return GF2_8_MUL[((unsigned)u << 8) + b];
#endif
}

OBL_FUSED_INLINE void obl_axpy2_ref(u8 *d0, u8 *d1, const u8 *src, u8 u0, u8 u1,
                                    unsigned len) {
  for (unsigned i = 0; i < len; i++) {
    u8 s = src[i];
    d0[i] ^= obl_fused_mul(u0, s);
    d1[i] ^= obl_fused_mul(u1, s);
  }
}

OBL_FUSED_INLINE void obl_axpy4_ref(u8 *d0, u8 *d1, u8 *d2, u8 *d3,
                                    const u8 *src, u8 u0, u8 u1, u8 u2, u8 u3,
                                    unsigned len) {
  for (unsigned i = 0; i < len; i++) {
    u8 s = src[i];
    d0[i] ^= obl_fused_mul(u0, s);
    d1[i] ^= obl_fused_mul(u1, s);
    d2[i] ^= obl_fused_mul(u2, s);
    d3[i] ^= obl_fused_mul(u3, s);
  }
}

static void obl_xor_set4_ref(u8 *dst, const u8 *r0, const u8 *r1, const u8 *r2,
                             const u8 *r3, unsigned len) {
  for (unsigned i = 0; i < len; i++)
    dst[i] = r0[i] ^ r1[i] ^ r2[i] ^ r3[i];
}

static void obl_xor_acc4_ref(u8 *dst, const u8 *r0, const u8 *r1, const u8 *r2,
                             const u8 *r3, unsigned len) {
  for (unsigned i = 0; i < len; i++)
    dst[i] ^= r0[i] ^ r1[i] ^ r2[i] ^ r3[i];
}

#if defined(OBLAS_ARCH_X86) && !defined(_MSC_VER)

#define OBL_FUSED_TARGET_AVX512_GFNI                                           \
  __attribute__((target("ssse3,avx2,avx512f,avx512bw,avx512dq,avx512vl,"       \
                        "gfni")))
#define OBL_FUSED_TARGET_AVX512                                                \
  __attribute__((target("ssse3,avx2,avx512f,avx512bw,avx512dq,avx512vl")))
#define OBL_FUSED_TARGET_AVX2_GFNI __attribute__((target("ssse3,avx2,gfni")))
#define OBL_FUSED_TARGET_AVX2 __attribute__((target("ssse3,avx2")))

OBL_FUSED_TARGET_AVX512 static inline void
obl_fused_xor512_fanout2(uint8_t *d0, uint8_t *d1, const uint8_t *src,
                         uint32_t len) {
  uint32_t i = 0;
  for (; len - i >= 64; i += 64) {
    __m512i s = _mm512_loadu_si512(src + i);
    _mm512_storeu_si512(d0 + i,
                        _mm512_xor_si512(_mm512_loadu_si512(d0 + i), s));
    _mm512_storeu_si512(d1 + i,
                        _mm512_xor_si512(_mm512_loadu_si512(d1 + i), s));
  }
  if (len - i >= 32) {
    __m256i s = _mm256_loadu_si256((const __m256i *)(src + i));
    _mm256_storeu_si256(
        (__m256i *)(d0 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d0 + i)), s));
    _mm256_storeu_si256(
        (__m256i *)(d1 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d1 + i)), s));
    i += 32;
  }
  if (len - i >= 16) {
    __m128i s = _mm_loadu_si128((const __m128i *)(src + i));
    _mm_storeu_si128(
        (__m128i *)(d0 + i),
        _mm_xor_si128(_mm_loadu_si128((const __m128i *)(d0 + i)), s));
    _mm_storeu_si128(
        (__m128i *)(d1 + i),
        _mm_xor_si128(_mm_loadu_si128((const __m128i *)(d1 + i)), s));
    i += 16;
  }
  for (; i < len; i++) {
    uint8_t s = src[i];
    d0[i] ^= s;
    d1[i] ^= s;
  }
}

OBL_FUSED_TARGET_AVX512 static inline void
obl_fused_xor512_fanout4(uint8_t *d0, uint8_t *d1, uint8_t *d2, uint8_t *d3,
                         const uint8_t *src, uint32_t len) {
  uint32_t i = 0;
  for (; len - i >= 64; i += 64) {
    __m512i s = _mm512_loadu_si512(src + i);
    _mm512_storeu_si512(d0 + i,
                        _mm512_xor_si512(_mm512_loadu_si512(d0 + i), s));
    _mm512_storeu_si512(d1 + i,
                        _mm512_xor_si512(_mm512_loadu_si512(d1 + i), s));
    _mm512_storeu_si512(d2 + i,
                        _mm512_xor_si512(_mm512_loadu_si512(d2 + i), s));
    _mm512_storeu_si512(d3 + i,
                        _mm512_xor_si512(_mm512_loadu_si512(d3 + i), s));
  }
  if (len - i >= 32) {
    __m256i s = _mm256_loadu_si256((const __m256i *)(src + i));
    _mm256_storeu_si256(
        (__m256i *)(d0 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d0 + i)), s));
    _mm256_storeu_si256(
        (__m256i *)(d1 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d1 + i)), s));
    _mm256_storeu_si256(
        (__m256i *)(d2 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d2 + i)), s));
    _mm256_storeu_si256(
        (__m256i *)(d3 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d3 + i)), s));
    i += 32;
  }
  if (len - i >= 16) {
    __m128i s = _mm_loadu_si128((const __m128i *)(src + i));
    _mm_storeu_si128(
        (__m128i *)(d0 + i),
        _mm_xor_si128(_mm_loadu_si128((const __m128i *)(d0 + i)), s));
    _mm_storeu_si128(
        (__m128i *)(d1 + i),
        _mm_xor_si128(_mm_loadu_si128((const __m128i *)(d1 + i)), s));
    _mm_storeu_si128(
        (__m128i *)(d2 + i),
        _mm_xor_si128(_mm_loadu_si128((const __m128i *)(d2 + i)), s));
    _mm_storeu_si128(
        (__m128i *)(d3 + i),
        _mm_xor_si128(_mm_loadu_si128((const __m128i *)(d3 + i)), s));
    i += 16;
  }
  for (; i < len; i++) {
    uint8_t s = src[i];
    d0[i] ^= s;
    d1[i] ^= s;
    d2[i] ^= s;
    d3[i] ^= s;
  }
}

OBL_FUSED_TARGET_AVX512 static inline void
obl_xor_set4_avx512(uint8_t *dst, const uint8_t *r0, const uint8_t *r1,
                    const uint8_t *r2, const uint8_t *r3, uint32_t len) {
  uint32_t i = 0;
  for (; len - i >= 64; i += 64) {
    __m512i v01 = _mm512_xor_si512(_mm512_loadu_si512(r0 + i),
                                   _mm512_loadu_si512(r1 + i));
    __m512i v23 = _mm512_xor_si512(_mm512_loadu_si512(r2 + i),
                                   _mm512_loadu_si512(r3 + i));
    _mm512_storeu_si512(dst + i, _mm512_xor_si512(v01, v23));
  }
  if (len - i >= 32) {
    __m256i v01 =
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(r0 + i)),
                         _mm256_loadu_si256((const __m256i *)(r1 + i)));
    __m256i v23 =
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(r2 + i)),
                         _mm256_loadu_si256((const __m256i *)(r3 + i)));
    _mm256_storeu_si256((__m256i *)(dst + i), _mm256_xor_si256(v01, v23));
    i += 32;
  }
  for (; i < len; i++)
    dst[i] = r0[i] ^ r1[i] ^ r2[i] ^ r3[i];
}

OBL_FUSED_TARGET_AVX512 static inline void
obl_xor_acc4_avx512(uint8_t *dst, const uint8_t *r0, const uint8_t *r1,
                    const uint8_t *r2, const uint8_t *r3, uint32_t len) {
  uint32_t i = 0;
  for (; len - i >= 64; i += 64) {
    __m512i v01 = _mm512_xor_si512(_mm512_loadu_si512(r0 + i),
                                   _mm512_loadu_si512(r1 + i));
    __m512i v23 = _mm512_xor_si512(_mm512_loadu_si512(r2 + i),
                                   _mm512_loadu_si512(r3 + i));
    __m512i acc = _mm512_xor_si512(_mm512_loadu_si512(dst + i), v01);
    _mm512_storeu_si512(dst + i, _mm512_xor_si512(acc, v23));
  }
  if (len - i >= 32) {
    __m256i v01 =
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(r0 + i)),
                         _mm256_loadu_si256((const __m256i *)(r1 + i)));
    __m256i v23 =
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(r2 + i)),
                         _mm256_loadu_si256((const __m256i *)(r3 + i)));
    __m256i acc =
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(dst + i)), v01);
    _mm256_storeu_si256((__m256i *)(dst + i), _mm256_xor_si256(acc, v23));
    i += 32;
  }
  for (; i < len; i++)
    dst[i] ^= r0[i] ^ r1[i] ^ r2[i] ^ r3[i];
}

OBL_FUSED_TARGET_AVX512_GFNI static inline void
obl_fused_gfni512_fanout2(uint8_t *d0, uint8_t *d1, uint8_t u0, uint8_t u1,
                          const uint8_t *src, uint32_t len) {
  const __m512i m0 = _mm512_set1_epi64(GF2_8_AFFINE_MAT[u0]);
  const __m512i m1 = _mm512_set1_epi64(GF2_8_AFFINE_MAT[u1]);
  uint32_t i = 0;
  for (; len - i >= 64; i += 64) {
    __m512i s = _mm512_loadu_si512(src + i);
    __m512i p0 = _mm512_gf2p8affine_epi64_epi8(s, m0, 0);
    __m512i p1 = _mm512_gf2p8affine_epi64_epi8(s, m1, 0);
    _mm512_storeu_si512(d0 + i,
                        _mm512_xor_si512(_mm512_loadu_si512(d0 + i), p0));
    _mm512_storeu_si512(d1 + i,
                        _mm512_xor_si512(_mm512_loadu_si512(d1 + i), p1));
  }
  if (len - i >= 32) {
    __m256i s = _mm256_loadu_si256((const __m256i *)(src + i));
    __m256i p0 =
        _mm256_gf2p8affine_epi64_epi8(s, _mm512_castsi512_si256(m0), 0);
    __m256i p1 =
        _mm256_gf2p8affine_epi64_epi8(s, _mm512_castsi512_si256(m1), 0);
    _mm256_storeu_si256(
        (__m256i *)(d0 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d0 + i)), p0));
    _mm256_storeu_si256(
        (__m256i *)(d1 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d1 + i)), p1));
    i += 32;
  }
  if (len - i >= 16) {
    __m128i s = _mm_loadu_si128((const __m128i *)(src + i));
    __m128i p0 = _mm_gf2p8affine_epi64_epi8(s, _mm512_castsi512_si128(m0), 0);
    __m128i p1 = _mm_gf2p8affine_epi64_epi8(s, _mm512_castsi512_si128(m1), 0);
    _mm_storeu_si128(
        (__m128i *)(d0 + i),
        _mm_xor_si128(_mm_loadu_si128((const __m128i *)(d0 + i)), p0));
    _mm_storeu_si128(
        (__m128i *)(d1 + i),
        _mm_xor_si128(_mm_loadu_si128((const __m128i *)(d1 + i)), p1));
    i += 16;
  }
  for (; i < len; i++) {
    uint8_t b = src[i];
    if (b) {
      d0[i] ^= obl_fused_mul(u0, b);
      d1[i] ^= obl_fused_mul(u1, b);
    }
  }
}

OBL_FUSED_TARGET_AVX512_GFNI static inline void
obl_fused_gfni512_fanout4(uint8_t *d0, uint8_t *d1, uint8_t *d2, uint8_t *d3,
                          uint8_t u0, uint8_t u1, uint8_t u2, uint8_t u3,
                          const uint8_t *src, uint32_t len) {
  const __m512i m0 = _mm512_set1_epi64(GF2_8_AFFINE_MAT[u0]);
  const __m512i m1 = _mm512_set1_epi64(GF2_8_AFFINE_MAT[u1]);
  const __m512i m2 = _mm512_set1_epi64(GF2_8_AFFINE_MAT[u2]);
  const __m512i m3 = _mm512_set1_epi64(GF2_8_AFFINE_MAT[u3]);
  uint32_t i = 0;
  for (; len - i >= 64; i += 64) {
    __m512i s = _mm512_loadu_si512(src + i);
    __m512i p0 = _mm512_gf2p8affine_epi64_epi8(s, m0, 0);
    __m512i p1 = _mm512_gf2p8affine_epi64_epi8(s, m1, 0);
    __m512i p2 = _mm512_gf2p8affine_epi64_epi8(s, m2, 0);
    __m512i p3 = _mm512_gf2p8affine_epi64_epi8(s, m3, 0);
    _mm512_storeu_si512(d0 + i,
                        _mm512_xor_si512(_mm512_loadu_si512(d0 + i), p0));
    _mm512_storeu_si512(d1 + i,
                        _mm512_xor_si512(_mm512_loadu_si512(d1 + i), p1));
    _mm512_storeu_si512(d2 + i,
                        _mm512_xor_si512(_mm512_loadu_si512(d2 + i), p2));
    _mm512_storeu_si512(d3 + i,
                        _mm512_xor_si512(_mm512_loadu_si512(d3 + i), p3));
  }
  if (len - i >= 32) {
    __m256i s = _mm256_loadu_si256((const __m256i *)(src + i));
    __m256i p0 =
        _mm256_gf2p8affine_epi64_epi8(s, _mm512_castsi512_si256(m0), 0);
    __m256i p1 =
        _mm256_gf2p8affine_epi64_epi8(s, _mm512_castsi512_si256(m1), 0);
    __m256i p2 =
        _mm256_gf2p8affine_epi64_epi8(s, _mm512_castsi512_si256(m2), 0);
    __m256i p3 =
        _mm256_gf2p8affine_epi64_epi8(s, _mm512_castsi512_si256(m3), 0);
    _mm256_storeu_si256(
        (__m256i *)(d0 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d0 + i)), p0));
    _mm256_storeu_si256(
        (__m256i *)(d1 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d1 + i)), p1));
    _mm256_storeu_si256(
        (__m256i *)(d2 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d2 + i)), p2));
    _mm256_storeu_si256(
        (__m256i *)(d3 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d3 + i)), p3));
    i += 32;
  }
  if (len - i >= 16) {
    __m128i s = _mm_loadu_si128((const __m128i *)(src + i));
    __m128i p0 = _mm_gf2p8affine_epi64_epi8(s, _mm512_castsi512_si128(m0), 0);
    __m128i p1 = _mm_gf2p8affine_epi64_epi8(s, _mm512_castsi512_si128(m1), 0);
    __m128i p2 = _mm_gf2p8affine_epi64_epi8(s, _mm512_castsi512_si128(m2), 0);
    __m128i p3 = _mm_gf2p8affine_epi64_epi8(s, _mm512_castsi512_si128(m3), 0);
    _mm_storeu_si128(
        (__m128i *)(d0 + i),
        _mm_xor_si128(_mm_loadu_si128((const __m128i *)(d0 + i)), p0));
    _mm_storeu_si128(
        (__m128i *)(d1 + i),
        _mm_xor_si128(_mm_loadu_si128((const __m128i *)(d1 + i)), p1));
    _mm_storeu_si128(
        (__m128i *)(d2 + i),
        _mm_xor_si128(_mm_loadu_si128((const __m128i *)(d2 + i)), p2));
    _mm_storeu_si128(
        (__m128i *)(d3 + i),
        _mm_xor_si128(_mm_loadu_si128((const __m128i *)(d3 + i)), p3));
    i += 16;
  }
  for (; i < len; i++) {
    uint8_t b = src[i];
    if (b) {
      d0[i] ^= obl_fused_mul(u0, b);
      d1[i] ^= obl_fused_mul(u1, b);
      d2[i] ^= obl_fused_mul(u2, b);
      d3[i] ^= obl_fused_mul(u3, b);
    }
  }
}

OBL_FUSED_TARGET_AVX512 static inline void
obl_fused_shuf512_fanout2(uint8_t *d0, uint8_t *d1, uint8_t u0, uint8_t u1,
                          const uint8_t *src, uint32_t len) {
  const __m512i mask = _mm512_set1_epi8(0x0f);
  const __m512i lo0 = _mm512_broadcast_i32x4(
      _mm_loadu_si128((const __m128i *)(GF2_8_SHUF_LO + u0 * 16)));
  const __m512i hi0 = _mm512_broadcast_i32x4(
      _mm_loadu_si128((const __m128i *)(GF2_8_SHUF_HI + u0 * 16)));
  const __m512i lo1 = _mm512_broadcast_i32x4(
      _mm_loadu_si128((const __m128i *)(GF2_8_SHUF_LO + u1 * 16)));
  const __m512i hi1 = _mm512_broadcast_i32x4(
      _mm_loadu_si128((const __m128i *)(GF2_8_SHUF_HI + u1 * 16)));
  uint32_t i = 0;
  for (; len - i >= 64; i += 64) {
    __m512i s = _mm512_loadu_si512(src + i);
    __m512i slo = _mm512_and_si512(s, mask);
    __m512i shi = _mm512_and_si512(_mm512_srli_epi64(s, 4), mask);
    __m512i p0 = _mm512_xor_si512(_mm512_shuffle_epi8(lo0, slo),
                                  _mm512_shuffle_epi8(hi0, shi));
    __m512i p1 = _mm512_xor_si512(_mm512_shuffle_epi8(lo1, slo),
                                  _mm512_shuffle_epi8(hi1, shi));
    _mm512_storeu_si512(d0 + i,
                        _mm512_xor_si512(_mm512_loadu_si512(d0 + i), p0));
    _mm512_storeu_si512(d1 + i,
                        _mm512_xor_si512(_mm512_loadu_si512(d1 + i), p1));
  }
  if (len - i >= 32) {
    __m256i m256 = _mm512_castsi512_si256(mask);
    __m256i s = _mm256_loadu_si256((const __m256i *)(src + i));
    __m256i slo = _mm256_and_si256(s, m256);
    __m256i shi = _mm256_and_si256(_mm256_srli_epi64(s, 4), m256);
    __m256i p0 =
        _mm256_xor_si256(_mm256_shuffle_epi8(_mm512_castsi512_si256(lo0), slo),
                         _mm256_shuffle_epi8(_mm512_castsi512_si256(hi0), shi));
    __m256i p1 =
        _mm256_xor_si256(_mm256_shuffle_epi8(_mm512_castsi512_si256(lo1), slo),
                         _mm256_shuffle_epi8(_mm512_castsi512_si256(hi1), shi));
    _mm256_storeu_si256(
        (__m256i *)(d0 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d0 + i)), p0));
    _mm256_storeu_si256(
        (__m256i *)(d1 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d1 + i)), p1));
    i += 32;
  }
  for (; i < len; i++) {
    uint8_t b = src[i];
    if (b) {
      d0[i] ^= obl_fused_mul(u0, b);
      d1[i] ^= obl_fused_mul(u1, b);
    }
  }
}

OBL_FUSED_TARGET_AVX512 static inline void
obl_fused_shuf512_fanout4(uint8_t *d0, uint8_t *d1, uint8_t *d2, uint8_t *d3,
                          uint8_t u0, uint8_t u1, uint8_t u2, uint8_t u3,
                          const uint8_t *src, uint32_t len) {
  const __m512i mask = _mm512_set1_epi8(0x0f);
  const __m512i lo0 = _mm512_broadcast_i32x4(
      _mm_loadu_si128((const __m128i *)(GF2_8_SHUF_LO + u0 * 16)));
  const __m512i hi0 = _mm512_broadcast_i32x4(
      _mm_loadu_si128((const __m128i *)(GF2_8_SHUF_HI + u0 * 16)));
  const __m512i lo1 = _mm512_broadcast_i32x4(
      _mm_loadu_si128((const __m128i *)(GF2_8_SHUF_LO + u1 * 16)));
  const __m512i hi1 = _mm512_broadcast_i32x4(
      _mm_loadu_si128((const __m128i *)(GF2_8_SHUF_HI + u1 * 16)));
  const __m512i lo2 = _mm512_broadcast_i32x4(
      _mm_loadu_si128((const __m128i *)(GF2_8_SHUF_LO + u2 * 16)));
  const __m512i hi2 = _mm512_broadcast_i32x4(
      _mm_loadu_si128((const __m128i *)(GF2_8_SHUF_HI + u2 * 16)));
  const __m512i lo3 = _mm512_broadcast_i32x4(
      _mm_loadu_si128((const __m128i *)(GF2_8_SHUF_LO + u3 * 16)));
  const __m512i hi3 = _mm512_broadcast_i32x4(
      _mm_loadu_si128((const __m128i *)(GF2_8_SHUF_HI + u3 * 16)));
  uint32_t i = 0;
  for (; len - i >= 64; i += 64) {
    __m512i s = _mm512_loadu_si512(src + i);
    __m512i slo = _mm512_and_si512(s, mask);
    __m512i shi = _mm512_and_si512(_mm512_srli_epi64(s, 4), mask);
    __m512i p0 = _mm512_xor_si512(_mm512_shuffle_epi8(lo0, slo),
                                  _mm512_shuffle_epi8(hi0, shi));
    __m512i p1 = _mm512_xor_si512(_mm512_shuffle_epi8(lo1, slo),
                                  _mm512_shuffle_epi8(hi1, shi));
    __m512i p2 = _mm512_xor_si512(_mm512_shuffle_epi8(lo2, slo),
                                  _mm512_shuffle_epi8(hi2, shi));
    __m512i p3 = _mm512_xor_si512(_mm512_shuffle_epi8(lo3, slo),
                                  _mm512_shuffle_epi8(hi3, shi));
    _mm512_storeu_si512(d0 + i,
                        _mm512_xor_si512(_mm512_loadu_si512(d0 + i), p0));
    _mm512_storeu_si512(d1 + i,
                        _mm512_xor_si512(_mm512_loadu_si512(d1 + i), p1));
    _mm512_storeu_si512(d2 + i,
                        _mm512_xor_si512(_mm512_loadu_si512(d2 + i), p2));
    _mm512_storeu_si512(d3 + i,
                        _mm512_xor_si512(_mm512_loadu_si512(d3 + i), p3));
  }
  if (len - i >= 32) {
    __m256i m256 = _mm512_castsi512_si256(mask);
    __m256i s = _mm256_loadu_si256((const __m256i *)(src + i));
    __m256i slo = _mm256_and_si256(s, m256);
    __m256i shi = _mm256_and_si256(_mm256_srli_epi64(s, 4), m256);
    __m256i p0 =
        _mm256_xor_si256(_mm256_shuffle_epi8(_mm512_castsi512_si256(lo0), slo),
                         _mm256_shuffle_epi8(_mm512_castsi512_si256(hi0), shi));
    __m256i p1 =
        _mm256_xor_si256(_mm256_shuffle_epi8(_mm512_castsi512_si256(lo1), slo),
                         _mm256_shuffle_epi8(_mm512_castsi512_si256(hi1), shi));
    __m256i p2 =
        _mm256_xor_si256(_mm256_shuffle_epi8(_mm512_castsi512_si256(lo2), slo),
                         _mm256_shuffle_epi8(_mm512_castsi512_si256(hi2), shi));
    __m256i p3 =
        _mm256_xor_si256(_mm256_shuffle_epi8(_mm512_castsi512_si256(lo3), slo),
                         _mm256_shuffle_epi8(_mm512_castsi512_si256(hi3), shi));
    _mm256_storeu_si256(
        (__m256i *)(d0 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d0 + i)), p0));
    _mm256_storeu_si256(
        (__m256i *)(d1 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d1 + i)), p1));
    _mm256_storeu_si256(
        (__m256i *)(d2 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d2 + i)), p2));
    _mm256_storeu_si256(
        (__m256i *)(d3 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d3 + i)), p3));
    i += 32;
  }
  for (; i < len; i++) {
    uint8_t b = src[i];
    if (b) {
      d0[i] ^= obl_fused_mul(u0, b);
      d1[i] ^= obl_fused_mul(u1, b);
      d2[i] ^= obl_fused_mul(u2, b);
      d3[i] ^= obl_fused_mul(u3, b);
    }
  }
}

OBL_FUSED_TARGET_AVX2 static inline void
obl_fused_xor256_fanout2(uint8_t *d0, uint8_t *d1, const uint8_t *src,
                         uint32_t len) {
  uint32_t i = 0;
  for (; len - i >= 32; i += 32) {
    __m256i s = _mm256_loadu_si256((const __m256i *)(src + i));
    _mm256_storeu_si256(
        (__m256i *)(d0 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d0 + i)), s));
    _mm256_storeu_si256(
        (__m256i *)(d1 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d1 + i)), s));
  }
  for (; i < len; i++) {
    uint8_t s = src[i];
    d0[i] ^= s;
    d1[i] ^= s;
  }
}

OBL_FUSED_TARGET_AVX2 static inline void
obl_fused_xor256_fanout4(uint8_t *d0, uint8_t *d1, uint8_t *d2, uint8_t *d3,
                         const uint8_t *src, uint32_t len) {
  uint32_t i = 0;
  for (; len - i >= 32; i += 32) {
    __m256i s = _mm256_loadu_si256((const __m256i *)(src + i));
    _mm256_storeu_si256(
        (__m256i *)(d0 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d0 + i)), s));
    _mm256_storeu_si256(
        (__m256i *)(d1 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d1 + i)), s));
    _mm256_storeu_si256(
        (__m256i *)(d2 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d2 + i)), s));
    _mm256_storeu_si256(
        (__m256i *)(d3 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d3 + i)), s));
  }
  for (; i < len; i++) {
    uint8_t s = src[i];
    d0[i] ^= s;
    d1[i] ^= s;
    d2[i] ^= s;
    d3[i] ^= s;
  }
}

OBL_FUSED_TARGET_AVX2 static inline void
obl_xor_set4_avx2(uint8_t *dst, const uint8_t *r0, const uint8_t *r1,
                  const uint8_t *r2, const uint8_t *r3, uint32_t len) {
  uint32_t i = 0;
  for (; len - i >= 32; i += 32) {
    __m256i v01 =
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(r0 + i)),
                         _mm256_loadu_si256((const __m256i *)(r1 + i)));
    __m256i v23 =
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(r2 + i)),
                         _mm256_loadu_si256((const __m256i *)(r3 + i)));
    _mm256_storeu_si256((__m256i *)(dst + i), _mm256_xor_si256(v01, v23));
  }
  for (; i < len; i++)
    dst[i] = r0[i] ^ r1[i] ^ r2[i] ^ r3[i];
}

OBL_FUSED_TARGET_AVX2 static inline void
obl_xor_acc4_avx2(uint8_t *dst, const uint8_t *r0, const uint8_t *r1,
                  const uint8_t *r2, const uint8_t *r3, uint32_t len) {
  uint32_t i = 0;
  for (; len - i >= 32; i += 32) {
    __m256i v01 =
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(r0 + i)),
                         _mm256_loadu_si256((const __m256i *)(r1 + i)));
    __m256i v23 =
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(r2 + i)),
                         _mm256_loadu_si256((const __m256i *)(r3 + i)));
    __m256i acc =
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(dst + i)), v01);
    _mm256_storeu_si256((__m256i *)(dst + i), _mm256_xor_si256(acc, v23));
  }
  for (; i < len; i++)
    dst[i] ^= r0[i] ^ r1[i] ^ r2[i] ^ r3[i];
}

OBL_FUSED_TARGET_AVX2_GFNI static inline void
obl_fused_gfni256_fanout2(uint8_t *d0, uint8_t *d1, uint8_t u0, uint8_t u1,
                          const uint8_t *src, uint32_t len) {
  const __m256i m0 = _mm256_set1_epi64x(GF2_8_AFFINE_MAT[u0]);
  const __m256i m1 = _mm256_set1_epi64x(GF2_8_AFFINE_MAT[u1]);
  uint32_t i = 0;
  for (; len - i >= 32; i += 32) {
    __m256i s = _mm256_loadu_si256((const __m256i *)(src + i));
    __m256i p0 = _mm256_gf2p8affine_epi64_epi8(s, m0, 0);
    __m256i p1 = _mm256_gf2p8affine_epi64_epi8(s, m1, 0);
    _mm256_storeu_si256(
        (__m256i *)(d0 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d0 + i)), p0));
    _mm256_storeu_si256(
        (__m256i *)(d1 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d1 + i)), p1));
  }
  for (; i < len; i++) {
    uint8_t b = src[i];
    if (b) {
      d0[i] ^= obl_fused_mul(u0, b);
      d1[i] ^= obl_fused_mul(u1, b);
    }
  }
}

OBL_FUSED_TARGET_AVX2_GFNI static inline void
obl_fused_gfni256_fanout4(uint8_t *d0, uint8_t *d1, uint8_t *d2, uint8_t *d3,
                          uint8_t u0, uint8_t u1, uint8_t u2, uint8_t u3,
                          const uint8_t *src, uint32_t len) {
  const __m256i m0 = _mm256_set1_epi64x(GF2_8_AFFINE_MAT[u0]);
  const __m256i m1 = _mm256_set1_epi64x(GF2_8_AFFINE_MAT[u1]);
  const __m256i m2 = _mm256_set1_epi64x(GF2_8_AFFINE_MAT[u2]);
  const __m256i m3 = _mm256_set1_epi64x(GF2_8_AFFINE_MAT[u3]);
  uint32_t i = 0;
  for (; len - i >= 32; i += 32) {
    __m256i s = _mm256_loadu_si256((const __m256i *)(src + i));
    __m256i p0 = _mm256_gf2p8affine_epi64_epi8(s, m0, 0);
    __m256i p1 = _mm256_gf2p8affine_epi64_epi8(s, m1, 0);
    __m256i p2 = _mm256_gf2p8affine_epi64_epi8(s, m2, 0);
    __m256i p3 = _mm256_gf2p8affine_epi64_epi8(s, m3, 0);
    _mm256_storeu_si256(
        (__m256i *)(d0 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d0 + i)), p0));
    _mm256_storeu_si256(
        (__m256i *)(d1 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d1 + i)), p1));
    _mm256_storeu_si256(
        (__m256i *)(d2 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d2 + i)), p2));
    _mm256_storeu_si256(
        (__m256i *)(d3 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d3 + i)), p3));
  }
  for (; i < len; i++) {
    uint8_t b = src[i];
    if (b) {
      d0[i] ^= obl_fused_mul(u0, b);
      d1[i] ^= obl_fused_mul(u1, b);
      d2[i] ^= obl_fused_mul(u2, b);
      d3[i] ^= obl_fused_mul(u3, b);
    }
  }
}

OBL_FUSED_TARGET_AVX2 static inline void
obl_fused_shuf256_fanout2(uint8_t *d0, uint8_t *d1, uint8_t u0, uint8_t u1,
                          const uint8_t *src, uint32_t len) {
  const __m256i mask = _mm256_set1_epi8(0x0f);
  const __m128i l0 =
      _mm_loadu_si128((const __m128i *)(GF2_8_SHUF_LO + u0 * 16));
  const __m128i h0 =
      _mm_loadu_si128((const __m128i *)(GF2_8_SHUF_HI + u0 * 16));
  const __m128i l1 =
      _mm_loadu_si128((const __m128i *)(GF2_8_SHUF_LO + u1 * 16));
  const __m128i h1 =
      _mm_loadu_si128((const __m128i *)(GF2_8_SHUF_HI + u1 * 16));
  const __m256i lo0 =
      _mm256_inserti128_si256(_mm256_castsi128_si256(l0), l0, 1);
  const __m256i hi0 =
      _mm256_inserti128_si256(_mm256_castsi128_si256(h0), h0, 1);
  const __m256i lo1 =
      _mm256_inserti128_si256(_mm256_castsi128_si256(l1), l1, 1);
  const __m256i hi1 =
      _mm256_inserti128_si256(_mm256_castsi128_si256(h1), h1, 1);
  uint32_t i = 0;
  for (; len - i >= 32; i += 32) {
    __m256i s = _mm256_loadu_si256((const __m256i *)(src + i));
    __m256i slo = _mm256_and_si256(s, mask);
    __m256i shi = _mm256_and_si256(_mm256_srli_epi64(s, 4), mask);
    __m256i p0 = _mm256_xor_si256(_mm256_shuffle_epi8(lo0, slo),
                                  _mm256_shuffle_epi8(hi0, shi));
    __m256i p1 = _mm256_xor_si256(_mm256_shuffle_epi8(lo1, slo),
                                  _mm256_shuffle_epi8(hi1, shi));
    _mm256_storeu_si256(
        (__m256i *)(d0 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d0 + i)), p0));
    _mm256_storeu_si256(
        (__m256i *)(d1 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d1 + i)), p1));
  }
  for (; i < len; i++) {
    uint8_t b = src[i];
    if (b) {
      d0[i] ^= obl_fused_mul(u0, b);
      d1[i] ^= obl_fused_mul(u1, b);
    }
  }
}

OBL_FUSED_TARGET_AVX2 static inline void
obl_fused_shuf256_fanout4(uint8_t *d0, uint8_t *d1, uint8_t *d2, uint8_t *d3,
                          uint8_t u0, uint8_t u1, uint8_t u2, uint8_t u3,
                          const uint8_t *src, uint32_t len) {
  const __m256i mask = _mm256_set1_epi8(0x0f);
  const __m128i l0 =
      _mm_loadu_si128((const __m128i *)(GF2_8_SHUF_LO + u0 * 16));
  const __m128i h0 =
      _mm_loadu_si128((const __m128i *)(GF2_8_SHUF_HI + u0 * 16));
  const __m128i l1 =
      _mm_loadu_si128((const __m128i *)(GF2_8_SHUF_LO + u1 * 16));
  const __m128i h1 =
      _mm_loadu_si128((const __m128i *)(GF2_8_SHUF_HI + u1 * 16));
  const __m128i l2 =
      _mm_loadu_si128((const __m128i *)(GF2_8_SHUF_LO + u2 * 16));
  const __m128i h2 =
      _mm_loadu_si128((const __m128i *)(GF2_8_SHUF_HI + u2 * 16));
  const __m128i l3 =
      _mm_loadu_si128((const __m128i *)(GF2_8_SHUF_LO + u3 * 16));
  const __m128i h3 =
      _mm_loadu_si128((const __m128i *)(GF2_8_SHUF_HI + u3 * 16));
  const __m256i lo0 =
      _mm256_inserti128_si256(_mm256_castsi128_si256(l0), l0, 1);
  const __m256i hi0 =
      _mm256_inserti128_si256(_mm256_castsi128_si256(h0), h0, 1);
  const __m256i lo1 =
      _mm256_inserti128_si256(_mm256_castsi128_si256(l1), l1, 1);
  const __m256i hi1 =
      _mm256_inserti128_si256(_mm256_castsi128_si256(h1), h1, 1);
  const __m256i lo2 =
      _mm256_inserti128_si256(_mm256_castsi128_si256(l2), l2, 1);
  const __m256i hi2 =
      _mm256_inserti128_si256(_mm256_castsi128_si256(h2), h2, 1);
  const __m256i lo3 =
      _mm256_inserti128_si256(_mm256_castsi128_si256(l3), l3, 1);
  const __m256i hi3 =
      _mm256_inserti128_si256(_mm256_castsi128_si256(h3), h3, 1);
  uint32_t i = 0;
  for (; len - i >= 32; i += 32) {
    __m256i s = _mm256_loadu_si256((const __m256i *)(src + i));
    __m256i slo = _mm256_and_si256(s, mask);
    __m256i shi = _mm256_and_si256(_mm256_srli_epi64(s, 4), mask);
    __m256i p0 = _mm256_xor_si256(_mm256_shuffle_epi8(lo0, slo),
                                  _mm256_shuffle_epi8(hi0, shi));
    __m256i p1 = _mm256_xor_si256(_mm256_shuffle_epi8(lo1, slo),
                                  _mm256_shuffle_epi8(hi1, shi));
    __m256i p2 = _mm256_xor_si256(_mm256_shuffle_epi8(lo2, slo),
                                  _mm256_shuffle_epi8(hi2, shi));
    __m256i p3 = _mm256_xor_si256(_mm256_shuffle_epi8(lo3, slo),
                                  _mm256_shuffle_epi8(hi3, shi));
    _mm256_storeu_si256(
        (__m256i *)(d0 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d0 + i)), p0));
    _mm256_storeu_si256(
        (__m256i *)(d1 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d1 + i)), p1));
    _mm256_storeu_si256(
        (__m256i *)(d2 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d2 + i)), p2));
    _mm256_storeu_si256(
        (__m256i *)(d3 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d3 + i)), p3));
  }
  for (; i < len; i++) {
    uint8_t b = src[i];
    if (b) {
      d0[i] ^= obl_fused_mul(u0, b);
      d1[i] ^= obl_fused_mul(u1, b);
      d2[i] ^= obl_fused_mul(u2, b);
      d3[i] ^= obl_fused_mul(u3, b);
    }
  }
}

OBL_FUSED_TARGET_AVX512_GFNI OBL_FUSED_INLINE void
obl_batch_axpy2_avx512_gfni(u8 *d0, u8 *d1, const u8 *src, u8 u0, u8 u1,
                            unsigned len) {
  if (u0 == 1 && u1 == 1)
    obl_fused_xor512_fanout2(d0, d1, src, len);
  else
    obl_fused_gfni512_fanout2(d0, d1, u0, u1, src, len);
}

OBL_FUSED_TARGET_AVX512_GFNI OBL_FUSED_INLINE void
obl_batch_axpy4_avx512_gfni(u8 *d0, u8 *d1, u8 *d2, u8 *d3, const u8 *src,
                            u8 u0, u8 u1, u8 u2, u8 u3, unsigned len) {
  if (u0 == 1 && u1 == 1 && u2 == 1 && u3 == 1)
    obl_fused_xor512_fanout4(d0, d1, d2, d3, src, len);
  else
    obl_fused_gfni512_fanout4(d0, d1, d2, d3, u0, u1, u2, u3, src, len);
}

OBL_FUSED_TARGET_AVX512 OBL_FUSED_INLINE void
obl_batch_axpy2_avx512_std(u8 *d0, u8 *d1, const u8 *src, u8 u0, u8 u1,
                           unsigned len) {
  if (u0 == 1 && u1 == 1)
    obl_fused_xor512_fanout2(d0, d1, src, len);
  else
    obl_fused_shuf512_fanout2(d0, d1, u0, u1, src, len);
}

OBL_FUSED_TARGET_AVX512 OBL_FUSED_INLINE void
obl_batch_axpy4_avx512_std(u8 *d0, u8 *d1, u8 *d2, u8 *d3, const u8 *src, u8 u0,
                           u8 u1, u8 u2, u8 u3, unsigned len) {
  if (u0 == 1 && u1 == 1 && u2 == 1 && u3 == 1)
    obl_fused_xor512_fanout4(d0, d1, d2, d3, src, len);
  else
    obl_fused_shuf512_fanout4(d0, d1, d2, d3, u0, u1, u2, u3, src, len);
}

OBL_FUSED_TARGET_AVX2_GFNI OBL_FUSED_INLINE void
obl_batch_axpy2_avx2_gfni(u8 *d0, u8 *d1, const u8 *src, u8 u0, u8 u1,
                          unsigned len) {
  if (u0 == 1 && u1 == 1)
    obl_fused_xor256_fanout2(d0, d1, src, len);
  else
    obl_fused_gfni256_fanout2(d0, d1, u0, u1, src, len);
}

OBL_FUSED_TARGET_AVX2_GFNI OBL_FUSED_INLINE void
obl_batch_axpy4_avx2_gfni(u8 *d0, u8 *d1, u8 *d2, u8 *d3, const u8 *src, u8 u0,
                          u8 u1, u8 u2, u8 u3, unsigned len) {
  if (u0 == 1 && u1 == 1 && u2 == 1 && u3 == 1)
    obl_fused_xor256_fanout4(d0, d1, d2, d3, src, len);
  else
    obl_fused_gfni256_fanout4(d0, d1, d2, d3, u0, u1, u2, u3, src, len);
}

OBL_FUSED_TARGET_AVX2 OBL_FUSED_INLINE void
obl_batch_axpy2_avx2_std(u8 *d0, u8 *d1, const u8 *src, u8 u0, u8 u1,
                         unsigned len) {
  if (u0 == 1 && u1 == 1)
    obl_fused_xor256_fanout2(d0, d1, src, len);
  else
    obl_fused_shuf256_fanout2(d0, d1, u0, u1, src, len);
}

OBL_FUSED_TARGET_AVX2 OBL_FUSED_INLINE void
obl_batch_axpy4_avx2_std(u8 *d0, u8 *d1, u8 *d2, u8 *d3, const u8 *src, u8 u0,
                         u8 u1, u8 u2, u8 u3, unsigned len) {
  if (u0 == 1 && u1 == 1 && u2 == 1 && u3 == 1)
    obl_fused_xor256_fanout4(d0, d1, d2, d3, src, len);
  else
    obl_fused_shuf256_fanout4(d0, d1, d2, d3, u0, u1, u2, u3, src, len);
}
/* Dispatch entries remain ordinary functions. Only the private batch calls
 * force inlining: generic callers may devirtualize these entries under LTO. */
#define OBL_FUSED_ENTRY(SUFFIX, ATTR)                                          \
  ATTR static void obl_axpy2_##SUFFIX(u8 *d0, u8 *d1, const u8 *src, u8 u0,    \
                                      u8 u1, unsigned len) {                   \
    obl_batch_axpy2_##SUFFIX(d0, d1, src, u0, u1, len);                        \
  }                                                                            \
  ATTR static void obl_axpy4_##SUFFIX(u8 *d0, u8 *d1, u8 *d2, u8 *d3,          \
                                      const u8 *src, u8 u0, u8 u1, u8 u2,      \
                                      u8 u3, unsigned len) {                   \
    obl_batch_axpy4_##SUFFIX(d0, d1, d2, d3, src, u0, u1, u2, u3, len);        \
  }
OBL_FUSED_ENTRY(avx2_std, OBL_FUSED_TARGET_AVX2)
OBL_FUSED_ENTRY(avx2_gfni, OBL_FUSED_TARGET_AVX2_GFNI)
OBL_FUSED_ENTRY(avx512_std, OBL_FUSED_TARGET_AVX512)
OBL_FUSED_ENTRY(avx512_gfni, OBL_FUSED_TARGET_AVX512_GFNI)
#undef OBL_FUSED_ENTRY

#endif

#if defined(OBLAS_ARCH_ARM) && (defined(__ARM_NEON) || defined(_MSC_VER))

static inline void obl_fused_neon_xor2(uint8_t *d0, uint8_t *d1,
                                       const uint8_t *src, uint32_t len) {
  uint32_t i = 0;
  for (; len - i >= 16; i += 16) {
    uint8x16_t s = vld1q_u8(src + i);
    vst1q_u8(d0 + i, veorq_u8(vld1q_u8(d0 + i), s));
    vst1q_u8(d1 + i, veorq_u8(vld1q_u8(d1 + i), s));
  }
  for (; i < len; i++) {
    uint8_t s = src[i];
    d0[i] ^= s;
    d1[i] ^= s;
  }
}

static inline void obl_fused_neon_xor4(uint8_t *d0, uint8_t *d1, uint8_t *d2,
                                       uint8_t *d3, const uint8_t *src,
                                       uint32_t len) {
  uint32_t i = 0;
  for (; len - i >= 16; i += 16) {
    uint8x16_t s = vld1q_u8(src + i);
    vst1q_u8(d0 + i, veorq_u8(vld1q_u8(d0 + i), s));
    vst1q_u8(d1 + i, veorq_u8(vld1q_u8(d1 + i), s));
    vst1q_u8(d2 + i, veorq_u8(vld1q_u8(d2 + i), s));
    vst1q_u8(d3 + i, veorq_u8(vld1q_u8(d3 + i), s));
  }
  for (; i < len; i++) {
    uint8_t s = src[i];
    d0[i] ^= s;
    d1[i] ^= s;
    d2[i] ^= s;
    d3[i] ^= s;
  }
}

static inline void obl_xor_set4_neon(uint8_t *dst, const uint8_t *r0,
                                     const uint8_t *r1, const uint8_t *r2,
                                     const uint8_t *r3, uint32_t len) {
  uint32_t i = 0;
  for (; len - i >= 16; i += 16) {
    uint8x16_t v01 = veorq_u8(vld1q_u8(r0 + i), vld1q_u8(r1 + i));
    uint8x16_t v23 = veorq_u8(vld1q_u8(r2 + i), vld1q_u8(r3 + i));
    vst1q_u8(dst + i, veorq_u8(v01, v23));
  }
  for (; i < len; i++)
    dst[i] = r0[i] ^ r1[i] ^ r2[i] ^ r3[i];
}

static inline void obl_xor_acc4_neon(uint8_t *dst, const uint8_t *r0,
                                     const uint8_t *r1, const uint8_t *r2,
                                     const uint8_t *r3, uint32_t len) {
  uint32_t i = 0;
  for (; len - i >= 16; i += 16) {
    uint8x16_t v01 = veorq_u8(vld1q_u8(r0 + i), vld1q_u8(r1 + i));
    uint8x16_t v23 = veorq_u8(vld1q_u8(r2 + i), vld1q_u8(r3 + i));
    uint8x16_t acc = veorq_u8(vld1q_u8(dst + i), v01);
    vst1q_u8(dst + i, veorq_u8(acc, v23));
  }
  for (; i < len; i++)
    dst[i] ^= r0[i] ^ r1[i] ^ r2[i] ^ r3[i];
}

static inline void obl_fused_neon_gf2(uint8_t *d0, uint8_t *d1, uint8_t u0,
                                      uint8_t u1, const uint8_t *src,
                                      uint32_t len) {
  const uint8x16_t mask = vdupq_n_u8(0x0f);
  const uint8x16_t lo0 = vld1q_u8(GF2_8_SHUF_LO + u0 * 16);
  const uint8x16_t hi0 = vld1q_u8(GF2_8_SHUF_HI + u0 * 16);
  const uint8x16_t lo1 = vld1q_u8(GF2_8_SHUF_LO + u1 * 16);
  const uint8x16_t hi1 = vld1q_u8(GF2_8_SHUF_HI + u1 * 16);
  uint32_t i = 0;
  for (; len - i >= 16; i += 16) {
    uint8x16_t s = vld1q_u8(src + i);
    uint8x16_t slo = vandq_u8(s, mask);
    uint8x16_t shi = vshrq_n_u8(s, 4);
    uint8x16_t p0 = veorq_u8(vqtbl1q_u8(lo0, slo), vqtbl1q_u8(hi0, shi));
    uint8x16_t p1 = veorq_u8(vqtbl1q_u8(lo1, slo), vqtbl1q_u8(hi1, shi));
    vst1q_u8(d0 + i, veorq_u8(vld1q_u8(d0 + i), p0));
    vst1q_u8(d1 + i, veorq_u8(vld1q_u8(d1 + i), p1));
  }
  for (; i < len; i++) {
    uint8_t b = src[i];
    if (b) {
      d0[i] ^= obl_fused_mul(u0, b);
      d1[i] ^= obl_fused_mul(u1, b);
    }
  }
}

static inline void obl_fused_neon_gf4(uint8_t *d0, uint8_t *d1, uint8_t *d2,
                                      uint8_t *d3, uint8_t u0, uint8_t u1,
                                      uint8_t u2, uint8_t u3,
                                      const uint8_t *src, uint32_t len) {
  const uint8x16_t mask = vdupq_n_u8(0x0f);
  const uint8x16_t lo0 = vld1q_u8(GF2_8_SHUF_LO + u0 * 16);
  const uint8x16_t hi0 = vld1q_u8(GF2_8_SHUF_HI + u0 * 16);
  const uint8x16_t lo1 = vld1q_u8(GF2_8_SHUF_LO + u1 * 16);
  const uint8x16_t hi1 = vld1q_u8(GF2_8_SHUF_HI + u1 * 16);
  const uint8x16_t lo2 = vld1q_u8(GF2_8_SHUF_LO + u2 * 16);
  const uint8x16_t hi2 = vld1q_u8(GF2_8_SHUF_HI + u2 * 16);
  const uint8x16_t lo3 = vld1q_u8(GF2_8_SHUF_LO + u3 * 16);
  const uint8x16_t hi3 = vld1q_u8(GF2_8_SHUF_HI + u3 * 16);
  uint32_t i = 0;
  for (; len - i >= 16; i += 16) {
    uint8x16_t s = vld1q_u8(src + i);
    uint8x16_t slo = vandq_u8(s, mask);
    uint8x16_t shi = vshrq_n_u8(s, 4);
    uint8x16_t p0 = veorq_u8(vqtbl1q_u8(lo0, slo), vqtbl1q_u8(hi0, shi));
    uint8x16_t p1 = veorq_u8(vqtbl1q_u8(lo1, slo), vqtbl1q_u8(hi1, shi));
    uint8x16_t p2 = veorq_u8(vqtbl1q_u8(lo2, slo), vqtbl1q_u8(hi2, shi));
    uint8x16_t p3 = veorq_u8(vqtbl1q_u8(lo3, slo), vqtbl1q_u8(hi3, shi));
    vst1q_u8(d0 + i, veorq_u8(vld1q_u8(d0 + i), p0));
    vst1q_u8(d1 + i, veorq_u8(vld1q_u8(d1 + i), p1));
    vst1q_u8(d2 + i, veorq_u8(vld1q_u8(d2 + i), p2));
    vst1q_u8(d3 + i, veorq_u8(vld1q_u8(d3 + i), p3));
  }
  for (; i < len; i++) {
    uint8_t b = src[i];
    if (b) {
      d0[i] ^= obl_fused_mul(u0, b);
      d1[i] ^= obl_fused_mul(u1, b);
      d2[i] ^= obl_fused_mul(u2, b);
      d3[i] ^= obl_fused_mul(u3, b);
    }
  }
}

OBL_FUSED_INLINE void obl_axpy2_neon(u8 *d0, u8 *d1, const u8 *src, u8 u0,
                                     u8 u1, unsigned len) {
  if (u0 == 1 && u1 == 1)
    obl_fused_neon_xor2(d0, d1, src, len);
  else
    obl_fused_neon_gf2(d0, d1, u0, u1, src, len);
}

OBL_FUSED_INLINE void obl_axpy4_neon(u8 *d0, u8 *d1, u8 *d2, u8 *d3,
                                     const u8 *src, u8 u0, u8 u1, u8 u2, u8 u3,
                                     unsigned len) {
  if (u0 == 1 && u1 == 1 && u2 == 1 && u3 == 1)
    obl_fused_neon_xor4(d0, d1, d2, d3, src, len);
  else
    obl_fused_neon_gf4(d0, d1, d2, d3, u0, u1, u2, u3, src, len);
}
#endif

#if defined(OBLAS_ARCH_RISCV) && defined(__riscv_vector)
static inline void obl_fused_rvv_xor2(uint8_t *d0, uint8_t *d1,
                                      const uint8_t *src, uint32_t len) {
  for (uint32_t i = 0; i < len;) {
    size_t vl = __riscv_vsetvl_e8m4(len - i);
    vuint8m4_t s = __riscv_vle8_v_u8m4(src + i, vl);
    __riscv_vse8_v_u8m4(
        d0 + i, __riscv_vxor_vv_u8m4(__riscv_vle8_v_u8m4(d0 + i, vl), s, vl),
        vl);
    __riscv_vse8_v_u8m4(
        d1 + i, __riscv_vxor_vv_u8m4(__riscv_vle8_v_u8m4(d1 + i, vl), s, vl),
        vl);
    i += (uint32_t)vl;
  }
}

static inline void obl_fused_rvv_xor4(uint8_t *d0, uint8_t *d1, uint8_t *d2,
                                      uint8_t *d3, const uint8_t *src,
                                      uint32_t len) {
  for (uint32_t i = 0; i < len;) {
    size_t vl = __riscv_vsetvl_e8m4(len - i);
    vuint8m4_t s = __riscv_vle8_v_u8m4(src + i, vl);
    __riscv_vse8_v_u8m4(
        d0 + i, __riscv_vxor_vv_u8m4(__riscv_vle8_v_u8m4(d0 + i, vl), s, vl),
        vl);
    __riscv_vse8_v_u8m4(
        d1 + i, __riscv_vxor_vv_u8m4(__riscv_vle8_v_u8m4(d1 + i, vl), s, vl),
        vl);
    __riscv_vse8_v_u8m4(
        d2 + i, __riscv_vxor_vv_u8m4(__riscv_vle8_v_u8m4(d2 + i, vl), s, vl),
        vl);
    __riscv_vse8_v_u8m4(
        d3 + i, __riscv_vxor_vv_u8m4(__riscv_vle8_v_u8m4(d3 + i, vl), s, vl),
        vl);
    i += (uint32_t)vl;
  }
}

static inline void obl_xor_set4_rvv(uint8_t *dst, const uint8_t *r0,
                                    const uint8_t *r1, const uint8_t *r2,
                                    const uint8_t *r3, uint32_t len) {
  for (uint32_t i = 0; i < len;) {
    size_t vl = __riscv_vsetvl_e8m4(len - i);
    vuint8m4_t v01 = __riscv_vxor_vv_u8m4(__riscv_vle8_v_u8m4(r0 + i, vl),
                                          __riscv_vle8_v_u8m4(r1 + i, vl), vl);
    vuint8m4_t v23 = __riscv_vxor_vv_u8m4(__riscv_vle8_v_u8m4(r2 + i, vl),
                                          __riscv_vle8_v_u8m4(r3 + i, vl), vl);
    __riscv_vse8_v_u8m4(dst + i, __riscv_vxor_vv_u8m4(v01, v23, vl), vl);
    i += (uint32_t)vl;
  }
}

static inline void obl_xor_acc4_rvv(uint8_t *dst, const uint8_t *r0,
                                    const uint8_t *r1, const uint8_t *r2,
                                    const uint8_t *r3, uint32_t len) {
  for (uint32_t i = 0; i < len;) {
    size_t vl = __riscv_vsetvl_e8m4(len - i);
    vuint8m4_t v01 = __riscv_vxor_vv_u8m4(__riscv_vle8_v_u8m4(r0 + i, vl),
                                          __riscv_vle8_v_u8m4(r1 + i, vl), vl);
    vuint8m4_t v23 = __riscv_vxor_vv_u8m4(__riscv_vle8_v_u8m4(r2 + i, vl),
                                          __riscv_vle8_v_u8m4(r3 + i, vl), vl);
    vuint8m4_t acc =
        __riscv_vxor_vv_u8m4(__riscv_vle8_v_u8m4(dst + i, vl), v01, vl);
    __riscv_vse8_v_u8m4(dst + i, __riscv_vxor_vv_u8m4(acc, v23, vl), vl);
    i += (uint32_t)vl;
  }
}

/* keep LMUL=1 for vrgather to avoid quadratic LMUL microcode penalties */
static inline void obl_fused_rvv_gf2(uint8_t *d0, uint8_t *d1, uint8_t u0,
                                     uint8_t u1, const uint8_t *src,
                                     uint32_t len) {
  size_t tvl = __riscv_vsetvl_e8m1(16);
  uint32_t i = 0;
  if (tvl >= 16) {
    vuint8m1_t lo0 = __riscv_vle8_v_u8m1(GF2_8_SHUF_LO + u0 * 16, tvl);
    vuint8m1_t hi0 = __riscv_vle8_v_u8m1(GF2_8_SHUF_HI + u0 * 16, tvl);
    vuint8m1_t lo1 = __riscv_vle8_v_u8m1(GF2_8_SHUF_LO + u1 * 16, tvl);
    vuint8m1_t hi1 = __riscv_vle8_v_u8m1(GF2_8_SHUF_HI + u1 * 16, tvl);
    for (; i < len;) {
      size_t vl = __riscv_vsetvl_e8m1(len - i);
      vuint8m1_t s = __riscv_vle8_v_u8m1(src + i, vl);
      vuint8m1_t slo = __riscv_vand_vx_u8m1(s, 0x0f, vl);
      vuint8m1_t shi = __riscv_vsrl_vx_u8m1(s, 4, vl);
      vuint8m1_t p0 =
          __riscv_vxor_vv_u8m1(__riscv_vrgather_vv_u8m1(lo0, slo, vl),
                               __riscv_vrgather_vv_u8m1(hi0, shi, vl), vl);
      vuint8m1_t p1 =
          __riscv_vxor_vv_u8m1(__riscv_vrgather_vv_u8m1(lo1, slo, vl),
                               __riscv_vrgather_vv_u8m1(hi1, shi, vl), vl);
      __riscv_vse8_v_u8m1(
          d0 + i, __riscv_vxor_vv_u8m1(__riscv_vle8_v_u8m1(d0 + i, vl), p0, vl),
          vl);
      __riscv_vse8_v_u8m1(
          d1 + i, __riscv_vxor_vv_u8m1(__riscv_vle8_v_u8m1(d1 + i, vl), p1, vl),
          vl);
      i += (uint32_t)vl;
    }
  }
  for (; i < len; i++) {
    uint8_t b = src[i];
    if (b) {
      d0[i] ^= obl_fused_mul(u0, b);
      d1[i] ^= obl_fused_mul(u1, b);
    }
  }
}

static inline void obl_fused_rvv_gf4(uint8_t *d0, uint8_t *d1, uint8_t *d2,
                                     uint8_t *d3, uint8_t u0, uint8_t u1,
                                     uint8_t u2, uint8_t u3, const uint8_t *src,
                                     uint32_t len) {
  size_t tvl = __riscv_vsetvl_e8m1(16);
  uint32_t i = 0;
  if (tvl >= 16) {
    vuint8m1_t lo0 = __riscv_vle8_v_u8m1(GF2_8_SHUF_LO + u0 * 16, tvl);
    vuint8m1_t hi0 = __riscv_vle8_v_u8m1(GF2_8_SHUF_HI + u0 * 16, tvl);
    vuint8m1_t lo1 = __riscv_vle8_v_u8m1(GF2_8_SHUF_LO + u1 * 16, tvl);
    vuint8m1_t hi1 = __riscv_vle8_v_u8m1(GF2_8_SHUF_HI + u1 * 16, tvl);
    vuint8m1_t lo2 = __riscv_vle8_v_u8m1(GF2_8_SHUF_LO + u2 * 16, tvl);
    vuint8m1_t hi2 = __riscv_vle8_v_u8m1(GF2_8_SHUF_HI + u2 * 16, tvl);
    vuint8m1_t lo3 = __riscv_vle8_v_u8m1(GF2_8_SHUF_LO + u3 * 16, tvl);
    vuint8m1_t hi3 = __riscv_vle8_v_u8m1(GF2_8_SHUF_HI + u3 * 16, tvl);
    for (; i < len;) {
      size_t vl = __riscv_vsetvl_e8m1(len - i);
      vuint8m1_t s = __riscv_vle8_v_u8m1(src + i, vl);
      vuint8m1_t slo = __riscv_vand_vx_u8m1(s, 0x0f, vl);
      vuint8m1_t shi = __riscv_vsrl_vx_u8m1(s, 4, vl);
      vuint8m1_t p0 =
          __riscv_vxor_vv_u8m1(__riscv_vrgather_vv_u8m1(lo0, slo, vl),
                               __riscv_vrgather_vv_u8m1(hi0, shi, vl), vl);
      vuint8m1_t p1 =
          __riscv_vxor_vv_u8m1(__riscv_vrgather_vv_u8m1(lo1, slo, vl),
                               __riscv_vrgather_vv_u8m1(hi1, shi, vl), vl);
      vuint8m1_t p2 =
          __riscv_vxor_vv_u8m1(__riscv_vrgather_vv_u8m1(lo2, slo, vl),
                               __riscv_vrgather_vv_u8m1(hi2, shi, vl), vl);
      vuint8m1_t p3 =
          __riscv_vxor_vv_u8m1(__riscv_vrgather_vv_u8m1(lo3, slo, vl),
                               __riscv_vrgather_vv_u8m1(hi3, shi, vl), vl);
      __riscv_vse8_v_u8m1(
          d0 + i, __riscv_vxor_vv_u8m1(__riscv_vle8_v_u8m1(d0 + i, vl), p0, vl),
          vl);
      __riscv_vse8_v_u8m1(
          d1 + i, __riscv_vxor_vv_u8m1(__riscv_vle8_v_u8m1(d1 + i, vl), p1, vl),
          vl);
      __riscv_vse8_v_u8m1(
          d2 + i, __riscv_vxor_vv_u8m1(__riscv_vle8_v_u8m1(d2 + i, vl), p2, vl),
          vl);
      __riscv_vse8_v_u8m1(
          d3 + i, __riscv_vxor_vv_u8m1(__riscv_vle8_v_u8m1(d3 + i, vl), p3, vl),
          vl);
      i += (uint32_t)vl;
    }
  }
  for (; i < len; i++) {
    uint8_t b = src[i];
    if (b) {
      d0[i] ^= obl_fused_mul(u0, b);
      d1[i] ^= obl_fused_mul(u1, b);
      d2[i] ^= obl_fused_mul(u2, b);
      d3[i] ^= obl_fused_mul(u3, b);
    }
  }
}

OBL_FUSED_INLINE void obl_axpy2_rvv(u8 *d0, u8 *d1, const u8 *src, u8 u0, u8 u1,
                                    unsigned len) {
  if (u0 == 1 && u1 == 1)
    obl_fused_rvv_xor2(d0, d1, src, len);
  else
    obl_fused_rvv_gf2(d0, d1, u0, u1, src, len);
}

OBL_FUSED_INLINE void obl_axpy4_rvv(u8 *d0, u8 *d1, u8 *d2, u8 *d3,
                                    const u8 *src, u8 u0, u8 u1, u8 u2, u8 u3,
                                    unsigned len) {
  if (u0 == 1 && u1 == 1 && u2 == 1 && u3 == 1)
    obl_fused_rvv_xor4(d0, d1, d2, d3, src, len);
  else
    obl_fused_rvv_gf4(d0, d1, d2, d3, u0, u1, u2, u3, src, len);
}
static void obl_axpyb32_rvv(u8 *a, u32 *b, u8 u, unsigned k) {
  if (u == 0 || k == 0)
    return;
  const uint8_t *bm = (const uint8_t *)b;
  unsigned i = 0;
  unsigned bulk = k & ~7u;
  while (i < bulk) {
    size_t vl = __riscv_vsetvl_e8m2(bulk - i);
    vl &= ~(size_t)7u;
    if (vl == 0)
      break;
    vbool4_t mask = __riscv_vlm_v_b4(bm + (i >> 3), vl);
    vuint8m2_t va = __riscv_vle8_v_u8m2(a + i, vl);
    va = __riscv_vxor_vx_u8m2_mu(mask, va, va, u, vl);
    __riscv_vse8_v_u8m2(a + i, va, vl);
    i += (unsigned)vl;
  }
  for (; i < k; i++) {
    if ((bm[i >> 3] >> (i & 7u)) & 1u)
      a[i] ^= u;
  }
}
#endif

/* Older x86 targets retain their existing single-row SIMD implementations. */
#if defined(OBLAS_ARCH_X86)
#define OBL_FUSED_FALLBACK(suffix)                                             \
  static void obl_axpy2_##suffix(u8 *d0, u8 *d1, const u8 *src, u8 u0, u8 u1,  \
                                 unsigned len) {                               \
    obl_axpy_##suffix(d0, (u8 *)src, u0, len);                                 \
    obl_axpy_##suffix(d1, (u8 *)src, u1, len);                                 \
  }                                                                            \
  static void obl_axpy4_##suffix(u8 *d0, u8 *d1, u8 *d2, u8 *d3,               \
                                 const u8 *src, u8 u0, u8 u1, u8 u2, u8 u3,    \
                                 unsigned len) {                               \
    obl_axpy2_##suffix(d0, d1, src, u0, u1, len);                              \
    obl_axpy2_##suffix(d2, d3, src, u2, u3, len);                              \
  }
OBL_FUSED_FALLBACK(ssse3_std)
OBL_FUSED_FALLBACK(ssse3_gfni)
#define obl_xor_set4_ssse3 obl_xor_set4_ref
#define obl_xor_acc4_ssse3 obl_xor_acc4_ref
#if defined(_MSC_VER)
OBL_FUSED_FALLBACK(avx2_std)
OBL_FUSED_FALLBACK(avx2_gfni)
OBL_FUSED_FALLBACK(avx512_std)
OBL_FUSED_FALLBACK(avx512_gfni)
#define obl_batch_axpy2_avx2_std obl_axpy2_avx2_std
#define obl_batch_axpy4_avx2_std obl_axpy4_avx2_std
#define obl_batch_axpy2_avx2_gfni obl_axpy2_avx2_gfni
#define obl_batch_axpy4_avx2_gfni obl_axpy4_avx2_gfni
#define obl_batch_axpy2_avx512_std obl_axpy2_avx512_std
#define obl_batch_axpy4_avx512_std obl_axpy4_avx512_std
#define obl_batch_axpy2_avx512_gfni obl_axpy2_avx512_gfni
#define obl_batch_axpy4_avx512_gfni obl_axpy4_avx512_gfni
#define obl_xor_set4_avx2 obl_xor_set4_ref
#define obl_xor_acc4_avx2 obl_xor_acc4_ref
#define obl_xor_set4_avx512 obl_xor_set4_ref
#define obl_xor_acc4_avx512 obl_xor_acc4_ref
#endif
#undef OBL_FUSED_FALLBACK
#endif

#undef OBL_FUSED_INLINE
#endif /* OBLAS_FUSED_H */
