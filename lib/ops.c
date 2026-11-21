#include "nanorq_ops.h"

#include <stdatomic.h>
#include <string.h>

#if defined(OBLAS_ARCH_X86) && !defined(_MSC_VER)
#include "obl/gf2_8_affine_mat.h"
#include <immintrin.h>
#include <tmmintrin.h>
#define NANORQ_OPS_X86 1
#elif defined(OBLAS_ARCH_ARM) && (defined(__ARM_NEON) || defined(_MSC_VER))
#include <arm_neon.h>
#define NANORQ_OPS_NEON 1
#endif

enum {
  OPS_SIMD_GENERIC = 0,
  OPS_SIMD_AVX2_STD = 1,
  OPS_SIMD_AVX2_GFNI = 2,
  OPS_SIMD_AVX512_STD = 3,
  OPS_SIMD_AVX512_GFNI = 4,
  OPS_SIMD_NEON = 5
};

static atomic_int ops_simd_state = ATOMIC_VAR_INIT(-1);

static int get_ops_simd(void) {
  int mode = atomic_load_explicit(&ops_simd_state, memory_order_relaxed);
  if (mode >= 0)
    return mode;
  nanorq_core_init();
  mode = OPS_SIMD_GENERIC;
#if defined(NANORQ_OPS_X86)
  __builtin_cpu_init();
  int has_avx512 =
      __builtin_cpu_supports("avx2") && __builtin_cpu_supports("avx512f") &&
      __builtin_cpu_supports("avx512bw") &&
      __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vl");
  if (has_avx512 && __builtin_cpu_supports("gfni")) {
    mode = OPS_SIMD_AVX512_GFNI;
  } else if (has_avx512) {
    mode = OPS_SIMD_AVX512_STD;
  } else if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("gfni")) {
    mode = OPS_SIMD_AVX2_GFNI;
  } else if (__builtin_cpu_supports("avx2")) {
    mode = OPS_SIMD_AVX2_STD;
  }
#elif defined(NANORQ_OPS_NEON)
  mode = OPS_SIMD_NEON;
#endif
  atomic_store_explicit(&ops_simd_state, mode, memory_order_relaxed);
  return mode;
}

static inline void prefetch_row_w(const uint8_t *p, uint32_t len) {
#if defined(__GNUC__) || defined(__clang__)
  for (uint32_t off = 0; off < len; off += 256)
    __builtin_prefetch(p + off, 1, 1);
#else
  (void)p;
  (void)len;
#endif
}

static inline uint8_t gf_mul_scalar(uint8_t u, uint8_t b) {
  if (!u || !b)
    return 0;
  return OCT_EXP[OCT_LOG[u] + OCT_LOG[b]];
}

#if defined(NANORQ_OPS_X86)

#define TARGET_AVX512_GFNI                                                     \
  __attribute__((target("ssse3,avx2,avx512f,avx512bw,avx512dq,avx512vl,"       \
                        "gfni")))
#define TARGET_AVX512                                                          \
  __attribute__((target("ssse3,avx2,avx512f,avx512bw,avx512dq,avx512vl")))
#define TARGET_AVX2_GFNI __attribute__((target("ssse3,avx2,gfni")))
#define TARGET_AVX2 __attribute__((target("ssse3,avx2")))

TARGET_AVX512 static inline void xor512_1(uint8_t *d0, const uint8_t *src,
                                          uint32_t len) {
  uint32_t i = 0;
  for (; i + 64 <= len; i += 64) {
    __m512i s = _mm512_loadu_si512(src + i);
    _mm512_storeu_si512(d0 + i,
                        _mm512_xor_si512(_mm512_loadu_si512(d0 + i), s));
  }
  if (i + 32 <= len) {
    __m256i s = _mm256_loadu_si256((const __m256i *)(src + i));
    _mm256_storeu_si256(
        (__m256i *)(d0 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d0 + i)), s));
    i += 32;
  }
  if (i + 16 <= len) {
    __m128i s = _mm_loadu_si128((const __m128i *)(src + i));
    _mm_storeu_si128(
        (__m128i *)(d0 + i),
        _mm_xor_si128(_mm_loadu_si128((const __m128i *)(d0 + i)), s));
    i += 16;
  }
  for (; i < len; i++)
    d0[i] ^= src[i];
}

TARGET_AVX512 static inline void
xor512_fanout2(uint8_t *d0, uint8_t *d1, const uint8_t *src, uint32_t len) {
  uint32_t i = 0;
  for (; i + 64 <= len; i += 64) {
    __m512i s = _mm512_loadu_si512(src + i);
    _mm512_storeu_si512(d0 + i,
                        _mm512_xor_si512(_mm512_loadu_si512(d0 + i), s));
    _mm512_storeu_si512(d1 + i,
                        _mm512_xor_si512(_mm512_loadu_si512(d1 + i), s));
  }
  if (i + 32 <= len) {
    __m256i s = _mm256_loadu_si256((const __m256i *)(src + i));
    _mm256_storeu_si256(
        (__m256i *)(d0 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d0 + i)), s));
    _mm256_storeu_si256(
        (__m256i *)(d1 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d1 + i)), s));
    i += 32;
  }
  if (i + 16 <= len) {
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

TARGET_AVX512 static inline void xor512_fanout4(uint8_t *d0, uint8_t *d1,
                                                uint8_t *d2, uint8_t *d3,
                                                const uint8_t *src,
                                                uint32_t len) {
  uint32_t i = 0;
  for (; i + 64 <= len; i += 64) {
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
  if (i + 32 <= len) {
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
  if (i + 16 <= len) {
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

TARGET_AVX512 static inline void xor512_set4(uint8_t *dst, const uint8_t *r0,
                                             const uint8_t *r1,
                                             const uint8_t *r2,
                                             const uint8_t *r3, uint32_t len) {
  uint32_t i = 0;
  for (; i + 64 <= len; i += 64) {
    __m512i v01 = _mm512_xor_si512(_mm512_loadu_si512(r0 + i),
                                   _mm512_loadu_si512(r1 + i));
    __m512i v23 = _mm512_xor_si512(_mm512_loadu_si512(r2 + i),
                                   _mm512_loadu_si512(r3 + i));
    _mm512_storeu_si512(dst + i, _mm512_xor_si512(v01, v23));
  }
  if (i + 32 <= len) {
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

TARGET_AVX512 static inline void xor512_acc4(uint8_t *dst, const uint8_t *r0,
                                             const uint8_t *r1,
                                             const uint8_t *r2,
                                             const uint8_t *r3, uint32_t len) {
  uint32_t i = 0;
  for (; i + 64 <= len; i += 64) {
    __m512i v01 = _mm512_xor_si512(_mm512_loadu_si512(r0 + i),
                                   _mm512_loadu_si512(r1 + i));
    __m512i v23 = _mm512_xor_si512(_mm512_loadu_si512(r2 + i),
                                   _mm512_loadu_si512(r3 + i));
    __m512i acc = _mm512_xor_si512(_mm512_loadu_si512(dst + i), v01);
    _mm512_storeu_si512(dst + i, _mm512_xor_si512(acc, v23));
  }
  if (i + 32 <= len) {
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

TARGET_AVX512_GFNI static inline void
gfni512_1(uint8_t *d0, uint8_t u0, const uint8_t *src, uint32_t len) {
  const __m512i m0 = _mm512_set1_epi64(GF2_8_AFFINE_MAT[u0]);
  uint32_t i = 0;
  for (; i + 64 <= len; i += 64) {
    __m512i s = _mm512_loadu_si512(src + i);
    __m512i p0 = _mm512_gf2p8affine_epi64_epi8(s, m0, 0);
    _mm512_storeu_si512(d0 + i,
                        _mm512_xor_si512(_mm512_loadu_si512(d0 + i), p0));
  }
  if (i + 32 <= len) {
    __m256i s = _mm256_loadu_si256((const __m256i *)(src + i));
    __m256i p0 =
        _mm256_gf2p8affine_epi64_epi8(s, _mm512_castsi512_si256(m0), 0);
    _mm256_storeu_si256(
        (__m256i *)(d0 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d0 + i)), p0));
    i += 32;
  }
  if (i + 16 <= len) {
    __m128i s = _mm_loadu_si128((const __m128i *)(src + i));
    __m128i p0 = _mm_gf2p8affine_epi64_epi8(s, _mm512_castsi512_si128(m0), 0);
    _mm_storeu_si128(
        (__m128i *)(d0 + i),
        _mm_xor_si128(_mm_loadu_si128((const __m128i *)(d0 + i)), p0));
    i += 16;
  }
  for (; i < len; i++) {
    uint8_t b = src[i];
    if (b)
      d0[i] ^= gf_mul_scalar(u0, b);
  }
}

TARGET_AVX512_GFNI static inline void gfni512_fanout2(uint8_t *d0, uint8_t *d1,
                                                      uint8_t u0, uint8_t u1,
                                                      const uint8_t *src,
                                                      uint32_t len) {
  const __m512i m0 = _mm512_set1_epi64(GF2_8_AFFINE_MAT[u0]);
  const __m512i m1 = _mm512_set1_epi64(GF2_8_AFFINE_MAT[u1]);
  uint32_t i = 0;
  for (; i + 64 <= len; i += 64) {
    __m512i s = _mm512_loadu_si512(src + i);
    __m512i p0 = _mm512_gf2p8affine_epi64_epi8(s, m0, 0);
    __m512i p1 = _mm512_gf2p8affine_epi64_epi8(s, m1, 0);
    _mm512_storeu_si512(d0 + i,
                        _mm512_xor_si512(_mm512_loadu_si512(d0 + i), p0));
    _mm512_storeu_si512(d1 + i,
                        _mm512_xor_si512(_mm512_loadu_si512(d1 + i), p1));
  }
  if (i + 32 <= len) {
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
  if (i + 16 <= len) {
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
      d0[i] ^= gf_mul_scalar(u0, b);
      d1[i] ^= gf_mul_scalar(u1, b);
    }
  }
}

TARGET_AVX512_GFNI static inline void
gfni512_fanout4(uint8_t *d0, uint8_t *d1, uint8_t *d2, uint8_t *d3, uint8_t u0,
                uint8_t u1, uint8_t u2, uint8_t u3, const uint8_t *src,
                uint32_t len) {
  const __m512i m0 = _mm512_set1_epi64(GF2_8_AFFINE_MAT[u0]);
  const __m512i m1 = _mm512_set1_epi64(GF2_8_AFFINE_MAT[u1]);
  const __m512i m2 = _mm512_set1_epi64(GF2_8_AFFINE_MAT[u2]);
  const __m512i m3 = _mm512_set1_epi64(GF2_8_AFFINE_MAT[u3]);
  uint32_t i = 0;
  for (; i + 64 <= len; i += 64) {
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
  if (i + 32 <= len) {
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
  if (i + 16 <= len) {
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
      d0[i] ^= gf_mul_scalar(u0, b);
      d1[i] ^= gf_mul_scalar(u1, b);
      d2[i] ^= gf_mul_scalar(u2, b);
      d3[i] ^= gf_mul_scalar(u3, b);
    }
  }
}

TARGET_AVX512 static inline void shuf512_1(uint8_t *d0, uint8_t u0,
                                           const uint8_t *src, uint32_t len) {
  const __m512i mask = _mm512_set1_epi8(0x0f);
  const __m512i lo0 = _mm512_broadcast_i32x4(
      _mm_loadu_si128((const __m128i *)(GF2_8_SHUF_LO + u0 * 16)));
  const __m512i hi0 = _mm512_broadcast_i32x4(
      _mm_loadu_si128((const __m128i *)(GF2_8_SHUF_HI + u0 * 16)));
  uint32_t i = 0;
  for (; i + 64 <= len; i += 64) {
    __m512i s = _mm512_loadu_si512(src + i);
    __m512i slo = _mm512_and_si512(s, mask);
    __m512i shi = _mm512_and_si512(_mm512_srli_epi64(s, 4), mask);
    __m512i p0 = _mm512_xor_si512(_mm512_shuffle_epi8(lo0, slo),
                                  _mm512_shuffle_epi8(hi0, shi));
    _mm512_storeu_si512(d0 + i,
                        _mm512_xor_si512(_mm512_loadu_si512(d0 + i), p0));
  }
  if (i + 32 <= len) {
    __m256i m256 = _mm512_castsi512_si256(mask);
    __m256i s = _mm256_loadu_si256((const __m256i *)(src + i));
    __m256i slo = _mm256_and_si256(s, m256);
    __m256i shi = _mm256_and_si256(_mm256_srli_epi64(s, 4), m256);
    __m256i p0 =
        _mm256_xor_si256(_mm256_shuffle_epi8(_mm512_castsi512_si256(lo0), slo),
                         _mm256_shuffle_epi8(_mm512_castsi512_si256(hi0), shi));
    _mm256_storeu_si256(
        (__m256i *)(d0 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d0 + i)), p0));
    i += 32;
  }
  for (; i < len; i++) {
    uint8_t b = src[i];
    if (b)
      d0[i] ^= gf_mul_scalar(u0, b);
  }
}

TARGET_AVX512 static inline void shuf512_fanout2(uint8_t *d0, uint8_t *d1,
                                                 uint8_t u0, uint8_t u1,
                                                 const uint8_t *src,
                                                 uint32_t len) {
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
  for (; i + 64 <= len; i += 64) {
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
  if (i + 32 <= len) {
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
      d0[i] ^= gf_mul_scalar(u0, b);
      d1[i] ^= gf_mul_scalar(u1, b);
    }
  }
}

TARGET_AVX512 static inline void
shuf512_fanout4(uint8_t *d0, uint8_t *d1, uint8_t *d2, uint8_t *d3, uint8_t u0,
                uint8_t u1, uint8_t u2, uint8_t u3, const uint8_t *src,
                uint32_t len) {
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
  for (; i + 64 <= len; i += 64) {
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
  if (i + 32 <= len) {
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
      d0[i] ^= gf_mul_scalar(u0, b);
      d1[i] ^= gf_mul_scalar(u1, b);
      d2[i] ^= gf_mul_scalar(u2, b);
      d3[i] ^= gf_mul_scalar(u3, b);
    }
  }
}

TARGET_AVX2 static inline void xor256_1(uint8_t *d0, const uint8_t *src,
                                        uint32_t len) {
  uint32_t i = 0;
  for (; i + 32 <= len; i += 32) {
    __m256i s = _mm256_loadu_si256((const __m256i *)(src + i));
    _mm256_storeu_si256(
        (__m256i *)(d0 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d0 + i)), s));
  }
  if (i + 16 <= len) {
    __m128i s = _mm_loadu_si128((const __m128i *)(src + i));
    _mm_storeu_si128(
        (__m128i *)(d0 + i),
        _mm_xor_si128(_mm_loadu_si128((const __m128i *)(d0 + i)), s));
    i += 16;
  }
  for (; i < len; i++)
    d0[i] ^= src[i];
}

TARGET_AVX2 static inline void
xor256_fanout2(uint8_t *d0, uint8_t *d1, const uint8_t *src, uint32_t len) {
  uint32_t i = 0;
  for (; i + 32 <= len; i += 32) {
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

TARGET_AVX2 static inline void xor256_fanout4(uint8_t *d0, uint8_t *d1,
                                              uint8_t *d2, uint8_t *d3,
                                              const uint8_t *src,
                                              uint32_t len) {
  uint32_t i = 0;
  for (; i + 32 <= len; i += 32) {
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

TARGET_AVX2 static inline void xor256_set4(uint8_t *dst, const uint8_t *r0,
                                           const uint8_t *r1, const uint8_t *r2,
                                           const uint8_t *r3, uint32_t len) {
  uint32_t i = 0;
  for (; i + 32 <= len; i += 32) {
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

TARGET_AVX2 static inline void xor256_acc4(uint8_t *dst, const uint8_t *r0,
                                           const uint8_t *r1, const uint8_t *r2,
                                           const uint8_t *r3, uint32_t len) {
  uint32_t i = 0;
  for (; i + 32 <= len; i += 32) {
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

TARGET_AVX2_GFNI static inline void
gfni256_1(uint8_t *d0, uint8_t u0, const uint8_t *src, uint32_t len) {
  const __m256i m0 = _mm256_set1_epi64x(GF2_8_AFFINE_MAT[u0]);
  uint32_t i = 0;
  for (; i + 32 <= len; i += 32) {
    __m256i s = _mm256_loadu_si256((const __m256i *)(src + i));
    __m256i p0 = _mm256_gf2p8affine_epi64_epi8(s, m0, 0);
    _mm256_storeu_si256(
        (__m256i *)(d0 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d0 + i)), p0));
  }
  if (i + 16 <= len) {
    __m128i s = _mm_loadu_si128((const __m128i *)(src + i));
    __m128i p0 = _mm_gf2p8affine_epi64_epi8(s, _mm256_castsi256_si128(m0), 0);
    _mm_storeu_si128(
        (__m128i *)(d0 + i),
        _mm_xor_si128(_mm_loadu_si128((const __m128i *)(d0 + i)), p0));
    i += 16;
  }
  for (; i < len; i++) {
    uint8_t b = src[i];
    if (b)
      d0[i] ^= gf_mul_scalar(u0, b);
  }
}

TARGET_AVX2_GFNI static inline void gfni256_fanout2(uint8_t *d0, uint8_t *d1,
                                                    uint8_t u0, uint8_t u1,
                                                    const uint8_t *src,
                                                    uint32_t len) {
  const __m256i m0 = _mm256_set1_epi64x(GF2_8_AFFINE_MAT[u0]);
  const __m256i m1 = _mm256_set1_epi64x(GF2_8_AFFINE_MAT[u1]);
  uint32_t i = 0;
  for (; i + 32 <= len; i += 32) {
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
      d0[i] ^= gf_mul_scalar(u0, b);
      d1[i] ^= gf_mul_scalar(u1, b);
    }
  }
}

TARGET_AVX2_GFNI static inline void
gfni256_fanout4(uint8_t *d0, uint8_t *d1, uint8_t *d2, uint8_t *d3, uint8_t u0,
                uint8_t u1, uint8_t u2, uint8_t u3, const uint8_t *src,
                uint32_t len) {
  const __m256i m0 = _mm256_set1_epi64x(GF2_8_AFFINE_MAT[u0]);
  const __m256i m1 = _mm256_set1_epi64x(GF2_8_AFFINE_MAT[u1]);
  const __m256i m2 = _mm256_set1_epi64x(GF2_8_AFFINE_MAT[u2]);
  const __m256i m3 = _mm256_set1_epi64x(GF2_8_AFFINE_MAT[u3]);
  uint32_t i = 0;
  for (; i + 32 <= len; i += 32) {
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
      d0[i] ^= gf_mul_scalar(u0, b);
      d1[i] ^= gf_mul_scalar(u1, b);
      d2[i] ^= gf_mul_scalar(u2, b);
      d3[i] ^= gf_mul_scalar(u3, b);
    }
  }
}

TARGET_AVX2 static inline void shuf256_1(uint8_t *d0, uint8_t u0,
                                         const uint8_t *src, uint32_t len) {
  const __m256i mask = _mm256_set1_epi8(0x0f);
  const __m128i l0 =
      _mm_loadu_si128((const __m128i *)(GF2_8_SHUF_LO + u0 * 16));
  const __m128i h0 =
      _mm_loadu_si128((const __m128i *)(GF2_8_SHUF_HI + u0 * 16));
  const __m256i lo0 =
      _mm256_inserti128_si256(_mm256_castsi128_si256(l0), l0, 1);
  const __m256i hi0 =
      _mm256_inserti128_si256(_mm256_castsi128_si256(h0), h0, 1);
  uint32_t i = 0;
  for (; i + 32 <= len; i += 32) {
    __m256i s = _mm256_loadu_si256((const __m256i *)(src + i));
    __m256i slo = _mm256_and_si256(s, mask);
    __m256i shi = _mm256_and_si256(_mm256_srli_epi64(s, 4), mask);
    __m256i p0 = _mm256_xor_si256(_mm256_shuffle_epi8(lo0, slo),
                                  _mm256_shuffle_epi8(hi0, shi));
    _mm256_storeu_si256(
        (__m256i *)(d0 + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(d0 + i)), p0));
  }
  for (; i < len; i++) {
    uint8_t b = src[i];
    if (b)
      d0[i] ^= gf_mul_scalar(u0, b);
  }
}

TARGET_AVX2 static inline void shuf256_fanout2(uint8_t *d0, uint8_t *d1,
                                               uint8_t u0, uint8_t u1,
                                               const uint8_t *src,
                                               uint32_t len) {
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
  for (; i + 32 <= len; i += 32) {
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
      d0[i] ^= gf_mul_scalar(u0, b);
      d1[i] ^= gf_mul_scalar(u1, b);
    }
  }
}

TARGET_AVX2 static inline void
shuf256_fanout4(uint8_t *d0, uint8_t *d1, uint8_t *d2, uint8_t *d3, uint8_t u0,
                uint8_t u1, uint8_t u2, uint8_t u3, const uint8_t *src,
                uint32_t len) {
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
  for (; i + 32 <= len; i += 32) {
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
      d0[i] ^= gf_mul_scalar(u0, b);
      d1[i] ^= gf_mul_scalar(u1, b);
      d2[i] ^= gf_mul_scalar(u2, b);
      d3[i] ^= gf_mul_scalar(u3, b);
    }
  }
}

#define DEFINE_X86_RUNNERS(NAME, ATTR, XOR1, XOR2, XOR4, GF1, GF2, GF4)        \
  ATTR static void run_fwd_##NAME(uint8_t *D, uint32_t stride,                 \
                                  const sched_op *ops, u32 beg, u32 end) {     \
    void (*scal)(uint8_t *, uint8_t, uint32_t) = nanorq_oblas.scal;            \
    u32 idx = beg;                                                             \
    while (idx < end) {                                                        \
      if (idx + 8 < end)                                                       \
        prefetch_row_w(D + (size_t)ops[idx + 8].i * stride, stride);           \
      u32 j = ops[idx].j;                                                      \
      u8 u0 = ops[idx].u;                                                      \
      if (u0 == 1 && idx + 3 < end && ops[idx + 1].u == 1 &&                   \
          ops[idx + 1].j == j && ops[idx + 2].u == 1 && ops[idx + 2].j == j && \
          ops[idx + 3].u == 1 && ops[idx + 3].j == j) {                        \
        XOR4(D + (size_t)ops[idx].i * stride,                                  \
             D + (size_t)ops[idx + 1].i * stride,                              \
             D + (size_t)ops[idx + 2].i * stride,                              \
             D + (size_t)ops[idx + 3].i * stride, D + (size_t)j * stride,      \
             stride);                                                          \
        idx += 4;                                                              \
      } else if (u0 > 0 && idx + 3 < end && ops[idx + 1].u > 0 &&              \
                 ops[idx + 1].j == j && ops[idx + 2].u > 0 &&                  \
                 ops[idx + 2].j == j && ops[idx + 3].u > 0 &&                  \
                 ops[idx + 3].j == j) {                                        \
        GF4(D + (size_t)ops[idx].i * stride,                                   \
            D + (size_t)ops[idx + 1].i * stride,                               \
            D + (size_t)ops[idx + 2].i * stride,                               \
            D + (size_t)ops[idx + 3].i * stride, u0, ops[idx + 1].u,           \
            ops[idx + 2].u, ops[idx + 3].u, D + (size_t)j * stride, stride);   \
        idx += 4;                                                              \
      } else if (u0 == 1 && idx + 1 < end && ops[idx + 1].u == 1 &&            \
                 ops[idx + 1].j == j) {                                        \
        XOR2(D + (size_t)ops[idx].i * stride,                                  \
             D + (size_t)ops[idx + 1].i * stride, D + (size_t)j * stride,      \
             stride);                                                          \
        idx += 2;                                                              \
      } else if (u0 > 0 && idx + 1 < end && ops[idx + 1].u > 0 &&              \
                 ops[idx + 1].j == j) {                                        \
        GF2(D + (size_t)ops[idx].i * stride,                                   \
            D + (size_t)ops[idx + 1].i * stride, u0, ops[idx + 1].u,           \
            D + (size_t)j * stride, stride);                                   \
        idx += 2;                                                              \
      } else if (u0 == 1) {                                                    \
        XOR1(D + (size_t)ops[idx].i * stride, D + (size_t)j * stride, stride); \
        idx++;                                                                 \
      } else if (u0 > 1) {                                                     \
        GF1(D + (size_t)ops[idx].i * stride, u0, D + (size_t)j * stride,       \
            stride);                                                           \
        idx++;                                                                 \
      } else {                                                                 \
        scal(D + (size_t)ops[idx].i * stride, j, stride);                      \
        idx++;                                                                 \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  ATTR static void run_rev_##NAME(uint8_t *D, uint32_t stride,                 \
                                  const sched_op *ops, u32 count) {            \
    u32 idx = count;                                                           \
    while (idx > 0) {                                                          \
      if (idx > 8)                                                             \
        prefetch_row_w(D + (size_t)ops[idx - 8].i * stride, stride);           \
      u32 j = ops[idx - 1].j;                                                  \
      if (idx >= 4 && ops[idx - 2].j == j && ops[idx - 3].j == j &&            \
          ops[idx - 4].j == j) {                                               \
        XOR4(D + (size_t)ops[idx - 1].i * stride,                              \
             D + (size_t)ops[idx - 2].i * stride,                              \
             D + (size_t)ops[idx - 3].i * stride,                              \
             D + (size_t)ops[idx - 4].i * stride, D + (size_t)j * stride,      \
             stride);                                                          \
        idx -= 4;                                                              \
      } else if (idx >= 2 && ops[idx - 2].j == j) {                            \
        XOR2(D + (size_t)ops[idx - 1].i * stride,                              \
             D + (size_t)ops[idx - 2].i * stride, D + (size_t)j * stride,      \
             stride);                                                          \
        idx -= 2;                                                              \
      } else {                                                                 \
        XOR1(D + (size_t)ops[idx - 1].i * stride, D + (size_t)j * stride,      \
             stride);                                                          \
        idx--;                                                                 \
      }                                                                        \
    }                                                                          \
  }

DEFINE_X86_RUNNERS(avx512_gfni, TARGET_AVX512_GFNI, xor512_1, xor512_fanout2,
                   xor512_fanout4, gfni512_1, gfni512_fanout2, gfni512_fanout4)
DEFINE_X86_RUNNERS(avx512_std, TARGET_AVX512, xor512_1, xor512_fanout2,
                   xor512_fanout4, shuf512_1, shuf512_fanout2, shuf512_fanout4)
DEFINE_X86_RUNNERS(avx2_gfni, TARGET_AVX2_GFNI, xor256_1, xor256_fanout2,
                   xor256_fanout4, gfni256_1, gfni256_fanout2, gfni256_fanout4)
DEFINE_X86_RUNNERS(avx2_std, TARGET_AVX2, xor256_1, xor256_fanout2,
                   xor256_fanout4, shuf256_1, shuf256_fanout2, shuf256_fanout4)

#endif /* NANORQ_OPS_X86 */

#if defined(NANORQ_OPS_NEON)
static inline void neon_xor4(uint8_t *d0, uint8_t *d1, uint8_t *d2, uint8_t *d3,
                             const uint8_t *src, uint32_t len) {
  uint32_t i = 0;
  for (; i + 16 <= len; i += 16) {
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

static void run_fwd_neon(uint8_t *D, uint32_t stride, const sched_op *ops,
                         u32 beg, u32 end) {
  void (*axpy)(uint8_t *, uint8_t *, uint8_t, uint32_t) = nanorq_oblas.axpy;
  void (*scal)(uint8_t *, uint8_t, uint32_t) = nanorq_oblas.scal;
  u32 idx = beg;
  while (idx < end) {
    if (idx + 8 < end)
      prefetch_row_w(D + (size_t)ops[idx + 8].i * stride, stride);
    u32 j = ops[idx].j;
    if (ops[idx].u == 1 && idx + 3 < end && ops[idx + 1].u == 1 &&
        ops[idx + 1].j == j && ops[idx + 2].u == 1 && ops[idx + 2].j == j &&
        ops[idx + 3].u == 1 && ops[idx + 3].j == j) {
      neon_xor4(
          D + (size_t)ops[idx].i * stride, D + (size_t)ops[idx + 1].i * stride,
          D + (size_t)ops[idx + 2].i * stride,
          D + (size_t)ops[idx + 3].i * stride, D + (size_t)j * stride, stride);
      idx += 4;
    } else {
      sched_op op = ops[idx++];
      if (op.u)
        axpy(D + (size_t)op.i * stride, D + (size_t)op.j * stride, op.u,
             stride);
      else
        scal(D + (size_t)op.i * stride, op.j, stride);
    }
  }
}

static void run_rev_neon(uint8_t *D, uint32_t stride, const sched_op *ops,
                         u32 count) {
  void (*axpy)(uint8_t *, uint8_t *, uint8_t, uint32_t) = nanorq_oblas.axpy;
  u32 idx = count;
  while (idx > 0) {
    if (idx > 8)
      prefetch_row_w(D + (size_t)ops[idx - 8].i * stride, stride);
    u32 j = ops[idx - 1].j;
    if (idx >= 4 && ops[idx - 2].j == j && ops[idx - 3].j == j &&
        ops[idx - 4].j == j) {
      neon_xor4(D + (size_t)ops[idx - 1].i * stride,
                D + (size_t)ops[idx - 2].i * stride,
                D + (size_t)ops[idx - 3].i * stride,
                D + (size_t)ops[idx - 4].i * stride, D + (size_t)j * stride,
                stride);
      idx -= 4;
    } else {
      sched_op op = ops[--idx];
      axpy(D + (size_t)op.i * stride, D + (size_t)op.j * stride, 1, stride);
    }
  }
}
#endif /* NANORQ_OPS_NEON */

static void run_fwd_generic(uint8_t *D, uint32_t stride, const sched_op *ops,
                            u32 beg, u32 end) {
  void (*axpy)(uint8_t *, uint8_t *, uint8_t, uint32_t) = nanorq_oblas.axpy;
  void (*scal)(uint8_t *, uint8_t, uint32_t) = nanorq_oblas.scal;
  for (u32 idx = beg; idx < end; idx++) {
    sched_op op = ops[idx];
    if (op.u)
      axpy(D + (size_t)op.i * stride, D + (size_t)op.j * stride, op.u, stride);
    else
      scal(D + (size_t)op.i * stride, op.j, stride);
  }
}

static void run_rev_generic(uint8_t *D, uint32_t stride, const sched_op *ops,
                            u32 count) {
  void (*axpy)(uint8_t *, uint8_t *, uint8_t, uint32_t) = nanorq_oblas.axpy;
  for (u32 idx = count; idx-- > 0;) {
    sched_op op = ops[idx];
    axpy(D + (size_t)op.i * stride, D + (size_t)op.j * stride, 1, stride);
  }
}

static void dispatch_fwd(int simd, uint8_t *D, uint32_t stride,
                         const sched_op *ops, u32 beg, u32 end) {
#if defined(NANORQ_OPS_X86)
  switch (simd) {
  case OPS_SIMD_AVX512_GFNI:
    run_fwd_avx512_gfni(D, stride, ops, beg, end);
    return;
  case OPS_SIMD_AVX512_STD:
    run_fwd_avx512_std(D, stride, ops, beg, end);
    return;
  case OPS_SIMD_AVX2_GFNI:
    run_fwd_avx2_gfni(D, stride, ops, beg, end);
    return;
  case OPS_SIMD_AVX2_STD:
    run_fwd_avx2_std(D, stride, ops, beg, end);
    return;
  default:
    break;
  }
#elif defined(NANORQ_OPS_NEON)
  if (simd == OPS_SIMD_NEON) {
    run_fwd_neon(D, stride, ops, beg, end);
    return;
  }
#else
  (void)simd;
#endif
  run_fwd_generic(D, stride, ops, beg, end);
}

static void dispatch_rev(int simd, uint8_t *D, uint32_t stride,
                         const sched_op *ops, u32 count) {
#if defined(NANORQ_OPS_X86)
  switch (simd) {
  case OPS_SIMD_AVX512_GFNI:
    run_rev_avx512_gfni(D, stride, ops, count);
    return;
  case OPS_SIMD_AVX512_STD:
    run_rev_avx512_std(D, stride, ops, count);
    return;
  case OPS_SIMD_AVX2_GFNI:
    run_rev_avx2_gfni(D, stride, ops, count);
    return;
  case OPS_SIMD_AVX2_STD:
    run_rev_avx2_std(D, stride, ops, count);
    return;
  default:
    break;
  }
#elif defined(NANORQ_OPS_NEON)
  if (simd == OPS_SIMD_NEON) {
    run_rev_neon(D, stride, ops, count);
    return;
  }
#else
  (void)simd;
#endif
  run_rev_generic(D, stride, ops, count);
}

void ops_push(void *arg, u32 i, u32 j, u8 u) {
  schedule *S = (schedule *)arg;
  if (!S || !S->ops.a)
    return;
  sched_op op = {u, i, j};
  if (i == 0 && j == 0 && u == 0) {
    if (S->cpidx < 2) {
      S->cp[S->cpidx++] = S->ops.n;
    } else {
      S->overflowed = true;
    }
  } else {
    if (S->ops.n < S->ops.m) {
      S->ops.a[S->ops.n++] = op;
    } else {
      S->overflowed = true;
    }
  }
}

void ops_run(nanorq_core *rq, uint8_t *D, uint32_t stride, schedule *S) {
  if (!rq || !D || stride == 0 || !S || !S->ops.a || S->cpidx != 2 ||
      S->overflowed)
    return;

  int simd = get_ops_simd();
  u32 rows = nanorq_core_get_pc_rows(rq);

  /* skip phase-1 undo pass when matrix exceeds L3 and workspace holds a copy */
  if (rq->W.work_base && rq->W.d.a && (size_t)rows <= SIZE_MAX / stride) {
    size_t mat_bytes = (size_t)rows * stride;
    if (mat_bytes >= (16U << 20) && rq->W.work_len >= mat_bytes) {
      uint8_t *bkp = rq->W.work_base;
      memcpy(bkp, D, mat_bytes);
      dispatch_fwd(simd, bkp, stride, S->ops.a, 0, S->cp[1]);
      for (u32 pos = rq->W.i; pos < rq->P.L; pos++) {
        u32 drow = uv_A(rq->W.d, pos);
        memcpy(D + (size_t)drow * stride, bkp + (size_t)drow * stride, stride);
      }
      dispatch_fwd(simd, D, stride, S->ops.a, S->cp[1], (u32)S->ops.n);
      dispatch_fwd(simd, D, stride, S->ops.a, 0, S->cp[0]);
      return;
    }
  }

  dispatch_fwd(simd, D, stride, S->ops.a, 0, S->cp[1]);
  if (S->cp[0] > 0)
    dispatch_rev(simd, D, stride, S->ops.a, S->cp[0]);
  dispatch_fwd(simd, D, stride, S->ops.a, S->cp[1], (u32)S->ops.n);
  dispatch_fwd(simd, D, stride, S->ops.a, 0, S->cp[0]);
}

void ops_mix(nanorq_core *rq, uint8_t *D, uint32_t stride, u32 esi, u8 *ptr) {
  if (!rq || !D || stride == 0 || !ptr)
    return;
  uint32_t mix[GENC_MAX];
  uint32_t num = nanorq_core_get_packet_mix(rq, esi, mix, GENC_MAX);

  if (num == 0) {
    memset(ptr, 0, stride);
    return;
  }

  /* resolve virtual row indices directly without physical matrix swaps */
  if (rq->W.d.a && rq->W.ci.a) {
    for (u32 it = 0; it < num; it++)
      mix[it] = uv_A(rq->W.d, uv_A(rq->W.ci, mix[it]));
  }

  bool aliases = false;
  for (u32 it = 0; it < num; it++) {
    if (ptr == D + (size_t)mix[it] * stride) {
      aliases = true;
      break;
    }
  }

#if defined(NANORQ_OPS_X86)
  int simd = get_ops_simd();
  if (!aliases && num >= 4) {
    if (simd == OPS_SIMD_AVX512_GFNI || simd == OPS_SIMD_AVX512_STD) {
      xor512_set4(ptr, D + (size_t)mix[0] * stride, D + (size_t)mix[1] * stride,
                  D + (size_t)mix[2] * stride, D + (size_t)mix[3] * stride,
                  stride);
      u32 it = 4;
      for (; it + 3 < num; it += 4) {
        xor512_acc4(ptr, D + (size_t)mix[it] * stride,
                    D + (size_t)mix[it + 1] * stride,
                    D + (size_t)mix[it + 2] * stride,
                    D + (size_t)mix[it + 3] * stride, stride);
      }
      for (; it < num; it++)
        xor512_1(ptr, D + (size_t)mix[it] * stride, stride);
      return;
    }
    if (simd == OPS_SIMD_AVX2_GFNI || simd == OPS_SIMD_AVX2_STD) {
      xor256_set4(ptr, D + (size_t)mix[0] * stride, D + (size_t)mix[1] * stride,
                  D + (size_t)mix[2] * stride, D + (size_t)mix[3] * stride,
                  stride);
      u32 it = 4;
      for (; it + 3 < num; it += 4) {
        xor256_acc4(ptr, D + (size_t)mix[it] * stride,
                    D + (size_t)mix[it + 1] * stride,
                    D + (size_t)mix[it + 2] * stride,
                    D + (size_t)mix[it + 3] * stride, stride);
      }
      for (; it < num; it++)
        xor256_1(ptr, D + (size_t)mix[it] * stride, stride);
      return;
    }
  }
#endif

  const uint8_t *first = D + (size_t)mix[0] * stride;
  if (ptr != first)
    memcpy(ptr, first, stride);
  for (u32 it = 1; it < num; it++) {
    u32 row = mix[it];
    nanorq_oblas.axpy(ptr, (D + (size_t)row * stride), 1, stride);
  }
}

size_t ops_estimate_schedule_bytes(uint32_t K) {
  /* allocate 40 * k + 1000 to conservatively bound schedule length. */
  if ((size_t)K > (SIZE_MAX - 1000) / 40)
    return 0;
  size_t estimated_ops = 40 * (size_t)K + 1000;
  if (estimated_ops > SIZE_MAX / sizeof(sched_op))
    return 0;
  return estimated_ops * sizeof(sched_op);
}

bool schedule_init(schedule *S, void *buf, size_t buf_bytes) {
  if (!S || !buf || buf_bytes < sizeof(sched_op))
    return false;
  *S = (schedule){0};
  S->ops.a = (sched_op *)buf;
  S->ops.m = buf_bytes / sizeof(sched_op);
  return true;
}

#ifndef NANORQ_NO_LIBC
#include <stdbool.h>
#include <stdlib.h>
#endif

void nanorq_core_get_memory_reqs(uint32_t K, uint32_t overhead, uint32_t stride,
                                 struct nanorq_core_mem_reqs *reqs) {
  if (!reqs)
    return;
  *reqs = (struct nanorq_core_mem_reqs){0};
  nanorq_core rq = {0};
  if (!nanorq_core_encoder_new(K, overhead, &rq) || stride == 0 ||
      overhead > UINT32_MAX - K)
    return;
  reqs->prepare_bytes = nanorq_core_calculate_prepare_memory(&rq);
  reqs->work_bytes = nanorq_core_calculate_work_memory(&rq);
  size_t rows = nanorq_core_get_pc_rows(&rq);
  if (rows > SIZE_MAX / stride) {
    *reqs = (struct nanorq_core_mem_reqs){0};
    return;
  }
  reqs->matrix_bytes = rows * stride;
  reqs->schedule_bytes = ops_estimate_schedule_bytes(K + overhead);
}

#ifndef NANORQ_NO_LIBC

bool nanorq_core_encode_simple(uint8_t *src_data, uint32_t K, uint16_t T,
                               uint32_t num_repair, uint8_t *repair_out) {
  if (!src_data || !repair_out || T == 0 || (size_t)num_repair > SIZE_MAX / T)
    return false;
  nanorq_core rq = {0};
  if (!nanorq_core_encoder_new(K, 0, &rq))
    return false;
  /* Repair ESIs map to internal symbol IDs starting at Kprime. */
  if (num_repair > UINT32_MAX - rq.P.Kprime + 1)
    return false;

  struct nanorq_core_mem_reqs reqs;
  nanorq_core_get_memory_reqs(K, 0, T, &reqs);
  if (!reqs.prepare_bytes || !reqs.work_bytes || !reqs.matrix_bytes ||
      !reqs.schedule_bytes)
    return false;

  uint8_t *prep_mem = (uint8_t *)malloc(reqs.prepare_bytes);
  if (!prep_mem)
    return false;
  if (!nanorq_core_prepare(&rq, prep_mem, reqs.prepare_bytes)) {
    free(prep_mem);
    return false;
  }

  uint8_t *D = (uint8_t *)calloc(1, reqs.matrix_bytes);
  if (!D) {
    free(prep_mem);
    return false;
  }
  uint32_t SH = nanorq_core_get_pc_genc_offset(&rq);

  for (uint32_t i = 0; i < K; i++) {
    memcpy(D + ((size_t)SH + i) * T, src_data + (size_t)i * T, T);
  }

  uint8_t *work_mem = (uint8_t *)malloc(reqs.work_bytes);
  schedule S_enc = {0};
  void *sched_mem = malloc(reqs.schedule_bytes);
  if (!work_mem || !schedule_init(&S_enc, sched_mem, reqs.schedule_bytes)) {
    free(prep_mem);
    free(D);
    free(work_mem);
    free(sched_mem);
    return false;
  }

  nanorq_core_set_op_callback(&rq, &S_enc, ops_push);

  if (!nanorq_core_precalculate(&rq, work_mem, reqs.work_bytes) ||
      S_enc.overflowed || S_enc.cpidx != 2) {
    free(prep_mem);
    free(D);
    free(work_mem);
    free(S_enc.ops.a);
    return false;
  }

  ops_run(&rq, D, T, &S_enc);

  for (uint32_t i = 0; i < num_repair; i++) {
    ops_mix(&rq, D, T, K + i, repair_out + (size_t)i * T);
  }

  free(prep_mem);
  free(D);
  free(work_mem);
  free(S_enc.ops.a);
  return true;
}

#endif /* NANORQ_NO_LIBC */
