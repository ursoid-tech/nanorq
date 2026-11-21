#ifdef __cplusplus
#define register
#define restrict __restrict
#endif

#include "oblas_lite.h"
#ifndef NANORQ_NO_LIBC
#include <stdlib.h>
#endif
#include <string.h>

#if defined(OBLAS_ARCH_X86)
#include <immintrin.h>
#include <tmmintrin.h>
#include "gf2_8_affine_mat.h"
#endif

#if defined(__GNUC__) || defined(__clang__)
#define OBL_INLINE static inline __attribute__((always_inline))
#else
#define OBL_INLINE static inline
#endif

static uint8_t GF2_8_MUL[65536];
static int gf2_8_mul_initialized = 0;

static void oblas_lite_init(void)
{
    if (gf2_8_mul_initialized)
        return;
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            if (i == 0 || j == 0) {
                GF2_8_MUL[(i << 8) + j] = 0;
            } else {
                GF2_8_MUL[(i << 8) + j] = GF2_8_EXP[GF2_8_LOG[i] + GF2_8_LOG[j]];
            }
        }
    }
    gf2_8_mul_initialized = 1;
}

#if defined(OBLAS_TINY)
static inline uint8_t gf2_8_mul(uint16_t a, uint16_t b)
{
    if (!a || !b) {
        return 0;
    }

    /* 8-bit carry-less multiplication in GF(2^8) */
    return GF2_8_EXP[GF2_8_LOG[a] + GF2_8_LOG[b]];
}

#define OBL_MUL_BYTE(u, b) gf2_8_mul((u), (b))

static void obl_axpy_ref(u8 *a, u8 *b, u8 u, unsigned k)
{
    register u8 *ap = a, *ae = &a[k], *bp = b;
    for (; ap != ae; ap++, bp++)
        *ap ^= gf2_8_mul(u, *bp);
}

static void obl_axiy_ref(u8 *a, u8 *b, u8 u, unsigned k)
{
    register u8 *ap = a, *ae = &a[k], *bp = b;
    for (; ap != ae; ap++, bp++)
        *ap = gf2_8_mul(u, *bp);
}

#else
#define OBL_MUL_BYTE(u, b) GF2_8_MUL[((unsigned)(u) << 8) + (b)]

static void obl_axpy_ref(u8 *restrict a, u8 *restrict b, u8 u, unsigned k)
{
    register const u8 *u_row = &GF2_8_MUL[u << 8];
    register u8 *restrict ap = a;
    register u8 *ae = &a[k];
    register const u8 *restrict bp = b;
    for (; ap < ae; ap++, bp++)
        *ap ^= u_row[*bp];
}

static void obl_axiy_ref(u8 *a, u8 *b, u8 u, unsigned k)
{
    register const u8 *u_row = &GF2_8_MUL[u << 8];
    register u8 *ap = a, *ae = &a[k], *bp = b;
    for (; ap != ae; ap++, bp++)
        *ap = u_row[*bp];
}
#endif

static void obl_scal_ref(u8 *a, u8 u, unsigned k)
{
    obl_axiy_ref(a, a, u, k);
}

OBL_INLINE void obl_axpy2_ref(u8 *d0, u8 *d1, const u8 *src, u8 u0, u8 u1, unsigned k)
{
    for (unsigned i = 0; i < k; i++) {
        u8 s = src[i];
        d0[i] ^= OBL_MUL_BYTE(u0, s);
        d1[i] ^= OBL_MUL_BYTE(u1, s);
    }
}

OBL_INLINE void obl_axpy4_ref(u8 *d0, u8 *d1, u8 *d2, u8 *d3, const u8 *src,
                              u8 u0, u8 u1, u8 u2, u8 u3, unsigned k)
{
    for (unsigned i = 0; i < k; i++) {
        u8 s = src[i];
        d0[i] ^= OBL_MUL_BYTE(u0, s);
        d1[i] ^= OBL_MUL_BYTE(u1, s);
        d2[i] ^= OBL_MUL_BYTE(u2, s);
        d3[i] ^= OBL_MUL_BYTE(u3, s);
    }
}

static void obl_xor_set4_ref(u8 *dst, const u8 *r0, const u8 *r1,
                             const u8 *r2, const u8 *r3, unsigned k)
{
    for (unsigned i = 0; i < k; i++)
        dst[i] = r0[i] ^ r1[i] ^ r2[i] ^ r3[i];
}

static void obl_xor_acc4_ref(u8 *dst, const u8 *r0, const u8 *r1,
                             const u8 *r2, const u8 *r3, unsigned k)
{
    for (unsigned i = 0; i < k; i++)
        dst[i] ^= r0[i] ^ r1[i] ^ r2[i] ^ r3[i];
}

static void obl_axpyb32_ref(u8 *a, u32 *b, u8 u, unsigned k)
{
    unsigned idx = 0, p = 0;
    unsigned k_fast = k & ~31;
    for (; idx < k_fast; idx += 32, p++) {
        u32 tmp = b[p];
        while (tmp > 0) {
            unsigned tz = __builtin_ctz(tmp);
            tmp = tmp & (tmp - 1);
            a[tz + idx] ^= u;
        }
    }
    if (idx < k) {
        u32 tmp = b[p];
        tmp &= (1U << (k - idx)) - 1;
        while (tmp > 0) {
            unsigned tz = __builtin_ctz(tmp);
            tmp = tmp & (tmp - 1);
            a[tz + idx] ^= u;
        }
    }
}

static inline void obl_prefetch_row_w(const u8 *p, unsigned k)
{
#if defined(__GNUC__) || defined(__clang__)
    unsigned count = k / 256 + (k % 256 != 0);
    for (unsigned n = 0; n < count; n++)
        __builtin_prefetch(p + (size_t)n * 256, 1, 1);
#else
    (void)p;
    (void)k;
#endif
}

#define OBL_ROW_AT(matrix, index, stride) ((matrix) + (size_t)(index) * (stride))

#define OBL_KERNEL(name) obl_##name##_ref
#define OBL_TARGET
#include "oblas_rows.inc"

#define OBL_NOOP(a, b) (b)

#define OBL_SHUF_TEMPLATE(op, a, b, f, T, LOAD, STORE)                               \
    do {                                                                             \
        VEC_MASK();                                                                  \
        VEC_COEF(0, u);                                                              \
        T *ap = (T *)a;                                                              \
        T *ae = (T *)(a + k - (k % (4 * sizeof(T))));                                \
        const T *bp = (const T *)b;                                                  \
        for (; ap < ae; ap += 4, bp += 4) {                                          \
            T bx0 = LOAD(bp + 0), bx1 = LOAD(bp + 1);                                \
            T bx2 = LOAD(bp + 2), bx3 = LOAD(bp + 3);                                \
            VEC_SPLIT(bx0, 0);                                                       \
            VEC_SPLIT(bx1, 1);                                                       \
            VEC_SPLIT(bx2, 2);                                                       \
            VEC_SPLIT(bx3, 3);                                                       \
            T prod0 = VEC_MUL(0, bx0, 0);                                            \
            T prod1 = VEC_MUL(0, bx1, 1);                                            \
            T prod2 = VEC_MUL(0, bx2, 2);                                            \
            T prod3 = VEC_MUL(0, bx3, 3);                                            \
            STORE(ap + 0, f(LOAD(ap + 0), prod0));                                   \
            STORE(ap + 1, f(LOAD(ap + 1), prod1));                                   \
            STORE(ap + 2, f(LOAD(ap + 2), prod2));                                   \
            STORE(ap + 3, f(LOAD(ap + 3), prod3));                                   \
        }                                                                            \
        T *ae2 = (T *)(a + k - (k % sizeof(T)));                                     \
        for (; ap < ae2; ap++, bp++) {                                               \
            T bx = LOAD(bp);                                                         \
            VEC_SPLIT(bx, 0);                                                        \
            T prod = VEC_MUL(0, bx, 0);                                              \
            STORE(ap, f(LOAD(ap), prod));                                            \
        }                                                                            \
        op##_ref((u8 *)ap, (u8 *)bp, u, k % sizeof(T));                              \
    } while (0)

/* operation flags are compile-time constants at each expansion */
#define OBL_SIMPLE_LOOP(swap, T, LOAD, STORE, XOR)                                   \
    do {                                                                             \
        unsigned i = 0, bulk = k - (k % sizeof(T));                                  \
        for (; i < bulk; i += sizeof(T)) {                                           \
            T av = LOAD((const T *)(a + i)), bv = LOAD((const T *)(b + i));          \
            STORE((T *)(a + i), (swap) ? bv : XOR(av, bv));                          \
            if (swap)                                                                \
                STORE((T *)(b + i), av);                                             \
        }                                                                            \
        for (; i < k; i++) {                                                         \
            u8 av = a[i], bv = b[i];                                                 \
            a[i] = (swap) ? bv : (u8)(av ^ bv);                                      \
            if (swap)                                                                \
                b[i] = av;                                                           \
        }                                                                            \
    } while (0)

#define OBL_XOR4_LOOP(acc, T, LOAD, STORE, XOR)                                      \
    do {                                                                             \
        unsigned i = 0, bulk = k - (k % sizeof(T));                                  \
        for (; i < bulk; i += sizeof(T)) {                                           \
            T v01 = XOR(LOAD((const T *)(r0 + i)), LOAD((const T *)(r1 + i)));       \
            T v23 = XOR(LOAD((const T *)(r2 + i)), LOAD((const T *)(r3 + i)));       \
            T v = XOR(v01, v23);                                                     \
            STORE((T *)(dst + i), (acc) ? XOR(LOAD((const T *)(dst + i)), v) : v);   \
        }                                                                            \
        for (; i < k; i++) {                                                         \
            u8 v = r0[i] ^ r1[i] ^ r2[i] ^ r3[i];                                    \
            dst[i] = (acc) ? (u8)(dst[i] ^ v) : v;                                   \
        }                                                                            \
    } while (0)

#define GENERATE_VEC_OPS(simd, attr, T, LOAD, STORE, XOR)                            \
    attr OBL_INLINE T obl_load_##simd(const u8 *src)                                 \
    {                                                                                \
        return LOAD((const T *)src);                                                 \
    }                                                                                \
    attr OBL_INLINE void obl_xor_into_##simd(u8 *dst, T value)                       \
    {                                                                                \
        STORE((T *)dst, XOR(LOAD((const T *)dst), value));                           \
    }                                                                                \
    attr static void obl_swap_##simd(u8 *restrict a, u8 *restrict b, unsigned k)     \
    {                                                                                \
        if (a == b || k == 0)                                                        \
            return;                                                                  \
        OBL_SIMPLE_LOOP(1, T, LOAD, STORE, XOR);                                     \
    }                                                                                \
    attr static void obl_xor_set4_##simd(u8 *dst, const u8 *r0, const u8 *r1,        \
                                         const u8 *r2, const u8 *r3, unsigned k)     \
    {                                                                                \
        OBL_XOR4_LOOP(0, T, LOAD, STORE, XOR);                                       \
    }                                                                                \
    attr static void obl_xor_acc4_##simd(u8 *dst, const u8 *r0, const u8 *r1,        \
                                         const u8 *r2, const u8 *r3, unsigned k)     \
    {                                                                                \
        OBL_XOR4_LOOP(1, T, LOAD, STORE, XOR);                                       \
    }

#define GENERATE_IMPL(suffix, attr, T, LOAD, STORE, XOR)                             \
    attr static inline void obl_axpy_##suffix(u8 *restrict a, u8 *restrict b,        \
                                              u8 u, unsigned k)                      \
    {                                                                                \
        if (u == 1) {                                                                \
            OBL_SIMPLE_LOOP(0, T, LOAD, STORE, XOR);                                 \
        } else {                                                                     \
            OBL_SHUF_TEMPLATE(obl_axpy, a, b, XOR, T, LOAD, STORE);                  \
        }                                                                            \
    }                                                                                \
    attr static void obl_axiy_##suffix(u8 *a, u8 *b, u8 u, unsigned k)               \
    {                                                                                \
        OBL_SHUF_TEMPLATE(obl_axiy, a, b, OBL_NOOP, T, LOAD, STORE);                 \
    }                                                                                \
    attr static void obl_scal_##suffix(u8 *a, u8 u, unsigned k)                      \
    {                                                                                \
        obl_axiy_##suffix(a, a, u, k);                                               \
    }

#if defined(OBLAS_ARCH_X86)

#define OBL_TGT_AVX512      __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl")))
#define OBL_TGT_AVX512_GFNI __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,gfni")))
#define OBL_TGT_AVX2        __attribute__((target("avx2")))
#define OBL_TGT_AVX2_GFNI   __attribute__((target("avx2,gfni")))
#define OBL_TGT_SSSE3       __attribute__((target("ssse3")))
#define OBL_TGT_SSSE3_GFNI  __attribute__((target("ssse3,gfni")))

GENERATE_VEC_OPS(avx512, OBL_TGT_AVX512, __m512i, _mm512_loadu_si512, _mm512_storeu_si512, _mm512_xor_si512)
#define VEC_TYPE __m512i
#define VEC_LOAD obl_load_avx512
#define VEC_XOR_INTO obl_xor_into_avx512

/* avx512 gfni */
#define VEC_MASK()            (void)0
#define VEC_COEF(id, c)       const __m512i m_##id = _mm512_set1_epi64(GF2_8_AFFINE_MAT[c])
#define VEC_SPLIT(bx, id)     (void)0
#define VEC_MUL(cid, bx, sid) _mm512_gf2p8affine_epi64_epi8(bx, m_##cid, 0)
GENERATE_IMPL(avx512_gfni, OBL_TGT_AVX512_GFNI, __m512i, _mm512_loadu_si512, _mm512_storeu_si512, _mm512_xor_si512)
#define OBL_KERNEL(name) obl_##name##_avx512_gfni
#define OBL_TARGET OBL_TGT_AVX512_GFNI
#include "oblas_rows.inc"

/* avx512 shuffle */
#define VEC_MASK()         const __m512i mask = _mm512_set1_epi8(0x0f)
#define VEC_COEF(id, c)    const __m512i lo_##id = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(GF2_8_SHUF_LO + (unsigned)(c) * 16))); \
                           const __m512i hi_##id = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(GF2_8_SHUF_HI + (unsigned)(c) * 16)))
#define VEC_SPLIT(bx, id)  __m512i slo_##id = _mm512_and_si512(bx, mask); \
                           __m512i shi_##id = _mm512_and_si512(_mm512_srli_epi64(bx, 4), mask)
#define VEC_MUL(c, bx, s)  _mm512_xor_si512(_mm512_shuffle_epi8(lo_##c, slo_##s), _mm512_shuffle_epi8(hi_##c, shi_##s))
GENERATE_IMPL(avx512_std, OBL_TGT_AVX512, __m512i, _mm512_loadu_si512, _mm512_storeu_si512, _mm512_xor_si512)
#define OBL_KERNEL(name) obl_##name##_avx512_std
#define OBL_TARGET OBL_TGT_AVX512
#include "oblas_rows.inc"
#undef VEC_TYPE
#undef VEC_LOAD
#undef VEC_XOR_INTO

GENERATE_VEC_OPS(avx2, OBL_TGT_AVX2, __m256i, _mm256_loadu_si256, _mm256_storeu_si256, _mm256_xor_si256)
#define VEC_TYPE __m256i
#define VEC_LOAD obl_load_avx2
#define VEC_XOR_INTO obl_xor_into_avx2

/* avx2 gfni */
#define VEC_MASK()            (void)0
#define VEC_COEF(id, c)       const __m256i m_##id = _mm256_set1_epi64x(GF2_8_AFFINE_MAT[c])
#define VEC_SPLIT(bx, id)     (void)0
#define VEC_MUL(cid, bx, sid) _mm256_gf2p8affine_epi64_epi8(bx, m_##cid, 0)
GENERATE_IMPL(avx2_gfni, OBL_TGT_AVX2_GFNI, __m256i, _mm256_loadu_si256, _mm256_storeu_si256, _mm256_xor_si256)
#define OBL_KERNEL(name) obl_##name##_avx2_gfni
#define OBL_TARGET OBL_TGT_AVX2_GFNI
#include "oblas_rows.inc"

/* avx2 shuffle */
#define VEC_MASK()         const __m256i mask = _mm256_set1_epi8(0x0f)
#define VEC_COEF(id, c)    const __m256i lo_##id = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(GF2_8_SHUF_LO + (unsigned)(c) * 16))); \
                           const __m256i hi_##id = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(GF2_8_SHUF_HI + (unsigned)(c) * 16)))
#define VEC_SPLIT(bx, id)  __m256i slo_##id = _mm256_and_si256(bx, mask); \
                           __m256i shi_##id = _mm256_and_si256(_mm256_srli_epi64(bx, 4), mask)
#define VEC_MUL(c, bx, s)  _mm256_xor_si256(_mm256_shuffle_epi8(lo_##c, slo_##s), _mm256_shuffle_epi8(hi_##c, shi_##s))
GENERATE_IMPL(avx2_std, OBL_TGT_AVX2, __m256i, _mm256_loadu_si256, _mm256_storeu_si256, _mm256_xor_si256)
#define OBL_KERNEL(name) obl_##name##_avx2_std
#define OBL_TARGET OBL_TGT_AVX2
#include "oblas_rows.inc"
#undef VEC_TYPE
#undef VEC_LOAD
#undef VEC_XOR_INTO

GENERATE_VEC_OPS(ssse3, OBL_TGT_SSSE3, __m128i, _mm_loadu_si128, _mm_storeu_si128, _mm_xor_si128)
#define VEC_TYPE __m128i
#define VEC_LOAD obl_load_ssse3
#define VEC_XOR_INTO obl_xor_into_ssse3

/* ssse3 gfni */
#define VEC_MASK()            (void)0
#define VEC_COEF(id, c)       const __m128i m_##id = _mm_set1_epi64x(GF2_8_AFFINE_MAT[c])
#define VEC_SPLIT(bx, id)     (void)0
#define VEC_MUL(cid, bx, sid) _mm_gf2p8affine_epi64_epi8(bx, m_##cid, 0)
GENERATE_IMPL(ssse3_gfni, OBL_TGT_SSSE3_GFNI, __m128i, _mm_loadu_si128, _mm_storeu_si128, _mm_xor_si128)
#define OBL_KERNEL(name) obl_##name##_ssse3_gfni
#define OBL_TARGET OBL_TGT_SSSE3_GFNI
#include "oblas_rows.inc"

/* ssse3 shuffle */
#define VEC_MASK()         const __m128i mask = _mm_set1_epi8(0x0f)
#define VEC_COEF(id, c)    const __m128i lo_##id = _mm_loadu_si128((const __m128i *)(GF2_8_SHUF_LO + (unsigned)(c) * 16)); \
                           const __m128i hi_##id = _mm_loadu_si128((const __m128i *)(GF2_8_SHUF_HI + (unsigned)(c) * 16))
#define VEC_SPLIT(bx, id)  __m128i slo_##id = _mm_and_si128(bx, mask); \
                           __m128i shi_##id = _mm_and_si128(_mm_srli_epi64(bx, 4), mask)
#define VEC_MUL(c, bx, s)  _mm_xor_si128(_mm_shuffle_epi8(lo_##c, slo_##s), _mm_shuffle_epi8(hi_##c, shi_##s))
GENERATE_IMPL(ssse3_std, OBL_TGT_SSSE3, __m128i, _mm_loadu_si128, _mm_storeu_si128, _mm_xor_si128)
#define OBL_KERNEL(name) obl_##name##_ssse3_std
#define OBL_TARGET OBL_TGT_SSSE3
#include "oblas_rows.inc"
#undef VEC_TYPE
#undef VEC_LOAD
#undef VEC_XOR_INTO

OBL_TGT_AVX512 static void obl_axpyb32_avx512(u8 *a, u32 *b, u8 u, unsigned k)
{
    __m512i *ap = (__m512i *)a;
    __m512i *ae = (__m512i *)(a + (k & ~63));
    __m512i scatter =
        _mm512_set_epi32(0x07070707, 0x07070707, 0x06060606, 0x06060606, 0x05050505, 0x05050505, 0x04040404, 0x04040404, 0x03030303,
                         0x03030303, 0x02020202, 0x02020202, 0x01010101, 0x01010101, 0x00000000, 0x00000000);
    __m512i cmpmask =
        _mm512_set_epi32(0x80402010, 0x08040201, 0x80402010, 0x08040201, 0x80402010, 0x08040201, 0x80402010, 0x08040201, 0x80402010,
                         0x08040201, 0x80402010, 0x08040201, 0x80402010, 0x08040201, 0x80402010, 0x08040201);
    __m512i up = _mm512_set1_epi8(u);
    unsigned p = 0;
    for (; ap < ae; p += 2, ap++) {
        __m128i low = _mm_loadl_epi64((const __m128i *)(b + p));
        __m512i bcast = _mm512_broadcastq_epi64(low);
        __m512i ret = _mm512_shuffle_epi8(bcast, scatter);
        ret = _mm512_andnot_si512(ret, cmpmask);
        __mmask64 tmp = _mm512_cmpeq_epi8_mask(ret, _mm512_setzero_si512());
        ret = _mm512_mask_blend_epi8(tmp, _mm512_setzero_si512(), up);
        _mm512_storeu_si512(ap, _mm512_xor_si512(_mm512_loadu_si512(ap), ret));
    }
    obl_axpyb32_ref((u8 *)ap, b + p, u, k & 63);
}

OBL_TGT_AVX2 static void obl_axpyb32_avx2(u8 *a, u32 *b, u8 u, unsigned k)
{
    __m256i *ap = (__m256i *)a;
    __m256i *ae = (__m256i *)(a + (k & ~31));
    __m256i scatter =
        _mm256_set_epi32(0x03030303, 0x03030303, 0x02020202, 0x02020202, 0x01010101, 0x01010101, 0x00000000, 0x00000000);
    __m256i cmpmask =
        _mm256_set_epi32(0x80402010, 0x08040201, 0x80402010, 0x08040201, 0x80402010, 0x08040201, 0x80402010, 0x08040201);
    __m256i up = _mm256_set1_epi8(u);
    unsigned p = 0;
    for (; ap < ae; p++, ap++) {
        __m256i bcast = _mm256_set1_epi32(b[p]);
        __m256i ret = _mm256_shuffle_epi8(bcast, scatter);
        ret = _mm256_andnot_si256(ret, cmpmask);
        ret = _mm256_and_si256(_mm256_cmpeq_epi8(ret, _mm256_setzero_si256()), up);
        _mm256_storeu_si256(ap, _mm256_xor_si256(_mm256_loadu_si256(ap), ret));
    }
    obl_axpyb32_ref((u8 *)ap, b + p, u, k & 31);
}

OBL_TGT_SSSE3 static void obl_axpyb32_ssse3(u8 *a, u32 *b, u8 u, unsigned k)
{
    __m128i *ap = (__m128i *)a;
    __m128i *ae = (__m128i *)(a + (k & ~31));
    __m128i scatter_hi = _mm_set_epi32(0x03030303, 0x03030303, 0x02020202, 0x02020202);
    __m128i scatter_lo = _mm_set_epi32(0x01010101, 0x01010101, 0x00000000, 0x00000000);
    __m128i cmpmask = _mm_set_epi32(0x80402010, 0x08040201, 0x80402010, 0x08040201);
    __m128i up = _mm_set1_epi8(u);
    unsigned p = 0;
    for (; ap < ae; p++, ap++) {
        __m128i bcast = _mm_set1_epi32(b[p]);
        __m128i ret_lo = _mm_shuffle_epi8(bcast, scatter_lo);
        __m128i ret_hi = _mm_shuffle_epi8(bcast, scatter_hi);
        ret_lo = _mm_andnot_si128(ret_lo, cmpmask);
        ret_hi = _mm_andnot_si128(ret_hi, cmpmask);
        ret_lo = _mm_and_si128(_mm_cmpeq_epi8(ret_lo, _mm_setzero_si128()), up);
        ret_hi = _mm_and_si128(_mm_cmpeq_epi8(ret_hi, _mm_setzero_si128()), up);
        _mm_storeu_si128(ap, _mm_xor_si128(_mm_loadu_si128(ap), ret_lo));
        ap++;
        _mm_storeu_si128(ap, _mm_xor_si128(_mm_loadu_si128(ap), ret_hi));
    }
    obl_axpyb32_ref((u8 *)ap, b + p, u, k & 31);
}

#endif

#if defined(OBLAS_ARCH_ARM) && (defined(__ARM_NEON) || defined(_MSC_VER))
#include <arm_neon.h>

#if !defined(__aarch64__) && !defined(_M_ARM64)
static inline uint8x16_t vqtbl1q_u8(uint8x16_t tbl, uint8x16_t idx)
{
    uint8x8x2_t tbl2;
    tbl2.val[0] = vget_low_u8(tbl);
    tbl2.val[1] = vget_high_u8(tbl);
    return vcombine_u8(vtbl2_u8(tbl2, vget_low_u8(idx)), vtbl2_u8(tbl2, vget_high_u8(idx)));
}
#endif

#define VEC_LOAD_neon(ptr)       vld1q_u8((const uint8_t *)(ptr))
#define VEC_STORE_neon(ptr, val) vst1q_u8((uint8_t *)(ptr), val)
#define VEC_MASK()          const uint8x16_t mask = vdupq_n_u8(0x0f)
#define VEC_COEF(id, c)     const uint8x16_t lo_##id = vld1q_u8(GF2_8_SHUF_LO + (unsigned)(c) * 16); \
                            const uint8x16_t hi_##id = vld1q_u8(GF2_8_SHUF_HI + (unsigned)(c) * 16)
#define VEC_SPLIT(bx, id)   uint8x16_t slo_##id = vandq_u8(bx, mask); \
                            uint8x16_t shi_##id = vshrq_n_u8(bx, 4)
#define VEC_MUL(c, bx, s)   veorq_u8(vqtbl1q_u8(lo_##c, slo_##s), vqtbl1q_u8(hi_##c, shi_##s))
GENERATE_VEC_OPS(neon, , uint8x16_t, VEC_LOAD_neon, VEC_STORE_neon, veorq_u8)
#define VEC_TYPE uint8x16_t
#define VEC_LOAD obl_load_neon
#define VEC_XOR_INTO obl_xor_into_neon
GENERATE_IMPL(neon, , uint8x16_t, VEC_LOAD_neon, VEC_STORE_neon, veorq_u8)
#define OBL_KERNEL(name) obl_##name##_neon
#define OBL_TARGET
#include "oblas_rows.inc"
#undef VEC_TYPE
#undef VEC_LOAD
#undef VEC_XOR_INTO

static void obl_axpyb32_neon(u8 *a, u32 *b, u8 u, unsigned k)
{
    uint8_t *ap = (uint8_t *)a;
    uint8_t *ae = (uint8_t *)(a + (k & ~31));
#ifdef _MSC_VER
    const uint8x16_t scatter_hi = {.n128_u8 = {2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3}};
    const uint8x16_t scatter_lo = {.n128_u8 = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1}};
    const uint8x16_t cmpmask = {.n128_u8 = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80}};
#else
    const uint8x16_t scatter_hi = {2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3};
    const uint8x16_t scatter_lo = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
    const uint8x16_t cmpmask = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
#endif
    const uint8x16_t up = vdupq_n_u8(u);
    unsigned p = 0;
    for (; ap < ae; p++) {
        uint8x16_t bcast = vreinterpretq_u8_u32(vdupq_n_u32(b[p]));
        uint8x16_t ret_lo = vceqq_u8(vbicq_u8(cmpmask, vqtbl1q_u8(bcast, scatter_lo)), vdupq_n_u8(0));
        uint8x16_t ret_hi = vceqq_u8(vbicq_u8(cmpmask, vqtbl1q_u8(bcast, scatter_hi)), vdupq_n_u8(0));
        ret_lo = vandq_u8(ret_lo, up);
        ret_hi = vandq_u8(ret_hi, up);
        vst1q_u8(ap, veorq_u8(vld1q_u8(ap), ret_lo));
        vst1q_u8(ap + 16, veorq_u8(vld1q_u8(ap + 16), ret_hi));
        ap += 32;
    }
    obl_axpyb32_ref((u8 *)ap, b + p, u, k & 31);
}
#endif

#if defined(OBLAS_ARCH_RISCV) && defined(__riscv_vector)
#include <riscv_vector.h>

static void obl_swap_rvv(u8 *a, u8 *b, unsigned k)
{
    if (a == b || k == 0)
        return;
    for (unsigned i = 0; i < k;) {
        size_t vl = __riscv_vsetvl_e8m1(k - i);
        vuint8m1_t av = __riscv_vle8_v_u8m1(a + i, vl);
        vuint8m1_t bv = __riscv_vle8_v_u8m1(b + i, vl);
        __riscv_vse8_v_u8m1(a + i, bv, vl);
        __riscv_vse8_v_u8m1(b + i, av, vl);
        i += (unsigned)vl;
    }
}

/* Gather the low- and high-nibble products from the coefficient tables. */
OBL_INLINE vuint8m1_t rvv_product(vuint8m1_t lo, vuint8m1_t hi,
                                  vuint8m1_t slo, vuint8m1_t shi, size_t vl)
{
    vuint8m1_t low = __riscv_vrgather_vv_u8m1(lo, slo, vl);
    vuint8m1_t high = __riscv_vrgather_vv_u8m1(hi, shi, vl);
    return __riscv_vxor_vv_u8m1(low, high, vl);
}

OBL_INLINE void rvv_multiply_row(u8 *a, u8 *b, u8 u, unsigned k, int add)
{
    size_t tvl = __riscv_vsetvl_e8m1(16);
    if (tvl < 16) {
        if (add)
            obl_axpy_ref(a, b, u, k);
        else
            obl_axiy_ref(a, b, u, k);
        return;
    }
    vuint8m1_t lo = __riscv_vle8_v_u8m1(GF2_8_SHUF_LO + (unsigned)u * 16, tvl);
    vuint8m1_t hi = __riscv_vle8_v_u8m1(GF2_8_SHUF_HI + (unsigned)u * 16, tvl);
    size_t vl;
    for (unsigned i = 0; i < k; i += (unsigned)vl) {
        vl = __riscv_vsetvl_e8m1(k - i);
        vuint8m1_t acc;
        if (add)
            acc = __riscv_vle8_v_u8m1(a + i, vl);
        vuint8m1_t src = __riscv_vle8_v_u8m1(b + i, vl);
        vuint8m1_t slo = __riscv_vand_vx_u8m1(src, 0x0f, vl);
        vuint8m1_t shi = __riscv_vsrl_vx_u8m1(src, 4, vl);
        vuint8m1_t value = rvv_product(lo, hi, slo, shi, vl);
        if (add)
            value = __riscv_vxor_vv_u8m1(acc, value, vl);
        __riscv_vse8_v_u8m1(a + i, value, vl);
    }
}

OBL_INLINE void rvv_xor_into_m4(u8 *dst, vuint8m4_t value, size_t vl)
{
    vuint8m4_t old = __riscv_vle8_v_u8m4(dst, vl);
    __riscv_vse8_v_u8m4(dst, __riscv_vxor_vv_u8m4(old, value, vl), vl);
}

static inline void obl_axpy_rvv(u8 *a, u8 *b, u8 u, unsigned k)
{
    if (u == 0)
        return;
    if (u == 1) {
        size_t vl;
        for (unsigned i = 0; i < k; i += (unsigned)vl) {
            vl = __riscv_vsetvl_e8m4(k - i);
            vuint8m4_t va = __riscv_vle8_v_u8m4(a + i, vl);
            vuint8m4_t vb = __riscv_vle8_v_u8m4(b + i, vl);
            __riscv_vse8_v_u8m4(a + i, __riscv_vxor_vv_u8m4(va, vb, vl), vl);
        }
        return;
    }
    rvv_multiply_row(a, b, u, k, 1);
}

static void obl_axiy_rvv(u8 *a, u8 *b, u8 u, unsigned k)
{
    if (u == 0) {
        memset(a, 0, k);
    } else if (u == 1) {
        if (a != b)
            memcpy(a, b, k);
    } else {
        rvv_multiply_row(a, b, u, k, 0);
    }
}

static void obl_scal_rvv(u8 *a, u8 u, unsigned k)
{
    obl_axiy_rvv(a, a, u, k);
}

OBL_INLINE void rvv_xor4(u8 *dst, const u8 *r0, const u8 *r1, const u8 *r2,
                         const u8 *r3, unsigned k, int add)
{
    for (unsigned i = 0; i < k;) {
        size_t vl = __riscv_vsetvl_e8m4(k - i);
        vuint8m4_t v01 = __riscv_vxor_vv_u8m4(__riscv_vle8_v_u8m4(r0 + i, vl), __riscv_vle8_v_u8m4(r1 + i, vl), vl);
        vuint8m4_t v23 = __riscv_vxor_vv_u8m4(__riscv_vle8_v_u8m4(r2 + i, vl), __riscv_vle8_v_u8m4(r3 + i, vl), vl);
        if (add)
            v01 = __riscv_vxor_vv_u8m4(__riscv_vle8_v_u8m4(dst + i, vl), v01, vl);
        __riscv_vse8_v_u8m4(dst + i, __riscv_vxor_vv_u8m4(v01, v23, vl), vl);
        i += (unsigned)vl;
    }
}

static void obl_xor_set4_rvv(u8 *dst, const u8 *r0, const u8 *r1, const u8 *r2, const u8 *r3, unsigned k)
{
    rvv_xor4(dst, r0, r1, r2, r3, k, 0);
}

static void obl_xor_acc4_rvv(u8 *dst, const u8 *r0, const u8 *r1, const u8 *r2, const u8 *r3, unsigned k)
{
    rvv_xor4(dst, r0, r1, r2, r3, k, 1);
}

OBL_INLINE void rvv_xor_into_m1(u8 *dst, vuint8m1_t value, size_t vl)
{
    vuint8m1_t old = __riscv_vle8_v_u8m1(dst, vl);
    __riscv_vse8_v_u8m1(dst, __riscv_vxor_vv_u8m1(old, value, vl), vl);
}

/* keep LMUL=1 for vrgather to avoid quadratic LMUL microcode penalties */
OBL_INLINE void obl_axpy2_rvv(u8 *d0, u8 *d1, const u8 *src, u8 u0, u8 u1, unsigned k)
{
    if (u0 == 1 && u1 == 1) {
        for (unsigned i = 0; i < k;) {
            size_t vl = __riscv_vsetvl_e8m4(k - i);
            vuint8m4_t s = __riscv_vle8_v_u8m4(src + i, vl);
            rvv_xor_into_m4(d0 + i, s, vl);
            rvv_xor_into_m4(d1 + i, s, vl);
            i += (unsigned)vl;
        }
        return;
    }
    size_t tvl = __riscv_vsetvl_e8m1(16);
    if (tvl < 16) {
        obl_axpy2_ref(d0, d1, src, u0, u1, k);
        return;
    }
    vuint8m1_t lo0 = __riscv_vle8_v_u8m1(GF2_8_SHUF_LO + (unsigned)u0 * 16, tvl);
    vuint8m1_t hi0 = __riscv_vle8_v_u8m1(GF2_8_SHUF_HI + (unsigned)u0 * 16, tvl);
    vuint8m1_t lo1 = __riscv_vle8_v_u8m1(GF2_8_SHUF_LO + (unsigned)u1 * 16, tvl);
    vuint8m1_t hi1 = __riscv_vle8_v_u8m1(GF2_8_SHUF_HI + (unsigned)u1 * 16, tvl);
    for (unsigned i = 0; i < k;) {
        size_t vl = __riscv_vsetvl_e8m1(k - i);
        vuint8m1_t s = __riscv_vle8_v_u8m1(src + i, vl);
        vuint8m1_t slo = __riscv_vand_vx_u8m1(s, 0x0f, vl);
        vuint8m1_t shi = __riscv_vsrl_vx_u8m1(s, 4, vl);
        vuint8m1_t p0 = rvv_product(lo0, hi0, slo, shi, vl);
        vuint8m1_t p1 = rvv_product(lo1, hi1, slo, shi, vl);
        rvv_xor_into_m1(d0 + i, p0, vl);
        rvv_xor_into_m1(d1 + i, p1, vl);
        i += (unsigned)vl;
    }
}

OBL_INLINE void obl_axpy4_rvv(u8 *d0, u8 *d1, u8 *d2, u8 *d3, const u8 *src,
                              u8 u0, u8 u1, u8 u2, u8 u3, unsigned k)
{
    if (u0 == 1 && u1 == 1 && u2 == 1 && u3 == 1) {
        for (unsigned i = 0; i < k;) {
            size_t vl = __riscv_vsetvl_e8m4(k - i);
            vuint8m4_t s = __riscv_vle8_v_u8m4(src + i, vl);
            rvv_xor_into_m4(d0 + i, s, vl);
            rvv_xor_into_m4(d1 + i, s, vl);
            rvv_xor_into_m4(d2 + i, s, vl);
            rvv_xor_into_m4(d3 + i, s, vl);
            i += (unsigned)vl;
        }
        return;
    }
    size_t tvl = __riscv_vsetvl_e8m1(16);
    if (tvl < 16) {
        obl_axpy4_ref(d0, d1, d2, d3, src, u0, u1, u2, u3, k);
        return;
    }
    vuint8m1_t lo0 = __riscv_vle8_v_u8m1(GF2_8_SHUF_LO + (unsigned)u0 * 16, tvl);
    vuint8m1_t hi0 = __riscv_vle8_v_u8m1(GF2_8_SHUF_HI + (unsigned)u0 * 16, tvl);
    vuint8m1_t lo1 = __riscv_vle8_v_u8m1(GF2_8_SHUF_LO + (unsigned)u1 * 16, tvl);
    vuint8m1_t hi1 = __riscv_vle8_v_u8m1(GF2_8_SHUF_HI + (unsigned)u1 * 16, tvl);
    vuint8m1_t lo2 = __riscv_vle8_v_u8m1(GF2_8_SHUF_LO + (unsigned)u2 * 16, tvl);
    vuint8m1_t hi2 = __riscv_vle8_v_u8m1(GF2_8_SHUF_HI + (unsigned)u2 * 16, tvl);
    vuint8m1_t lo3 = __riscv_vle8_v_u8m1(GF2_8_SHUF_LO + (unsigned)u3 * 16, tvl);
    vuint8m1_t hi3 = __riscv_vle8_v_u8m1(GF2_8_SHUF_HI + (unsigned)u3 * 16, tvl);
    for (unsigned i = 0; i < k;) {
        size_t vl = __riscv_vsetvl_e8m1(k - i);
        vuint8m1_t s = __riscv_vle8_v_u8m1(src + i, vl);
        vuint8m1_t slo = __riscv_vand_vx_u8m1(s, 0x0f, vl);
        vuint8m1_t shi = __riscv_vsrl_vx_u8m1(s, 4, vl);
        vuint8m1_t p0 = rvv_product(lo0, hi0, slo, shi, vl);
        vuint8m1_t p1 = rvv_product(lo1, hi1, slo, shi, vl);
        vuint8m1_t p2 = rvv_product(lo2, hi2, slo, shi, vl);
        vuint8m1_t p3 = rvv_product(lo3, hi3, slo, shi, vl);
        rvv_xor_into_m1(d0 + i, p0, vl);
        rvv_xor_into_m1(d1 + i, p1, vl);
        rvv_xor_into_m1(d2 + i, p2, vl);
        rvv_xor_into_m1(d3 + i, p3, vl);
        i += (unsigned)vl;
    }
}

static void obl_axpyb32_rvv(u8 *a, u32 *b, u8 u, unsigned k)
{
    if (u == 0 || k == 0)
        return;
    const uint8_t *bm = (const uint8_t *)b;
    unsigned i = 0, bulk = k & ~7u;
    while (i < bulk) {
        size_t vl = __riscv_vsetvl_e8m2(bulk - i) & ~(size_t)7u;
        if (vl == 0)
            break;
        vbool4_t mask = __riscv_vlm_v_b4(bm + (i >> 3), vl);
        vuint8m2_t va = __riscv_vle8_v_u8m2(a + i, vl);
        __riscv_vse8_v_u8m2(a + i, __riscv_vxor_vx_u8m2_mu(mask, va, va, u, vl), vl);
        i += (unsigned)vl;
    }
    for (; i < k; i++) {
        if ((bm[i >> 3] >> (i & 7u)) & 1u)
            a[i] ^= u;
    }
}

#define OBL_KERNEL(name) obl_##name##_rvv
#define OBL_TARGET
#include "oblas_rows.inc"
#endif

#define OBL_SET_IMPL(suffix, simd, alignment)                                \
    *impl = (struct oblas_impl){                                             \
        .axpy = obl_axpy_##suffix, .scal = obl_scal_##suffix,                \
        .axiy = obl_axiy_##suffix, .axpyb32 = obl_axpyb32_##simd,            \
        .align_size = (alignment), .swap = obl_swap_##simd,                  \
        .row_ops = obl_row_ops_##suffix,                                     \
        .xor_set4 = obl_xor_set4_##simd, .xor_acc4 = obl_xor_acc4_##simd }

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4113)
#endif
void oblas_get_impl(struct oblas_impl *impl)
{
    oblas_lite_init();
#if defined(OBLAS_ARCH_X86) && !defined(_MSC_VER)
    __builtin_cpu_init();
#endif
    /* fallback */
    impl->axpy = obl_axpy_ref;
    impl->scal = obl_scal_ref;
    impl->axiy = obl_axiy_ref;
    impl->axpyb32 = obl_axpyb32_ref;
    impl->align_size = sizeof(void *);
    impl->swap = obl_swap;
    impl->row_ops = obl_row_ops_ref;
    impl->xor_set4 = obl_xor_set4_ref;
    impl->xor_acc4 = obl_xor_acc4_ref;

#if defined(OBLAS_ARCH_X86)
    int has_avx512 = __builtin_cpu_supports("avx2") &&
                     __builtin_cpu_supports("avx512f") &&
                     __builtin_cpu_supports("avx512bw") &&
                     __builtin_cpu_supports("avx512dq") &&
                     __builtin_cpu_supports("avx512vl");
    if (has_avx512 && __builtin_cpu_supports("gfni")) {
        OBL_SET_IMPL(avx512_gfni, avx512, 64);
    } else if (has_avx512) {
        OBL_SET_IMPL(avx512_std, avx512, 64);
    } else if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("gfni")) {
        OBL_SET_IMPL(avx2_gfni, avx2, 32);
    } else if (__builtin_cpu_supports("avx2")) {
        OBL_SET_IMPL(avx2_std, avx2, 32);
    } else if (__builtin_cpu_supports("ssse3") && __builtin_cpu_supports("gfni")) {
        OBL_SET_IMPL(ssse3_gfni, ssse3, 16);
    } else if (__builtin_cpu_supports("ssse3")) {
        OBL_SET_IMPL(ssse3_std, ssse3, 16);
    }
#elif defined(OBLAS_ARCH_ARM) && (defined(__ARM_NEON) || defined(_MSC_VER))
    OBL_SET_IMPL(neon, neon, 16);
#elif defined(OBLAS_ARCH_RISCV) && defined(__riscv_vector)
    OBL_SET_IMPL(rvv, rvv, 16);
#endif
#ifdef _MSC_VER
#pragma warning(pop)
#endif
}
