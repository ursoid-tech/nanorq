/* include implementation so every backend can be tested */
#include "obl/oblas_lite.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { BUFFER_SIZE = 2200, PACKED_WORDS = (BUFFER_SIZE + 31) / 32 + 8 };

typedef void (*axpy2_fn)(uint8_t *, uint8_t *, const uint8_t *, uint8_t,
                         uint8_t, unsigned);
typedef void (*axpy4_fn)(uint8_t *, uint8_t *, uint8_t *, uint8_t *,
                         const uint8_t *, uint8_t, uint8_t, uint8_t, uint8_t,
                         unsigned);

struct kernel_suite {
  struct oblas_impl impl;
  axpy2_fn axpy2;
  axpy4_fn axpy4;
};

#define WRAP_FUSED(suffix, attr)                                               \
  attr static void test_axpy2_##suffix(u8 *d0, u8 *d1, const u8 *s, u8 u0,     \
                                       u8 u1, unsigned k) {                    \
    obl_axpy2_##suffix(d0, d1, s, u0, u1, k);                                  \
  }                                                                            \
  attr static void test_axpy4_##suffix(u8 *d0, u8 *d1, u8 *d2, u8 *d3,         \
                                       const u8 *s, u8 u0, u8 u1, u8 u2,       \
                                       u8 u3, unsigned k) {                    \
    obl_axpy4_##suffix(d0, d1, d2, d3, s, u0, u1, u2, u3, k);                  \
  }

WRAP_FUSED(ref, )
#if defined(OBLAS_ARCH_X86)
WRAP_FUSED(ssse3_std, OBL_TGT_SSSE3)
WRAP_FUSED(ssse3_gfni, OBL_TGT_SSSE3_GFNI)
WRAP_FUSED(avx2_std, OBL_TGT_AVX2)
WRAP_FUSED(avx2_gfni, OBL_TGT_AVX2_GFNI)
WRAP_FUSED(avx512_std, OBL_TGT_AVX512)
WRAP_FUSED(avx512_gfni, OBL_TGT_AVX512_GFNI)
#elif defined(OBLAS_ARCH_ARM) && (defined(__ARM_NEON) || defined(_MSC_VER))
WRAP_FUSED(neon, )
#elif defined(OBLAS_ARCH_RISCV) && defined(__riscv_vector)
WRAP_FUSED(rvv, )
#endif

static const unsigned lengths[] = {
    0,    1,    2,    7,    8,    15,   16,   17,   31,   32,
    33,   63,   64,   65,   127,  128,  129,  255,  256,  257,
    1279, 1280, 1281, 1295, 1296, 1297, 1311, 1312, 1313, 2049};
static const unsigned offsets[] = {0, 1, 7, 15, 31};
static const uint8_t coefficients[] = {0, 1, 2, 127, 255};
static uint8_t reference_products[256][256];

#define COUNT(a) (sizeof(a) / sizeof((a)[0]))

static uint8_t multiply(uint8_t a, uint8_t b) {
  uint16_t x = a, result = 0;
  for (unsigned bit = 0; bit < 8; bit++) {
    if (b & 1)
      result ^= x;
    b >>= 1;
    x <<= 1;
    if (x & 0x100)
      x ^= 0x11d;
  }
  return (uint8_t)result;
}

static void check(const char *name, struct oblas_impl impl) {
  uint8_t a[BUFFER_SIZE], b[BUFFER_SIZE], expected[BUFFER_SIZE];
  uint8_t source[BUFFER_SIZE];
  for (unsigned n = 0; n < sizeof(lengths) / sizeof(lengths[0]); n++) {
    unsigned len = lengths[n];
    for (unsigned o = 0; o < sizeof(offsets) / sizeof(offsets[0]); o++) {
      unsigned off = offsets[o];
      for (unsigned i = 0; i < sizeof(a); i++) {
        a[i] = expected[i] = (uint8_t)(i * 73 + i / 13);
        b[i] = source[i] = (uint8_t)(i * 29 + i / 7);
      }
      for (unsigned i = 0; i < len; i++) {
        expected[off + i] = b[off + i];
        source[off + i] = a[off + i];
      }
      impl.swap(a + off, b + off, len);
      impl.swap(a + off, a + off, len);
      if (memcmp(a, expected, sizeof(a)) || memcmp(b, source, sizeof(b))) {
        fprintf(stderr, "%s swap failed: length=%u offset=%u\n", name, len,
                off);
        exit(1);
      }
      for (unsigned c = 0; c < sizeof(coefficients); c++) {
        uint8_t coefficient = coefficients[c];
        for (unsigned op = 0; op < 3; op++) {
          for (unsigned i = 0; i < sizeof(a); i++) {
            a[i] = expected[i] = (uint8_t)(i * 73 + i / 13);
            b[i] = source[i] = (uint8_t)(i * 29 + i / 7);
          }
          for (unsigned i = 0; i < len; i++) {
            uint8_t value =
                multiply(coefficient, op == 2 ? a[off + i] : b[off + i]);
            expected[off + i] = op == 0 ? a[off + i] ^ value : value;
          }
          if (op == 0)
            impl.axpy(a + off, b + off, coefficient, len);
          else if (op == 1)
            impl.axiy(a + off, b + off, coefficient, len);
          else
            impl.scal(a + off, coefficient, len);
          if (memcmp(a, expected, sizeof(a)) || memcmp(b, source, sizeof(b))) {
            fprintf(
                stderr,
                "%s failed: operation=%u length=%u offset=%u coefficient=%u\n",
                name, op, len, off, coefficient);
            exit(1);
          }
        }
      }
    }
  }
  impl.swap(NULL, NULL, 0);
  impl.swap(NULL, a, 0);
  impl.swap(a, NULL, 0);
  printf("%s kernels passed\n", name);
}

/* every equality pattern for four destinations/sources */
static const unsigned aliases[][4] = {
    {0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}, {0, 0, 1, 1}, {0, 0, 1, 2},
    {0, 1, 0, 0}, {0, 1, 0, 1}, {0, 1, 0, 2}, {0, 1, 1, 0}, {0, 1, 1, 1},
    {0, 1, 1, 2}, {0, 1, 2, 0}, {0, 1, 2, 1}, {0, 1, 2, 2}, {0, 1, 2, 3}};

static void check_fanout_case(const char *name, struct kernel_suite suite,
                              unsigned len, unsigned off, unsigned alias,
                              const uint8_t coefficients[4]) {
  uint8_t rows[4][BUFFER_SIZE], expected[4][BUFFER_SIZE];
  uint8_t source[BUFFER_SIZE], original_source[BUFFER_SIZE];
  const unsigned *row = aliases[alias];
  /* mismatch source and destination alignment */
  unsigned src_off = (off + 13) % 32;
  for (unsigned i = 0; i < BUFFER_SIZE; i++)
    source[i] = original_source[i] = (uint8_t)(i * 29 + i / 7);
  for (unsigned count = 2; count <= 4; count += 2) {
    for (unsigned r = 0; r < 4; r++)
      for (unsigned i = 0; i < BUFFER_SIZE; i++)
        rows[r][i] = expected[r][i] = (uint8_t)(i * 73 + i / 13 + r * 41);
    for (unsigned r = 0; r < count; r++)
      for (unsigned i = 0; i < len; i++)
        expected[row[r]][off + i] ^=
            reference_products[coefficients[r]][source[src_off + i]];
    if (count == 2)
      suite.axpy2(rows[row[0]] + off, rows[row[1]] + off, source + src_off,
                  coefficients[0], coefficients[1], len);
    else
      suite.axpy4(rows[row[0]] + off, rows[row[1]] + off, rows[row[2]] + off,
                  rows[row[3]] + off, source + src_off, coefficients[0],
                  coefficients[1], coefficients[2], coefficients[3], len);
    if (memcmp(rows, expected, sizeof(rows)) ||
        memcmp(source, original_source, sizeof(source))) {
      fprintf(stderr,
              "%s axpy%u failed: length=%u offset=%u alias=%u "
              "coefficients=%u,%u,%u,%u\n",
              name, count, len, off, alias, coefficients[0], coefficients[1],
              coefficients[2], coefficients[3]);
      exit(1);
    }
  }
}

static void check_fanout(const char *name, struct kernel_suite suite) {
  if (!suite.axpy2 || !suite.axpy4)
    return;
  static const uint8_t cases[][4] = {
      {0, 0, 0, 0},     {1, 1, 1, 1}, {0, 1, 127, 255},     {1, 0, 2, 0},
      {255, 127, 1, 0}, {2, 2, 2, 2}, {255, 255, 255, 255}, {128, 254, 3, 17}};
  static const unsigned coefficient_lengths[] = {1, 17, 33, 65, 129, 1313};
  for (unsigned n = 0; n < COUNT(lengths); n++)
    for (unsigned o = 0; o < COUNT(offsets); o++)
      for (unsigned a = 0; a < COUNT(aliases); a++)
        for (unsigned c = 0; c < COUNT(cases); c++)
          check_fanout_case(name, suite, lengths[n], offsets[o], a, cases[c]);
  /* odd multipliers visit every GF256 coefficient in every destination */
  for (unsigned c = 0; c < 256; c++) {
    uint8_t all_coefficients[] = {(uint8_t)c, (uint8_t)(c * 73 + 1),
                                  (uint8_t)(c * 151 + 19),
                                  (uint8_t)(c * 229 + 101)};
    for (unsigned n = 0; n < COUNT(coefficient_lengths); n++)
      check_fanout_case(name, suite, coefficient_lengths[n],
                        offsets[c % COUNT(offsets)], c % COUNT(aliases),
                        all_coefficients);
  }
}

static void check_xor4(const char *name, struct oblas_impl impl) {
  uint8_t rows[4][BUFFER_SIZE], original_rows[4][BUFFER_SIZE];
  uint8_t output[BUFFER_SIZE], expected[BUFFER_SIZE];
  for (unsigned r = 0; r < 4; r++)
    for (unsigned i = 0; i < BUFFER_SIZE; i++)
      rows[r][i] = original_rows[r][i] =
          (uint8_t)(i * (29 + r * 2) + i / (7 + r) + r * 41);
  for (unsigned n = 0; n < COUNT(lengths); n++) {
    unsigned len = lengths[n];
    for (unsigned o = 0; o < COUNT(offsets); o++) {
      unsigned off = offsets[o];
      for (unsigned a = 0; a < 2 * COUNT(aliases); a++) {
        unsigned alias = a % COUNT(aliases);
        const uint8_t *src[4];
        for (unsigned r = 0; r < 4; r++) {
          unsigned shift = a < COUNT(aliases) ? aliases[alias][r] : r;
          /* exercise exact duplicates and partially overlapping sources */
          src[r] = rows[aliases[alias][r]] + (off + shift * 7) % 32;
        }
        for (unsigned accumulate = 0; accumulate < 2; accumulate++) {
          for (unsigned i = 0; i < BUFFER_SIZE; i++)
            output[i] = expected[i] = (uint8_t)(i * 73 + i / 13);
          for (unsigned i = 0; i < len; i++) {
            uint8_t value = src[0][i] ^ src[1][i] ^ src[2][i] ^ src[3][i];
            expected[off + i] = accumulate ? expected[off + i] ^ value : value;
          }
          if (accumulate)
            impl.xor_acc4(output + off, src[0], src[1], src[2], src[3], len);
          else
            impl.xor_set4(output + off, src[0], src[1], src[2], src[3], len);
          if (memcmp(output, expected, sizeof(output)) ||
              memcmp(rows, original_rows, sizeof(rows))) {
            fprintf(stderr, "%s xor_%s4 failed: length=%u offset=%u alias=%u\n",
                    name, accumulate ? "acc" : "set", len, off, a);
            exit(1);
          }
        }
      }
    }
  }
}

static uint32_t packed_pattern(unsigned pattern, unsigned word) {
  switch (pattern) {
  case 0:
    return 0;
  case 1:
    return UINT32_MAX;
  case 2:
    return UINT32_C(0xaaaaaaaa);
  case 3:
    return UINT32_C(0x55555555);
  case 4:
    return UINT32_C(1) << (word % 32);
  case 5:
    return ~(UINT32_C(1) << (word % 32));
  default:
    return UINT32_C(0x89722dfd) * (word + 1) ^ UINT32_C(0xabe893c7);
  }
}

static void check_packed_case(const char *name, struct oblas_impl impl,
                              unsigned len, unsigned off, unsigned pattern,
                              uint8_t coefficient) {
  uint8_t output[BUFFER_SIZE], expected[BUFFER_SIZE];
  uint32_t packed[PACKED_WORDS], original_packed[PACKED_WORDS];
  unsigned word_offset = off % 8;
  for (unsigned i = 0; i < BUFFER_SIZE; i++)
    output[i] = expected[i] = (uint8_t)(i * 73 + i / 13);
  for (unsigned i = 0; i < PACKED_WORDS; i++)
    packed[i] = original_packed[i] = packed_pattern(pattern, i);
  for (unsigned i = 0; i < len; i++)
    if ((packed[word_offset + i / 32] >> (i % 32)) & 1)
      expected[off + i] ^= coefficient;
  impl.axpyb32(output + off, packed + word_offset, coefficient, len);
  if (memcmp(output, expected, sizeof(output)) ||
      memcmp(packed, original_packed, sizeof(packed))) {
    fprintf(
        stderr,
        "%s axpyb32 failed: length=%u offset=%u pattern=%u coefficient=%u\n",
        name, len, off, pattern, coefficient);
    exit(1);
  }
}

static void check_packed(const char *name, struct oblas_impl impl) {
  static const unsigned coefficient_lengths[] = {1,  31, 32,  33,  63,
                                                 64, 65, 129, 1313};
  for (unsigned n = 0; n < COUNT(lengths); n++)
    for (unsigned o = 0; o < COUNT(offsets); o++)
      for (unsigned p = 0; p < 7; p++)
        for (unsigned c = 0; c < COUNT(coefficients); c++)
          check_packed_case(name, impl, lengths[n], offsets[o], p,
                            coefficients[c]);
  for (unsigned c = 0; c < 256; c++)
    for (unsigned n = 0; n < COUNT(coefficient_lengths); n++)
      check_packed_case(name, impl, coefficient_lengths[n],
                        offsets[c % COUNT(offsets)], c % 7, (uint8_t)c);
}

/* exact allocations expose vector tail overreads under ASan */
static void check_bounds(const char *name, struct kernel_suite suite) {
  struct oblas_impl impl = suite.impl;
  for (unsigned n = 0; n < COUNT(lengths); n++) {
    unsigned len = lengths[n];
    uint8_t *storage[5], *row[5];
    unsigned words = (len + 31) / 32;
    uint32_t *packed = malloc((words ? words : 1) * sizeof(*packed));
    uint8_t *mat_storage = malloc(5 * (size_t)len + 1);
    if (!packed || !mat_storage)
      exit(1);
    memset(mat_storage, 0xa5, 5 * (size_t)len + 1);
    for (unsigned i = 0; i < words; i++)
      packed[i] = UINT32_MAX;
    for (unsigned r = 0; r < 5; r++) {
      storage[r] = malloc(len + 1);
      if (!storage[r])
        exit(1);
      memset(storage[r], 0xa5, len + 1);
      row[r] = storage[r] + 1;
    }
    struct oblas_row_op rops[4] = {
        {1, 0, 4}, {255, 1, 4}, {127, 2, 4}, {2, 3, 4}};
    impl.row_ops(mat_storage + 1, len, rops, 4, 0);
    impl.row_ops(mat_storage + 1, len, rops, 4, 1);
    for (unsigned c = 0; c < COUNT(coefficients); c++) {
      uint8_t u = coefficients[c];
      if (suite.axpy2 && suite.axpy4) {
        suite.axpy2(row[0], row[1], row[4], u, u, len);
        suite.axpy4(row[0], row[1], row[2], row[3], row[4], u, u, u, u, len);
      }
      impl.axpyb32(row[0], packed, u, len);
    }
    if (suite.axpy2 && suite.axpy4) {
      suite.axpy2(row[0], row[1], row[4], 1, 255, len);
      suite.axpy4(row[0], row[1], row[2], row[3], row[4], 0, 1, 127, 255, len);
    }
    impl.xor_set4(row[0], row[1], row[2], row[3], row[4], len);
    impl.xor_acc4(row[0], row[1], row[2], row[3], row[4], len);
    if (mat_storage[0] != 0xa5) {
      fprintf(stderr, "%s row_ops wrote before matrix: length=%u\n", name, len);
      exit(1);
    }
    for (unsigned r = 0; r < 5; r++) {
      if (storage[r][0] != 0xa5) {
        fprintf(stderr, "%s wrote before a row: length=%u row=%u\n", name, len,
                r);
        exit(1);
      }
      free(storage[r]);
    }
    free(mat_storage);
    free(packed);
  }
}

static void check_row_ops_case(const char *name, struct oblas_impl impl,
                               unsigned stride, unsigned off, size_t count,
                               unsigned trial, int reverse) {
  enum { ROWS = 12, MAX_OPS = 65 };
  static const unsigned dest_offsets[] = {1, 1, 2, 3, 1, 2, 1, 2};
  uint8_t matrix[ROWS * BUFFER_SIZE], expected[ROWS * BUFFER_SIZE];
  struct oblas_row_op ops[MAX_OPS];
  size_t bytes = off + 16 + ROWS * stride + 32;
  uint8_t *base = matrix + off + 16, *ref = expected + off + 16;
  for (size_t i = 0; i < bytes; i++)
    matrix[i] = expected[i] = (uint8_t)(i * 73 + i / 13 + trial * 17);
  for (unsigned i = 0; i < MAX_OPS; i++) {
    unsigned source = (i / 8 + trial) % ROWS;
    unsigned dest = (source + dest_offsets[i % COUNT(dest_offsets)]) % ROWS;
    uint8_t coefficient = i % 8 < 4 ? 1 : (uint8_t)(trial + i * 73);
    if (!coefficient)
      coefficient = 255;
    ops[i] = (struct oblas_row_op){reverse ? 1 : coefficient, dest, source};
    if (!reverse && i % 11 == 8) {
      /* scales interrupt groups and consume only the low byte of j */
      ops[i].u = 0;
      ops[i].j = 256u + (uint8_t)(trial + i * 37);
    }
  }
  /* execute reference ops sequentially, including repeated destinations */
  for (size_t step = 0; step < count; step++) {
    const struct oblas_row_op *op = &ops[reverse ? count - 1 - step : step];
    uint8_t *dst = ref + (size_t)op->i * stride;
    if (reverse || op->u) {
      const uint8_t *src = ref + (size_t)op->j * stride;
      uint8_t coefficient = reverse ? 1 : op->u;
      for (unsigned i = 0; i < stride; i++)
        dst[i] ^= reference_products[coefficient][src[i]];
    } else {
      for (unsigned i = 0; i < stride; i++)
        dst[i] = reference_products[(uint8_t)op->j][dst[i]];
    }
  }
  impl.row_ops(base, stride, ops, count, reverse);
  if (memcmp(matrix, expected, bytes)) {
    fprintf(stderr,
            "%s row_ops failed: stride=%u offset=%u count=%zu trial=%u "
            "reverse=%d\n",
            name, stride, off, count, trial, reverse);
    exit(1);
  }
}

static void check_row_ops(const char *name, struct oblas_impl impl) {
  static const size_t counts[] = {0,  1,  2,  3,  4,  5,  7,  8, 9,
                                  15, 16, 17, 31, 32, 33, 64, 65};
  for (unsigned n = 0; n < COUNT(lengths); n++)
    for (unsigned o = 0; o < COUNT(offsets); o++)
      for (unsigned c = 0; c < COUNT(counts); c++)
        for (int reverse = 0; reverse < 2; reverse++)
          check_row_ops_case(name, impl, lengths[n], offsets[o], counts[c],
                             n * 11 + o * 17 + c, reverse);
  /* sweep every GF256 coefficient, including scale-by-zero and scale-by-one */
  for (unsigned c = 0; c < 256; c++)
    check_row_ops_case(name, impl, 65, offsets[c % COUNT(offsets)], 65, c, 0);
}

static void check_all(const char *name, struct kernel_suite suite) {
  struct oblas_impl impl = suite.impl;
  if (!impl.xor_set4 || !impl.xor_acc4 || !impl.axpyb32 || !impl.row_ops) {
    fprintf(stderr, "%s omitted a fused, packed-bit, or row-batch kernel\n",
            name);
    exit(1);
  }
  check(name, impl);
  check_fanout(name, suite);
  check_xor4(name, impl);
  check_packed(name, impl);
  check_bounds(name, suite);
  check_row_ops(name, impl);
  printf("%s fused, packed-bit, and row-batch kernels passed\n", name);
}

#define CHECK_BACKEND(suffix, simd)                                            \
  do {                                                                         \
    struct kernel_suite suite = {0};                                           \
    suite.impl.axpy = obl_axpy_##suffix;                                       \
    suite.impl.axiy = obl_axiy_##suffix;                                       \
    suite.impl.swap = obl_swap_##simd;                                         \
    suite.impl.scal = obl_scal_##suffix;                                       \
    suite.impl.axpyb32 = obl_axpyb32_##simd;                                   \
    suite.impl.xor_set4 = obl_xor_set4_##simd;                                 \
    suite.impl.xor_acc4 = obl_xor_acc4_##simd;                                 \
    suite.impl.row_ops = obl_row_ops_##suffix;                                 \
    suite.axpy2 = test_axpy2_##suffix;                                         \
    suite.axpy4 = test_axpy4_##suffix;                                         \
    check_all(#suffix, suite);                                                 \
  } while (0)

int main(void) {
  struct kernel_suite suite = {0};
  for (unsigned a = 0; a < 256; a++)
    for (unsigned b = 0; b < 256; b++)
      reference_products[a][b] = multiply((uint8_t)a, (uint8_t)b);
  oblas_get_impl(&suite.impl);
  check_all("automatic", suite);
  suite.impl.axpy = obl_axpy_ref;
  suite.impl.axiy = obl_axiy_ref;
  suite.impl.scal = obl_scal_ref_wrapper;
  suite.impl.swap = obl_swap;
  suite.impl.axpyb32 = obl_axpyb32_ref;
  suite.impl.xor_set4 = obl_xor_set4_ref;
  suite.impl.xor_acc4 = obl_xor_acc4_ref;
  suite.impl.row_ops = obl_row_ops_ref;
  suite.axpy2 = test_axpy2_ref;
  suite.axpy4 = test_axpy4_ref;
  check_all("scalar", suite);
#if defined(OBLAS_ARCH_X86)
  if (__builtin_cpu_supports("ssse3"))
    CHECK_BACKEND(ssse3_std, ssse3);
  if (__builtin_cpu_supports("avx2"))
    CHECK_BACKEND(avx2_std, avx2);
  if (__builtin_cpu_supports("ssse3") && __builtin_cpu_supports("gfni"))
    CHECK_BACKEND(ssse3_gfni, ssse3);
  if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("gfni"))
    CHECK_BACKEND(avx2_gfni, avx2);
  if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("avx512f") &&
      __builtin_cpu_supports("avx512bw") &&
      __builtin_cpu_supports("avx512dq") &&
      __builtin_cpu_supports("avx512vl")) {
    CHECK_BACKEND(avx512_std, avx512);
    if (__builtin_cpu_supports("gfni"))
      CHECK_BACKEND(avx512_gfni, avx512);
  }
#elif defined(OBLAS_ARCH_ARM) && (defined(__ARM_NEON) || defined(_MSC_VER))
  CHECK_BACKEND(neon, neon);
#elif defined(OBLAS_ARCH_RISCV) && defined(__riscv_vector)
  CHECK_BACKEND(rvv, rvv);
#endif
  puts("SIMD kernel regressions passed");
  return 0;
}
