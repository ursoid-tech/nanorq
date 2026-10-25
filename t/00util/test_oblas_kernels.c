/* include implementation so every backend can be tested */
#include "obl/oblas_lite.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
  static const unsigned lengths[] = {
      0,    1,    2,    7,    8,    15,   16,   17,   31,   32,
      33,   63,   64,   65,   127,  128,  129,  255,  256,  257,
      1279, 1280, 1281, 1295, 1296, 1297, 1311, 1312, 1313, 2049};
  static const unsigned offsets[] = {0, 1, 7, 15, 31};
  static const uint8_t coefficients[] = {0, 1, 2, 127, 255};
  uint8_t a[2200], b[2200], expected[2200], source[2200];
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

#define CHECK_BACKEND(suffix, simd)                                            \
  do {                                                                         \
    struct oblas_impl impl = {0};                                              \
    impl.axpy = obl_axpy_##suffix;                                             \
    impl.axiy = obl_axiy_##suffix;                                             \
    impl.swap = obl_swap_##simd;                                               \
    impl.scal = obl_scal_##suffix;                                             \
    check(#suffix, impl);                                                      \
  } while (0)

int main(void) {
  struct oblas_impl impl;
  oblas_get_impl(&impl);
  check("automatic", impl);
  impl.axpy = obl_axpy_ref;
  impl.axiy = obl_axiy_ref;
  impl.scal = obl_scal_ref_wrapper;
  impl.swap = obl_swap;
  check("scalar", impl);
#if defined(OBLAS_ARCH_X86)
  if (__builtin_cpu_supports("ssse3"))
    CHECK_BACKEND(ssse3_std, ssse3);
  if (__builtin_cpu_supports("avx2"))
    CHECK_BACKEND(avx2_std, avx2);
  if (__builtin_cpu_supports("ssse3") && __builtin_cpu_supports("gfni"))
    CHECK_BACKEND(ssse3_gfni, ssse3);
  if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("gfni"))
    CHECK_BACKEND(avx2_gfni, avx2);
  if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
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
