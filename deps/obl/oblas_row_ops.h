/* Private row-operation batching. This layer knows only elementary GF256
 * row operations; callers own solver phases, permutations, and workspaces.
 * Keeping the batch with the kernels avoids an indirect call for every row. */
#ifndef OBLAS_ROW_OPS_H
#define OBLAS_ROW_OPS_H

static inline void obl_prefetch_row_w(const uint8_t *p, unsigned len) {
#if defined(__GNUC__) || defined(__clang__)
  unsigned count = len / 256 + (len % 256 != 0);
  for (unsigned n = 0; n < count; n++)
    __builtin_prefetch(p + (size_t)n * 256, 1, 1);
#else
  (void)p;
  (void)len;
#endif
}

#define OBL_ROW_OPS(NAME, ATTR, AXPY, SCAL, AXPY2, AXPY4)                      \
  ATTR static void obl_row_ops_##NAME(uint8_t *D, unsigned stride,             \
                                      const struct oblas_row_op *ops,          \
                                      size_t count, int reverse_xor) {         \
    if (reverse_xor) {                                                         \
      size_t idx = count;                                                      \
      while (idx > 0) {                                                        \
        if (idx > 8)                                                           \
          obl_prefetch_row_w(D + (size_t)ops[idx - 8].i * stride, stride);     \
        uint32_t j = ops[idx - 1].j;                                           \
        if (idx >= 4 && ops[idx - 2].j == j && ops[idx - 3].j == j &&          \
            ops[idx - 4].j == j) {                                             \
          AXPY4(D + (size_t)ops[idx - 1].i * stride,                           \
                D + (size_t)ops[idx - 2].i * stride,                           \
                D + (size_t)ops[idx - 3].i * stride,                           \
                D + (size_t)ops[idx - 4].i * stride, D + (size_t)j * stride,   \
                1, 1, 1, 1, stride);                                           \
          idx -= 4;                                                            \
        } else if (idx >= 2 && ops[idx - 2].j == j) {                          \
          AXPY2(D + (size_t)ops[idx - 1].i * stride,                           \
                D + (size_t)ops[idx - 2].i * stride, D + (size_t)j * stride,   \
                1, 1, stride);                                                 \
          idx -= 2;                                                            \
        } else {                                                               \
          AXPY(D + (size_t)ops[idx - 1].i * stride, D + (size_t)j * stride, 1, \
               stride);                                                        \
          idx--;                                                               \
        }                                                                      \
      }                                                                        \
      return;                                                                  \
    }                                                                          \
    size_t idx = 0;                                                            \
    while (idx < count) {                                                      \
      if (count - idx > 8)                                                     \
        obl_prefetch_row_w(D + (size_t)ops[idx + 8].i * stride, stride);       \
      uint32_t j = ops[idx].j;                                                 \
      uint8_t u = ops[idx].u;                                                  \
      if (u && count - idx >= 4 && ops[idx + 1].u && ops[idx + 1].j == j &&    \
          ops[idx + 2].u && ops[idx + 2].j == j && ops[idx + 3].u &&           \
          ops[idx + 3].j == j) {                                               \
        AXPY4(D + (size_t)ops[idx].i * stride,                                 \
              D + (size_t)ops[idx + 1].i * stride,                             \
              D + (size_t)ops[idx + 2].i * stride,                             \
              D + (size_t)ops[idx + 3].i * stride, D + (size_t)j * stride, u,  \
              ops[idx + 1].u, ops[idx + 2].u, ops[idx + 3].u, stride);         \
        idx += 4;                                                              \
      } else if (u && count - idx >= 2 && ops[idx + 1].u &&                    \
                 ops[idx + 1].j == j) {                                        \
        AXPY2(D + (size_t)ops[idx].i * stride,                                 \
              D + (size_t)ops[idx + 1].i * stride, D + (size_t)j * stride, u,  \
              ops[idx + 1].u, stride);                                         \
        idx += 2;                                                              \
      } else {                                                                 \
        if (u)                                                                 \
          AXPY(D + (size_t)ops[idx].i * stride, D + (size_t)j * stride, u,     \
               stride);                                                        \
        else                                                                   \
          SCAL(D + (size_t)ops[idx].i * stride, (uint8_t)j, stride);           \
        idx++;                                                                 \
      }                                                                        \
    }                                                                          \
  }

OBL_ROW_OPS(ref, , obl_axpy_ref, obl_scal_ref_wrapper, obl_axpy2_ref,
            obl_axpy4_ref)
#if defined(OBLAS_ARCH_X86)
OBL_ROW_OPS(ssse3_std, __attribute__((target("ssse3"))), obl_axpy_ssse3_std,
            obl_scal_ssse3_std, obl_axpy2_ssse3_std, obl_axpy4_ssse3_std)
OBL_ROW_OPS(ssse3_gfni, __attribute__((target("ssse3,gfni"))),
            obl_axpy_ssse3_gfni, obl_scal_ssse3_gfni, obl_axpy2_ssse3_gfni,
            obl_axpy4_ssse3_gfni)
OBL_ROW_OPS(avx2_std, __attribute__((target("ssse3,avx2"))), obl_axpy_avx2_std,
            obl_scal_avx2_std, obl_batch_axpy2_avx2_std,
            obl_batch_axpy4_avx2_std)
OBL_ROW_OPS(avx2_gfni, __attribute__((target("ssse3,avx2,gfni"))),
            obl_axpy_avx2_gfni, obl_scal_avx2_gfni, obl_batch_axpy2_avx2_gfni,
            obl_batch_axpy4_avx2_gfni)
OBL_ROW_OPS(
    avx512_std,
    __attribute__((target("ssse3,avx2,avx512f,avx512bw,avx512dq,avx512vl"))),
    obl_axpy_avx512_std, obl_scal_avx512_std, obl_batch_axpy2_avx512_std,
    obl_batch_axpy4_avx512_std)
OBL_ROW_OPS(avx512_gfni,
            __attribute__((
                target("ssse3,avx2,avx512f,avx512bw,avx512dq,avx512vl,gfni"))),
            obl_axpy_avx512_gfni, obl_scal_avx512_gfni,
            obl_batch_axpy2_avx512_gfni, obl_batch_axpy4_avx512_gfni)
#elif defined(OBLAS_ARCH_ARM) && (defined(__ARM_NEON) || defined(_MSC_VER))
OBL_ROW_OPS(neon, , obl_axpy_neon, obl_scal_neon, obl_axpy2_neon,
            obl_axpy4_neon)
#elif defined(OBLAS_ARCH_RISCV) && defined(__riscv_vector)
OBL_ROW_OPS(rvv, , obl_axpy_rvv, obl_scal_rvv, obl_axpy2_rvv, obl_axpy4_rvv)
#endif

#undef OBL_ROW_OPS
#endif /* OBLAS_ROW_OPS_H */
