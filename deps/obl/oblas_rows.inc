/* This file is included once per backend. OBL_KERNEL selects its functions
 * and OBL_TARGET supplies their instruction set. Fixed-width backends also
 * supply the vector primitives used by the two- and four-row kernels. */
#ifdef VEC_TYPE
/* Load the source once per vector. Destinations can repeat, so update in order. */
OBL_TARGET OBL_INLINE void OBL_KERNEL(axpy2)(u8 *d0, u8 *d1, const u8 *src,
                                             u8 u0, u8 u1, unsigned k)
{
    unsigned i = 0, bulk = k - (k % sizeof(VEC_TYPE));
    if (u0 == 1 && u1 == 1) {
        for (; i < bulk; i += sizeof(VEC_TYPE)) {
            VEC_TYPE s = VEC_LOAD(src + i);
            VEC_XOR_INTO(d0 + i, s);
            VEC_XOR_INTO(d1 + i, s);
        }
        for (; i < k; i++) {
            u8 s = src[i];
            d0[i] ^= s;
            d1[i] ^= s;
        }
    } else {
        VEC_MASK();
        VEC_COEF(0, u0);
        VEC_COEF(1, u1);
        for (; i < bulk; i += sizeof(VEC_TYPE)) {
            VEC_TYPE s = VEC_LOAD(src + i);
            VEC_SPLIT(s, 0);
            VEC_TYPE p0 = VEC_MUL(0, s, 0);
            VEC_TYPE p1 = VEC_MUL(1, s, 0);
            VEC_XOR_INTO(d0 + i, p0);
            VEC_XOR_INTO(d1 + i, p1);
        }
        obl_axpy2_ref(d0 + i, d1 + i, src + i, u0, u1, k - i);
    }
}

OBL_TARGET OBL_INLINE void OBL_KERNEL(axpy4)(u8 *d0, u8 *d1, u8 *d2, u8 *d3,
                                             const u8 *src, u8 u0, u8 u1,
                                             u8 u2, u8 u3, unsigned k)
{
    unsigned i = 0, bulk = k - (k % sizeof(VEC_TYPE));
    if (u0 == 1 && u1 == 1 && u2 == 1 && u3 == 1) {
        for (; i < bulk; i += sizeof(VEC_TYPE)) {
            VEC_TYPE s = VEC_LOAD(src + i);
            VEC_XOR_INTO(d0 + i, s);
            VEC_XOR_INTO(d1 + i, s);
            VEC_XOR_INTO(d2 + i, s);
            VEC_XOR_INTO(d3 + i, s);
        }
        for (; i < k; i++) {
            u8 s = src[i];
            d0[i] ^= s;
            d1[i] ^= s;
            d2[i] ^= s;
            d3[i] ^= s;
        }
    } else {
        VEC_MASK();
        VEC_COEF(0, u0);
        VEC_COEF(1, u1);
        VEC_COEF(2, u2);
        VEC_COEF(3, u3);
        for (; i < bulk; i += sizeof(VEC_TYPE)) {
            VEC_TYPE s = VEC_LOAD(src + i);
            VEC_SPLIT(s, 0);
            VEC_TYPE p0 = VEC_MUL(0, s, 0);
            VEC_TYPE p1 = VEC_MUL(1, s, 0);
            VEC_TYPE p2 = VEC_MUL(2, s, 0);
            VEC_TYPE p3 = VEC_MUL(3, s, 0);
            VEC_XOR_INTO(d0 + i, p0);
            VEC_XOR_INTO(d1 + i, p1);
            VEC_XOR_INTO(d2 + i, p2);
            VEC_XOR_INTO(d3 + i, p3);
        }
        obl_axpy4_ref(d0 + i, d1 + i, d2 + i, d3 + i, src + i,
                      u0, u1, u2, u3, k - i);
    }
}

#undef VEC_MASK
#undef VEC_COEF
#undef VEC_SPLIT
#undef VEC_MUL
#endif

OBL_TARGET OBL_INLINE void OBL_KERNEL(replay)(u8 *D, unsigned stride,
                                              const struct oblas_row_op *ops,
                                              size_t count, int step)
{
    size_t left = count;
    while (left) {
        const struct oblas_row_op *op = ops + (step > 0 ? count - left : left - 1);
        if (left > 8)
            obl_prefetch_row_w(OBL_ROW_AT(D, op[8 * step].i, stride), stride);
        u8 *dst = OBL_ROW_AT(D, op->i, stride);
        u32 j = op->j;
        if (!op->u) {
            OBL_KERNEL(scal)(dst, (u8)j, stride);
            left--;
        } else if (op->i == j) {
            OBL_KERNEL(scal)(dst, (u8)(1 ^ op->u), stride);
            left--;
        } else if (left >= 2 && op[step].u && op[step].j == j) {
            u8 *src = OBL_ROW_AT(D, j, stride);
            u8 *d1 = OBL_ROW_AT(D, op[step].i, stride);
            /* Only the last destination may overwrite the shared source. */
            if (left >= 4 && op[step].i != j && op[2 * step].u && op[2 * step].j == j &&
                op[2 * step].i != j && op[3 * step].u && op[3 * step].j == j) {
                OBL_KERNEL(axpy4)(dst, d1, OBL_ROW_AT(D, op[2 * step].i, stride),
                                  OBL_ROW_AT(D, op[3 * step].i, stride), src, op->u,
                                  op[step].u, op[2 * step].u, op[3 * step].u, stride);
                left -= 4;
            } else {
                OBL_KERNEL(axpy2)(dst, d1, src, op->u, op[step].u, stride);
                left -= 2;
            }
        } else {
            OBL_KERNEL(axpy)(dst, OBL_ROW_AT(D, j, stride), op->u, stride);
            left--;
        }
    }
}

OBL_TARGET static void OBL_KERNEL(row_ops)(u8 *D, unsigned stride,
                                           const struct oblas_row_op *ops,
                                           size_t count, int reverse)
{
    if (reverse)
        OBL_KERNEL(replay)(D, stride, ops, count, -1);
    else
        OBL_KERNEL(replay)(D, stride, ops, count, 1);
}

#undef OBL_KERNEL
#undef OBL_TARGET
