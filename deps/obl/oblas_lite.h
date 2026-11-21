#ifndef OBLAS_LITE_H
#define OBLAS_LITE_H

#include <stdint.h>
#include <stddef.h>

#include "gf2_8_tables.h"
#include "oblas_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* A compact elementary row operation: i ^= u*j when u is nonzero;
 * when u is zero, scale row i by the byte value in j. AXPY source and
 * destination rows must differ. Indices refer to rows of the supplied matrix. */
struct oblas_row_op {
    uint8_t u;
    uint32_t i;
    uint32_t j;
};

struct oblas_impl {
    void (*axpy)(uint8_t *a, uint8_t *b, uint8_t u, unsigned k);
    void (*scal)(uint8_t *a, uint8_t u, unsigned k);
    void (*axiy)(uint8_t *a, uint8_t *b, uint8_t u, unsigned k);
    void (*axpyb32)(uint8_t *a, uint32_t *b, uint8_t u, unsigned k);
    size_t align_size;
    void (*swap)(uint8_t *a, uint8_t *b, unsigned k);
    /* Execute an ordered batch, fusing adjacent operations with one source.
     * reverse_xor executes an XOR-only batch in reverse with coefficient 1.
     * All row indices must be valid; operations and matrix must be disjoint. */
    void (*row_ops)(uint8_t *matrix, unsigned stride,
                    const struct oblas_row_op *ops, size_t count,
                    int reverse_xor);
    /* Fused dN ^= uN * src. Source must be disjoint from every destination;
     * destinations may be identical or disjoint, but not partially overlap.
     * Identical destinations accumulate each product in argument order.
     * All fused primitives accept unaligned buffers and arbitrary byte lengths. */
    void (*axpy2)(uint8_t *d0, uint8_t *d1, const uint8_t *src,
                  uint8_t u0, uint8_t u1, unsigned k);
    void (*axpy4)(uint8_t *d0, uint8_t *d1, uint8_t *d2, uint8_t *d3,
                  const uint8_t *src, uint8_t u0, uint8_t u1, uint8_t u2,
                  uint8_t u3, unsigned k);
    /* Set/accumulate the XOR of four sources. Destination must be disjoint
     * from all sources; the sources may repeat or overlap each other. */
    void (*xor_set4)(uint8_t *dst, const uint8_t *r0, const uint8_t *r1,
                     const uint8_t *r2, const uint8_t *r3, unsigned k);
    void (*xor_acc4)(uint8_t *dst, const uint8_t *r0, const uint8_t *r1,
                     const uint8_t *r2, const uint8_t *r3, unsigned k);
};

void oblas_get_impl(struct oblas_impl *impl);

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;

#ifdef __cplusplus
}
#endif

#endif /* OBLAS_LITE_H */
