#ifndef OBLAS_LITE_H
#define OBLAS_LITE_H

#include <stdint.h>
#include <stddef.h>

#include "gf2_8_tables.h"
#include "oblas_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* elementary row op: i ^= u*j when u != 0, else scale row i by (uint8_t)j */
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
    /* run the recorded row operations in forward or reverse order */
    void (*row_ops)(uint8_t *matrix, unsigned stride,
                    const struct oblas_row_op *ops, size_t count,
                    int reverse);
    /* set/accumulate 4-way XOR into disjoint dst; sources may alias each other */
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
