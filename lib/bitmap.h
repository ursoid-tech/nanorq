#ifndef NANORQ_BITMAP_H
#define NANORQ_BITMAP_H

#include "uvec.h"

/* Internal C helper: callers supply distinct rows. */
static inline void bm_xor_raw(u32 *restrict ap, const u32 *restrict bp, u32 s) {
  for (u32 idx = 0; idx < s; idx++)
    ap[idx] ^= bp[idx];
}

#endif
