#include "sopi.h"

static inline uint64_t fold_m31(uint64_t x) {
  x = (x & 0x7FFFFFFFULL) + (x >> 31);
  x = (x & 0x7FFFFFFFULL) + (x >> 31);
  if (x >= SOPI_N)
    x -= SOPI_N;
  return x;
}

uint32_t sopi_mersenne_mod(uint64_t A, uint64_t i, uint64_t B) {
  uint64_t a = fold_m31(A);
  uint64_t im = fold_m31(i);
  uint64_t bm = fold_m31(B);
  uint64_t c = a + fold_m31(im * bm);
  if (c >= SOPI_N)
    c -= SOPI_N;
  return (uint32_t)c;
}

uint32_t sopi_get_esi_single(const rq_sopi *sopi, uint64_t i) {
  if (!sopi)
    return 0;
  return sopi_mersenne_mod(sopi->a, i, sopi->b);
}

void sopi_get_mapping_multi(const rq_sopi *sopi, uint64_t i, uint32_t Z,
                            uint32_t *out_sbn, uint32_t *out_esi) {
  if (!sopi || Z == 0)
    return;
  uint64_t r = i / Z;
  if (out_esi) {
    *out_esi = sopi_mersenne_mod(sopi->a, r, sopi->b);
  }
  if (out_sbn) {
    uint64_t sbn = (i % Z) + (sopi->c % Z) + ((r % Z) * (sopi->d % Z)) % Z;
    *out_sbn = (uint32_t)(sbn % Z);
  }
}
