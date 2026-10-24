#include "nanorq_core.h"
#include "partition.h"
#include "sopi.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

void test_partition() {
  printf("Testing partition...\n");
  struct partition p = partition_fill(1000, 3);
  assert(p.IL == 334);
  assert(p.IS == 333);
  assert(p.JL == 1);
  assert(p.JS == 2);

  p = partition_fill(100, 10);
  assert(p.IL == 0);
  assert(p.IS == 10);
  assert(p.JL == 0);
  assert(p.JS == 10);

  printf("Partition tests passed.\n");
}

void test_sopi() {
  printf("Testing SOPI...\n");
  rq_sopi sopi = {100, 50, 200, 10};

  // Single ESI mapping test
  uint32_t esi = sopi_get_esi_single(&sopi, 5);
  uint64_t expected_esi = (100ULL + 5ULL * 50ULL) % SOPI_N;
  assert(esi == expected_esi);

  // Large index that overflows 32-bit but stays within limits
  uint64_t large_i = 100000000ULL;
  uint64_t large_b = 300ULL;
  uint32_t val = sopi_mersenne_mod(sopi.a, large_i, large_b);
  uint64_t expected_val = (sopi.a + large_i * large_b) % SOPI_N;
  assert(val == expected_val);

  // Multi block mapping test
  uint32_t Z = 5;
  uint32_t out_sbn, out_esi;
  sopi_get_mapping_multi(&sopi, 12, Z, &out_sbn, &out_esi);

  uint64_t r = 12 / Z;
  uint64_t expected_multi_esi = (sopi.a + r * sopi.b) % SOPI_N;
  uint32_t expected_multi_sbn = (12 + sopi.c + r * sopi.d) % Z;
  assert(out_esi == expected_multi_esi);
  assert(out_sbn == expected_multi_sbn);

  printf("SOPI tests passed.\n");
}

static u32 choose_no_rows(void *arg, pc *W, u32 V0, u32 Vrows, u32 Srows,
                          u32 Vcols) {
  return Srows;
}

static void test_no_preconditioned_rows(void) {
  nanorq_core rq = {0};
  assert(nanorq_core_encoder_new(100, 0, &rq));
  size_t prep_len = nanorq_core_calculate_prepare_memory(&rq);
  size_t work_len = nanorq_core_calculate_work_memory(&rq);
  uint8_t *prep = malloc(prep_len), *work = malloc(work_len);
  assert(prep && work);
  assert(nanorq_core_prepare(&rq, prep, prep_len));
  nanorq_core_set_choose_callback(&rq, NULL, choose_no_rows);
  assert(nanorq_core_patch_matrix(&rq));
  assert(rq.W.i == 0);
  assert(nanorq_core_precalculate(&rq, work, work_len));
  free(work);
  free(prep);
}

int main() {
  test_partition();
  test_sopi();
  test_no_preconditioned_rows();
  printf("All utility tests passed successfully!\n");
  return 0;
}
