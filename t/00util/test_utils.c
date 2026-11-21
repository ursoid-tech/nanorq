#include "nanorq_core.h"
#include "nanorq_ops.h"
#include "partition.h"
#include "sopi.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

  /* single ESI mapping test */
  uint32_t esi = sopi_get_esi_single(&sopi, 5);
  uint64_t expected_esi = (100ULL + 5ULL * 50ULL) % SOPI_N;
  assert(esi == expected_esi);

  /* large index that stays within 64-bit product */
  uint64_t large_i = 100000000ULL;
  uint64_t large_b = 300ULL;
  uint32_t val = sopi_mersenne_mod(sopi.a, large_i, large_b);
  uint64_t expected_val = (sopi.a + large_i * large_b) % SOPI_N;
  assert(val == expected_val);

  /* 64-bit i * B overflow and unreduced 32-bit inputs */
  uint64_t huge_i = (1ULL << 50) + 1234567ULL;
  uint64_t huge_b = 0xFFFFFFFFULL;
  uint64_t huge_a = 0xFFFFFFFFULL;
  uint32_t huge_val = sopi_mersenne_mod(huge_a, huge_i, huge_b);
  uint64_t exp_huge =
      ((huge_a % SOPI_N) + ((huge_i % SOPI_N) * (huge_b % SOPI_N)) % SOPI_N) %
      SOPI_N;
  assert(huge_val == exp_huge);
  assert(sopi_mersenne_mod(UINT64_MAX, UINT64_MAX, UINT64_MAX) < SOPI_N);

  /* multi block mapping test */
  uint32_t Z = 5;
  uint32_t out_sbn, out_esi;
  sopi_get_mapping_multi(&sopi, 12, Z, &out_sbn, &out_esi);

  uint64_t r = 12 / Z;
  uint64_t expected_multi_esi = (sopi.a + r * sopi.b) % SOPI_N;
  uint32_t expected_multi_sbn = (12 + sopi.c + r * (uint64_t)sopi.d) % Z;
  assert(out_esi == expected_multi_esi);
  assert(out_sbn == expected_multi_sbn);

  /* huge index multi-block sbn wrap check */
  rq_sopi big_sopi = {17, 0x7FFFFFF0U, 0xFFFFFFFAU, 0xFFFFFFFBU};
  sopi_get_mapping_multi(&big_sopi, UINT64_MAX - 3ULL, 7, &out_sbn, &out_esi);
  assert(out_sbn < 7 && out_esi < SOPI_N);

  printf("SOPI tests passed.\n");
}

static void test_high_overhead_mem_reqs(void) {
  enum { K = 10, OH = 200, STRIDE = 64 };
  struct nanorq_core_mem_reqs reqs = {0};
  nanorq_core_get_memory_reqs(K, OH, STRIDE, &reqs);
  assert(reqs.schedule_bytes >= ops_estimate_schedule_bytes(K + OH));
  nanorq_core rq = {0};
  assert(nanorq_core_encoder_new(K, OH, &rq));
  uint8_t *prep = malloc(reqs.prepare_bytes);
  uint8_t *work = malloc(reqs.work_bytes);
  void *sched_buf = malloc(reqs.schedule_bytes);
  schedule S = {0};
  assert(prep && work && sched_buf);
  assert(nanorq_core_prepare(&rq, prep, reqs.prepare_bytes));
  assert(schedule_init(&S, sched_buf, reqs.schedule_bytes));
  nanorq_core_set_op_callback(&rq, &S, ops_push);
  assert(nanorq_core_precalculate(&rq, work, reqs.work_bytes));
  assert(!S.overflowed && S.cpidx == 2);
  free(sched_buf);
  free(work);
  free(prep);
}

void test_prepared_workspace() {
  const unsigned ks[] = {100, 1000, 5000};
  for (unsigned t = 0; t < sizeof(ks) / sizeof(*ks); t++) {
    nanorq_core rq;
    memset(&rq, 0xa5, sizeof(rq));
    assert(nanorq_core_encoder_new(ks[t], 0, &rq));
    size_t bound = nanorq_core_calculate_work_memory(&rq);
    size_t prep_len = nanorq_core_calculate_prepare_memory(&rq);
    uint8_t *prep = malloc(prep_len);
    assert(prep && nanorq_core_prepare(&rq, prep, prep_len));
    size_t exact = nanorq_core_calculate_prepared_work_memory(&rq);
    assert(nanorq_core_calculate_work_memory(&rq) == bound);
    assert(exact > 0 && exact < bound);
    uint8_t *work = malloc(exact + 128);
    assert(work);
    memset(work, 0xa5, exact + 128);
    assert(nanorq_core_precalculate(&rq, work + 1, exact));
    assert(work[0] == 0xa5);
    for (size_t i = exact + 1; i < exact + 128; i++)
      assert(work[i] == 0xa5);
    /* new equation matrix invalidates prepared width */
    assert(nanorq_core_generate(&rq, prep, prep_len));
    assert(nanorq_core_calculate_work_memory(&rq) == bound);
    assert(nanorq_core_calculate_prepared_work_memory(&rq) == 0);
    nanorq_core_replace_symbol(&rq, 0, 0);
    assert(nanorq_core_patch_matrix(&rq));
    assert(nanorq_core_calculate_prepared_work_memory(&rq) == exact);
    nanorq_core_replace_symbol(&rq, 0, 0);
    assert(nanorq_core_calculate_prepared_work_memory(&rq) == 0);
    assert(nanorq_core_encoder_new(ks[t], 0, &rq));
    assert(nanorq_core_calculate_work_memory(&rq) == bound);
    assert(nanorq_core_calculate_prepared_work_memory(&rq) == 0);
    free(work);
    free(prep);
  }
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
  test_high_overhead_mem_reqs();
  test_prepared_workspace();
  test_no_preconditioned_rows();
  printf("All utility tests passed successfully!\n");
  return 0;
}
