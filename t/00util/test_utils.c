#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif
#include "nanorq_core.h"
#include "nanorq_ops.h"
#include "partition.h"
#include "sopi.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#if !defined(MAP_ANONYMOUS) && defined(MAP_ANON)
#define MAP_ANONYMOUS MAP_ANON
#endif
#endif

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

#if SIZE_MAX > UINT32_MAX && defined(MAP_ANONYMOUS) && defined(PROT_NONE)
/* trace operations without touching GiB-sized rows */
static uint8_t *offset_matrix;
static size_t offset_call;
static const unsigned offset_stride = 1U << 30;
static const struct {
  char kind;
  unsigned i, j, u;
} offset_calls[] = {
    {'a', 4, 5, 1},                 /* forward */
    {'s', 5, 0, 2}, {'s', 5, 0, 2}, /* undo */
    {'a', 4, 5, 1}, {'a', 5, 4, 3}, /* backsolve */
    {'s', 4, 0, 7}, {'a', 4, 5, 1}, /* reapply */
    {'s', 5, 0, 2}, {'w', 0, 0, 0}, /* rows */
    {'w', 1, 1, 0}, {'w', 2, 2, 0}, {'w', 3, 3, 0}, {'w', 4, 5, 0},
    {'w', 4, 4, 0}, {'w', 0, 4, 0}, /* cols */
    {'w', 0, 0, 0}, {'w', 1, 1, 0}, {'w', 2, 2, 0}, {'w', 3, 3, 0},
};

static void check_offset_call(char kind, uint8_t *dst, uint8_t *src, unsigned u,
                              unsigned len) {
  assert(offset_call < sizeof(offset_calls) / sizeof(*offset_calls));
  assert(kind == offset_calls[offset_call].kind);
  assert((kind == 's') == (src == NULL));
  assert(u == offset_calls[offset_call].u && len == offset_stride);
  assert(dst == offset_matrix + (size_t)offset_calls[offset_call].i * len);
  if (src)
    assert(src == offset_matrix + (size_t)offset_calls[offset_call].j * len);
  offset_call++;
}

static void check_offset_axpy(uint8_t *dst, uint8_t *src, uint8_t u,
                              unsigned len) {
  check_offset_call('a', dst, src, u, len);
}

static void check_offset_scal(uint8_t *dst, uint8_t u, unsigned len) {
  check_offset_call('s', dst, NULL, u, len);
}

static void check_offset_swap(uint8_t *dst, uint8_t *src, unsigned len) {
  check_offset_call('w', dst, src, 0, len);
}
#endif

static void test_large_matrix_offsets(void) {
#if SIZE_MAX > UINT32_MAX && defined(MAP_ANONYMOUS) && defined(PROT_NONE)
  enum { STRIDE = 131072 };
  nanorq_core rq = {0};
  bool ok = nanorq_core_encoder_new(50000, 0, &rq);
  assert(ok);
  (void)ok;
  size_t bytes = (size_t)nanorq_core_get_pc_rows(&rq) * STRIDE;
  /* reserve address space; only mixed rows are mapped read-write */
  uint8_t *D = mmap(NULL, bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (D == MAP_FAILED) {
    puts("SKIP: large matrix offsets (virtual memory reservation unavailable)");
    return;
  }
  uint32_t mix[GENC_MAX];
  uint32_t num = nanorq_core_get_packet_mix(&rq, 0, mix, GENC_MAX);
  uint8_t expected = 0;
  uint8_t expected_tail = 0;
  uint8_t high_expected = 0;
  for (uint32_t i = 0; i < num; i++) {
    size_t offset = (size_t)mix[i] * STRIDE;
    if (mprotect(D + offset, STRIDE, PROT_READ | PROT_WRITE) != 0) {
      int unmapped = munmap(D, bytes);
      assert(unmapped == 0);
      (void)unmapped;
      puts("SKIP: large matrix offsets (mprotect unavailable)");
      return;
    }
    uint8_t val = (uint8_t)(((mix[i] * 2654435761U) >> 24) | 1U);
    if (offset > UINT32_MAX) {
      if ((high_expected ^ val) == 0)
        val ^= 2U;
      high_expected ^= val;
    }
    uint8_t tail = (uint8_t)(i + 1);
    D[offset] = val;
    D[offset + STRIDE - 1] = tail;
    expected ^= val;
    expected_tail ^= tail;
  }
  assert(high_expected != 0);
  uint8_t *out = malloc(STRIDE);
  assert(out);
  memset(out, 0xa5, STRIDE);
  ops_mix(&rq, D, STRIDE, 0, out);
  assert(out[0] == expected);
  for (unsigned i = 1; i < STRIDE - 1; i++)
    assert(out[i] == 0);
  assert(out[STRIDE - 1] == expected_tail);
  free(out);
  (void)expected;
  (void)expected_tail;
  (void)high_expected;

  /* six synthetic rows with 1 GiB stride cross the 32-bit offset boundary */
  assert(bytes >= (size_t)6 * offset_stride);
  uint32_t rows[] = {0, 1, 2, 3, 5, 4};
  uint32_t cols[] = {4, 1, 2, 3, 0};
  const uint32_t saved_rows[] = {0, 1, 2, 3, 5, 4};
  const uint32_t saved_cols[] = {4, 1, 2, 3, 0};
  nanorq_core scheduled = {.P.L = 5, .overhead = 1};
  scheduled.W.di.a = rows;
  scheduled.W.c.a = cols;
  sched_op operations[] = {{1, 4, 5}, {0, 5, 2}, {3, 5, 4}, {0, 4, 7}};
  schedule S = {
      .cp = {2, 2},
      .cpidx = 2,
      .ops = {.a = operations, .n = 4, .m = 4},
  };
  struct oblas_impl saved = nanorq_oblas;
  nanorq_oblas.axpy = check_offset_axpy;
  nanorq_oblas.scal = check_offset_scal;
  nanorq_oblas.swap = check_offset_swap;
  offset_matrix = D;
  offset_call = 0;
  ops_run(&scheduled, D, offset_stride, &S);
  nanorq_oblas = saved;
  assert(offset_call == sizeof(offset_calls) / sizeof(*offset_calls));
  assert(memcmp(rows, saved_rows, sizeof(rows)) == 0);
  assert(memcmp(cols, saved_cols, sizeof(cols)) == 0);
  (void)saved_rows;
  (void)saved_cols;
  int unmapped = munmap(D, bytes);
  assert(unmapped == 0);
  (void)unmapped;
  puts("Large matrix offset tests passed.");
#else
  puts("SKIP: large matrix offsets require 64-bit size_t and anonymous mmap");
#endif
}

int main() {
  test_partition();
  test_sopi();
  test_high_overhead_mem_reqs();
  test_prepared_workspace();
  test_no_preconditioned_rows();
  test_large_matrix_offsets();
  printf("All utility tests passed successfully!\n");
  return 0;
}
