#include "nanorq_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition, message)                                              \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "FAIL: %s (line %d)\n", message, __LINE__);              \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

struct fixture {
  nanorq_core rq;
  schedule schedule;
  uint8_t *prepare, *work;
  void *schedule_mem;
  size_t work_bytes;
};

static void *checked_malloc(size_t bytes) {
  void *ptr = malloc(bytes);
  CHECK(ptr != NULL, "allocate test memory");
  return ptr;
}

static void fixture_init(struct fixture *f, uint32_t k, uint32_t overhead) {
  memset(f, 0, sizeof(*f));
  CHECK(nanorq_core_encoder_new(k, overhead, &f->rq), "initialize core");
  size_t prep_bytes = nanorq_core_calculate_prepare_memory(&f->rq) + 64;
  f->prepare = checked_malloc(prep_bytes);
  CHECK(nanorq_core_prepare(&f->rq, f->prepare, prep_bytes), "prepare core");
  for (uint32_t extra = 0; extra < overhead; extra++)
    nanorq_core_replace_symbol(&f->rq, f->rq.P.Kprime + extra, k + extra);
  if (overhead > 0)
    CHECK(nanorq_core_patch_matrix(&f->rq), "patch overhead core");
  size_t schedule_bytes = ops_estimate_schedule_bytes(k + overhead);
  f->schedule_mem = checked_malloc(schedule_bytes);
  CHECK(schedule_init(&f->schedule, f->schedule_mem, schedule_bytes),
        "initialize schedule");
  f->work_bytes = nanorq_core_calculate_prepared_work_memory(&f->rq) + 64;
  f->work = checked_malloc(f->work_bytes);
  nanorq_core_set_op_callback(&f->rq, &f->schedule, ops_push);
  CHECK(nanorq_core_precalculate(&f->rq, f->work, f->work_bytes),
        "invert core");
  CHECK(f->schedule.cpidx == 2 && !f->schedule.overflowed, "complete schedule");
}

static void fixture_free(struct fixture *f) {
  free(f->work);
  free(f->schedule_mem);
  free(f->prepare);
}

/* independent of OBL tables and kernels */
static uint8_t multiply(uint8_t a, uint8_t b) {
  unsigned result = 0, x = a;
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

static void reference_forward(uint8_t *d, uint32_t stride, const schedule *s,
                              size_t begin, size_t end) {
  for (size_t opidx = begin; opidx < end; opidx++) {
    const sched_op *op = &s->ops.a[opidx];
    uint8_t *dst = d + (size_t)op->i * stride;
    if (op->u) {
      const uint8_t *src = d + (size_t)op->j * stride;
      for (uint32_t byte = 0; byte < stride; byte++)
        dst[byte] ^= multiply(op->u, src[byte]);
    } else {
      for (uint32_t byte = 0; byte < stride; byte++)
        dst[byte] = multiply((uint8_t)op->j, dst[byte]);
    }
  }
}

static void reference_replay(uint8_t *d, uint32_t stride, const schedule *s) {
  reference_forward(d, stride, s, 0, s->cp[1]);
  for (size_t opidx = s->cp[0]; opidx-- > 0;)
    reference_forward(d, stride, s, opidx, opidx + 1);
  reference_forward(d, stride, s, s->cp[1], s->ops.n);
  reference_forward(d, stride, s, 0, s->cp[0]);
}

static void fill_bytes(uint8_t *d, size_t bytes) {
  for (size_t i = 0; i < bytes; i++)
    d[i] = (uint8_t)(i * 73U + i / 13U + 17U);
}

static bool all_bytes(const uint8_t *d, size_t bytes, uint8_t value) {
  for (size_t i = 0; i < bytes; i++)
    if (d[i] != value)
      return false;
  return true;
}

static void compare_schedule(nanorq_core *rq, schedule *s, uint32_t stride) {
  size_t bytes = (size_t)nanorq_core_get_pc_rows(rq) * stride;
  uint8_t *storage = checked_malloc(bytes + 16);
  uint8_t *reference = checked_malloc(bytes + 16);
  fill_bytes(storage, bytes + 16);
  memcpy(reference, storage, bytes + 16);
  /* offset base to test unaligned SIMD rows and tails */
  CHECK(ops_run_pivoted(storage + 5, stride, s), "replay in pivot order");
  reference_replay(reference + 5, stride, s);
  CHECK(memcmp(storage, reference, bytes + 16) == 0,
        "replay including guards matches independent GF256 arithmetic");
  free(reference);
  free(storage);
}

static void check_mixing(struct fixture *f, uint32_t stride) {
  nanorq_core *rq = &f->rq;
  size_t bytes = (size_t)nanorq_core_get_pc_rows(rq) * stride;
  uint8_t *d = checked_malloc(bytes);
  uint8_t *logical = checked_malloc(bytes);
  uint8_t *aliased = checked_malloc(bytes);
  uint8_t *output = checked_malloc(stride + 16);
  uint8_t *reference = checked_malloc(stride + 16);
  fill_bytes(d, bytes);
  /* materialize logical matrix using c and d independently of ci */
  bool permuted = false;
  for (uint32_t pos = 0; pos < rq->P.L; pos++) {
    uint32_t col = rq->W.c.a[pos], row = rq->W.d.a[pos];
    memcpy(logical + (size_t)col * stride, d + (size_t)row * stride, stride);
    permuted |= col != row;
  }
  CHECK(permuted, "mixing fixture must have a nonidentity permutation");
  for (uint32_t esi = 0; esi < rq->P.K + 40; esi++) {
    uint32_t mix[GENC_MAX];
    uint32_t count = nanorq_core_get_packet_mix(rq, esi, mix, GENC_MAX);
    memset(output, 0xa5, stride + 16);
    memset(reference, 0xa5, stride + 16);
    memset(reference + 3, 0, stride);
    for (uint32_t idx = 0; idx < count; idx++)
      for (uint32_t byte = 0; byte < stride; byte++)
        reference[3 + byte] ^= logical[(size_t)mix[idx] * stride + byte];
    ops_mix_pivoted(rq, d, stride, esi, output + 3);
    CHECK(memcmp(output, reference, stride + 16) == 0,
          "virtual row mixing and tails match materialized logical matrix");
    for (unsigned which = 0; which < 2; which++) {
      uint32_t col = mix[which ? count - 1 : 0], row = 0;
      for (uint32_t pos = 0; pos < rq->P.L; pos++)
        if (rq->W.c.a[pos] == col)
          row = rq->W.d.a[pos];
      memcpy(aliased, d, bytes);
      size_t offset = (size_t)row * stride;
      ops_mix_pivoted(rq, aliased, stride, esi, aliased + offset);
      CHECK(memcmp(aliased + offset, reference + 3, stride) == 0,
            "mixing may overwrite the first or a later input row");
      CHECK(memcmp(aliased, d, offset) == 0 &&
                memcmp(aliased + offset + stride, d + offset + stride,
                       bytes - offset - stride) == 0,
            "aliased mixing changes only its destination row");
    }
  }
  free(reference);
  free(output);
  free(aliased);
  free(logical);
  free(d);
}

/* The public replay matrix can outlive the schedule and prepare arena. */
static void canonical_replay(void) {
  enum { K = 23, STRIDE = 17 };
  struct fixture f;
  fixture_init(&f, K, 0);
  size_t bytes = (size_t)nanorq_core_get_pc_rows(&f.rq) * STRIDE;
  uint8_t *d = checked_malloc(bytes);
  uint8_t src[K][STRIDE], output[STRIDE];
  fill_bytes((uint8_t *)src, sizeof(src));
  nanorq_core_init_matrix(&f.rq, d, STRIDE);
  for (u32 esi = 0; esi < K; esi++)
    nanorq_core_place_symbol(&f.rq, d, STRIDE, esi, src[esi], STRIDE);
  ops_run(&f.rq, d, STRIDE, &f.schedule);
  fixture_free(&f);

  nanorq_core other;
  CHECK(nanorq_core_encoder_new(K, 0, &other), "fresh mixing core");
  for (u32 esi = 0; esi < K; esi++) {
    u32 mix[GENC_MAX];
    u32 count = nanorq_core_get_packet_mix(&other, esi, mix, GENC_MAX);
    memset(output, 0, sizeof(output));
    for (u32 i = 0; i < count; i++)
      for (u32 j = 0; j < STRIDE; j++)
        output[j] ^= d[(size_t)mix[i] * STRIDE + j];
    CHECK(memcmp(output, src[esi], STRIDE) == 0, "canonical intermediate rows");
    ops_mix(&f.rq, d, STRIDE, esi, output);
    CHECK(memcmp(output, src[esi], STRIDE) == 0, "mix after prepare arena freed");
    ops_mix(&other, d, STRIDE, esi, output);
    CHECK(memcmp(output, src[esi], STRIDE) == 0, "mix with a fresh core");
  }
  free(d);
}

static void canonical_mixing_before_inversion(void) {
  enum { STRIDE = 65 };
  nanorq_core rq;
  /* constructors must establish deterministic state without caller zeroing */
  memset(&rq, 0xa5, sizeof(rq));
  CHECK(nanorq_core_encoder_new(10, 0, &rq), "canonical mixing core");
  CHECK(!nanorq_core_patch_matrix(&rq), "unprepared core rejects patch");
  size_t prep_bytes = nanorq_core_calculate_prepare_memory(&rq);
  uint8_t *prep_mem = checked_malloc(prep_bytes);
  size_t bytes = (size_t)nanorq_core_get_pc_rows(&rq) * STRIDE;
  uint8_t *d = checked_malloc(bytes);
  uint8_t actual[STRIDE], expected[STRIDE];
  fill_bytes(d, bytes);
  for (unsigned stage = 0; stage < 3; stage++) {
    if (stage == 1)
      CHECK(nanorq_core_generate(&rq, prep_mem, prep_bytes), "generate core");
    else if (stage == 2)
      CHECK(nanorq_core_prepare(&rq, prep_mem, prep_bytes), "prepare core");
    for (uint32_t esi = 0; esi < rq.P.K + 20; esi++) {
      uint32_t mix[GENC_MAX];
      uint32_t count = nanorq_core_get_packet_mix(&rq, esi, mix, GENC_MAX);
      memset(expected, 0, sizeof(expected));
      for (uint32_t idx = 0; idx < count; idx++)
        for (uint32_t byte = 0; byte < STRIDE; byte++)
          expected[byte] ^= d[(size_t)mix[idx] * STRIDE + byte];
      ops_mix(&rq, d, STRIDE, esi, actual);
      CHECK(memcmp(actual, expected, STRIDE) == 0,
            "uninverted core mixes canonical rows");
    }
  }
  free(d);
  free(prep_mem);
}

static void small_replay_and_mixing(void) {
  static const uint32_t strides[] = {1,  2,  7,  15,  16,  17,  31,  32,  33,
                                     63, 64, 65, 127, 128, 129, 257, 1281};
  struct fixture f;
  fixture_init(&f, 64, 2);
  /* callers may immediately repurpose or free inversion workspace */
  memset(f.work, 0xa5, f.work_bytes);
  compare_schedule(&f.rq, &f.schedule, 65);
  CHECK(all_bytes(f.work, f.work_bytes, 0xa5),
        "replay must not access repurposed inversion workspace");
  free(f.work);
  f.work = NULL;
  for (size_t idx = 0; idx < sizeof(strides) / sizeof(strides[0]); idx++) {
    compare_schedule(&f.rq, &f.schedule, strides[idx]);
    check_mixing(&f, strides[idx]);
  }
  fixture_free(&f);

  /* test 4- and 2-way batches, repeated destinations, scales, and splits */
  sched_op operations[] = {{1, 1, 0},   {13, 2, 0},  {1, 0, 0},    {0, 4, 99},
                           {1, 5, 0},   {1, 6, 0},   {1, 7, 0},    {1, 8, 0},
                           {1, 8, 0},   {1, 0, 8},   {73, 9, 1},   {73, 10, 1},
                           {73, 11, 1}, {73, 12, 1}, {255, 13, 2}, {255, 14, 2},
                           {2, 14, 2},  {0, 4, 99},  {1, 6, 5},    {1, 5, 6},
                           {19, 15, 3}, {19, 16, 3}, {19, 15, 3},  {19, 17, 3}};
  schedule s = {.cp = {5, 17},
                .cpidx = 2,
                .ops = {.a = operations,
                        .n = sizeof(operations) / sizeof(operations[0])}};
  nanorq_core rq = {0};
  CHECK(nanorq_core_encoder_new(10, 0, &rq), "synthetic schedule core");
  for (size_t idx = 0; idx < sizeof(strides) / sizeof(strides[0]); idx++)
    compare_schedule(&rq, &s, strides[idx]);
}

static void reject_invalid_checkpoints(void) {
  sched_op operation = {1, 1, 0};
  schedule s = {.cp = {1, 0}, .cpidx = 2, .ops = {.a = &operation, .n = 1}};
  uint8_t d[2] = {7, 11};
  CHECK(!ops_run_pivoted(d, 1, &s), "reject invalid schedule");
  CHECK(d[0] == 7 && d[1] == 11, "reject reversed checkpoints");
  s.cp[0] = 0;
  s.cp[1] = 2;
  CHECK(!ops_run_pivoted(d, 1, &s), "reject invalid schedule");
  CHECK(d[0] == 7 && d[1] == 11, "reject checkpoint past schedule");
}

int main(void) {
  canonical_replay();
  canonical_mixing_before_inversion();
  small_replay_and_mixing();
  reject_invalid_checkpoints();
  puts("schedule replay regressions passed");
  return 0;
}
