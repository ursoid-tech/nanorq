#include "nanorq_ops.h"

#include <string.h>

void ops_push(void *arg, u32 i, u32 j, u8 u) {
  schedule *S = (schedule *)arg;
  if (!S || !S->ops.a)
    return;
  sched_op op = {u, i, j};
  if (i == 0 && j == 0 && u == 0) {
    if (S->cpidx < 2) {
      S->cp[S->cpidx++] = S->ops.n;
    } else {
      S->overflowed = true;
    }
  } else {
    if (S->ops.n < S->ops.m) {
      S->ops.a[S->ops.n++] = op;
    } else {
      S->overflowed = true;
    }
  }
}

bool ops_run_pivoted(uint8_t *D, uint32_t stride, schedule *S) {
  if (!D || stride == 0 || !S || !S->ops.a || S->cpidx != 2 ||
      S->overflowed || S->cp[0] > S->cp[1] || S->cp[1] > S->ops.n)
    return false;
  nanorq_core_init();
  void (*row_ops)(u8 *, unsigned, const struct oblas_row_op *, size_t, int) =
      nanorq_oblas.row_ops;

  row_ops(D, stride, S->ops.a, S->cp[1], 0);
  if (S->cp[0] > 0)
    row_ops(D, stride, S->ops.a, S->cp[0], 1);
  row_ops(D, stride, S->ops.a + S->cp[1], S->ops.n - S->cp[1], 0);
  row_ops(D, stride, S->ops.a, S->cp[0], 0);
  return true;
}

/* Mark visited rows in the pivot vector while rotating each cycle. */
static void permute_rows(u8 *D, u32 stride, u32 *perm, u32 count) {
  const u32 visited = UINT32_C(1) << 31;
  for (u32 i = 0; i < count; i++) {
    u32 at = i;
    while (!(perm[at] & visited)) {
      u32 next = perm[at];
      if (next != i)
        nanorq_oblas.swap(D + (size_t)i * stride, D + (size_t)next * stride,
                          stride);
      perm[at] |= visited;
      at = next;
    }
  }
  for (u32 i = 0; i < count; i++)
    perm[i] &= ~visited;
}

void ops_run(nanorq_core *rq, uint8_t *D, uint32_t stride, schedule *S) {
  if (!rq || !rq->W.di.a || !rq->W.c.a || !ops_run_pivoted(D, stride, S))
    return;
  permute_rows(D, stride, rq->W.di.a, nanorq_core_get_pc_rows(rq));
  permute_rows(D, stride, rq->W.c.a, nanorq_core_get_pc_cols(rq));
}

static void mix_rows(nanorq_core *rq, u8 *D, u32 stride, u32 esi, u8 *ptr,
                     bool pivoted) {
  if (!rq || !D || stride == 0 || !ptr)
    return;
  uint32_t mix[GENC_MAX];
  uint32_t num = nanorq_core_get_packet_mix(rq, esi, mix, GENC_MAX);
  if (num == 0) {
    memset(ptr, 0, stride);
    return;
  }
  nanorq_core_init();

  /* resolve logical intermediate symbols through the inverting core */
  if (pivoted) {
    for (u32 it = 0; it < num; it++)
      mix[it] = uv_A(rq->W.d, uv_A(rq->W.ci, mix[it]));
  }

  for (u32 it = 0; it < num; it++) {
    if (ptr == D + (size_t)mix[it] * stride) {
      /* read every input byte before overwriting an aliased source row */
      for (u32 col = 0; col < stride; col++) {
        u8 value = 0;
        for (u32 n = 0; n < num; n++)
          value ^= D[(size_t)mix[n] * stride + col];
        ptr[col] = value;
      }
      return;
    }
  }

  u32 it = 1;
  if (num >= 4) {
    nanorq_oblas.xor_set4(
        ptr, D + (size_t)mix[0] * stride, D + (size_t)mix[1] * stride,
        D + (size_t)mix[2] * stride, D + (size_t)mix[3] * stride, stride);
    it = 4;
    for (; num - it >= 4; it += 4)
      nanorq_oblas.xor_acc4(ptr, D + (size_t)mix[it] * stride,
                            D + (size_t)mix[it + 1] * stride,
                            D + (size_t)mix[it + 2] * stride,
                            D + (size_t)mix[it + 3] * stride, stride);
  } else {
    memcpy(ptr, D + (size_t)mix[0] * stride, stride);
  }
  for (; it < num; it++)
    nanorq_oblas.axpy(ptr, D + (size_t)mix[it] * stride, 1, stride);
}

void ops_mix(nanorq_core *rq, uint8_t *D, uint32_t stride, u32 esi, u8 *ptr) {
  mix_rows(rq, D, stride, esi, ptr, false);
}

void ops_mix_pivoted(nanorq_core *rq, uint8_t *D, uint32_t stride, u32 esi,
                     u8 *ptr) {
  if (rq && rq->W.d.a && rq->W.ci.a)
    mix_rows(rq, D, stride, esi, ptr, true);
}

size_t ops_estimate_schedule_bytes(uint32_t K) {
  /* allocate 40 * k + 1000 to conservatively bound schedule length. */
  if ((size_t)K > (SIZE_MAX - 1000) / 40)
    return 0;
  size_t estimated_ops = 40 * (size_t)K + 1000;
  if (estimated_ops > SIZE_MAX / sizeof(sched_op))
    return 0;
  return estimated_ops * sizeof(sched_op);
}

bool schedule_init(schedule *S, void *buf, size_t buf_bytes) {
  if (!S || !buf || buf_bytes < sizeof(sched_op))
    return false;
  *S = (schedule){0};
  S->ops.a = (sched_op *)buf;
  S->ops.m = buf_bytes / sizeof(sched_op);
  return true;
}

#ifndef NANORQ_NO_LIBC
#include <stdbool.h>
#include <stdlib.h>
#endif

void nanorq_core_get_memory_reqs(uint32_t K, uint32_t overhead, uint32_t stride,
                                 struct nanorq_core_mem_reqs *reqs) {
  if (!reqs)
    return;
  *reqs = (struct nanorq_core_mem_reqs){0};
  nanorq_core rq = {0};
  if (!nanorq_core_encoder_new(K, overhead, &rq) || stride == 0 ||
      overhead > UINT32_MAX - K)
    return;
  reqs->prepare_bytes = nanorq_core_calculate_prepare_memory(&rq);
  reqs->work_bytes = nanorq_core_calculate_work_memory(&rq);
  size_t rows = nanorq_core_get_pc_rows(&rq);
  if (rows > SIZE_MAX / stride) {
    *reqs = (struct nanorq_core_mem_reqs){0};
    return;
  }
  reqs->matrix_bytes = rows * stride;
  reqs->schedule_bytes = ops_estimate_schedule_bytes(K + overhead);
}

#ifndef NANORQ_NO_LIBC

bool nanorq_core_encode_simple(uint8_t *src_data, uint32_t K, uint16_t T,
                               uint32_t num_repair, uint8_t *repair_out) {
  if (!src_data || !repair_out || T == 0 || (size_t)num_repair > SIZE_MAX / T)
    return false;
  nanorq_core rq = {0};
  if (!nanorq_core_encoder_new(K, 0, &rq))
    return false;
  /* Repair ESIs map to internal symbol IDs starting at Kprime. */
  if (num_repair > UINT32_MAX - rq.P.Kprime + 1)
    return false;

  struct nanorq_core_mem_reqs reqs;
  nanorq_core_get_memory_reqs(K, 0, T, &reqs);
  if (!reqs.prepare_bytes || !reqs.work_bytes || !reqs.matrix_bytes ||
      !reqs.schedule_bytes)
    return false;

  uint8_t *prep_mem = (uint8_t *)malloc(reqs.prepare_bytes);
  if (!prep_mem)
    return false;
  if (!nanorq_core_prepare(&rq, prep_mem, reqs.prepare_bytes)) {
    free(prep_mem);
    return false;
  }

  uint8_t *D = (uint8_t *)calloc(1, reqs.matrix_bytes);
  if (!D) {
    free(prep_mem);
    return false;
  }
  uint32_t SH = nanorq_core_get_pc_genc_offset(&rq);

  for (uint32_t i = 0; i < K; i++) {
    memcpy(D + ((size_t)SH + i) * T, src_data + (size_t)i * T, T);
  }

  uint8_t *work_mem = (uint8_t *)malloc(reqs.work_bytes);
  schedule S_enc = {0};
  void *sched_mem = malloc(reqs.schedule_bytes);
  if (!work_mem || !schedule_init(&S_enc, sched_mem, reqs.schedule_bytes)) {
    free(prep_mem);
    free(D);
    free(work_mem);
    free(sched_mem);
    return false;
  }

  nanorq_core_set_op_callback(&rq, &S_enc, ops_push);

  if (!nanorq_core_precalculate(&rq, work_mem, reqs.work_bytes) ||
      !ops_run_pivoted(D, T, &S_enc)) {
    free(prep_mem);
    free(D);
    free(work_mem);
    free(S_enc.ops.a);
    return false;
  }

  for (uint32_t i = 0; i < num_repair; i++) {
    ops_mix_pivoted(&rq, D, T, K + i, repair_out + (size_t)i * T);
  }

  free(prep_mem);
  free(D);
  free(work_mem);
  free(S_enc.ops.a);
  return true;
}

#endif /* NANORQ_NO_LIBC */
