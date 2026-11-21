#include "nanorq_core.h"

#ifndef NANORQ_OPS_H
#define NANORQ_OPS_H

typedef struct oblas_row_op sched_op;

typedef struct {
  sched_op *a;
  size_t n;
  size_t m;
} oplist;

typedef struct {
  u32 cp[2];
  u32 cpidx;
  bool overflowed;
  oplist ops;
} schedule;

/* initialise schedule with caller memory. buf needs
 * ops_estimate_schedule_bytes(k). */
bool schedule_init(schedule *S, void *buf, size_t buf_bytes);

void ops_push(void *arg, u32 i, u32 j, u8 u);
/* Replay requires the schedule and core preparation memory to remain live.
 * The inversion workspace may be freed or reused after precalculate.
 * D remains in physical row order; use ops_mix with this same core. */
void ops_run(nanorq_core *rq, uint8_t *D, uint32_t stride, schedule *S);

/* Optional replay copy size, or zero when the optimization is not useful.
 * This is not a minimum memory requirement for ops_run. */
size_t ops_replay_scratch_bytes(nanorq_core *rq, uint32_t stride);

/* Scratch is borrowed only for this call and its contents are overwritten.
 * It must not overlap the schedule or core preparation memory. The completed
 * inversion workspace may be explicitly reused here. Null, undersized, or
 * D-overlapping scratch falls back to ordinary replay. */
void ops_run_with_scratch(nanorq_core *rq, uint8_t *D, uint32_t stride,
                          schedule *S, void *scratch, size_t scratch_bytes);
/* Output must be disjoint from D or exactly one complete matrix row.
 * Partial overlap is unsupported. */
void ops_mix(nanorq_core *rq, uint8_t *D, uint32_t stride, u32 esi, u8 *ptr);
size_t ops_estimate_schedule_bytes(uint32_t K);

/* mem reqs are defined in nanorq_core.h */

bool nanorq_core_encode_simple(uint8_t *src_data, uint32_t K, uint16_t T,
                               uint32_t num_repair, uint8_t *repair_out);

#endif
