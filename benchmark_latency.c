#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "nanorq.h"

/* back-to-back host processing without network simulation */
#define SYMBOL_SIZE 1200
#define REPAIR_SYMBOLS 4
#define DEFAULT_ITERATIONS 1000
#define MAX_ITERATIONS 1000000

struct sample {
  uint64_t encode;
  uint64_t decode;
  uint64_t total;
};

struct context {
  nanorq *encoder;
  nanorq *decoder;
  struct ioctx *input;
  struct ioctx *output;
};

static void fail(const char *message) {
  fprintf(stderr, "benchmark_latency: %s\n", message);
  exit(EXIT_FAILURE);
}

static uint64_t monotonic_ns(void) {
  struct timespec now;
  if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
    fail("clock_gettime failed");
  return (uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec;
}

static void init_encoder(struct context *ctx, uint8_t *payload, size_t len,
                         uint16_t k) {
  ctx->encoder = nanorq_encoder_new_ex(len, SYMBOL_SIZE, k, 1, 8);
  ctx->input = ioctx_from_mem_ro(payload, len);
  if (!ctx->encoder || !ctx->input)
    fail("encoder allocation failed");
  if (nanorq_blocks(ctx->encoder) != 1 ||
      nanorq_block_symbols(ctx->encoder, 0) != k ||
      nanorq_symbol_size(ctx->encoder) != SYMBOL_SIZE)
    fail("unexpected encoder geometry");
}

static void init_decoder(struct context *ctx, uint8_t *decoded, size_t len,
                         uint64_t common, uint32_t scheme) {
  ctx->decoder = nanorq_decoder_new(common, scheme);
  ctx->output = ioctx_from_mem(decoded, len);
  if (!ctx->decoder || !ctx->output)
    fail("decoder allocation failed");
}

static void destroy_encoder(struct context *ctx) {
  ctx->input->destroy(ctx->input);
  nanorq_free(ctx->encoder);
  ctx->input = NULL;
  ctx->encoder = NULL;
}

static void destroy_decoder(struct context *ctx) {
  ctx->output->destroy(ctx->output);
  nanorq_free(ctx->decoder);
  ctx->output = NULL;
  ctx->decoder = NULL;
}

/* vary every symbol per iteration so reuse cannot hide stale data */
static void fill_payload(uint8_t *payload, size_t len, uint64_t iteration) {
  uint64_t state = iteration + UINT64_C(0x9e3779b97f4a7c15);
  for (size_t i = 0; i < len; i++) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    payload[i] = (uint8_t)((state * UINT64_C(2685821657736338717)) >> 56);
  }
}

static struct sample run_block(struct context *ctx, int reuse, uint16_t k,
                               uint8_t *payload, uint8_t *decoded,
                               uint8_t *packets, size_t iteration) {
  size_t len = (size_t)k * SYMBOL_SIZE;
  size_t missing = iteration % k;
  fill_payload(payload, len, iteration);
  memset(decoded, 0xa5, len);

  uint64_t begin = monotonic_ns();
  if (reuse)
    nanorq_encoder_reset(ctx->encoder, 0);
  else
    init_encoder(ctx, payload, len, k);

  uint64_t common = nanorq_oti_common(ctx->encoder);
  uint32_t scheme = nanorq_oti_scheme_specific(ctx->encoder);
  /* encode all symbols including the dropped source symbol */
  for (uint32_t esi = 0; esi < (uint32_t)k + REPAIR_SYMBOLS; esi++) {
    if (nanorq_encode(ctx->encoder, packets + (size_t)esi * SYMBOL_SIZE, esi, 0,
                      ctx->input) != SYMBOL_SIZE)
      fail("symbol encoding failed");
  }
  if (!reuse)
    destroy_encoder(ctx);
  uint64_t encoded = monotonic_ns();

  if (reuse)
    nanorq_encoder_reset(ctx->decoder, 0);
  else
    init_decoder(ctx, decoded, len, common, scheme);

  for (uint32_t esi = 0; esi < (uint32_t)k + REPAIR_SYMBOLS; esi++) {
    if (esi == missing)
      continue;
    if (nanorq_decoder_add_symbol(
            ctx->decoder, packets + (size_t)esi * SYMBOL_SIZE,
            nanorq_tag(0, esi), ctx->output) == NANORQ_SYM_ERR)
      fail("symbol ingestion failed");
  }
  if (!nanorq_repair_block(ctx->decoder, ctx->output, 0))
    fail("fixed repair budget did not recover the block");
  if (!reuse)
    destroy_decoder(ctx);
  uint64_t end = monotonic_ns();

  /* verify outside timed region */
  if (memcmp(payload, decoded, len) != 0)
    fail("decoded bytes differ from this iteration's payload");

  return (struct sample){encoded - begin, end - encoded, end - begin};
}

static int compare_u64(const void *left, const void *right) {
  uint64_t a = *(const uint64_t *)left;
  uint64_t b = *(const uint64_t *)right;
  return (a > b) - (a < b);
}

/* nearest-rank percentile in [1, count] */
static double percentile_us(const uint64_t *sorted, size_t count,
                            size_t percentile) {
  size_t rank = (count * percentile + 99) / 100;
  return (double)sorted[rank - 1] / 1000.0;
}

static void report(const char *mode, uint16_t k, const struct sample *samples,
                   size_t count) {
  uint64_t *times = malloc(count * sizeof(*times));
  if (!times)
    fail("statistics allocation failed");
  printf("%s,%u,%u,%u,%zu", mode, k, SYMBOL_SIZE, REPAIR_SYMBOLS, count);
  for (int metric = 0; metric < 3; metric++) {
    for (size_t i = 0; i < count; i++)
      times[i] = metric == 0   ? samples[i].encode
                 : metric == 1 ? samples[i].decode
                               : samples[i].total;
    qsort(times, count, sizeof(*times), compare_u64);
    printf(",%.3f,%.3f,%.3f", percentile_us(times, count, 50),
           percentile_us(times, count, 95), percentile_us(times, count, 99));
  }
  putchar('\n');
  free(times);
}

static void benchmark(uint16_t k, size_t iterations, int reuse) {
  size_t len = (size_t)k * SYMBOL_SIZE;
  uint8_t *payload = malloc(len);
  uint8_t *decoded = malloc(len);
  uint8_t *packets = malloc(((size_t)k + REPAIR_SYMBOLS) * SYMBOL_SIZE);
  struct sample *samples = malloc(iterations * sizeof(*samples));
  if (!payload || !decoded || !packets || !samples)
    fail("benchmark buffer allocation failed");

  struct context ctx = {0};
  if (reuse) {
    init_encoder(&ctx, payload, len, k);
    init_decoder(&ctx, decoded, len, nanorq_oti_common(ctx.encoder),
                 nanorq_oti_scheme_specific(ctx.encoder));
    /* warm up lazy allocations without nanorq_precalculate */
    (void)run_block(&ctx, 1, k, payload, decoded, packets, 0);
  }

  for (size_t i = 0; i < iterations; i++)
    samples[i] = run_block(&ctx, reuse, k, payload, decoded, packets, i + 1);

  if (reuse) {
    destroy_encoder(&ctx);
    destroy_decoder(&ctx);
  }
  report(reuse ? "reuse" : "cold", k, samples, iterations);
  free(samples);
  free(packets);
  free(decoded);
  free(payload);
}

int main(int argc, char **argv) {
  size_t iterations = DEFAULT_ITERATIONS;
  if (argc > 2) {
    fprintf(stderr, "usage: %s [iterations: 1..%u]\n", argv[0], MAX_ITERATIONS);
    return EXIT_FAILURE;
  }
  if (argc == 2) {
    char *end = NULL;
    errno = 0;
    unsigned long value = strtoul(argv[1], &end, 10);
    if (errno || argv[1][0] < '0' || argv[1][0] > '9' || *end || value == 0 ||
        value > MAX_ITERATIONS)
      fail("iterations must be an integer from 1 to 1000000");
    iterations = (size_t)value;
  }

  puts("# Back-to-back host latency in microseconds; one source lost, four "
       "repairs sent.");
  puts("# cold includes object/io construction and destruction; reuse includes "
       "resets,");
  puts("# excludes one-time setup/warmup, and does not call "
       "nanorq_precalculate.");
  puts("# Both include source/repair encoding and packet ingestion. Payload "
       "generation,");
  puts("# caller buffers, verification, transport, and block-fill delay are "
       "excluded.");
  puts("mode,K,T,repairs,iterations,encode_p50_us,encode_p95_us,encode_p99_us,"
       "decode_p50_us,decode_p95_us,decode_p99_us,total_p50_us,total_p95_us,"
       "total_p99_us");
  const uint16_t sizes[] = {1, 4, 8, 16, 32, 64};
  for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
    benchmark(sizes[i], iterations, 0);
    benchmark(sizes[i], iterations, 1);
  }
  return EXIT_SUCCESS;
}
