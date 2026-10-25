#include "nanorq.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* compile high-level code directly to inject scratch allocation faults */
static unsigned alloc_fail_at, alloc_calls;
static void *api_test_malloc(size_t bytes) {
  if (alloc_fail_at && ++alloc_calls == alloc_fail_at)
    return NULL;
  return malloc(bytes);
}
static void *api_test_realloc(void *ptr, size_t bytes) {
  if (alloc_fail_at && ++alloc_calls == alloc_fail_at)
    return NULL;
  return realloc(ptr, bytes);
}
#define malloc api_test_malloc
#define realloc api_test_realloc
#include "../../lib/nanorq.c"
#undef malloc
#undef realloc

#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__);                  \
      return false;                                                            \
    }                                                                          \
  } while (0)

static bool reject_invalid_oti(void) {
  uint64_t common = (128ULL << 24) | 127U;
  uint32_t scheme_with_65536_subblocks = (65535U << 8) | 1U;
  nanorq *rq = nanorq_decoder_new(common, scheme_with_65536_subblocks);
  CHECK(rq == NULL, "N=65536 OTI must be rejected");
  rq = nanorq_decoder_new((128ULL << 24) | 65535U, 1U);
  CHECK(rq == NULL, "T=65536 OTI must be rejected before division");
  return true;
}

static bool grow_t_without_looping(void) {
  nanorq *rq = nanorq_encoder_new_ex(20000000, 1, 0, 0, 1);
  CHECK(rq != NULL, "large transfer with T=1 must construct");
  CHECK(nanorq_symbol_size(rq) > 1, "constructor must increase T");
  nanorq_free(rq);
  return true;
}

static bool zero_pad_last_symbol(void) {
  uint8_t source[] = {0x11, 0x22, 0x33};
  uint8_t encoded[4] = {0xaa, 0xaa, 0xaa, 0xaa};
  struct ioctx *io = ioctx_from_mem_ro(source, sizeof(source));
  nanorq *rq = nanorq_encoder_new_ex(sizeof(source), 4, 1, 1, 1);
  CHECK(io && rq, "padding setup");
  CHECK(nanorq_encode(rq, encoded, 0, 0, io) == sizeof(encoded),
        "encode final symbol");
  CHECK(memcmp(encoded, source, sizeof(source)) == 0 && encoded[3] == 0,
        "final symbol must be zero padded");
  io->destroy(io);
  nanorq_free(rq);
  return true;
}

static bool recode_single_symbol(void) {
  uint8_t source[] = {1, 2, 3, 4};
  uint8_t coef[1] = {0};
  uint8_t payload[4] = {0};
  struct ioctx *io = ioctx_from_mem_ro(source, sizeof(source));
  nanorq *rq = nanorq_encoder_new_ex(sizeof(source), 4, 1, 1, 1);
  CHECK(io && rq, "K=1 setup");
  CHECK(nanorq_generate_recoded_symbol(rq, io, 0, 0, coef, payload),
        "K=1 recode must not divide by zero");
  CHECK(coef[0] != 0, "K=1 recode coefficient");
  io->destroy(io);
  nanorq_free(rq);
  return true;
}

static bool recode_uses_requested_block(void) {
  uint8_t source[] = {0, 0, 0, 0, 0x10, 0x20, 0x40, 0x80};
  struct ioctx *io = ioctx_from_mem_ro(source, sizeof(source));
  nanorq *rq = nanorq_encoder_new_ex(sizeof(source), 2, 2, 2, 1);
  uint8_t coef0[2], coef1[2], payload0[2], payload1[2];
  CHECK(io && rq && nanorq_blocks(rq) == 2, "multi-block setup");
  bool differs = false;
  for (uint32_t esi = 0; esi < 16 && !differs; esi++) {
    CHECK(nanorq_generate_recoded_symbol(rq, io, 0, esi, coef0, payload0),
          "recode block zero");
    CHECK(nanorq_generate_recoded_symbol(rq, io, 1, esi, coef1, payload1),
          "recode block one");
    differs = memcmp(payload0, payload1, sizeof(payload0)) != 0;
  }
  CHECK(differs, "recoding must read from the requested source block");
  io->destroy(io);
  nanorq_free(rq);
  return true;
}

static bool hybrid_repair_uses_known_symbols(void) {
  enum { K = 10, T = 8 };
  uint8_t source[K * T], decoded[K * T], coef[K], payload[T];
  for (size_t i = 0; i < sizeof(source); i++)
    source[i] = (uint8_t)(i * 17U + 3U);
  memset(decoded, 0, sizeof(decoded));
  struct ioctx *src_io = ioctx_from_mem_ro(source, sizeof(source));
  struct ioctx *dec_io = ioctx_from_mem(decoded, sizeof(decoded));
  nanorq *enc = nanorq_encoder_new_ex(sizeof(source), T, K, 1, 1);
  CHECK(src_io && dec_io && enc, "hybrid encoder setup");
  nanorq *dec = nanorq_decoder_new(nanorq_oti_common(enc),
                                   nanorq_oti_scheme_specific(enc));
  CHECK(dec, "hybrid decoder setup");
  for (uint32_t esi = 0; esi < K - 1; esi++)
    CHECK(nanorq_decoder_add_symbol(dec, source + esi * T, nanorq_tag(0, esi),
                                    dec_io) == NANORQ_SYM_ADDED,
          "add known source symbol");

  uint32_t recode_esi = 0;
  do {
    CHECK(nanorq_generate_recoded_symbol(enc, src_io, 0, recode_esi, coef,
                                         payload),
          "generate hybrid equation");
    recode_esi++;
  } while (coef[K - 1] == 0 && recode_esi < 1000);
  CHECK(coef[K - 1] != 0, "find equation covering missing symbol");
  CHECK(nanorq_decoder_add_recoded_symbol(dec, payload,
                                          nanorq_tag(0, recode_esi), coef,
                                          dec_io) == NANORQ_SYM_ADDED,
        "add hybrid equation");
  CHECK(nanorq_repair_block(dec, dec_io, 0), "hybrid repair");
  CHECK(memcmp(source, decoded, sizeof(source)) == 0,
        "hybrid repair output must match source");

  nanorq_free(dec);
  nanorq_free(enc);
  dec_io->destroy(dec_io);
  src_io->destroy(src_io);
  return true;
}

static bool fail_seek(struct ioctx *io, size_t offset) {
  (void)io;
  (void)offset;
  return true;
}

static size_t fail_write(struct ioctx *io, const uint8_t *buf, size_t len) {
  (void)io;
  (void)buf;
  (void)len;
  return 0;
}

static bool output_failure_is_reported(void) {
  uint8_t source[8] = {0};
  nanorq *enc = nanorq_encoder_new_ex(sizeof(source), 4, 2, 1, 1);
  CHECK(enc, "failed-output encoder setup");
  nanorq *dec = nanorq_decoder_new(nanorq_oti_common(enc),
                                   nanorq_oti_scheme_specific(enc));
  CHECK(dec, "failed-output decoder setup");
  struct ioctx failing = {.write = fail_write,
                          .seek = fail_seek,
                          .writable = true,
                          .seekable = true};
  CHECK(nanorq_decoder_add_symbol(dec, source, nanorq_tag(0, 0), &failing) ==
            NANORQ_SYM_ERR,
        "short output write must fail");
  CHECK(nanorq_num_missing(dec, 0) == 2,
        "failed output must not mark symbol present");
  nanorq_free(dec);
  nanorq_free(enc);
  return true;
}

static bool conventional_repair_can_retry(void) {
  enum { K = 10, T = 8 };
  uint8_t source[K * T], decoded[K * T], repair[T];
  for (size_t i = 0; i < sizeof(source); i++)
    source[i] = (uint8_t)(i * 29U + 7U);
  memset(decoded, 0, sizeof(decoded));
  struct ioctx *src_io = ioctx_from_mem_ro(source, sizeof(source));
  struct ioctx *dec_io = ioctx_from_mem(decoded, sizeof(decoded));
  nanorq *enc = nanorq_encoder_new_ex(sizeof(source), T, K, 1, 1);
  CHECK(src_io && dec_io && enc, "retry encoder setup");
  nanorq *dec = nanorq_decoder_new(nanorq_oti_common(enc),
                                   nanorq_oti_scheme_specific(enc));
  CHECK(dec, "retry decoder setup");
  for (uint32_t esi = 1; esi < K; esi++)
    CHECK(nanorq_decoder_add_symbol(dec, source + esi * T, nanorq_tag(0, esi),
                                    dec_io) == NANORQ_SYM_ADDED,
          "add retry source symbol");
  CHECK(nanorq_encode(enc, repair, K, 0, src_io) == T,
        "generate conventional repair symbol");
  CHECK(nanorq_decoder_add_symbol(dec, repair, nanorq_tag(0, K), dec_io) ==
            NANORQ_SYM_ADDED,
        "add conventional repair symbol");

  struct ioctx failing = {.write = fail_write,
                          .seek = fail_seek,
                          .writable = true,
                          .seekable = true};
  CHECK(!nanorq_repair_block(dec, &failing, 0),
        "repair must report output failure");
  CHECK(nanorq_num_missing(dec, 0) == 1,
        "failed repair must preserve missing state");
  CHECK(nanorq_repair_block(dec, dec_io, 0),
        "repair must be retryable after output failure");
  CHECK(memcmp(source, decoded, sizeof(source)) == 0,
        "retried repair output must match source");

  nanorq_free(dec);
  nanorq_free(enc);
  dec_io->destroy(dec_io);
  src_io->destroy(src_io);
  return true;
}

/* test matrix growth, reuse, and shrink across resets */
static bool repair_capacity_survives_reset(void) {
  enum { K = 10, T = 19, BYTES = K * T - 3 };
  const unsigned overheads[] = {0, 5, 1, 8, 0, 4};
  uint8_t source[BYTES] = {0}, decoded[BYTES], packet[T];
  struct ioctx *src_io = ioctx_from_mem_ro(source, sizeof(source));
  struct ioctx *dec_io = ioctx_from_mem(decoded, sizeof(decoded));
  nanorq *enc = nanorq_encoder_new_ex(sizeof(source), T, K, 1, 1);
  CHECK(src_io && dec_io && enc, "capacity encoder setup");
  nanorq *dec = nanorq_decoder_new(nanorq_oti_common(enc),
                                   nanorq_oti_scheme_specific(enc));
  CHECK(dec, "capacity decoder setup");
  for (unsigned trial = 0; trial < sizeof(overheads) / sizeof(*overheads);
       trial++) {
    for (size_t i = 0; i < sizeof(source); i++)
      source[i] = (uint8_t)(i * 29U + trial * 37U + 7U);
    memset(decoded, 0xa5, sizeof(decoded));
    CHECK(nanorq_generate_symbols(enc, 0, src_io), "load partial source block");
    for (unsigned esi = 1; esi < K + 1 + overheads[trial]; esi++) {
      memset(packet, 0xa5, sizeof(packet));
      CHECK(nanorq_encode(enc, packet, esi, 0, src_io) == T,
            "generate capacity symbol");
      CHECK(nanorq_decoder_add_symbol(dec, packet, nanorq_tag(0, esi),
                                      dec_io) == NANORQ_SYM_ADDED,
            "add capacity symbol");
    }
    CHECK(nanorq_repair_block(dec, dec_io, 0), "repair after capacity change");
    CHECK(memcmp(source, decoded, sizeof(source)) == 0,
          "capacity reuse must preserve recovered bytes");
    nanorq_encoder_reset(enc, 0);
    nanorq_encoder_reset(dec, 0);
  }
  nanorq_free(dec);
  nanorq_free(enc);
  dec_io->destroy(dec_io);
  src_io->destroy(src_io);
  return true;
}

static bool rank_failure_can_retry_after_growth(void) {
  enum { K = 10, T = 8 };
  uint8_t source[K * T] = {0}, decoded[K * T], packet[T];
  struct ioctx *src_io = ioctx_from_mem_ro(source, sizeof(source));
  struct ioctx *dec_io = ioctx_from_mem(decoded, sizeof(decoded));
  nanorq *enc = nanorq_encoder_new_ex(sizeof(source), T, K, 1, 1);
  CHECK(src_io && dec_io && enc, "rank retry encoder setup");
  nanorq *dec = nanorq_decoder_new(nanorq_oti_common(enc),
                                   nanorq_oti_scheme_specific(enc));
  CHECK(dec, "rank retry decoder setup");
  for (unsigned trial = 0; trial < 2; trial++) {
    for (size_t i = 0; i < sizeof(source); i++)
      source[i] = (uint8_t)(i * 17U + trial * 37U + 3U);
    memset(decoded, 0, sizeof(decoded));
    for (unsigned esi = 1; esi < K; esi++)
      CHECK(nanorq_decoder_add_symbol(dec, source + esi * T, nanorq_tag(0, esi),
                                      dec_io) == NANORQ_SYM_ADDED,
            "add known symbols before rank failure");
    /* ESI 142 with source ESIs 1..9 is rank deficient for K=10 */
    CHECK(nanorq_encode(enc, packet, 142, 0, src_io) == T,
          "generate deficient equation");
    CHECK(nanorq_decoder_add_symbol(dec, packet, nanorq_tag(0, 142), dec_io) ==
              NANORQ_SYM_ADDED,
          "add deficient equation");
    for (unsigned retry = 0; retry < 2; retry++) {
      CHECK(!nanorq_repair_block(dec, dec_io, 0), "report deficient rank");
      CHECK(nanorq_num_missing(dec, 0) == 1, "preserve missing state");
      CHECK(memcmp(source + T, decoded + T, sizeof(source) - T) == 0,
            "rank failure must preserve known symbols");
    }
    CHECK(nanorq_encode(enc, packet, K, 0, src_io) == T,
          "generate independent equation");
    CHECK(nanorq_decoder_add_symbol(dec, packet, nanorq_tag(0, K), dec_io) ==
              NANORQ_SYM_ADDED,
          "add independent equation");
    CHECK(nanorq_repair_block(dec, dec_io, 0),
          "retry with additional equation");
    CHECK(memcmp(source, decoded, sizeof(source)) == 0,
          "rank retry must recover original bytes");
    nanorq_encoder_reset(enc, 0);
    nanorq_encoder_reset(dec, 0);
  }
  nanorq_free(dec);
  nanorq_free(enc);
  dec_io->destroy(dec_io);
  src_io->destroy(src_io);
  return true;
}

static bool unequal_blocks_use_contiguous_io(void) {
  enum { T = 19, BYTES = 11 * T - 3 };
  uint8_t source[BYTES], decoded[BYTES] = {0}, packet[T];
  for (size_t i = 0; i < sizeof(source); i++)
    source[i] = (uint8_t)(i * 19U + 11U);
  struct ioctx *src_io = ioctx_from_mem_ro(source, sizeof(source));
  struct ioctx *dec_io = ioctx_from_mem(decoded, sizeof(decoded));
  nanorq *enc = nanorq_encoder_new_ex(sizeof(source), T, 6, 2, 1);
  CHECK(src_io && dec_io && enc && nanorq_blocks(enc) == 2,
        "unequal blocks encoder setup");
  nanorq *dec = nanorq_decoder_new(nanorq_oti_common(enc),
                                   nanorq_oti_scheme_specific(enc));
  CHECK(dec, "unequal blocks decoder setup");
  for (unsigned sbn = 0; sbn < 2; sbn++) {
    unsigned K = nanorq_block_symbols(enc, sbn);
    CHECK(K == (sbn == 0 ? 6 : 5), "unequal block dimensions");
    CHECK(nanorq_generate_symbols(enc, sbn, src_io),
          "load unequal source block");
    for (unsigned esi = 1; esi < K + 3; esi++) {
      CHECK(nanorq_encode(enc, packet, esi, sbn, src_io) == T,
            "encode unequal block symbol");
      CHECK(nanorq_decoder_add_symbol(dec, packet, nanorq_tag(sbn, esi),
                                      dec_io) == NANORQ_SYM_ADDED,
            "add unequal block symbol");
    }
    CHECK(nanorq_repair_block(dec, dec_io, sbn), "repair unequal block");
  }
  CHECK(memcmp(source, decoded, sizeof(source)) == 0,
        "unequal partial blocks must match source");
  nanorq_free(dec);
  nanorq_free(enc);
  dec_io->destroy(dec_io);
  src_io->destroy(src_io);
  return true;
}

static bool scratch_allocation_failures_can_retry(void) {
  enum { K = 10, T = 8 };
  uint8_t source[K * T], decoded[K * T], packet[T];
  for (size_t i = 0; i < sizeof(source); i++)
    source[i] = (uint8_t)(i * 17U + 3U);
  struct ioctx *src_io = ioctx_from_mem_ro(source, sizeof(source));
  struct ioctx *dec_io = ioctx_from_mem(decoded, sizeof(decoded));
  nanorq *enc = nanorq_encoder_new_ex(sizeof(source), T, K, 1, 1);
  CHECK(src_io && dec_io && enc, "allocation retry encoder setup");
  CHECK(nanorq_encode(enc, packet, K, 0, src_io) == T,
        "generate allocation retry equation");
  for (unsigned failure = 1; failure <= 4; failure++) {
    memset(decoded, 0, sizeof(decoded));
    nanorq *dec = nanorq_decoder_new(nanorq_oti_common(enc),
                                     nanorq_oti_scheme_specific(enc));
    CHECK(dec, "allocation retry decoder setup");
    for (unsigned esi = 1; esi < K; esi++)
      CHECK(nanorq_decoder_add_symbol(dec, source + esi * T, nanorq_tag(0, esi),
                                      dec_io) == NANORQ_SYM_ADDED,
            "add allocation retry source");
    CHECK(nanorq_decoder_add_symbol(dec, packet, nanorq_tag(0, K), dec_io) ==
              NANORQ_SYM_ADDED,
          "add allocation retry equation");
    alloc_calls = 0;
    alloc_fail_at = failure;
    bool repaired = nanorq_repair_block(dec, dec_io, 0);
    alloc_fail_at = 0;
    CHECK(alloc_calls == failure && !repaired,
          "report scratch allocation failure");
    CHECK(nanorq_num_missing(dec, 0) == 1, "allocation failure preserves gaps");
    CHECK(memcmp(source + T, decoded + T, sizeof(source) - T) == 0,
          "allocation failure preserves known data");
    CHECK(nanorq_repair_block(dec, dec_io, 0),
          "retry after allocation failure");
    CHECK(memcmp(source, decoded, sizeof(source)) == 0,
          "allocation retry recovers original bytes");
    if (failure == 1) {
      /* fail realloc of scratch buffer after D growth */
      uint8_t extra[T];
      nanorq_encoder_reset(dec, 0);
      memset(decoded, 0, sizeof(decoded));
      for (unsigned esi = 1; esi < K; esi++)
        CHECK(nanorq_decoder_add_symbol(dec, source + esi * T,
                                        nanorq_tag(0, esi),
                                        dec_io) == NANORQ_SYM_ADDED,
              "add source before scratch growth");
      for (unsigned esi = K; esi < K + 9; esi++) {
        CHECK(nanorq_encode(enc, extra, esi, 0, src_io) == T,
              "generate scratch growth equation");
        CHECK(nanorq_decoder_add_symbol(dec, extra, nanorq_tag(0, esi),
                                        dec_io) == NANORQ_SYM_ADDED,
              "add scratch growth equation");
      }
      alloc_calls = 0;
      alloc_fail_at = 1;
      repaired = nanorq_repair_block(dec, dec_io, 0);
      alloc_fail_at = 0;
      CHECK(!repaired && alloc_calls == 1, "report scratch growth failure");
      CHECK(nanorq_num_missing(dec, 0) == 1, "scratch growth preserves gaps");
      CHECK(nanorq_repair_block(dec, dec_io, 0), "retry failed scratch growth");
      CHECK(memcmp(source, decoded, sizeof(source)) == 0,
            "scratch growth retry recovers original bytes");
    }
    nanorq_free(dec);
  }
  nanorq_free(enc);
  dec_io->destroy(dec_io);
  src_io->destroy(src_io);
  return true;
}

static bool mmap_tracks_logical_size(void) {
  char path[] = "/tmp/nanorq-io-XXXXXX";
  int fd = mkstemp(path);
  CHECK(fd >= 0, "create mmap regression file");
  close(fd);

  struct ioctx *io = ioctx_mmap_file(path, IOCTX_MODE_READ);
  CHECK(io && io->size(io) == 0, "open an empty mmap input");
  CHECK(io->seek(io, 0), "seek to empty mmap EOF");
  io->destroy(io);

  uint8_t byte = 0x5a;
  io = ioctx_mmap_file(path, IOCTX_MODE_WRITE);
  CHECK(io && io->size(io) == 0, "new mmap output has logical size zero");
  CHECK(io->seek(io, 100), "seek within mmap output capacity");
  CHECK(io->size(io) == 0, "seek alone must not grow logical size");
  CHECK(io->write(io, &byte, 1) == 1 && io->size(io) == 101,
        "write grows mmap logical size");
  io->destroy(io);

  io = ioctx_mmap_file(path, IOCTX_MODE_READ);
  CHECK(io && io->size(io) == 101, "mmap output truncates to logical size");
  io->destroy(io);
  unlink(path);
  return true;
}

int main(void) {
  CHECK(reject_invalid_oti(), "invalid OTI regression");
  CHECK(grow_t_without_looping(), "T growth regression");
  CHECK(zero_pad_last_symbol(), "padding regression");
  CHECK(recode_single_symbol(), "K=1 regression");
  CHECK(recode_uses_requested_block(), "source block regression");
  CHECK(hybrid_repair_uses_known_symbols(), "hybrid repair regression");
  CHECK(output_failure_is_reported(), "output failure regression");
  CHECK(conventional_repair_can_retry(), "repair retry regression");
  CHECK(repair_capacity_survives_reset(), "capacity reset regression");
  CHECK(rank_failure_can_retry_after_growth(), "rank retry regression");
  CHECK(unequal_blocks_use_contiguous_io(), "unequal direct I/O regression");
  CHECK(scratch_allocation_failures_can_retry(),
        "scratch allocation retry regression");
  CHECK(mmap_tracks_logical_size(), "mmap logical size regression");
  puts("all API regressions passed");
  return EXIT_SUCCESS;
}
