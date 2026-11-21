#include "nanorq.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
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
  /* Each equation is independent of source ESI 0 for K=10. Together
   * these fill the capped first attempt without recovering that symbol. */
  const unsigned deficient[] = {142,  232,  244,  245,  375,  583,  848,
                                868,  1184, 1503, 1984, 2400, 2546, 2802,
                                2910, 3069, 3766, 4014, 4310};
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
    size_t count = trial == 0 ? 1 : sizeof(deficient) / sizeof(deficient[0]);
    for (size_t i = 0; i < count; i++) {
      unsigned esi = deficient[i];
      CHECK(nanorq_encode(enc, packet, esi, 0, src_io) == T,
            "generate deficient equation");
      CHECK(nanorq_decoder_add_symbol(dec, packet, nanorq_tag(0, esi),
                                      dec_io) == NANORQ_SYM_ADDED,
            "add deficient equation");
    }
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
  CHECK(nanorq_generate_symbols(enc, 0, src_io),
        "prepare encoder before mix_buf failure");
  alloc_calls = 0;
  alloc_fail_at = 1;
  CHECK(nanorq_encode(enc, packet, K, 0, src_io) == 0,
        "report mix_buf allocation failure");
  alloc_fail_at = 0;
  CHECK(alloc_calls == 1, "encode mix_buf uses malloc");
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
  uint8_t readback[2];
  CHECK(io->seek(io, 100) && io->read(io, readback, sizeof(readback)) == 1 &&
            readback[0] == byte,
        "read from writable mmap stops at logical EOF");

  /* Opening the output reserves one mapping window in the backing file. */
  struct stat st;
  CHECK(stat(path, &st) == 0 && st.st_size > 101, "get mmap window size");
  size_t boundary = (size_t)st.st_size;
  const uint8_t edge[] = {0x42, 0xc7};
  CHECK(io->seek(io, boundary - 1) && io->write(io, edge, 1) == 1 &&
            io->write(io, edge + 1, 1) == 1,
        "consecutive mmap writes across a window boundary");
  io->destroy(io);

  io = ioctx_mmap_file(path, IOCTX_MODE_READ);
  CHECK(io && io->size(io) == boundary + 1,
        "mmap output truncates to logical size");
  CHECK(io->read(io, readback, 1) == 1 && readback[0] == 0,
        "second mmap write preserves the start of the first window");
  CHECK(io->seek(io, boundary - 1) && io->read(io, readback, 1) == 1 &&
            io->read(io, readback + 1, 1) == 1 &&
            memcmp(readback, edge, sizeof(edge)) == 0,
        "consecutive mmap reads across a window boundary");
  CHECK(!io->seek(io, boundary + 2), "mmap read seek past EOF must fail");
  io->destroy(io);
  unlink(path);
  return true;
}

static bool tsnc_relay_large_k(void) {
  enum { K = 260, T = 16 };
  uint8_t *src = calloc(K, T), *dec_buf = calloc(K, T);
  uint8_t coef[K], pkt[T];
  CHECK(src && dec_buf, "tsnc large-k alloc");
  for (size_t i = 0; i < (size_t)K * T; i++)
    src[i] = (uint8_t)(i * 31U + 5U);
  struct ioctx *sio = ioctx_from_mem_ro(src, (size_t)K * T);
  struct ioctx *dio = ioctx_from_mem(dec_buf, (size_t)K * T);
  nanorq *enc = nanorq_encoder_new_ex((size_t)K * T, T, K, 1, 1);
  CHECK(sio && dio && enc, "tsnc large-k setup");
  uint64_t common = nanorq_oti_common(enc);
  uint32_t scheme = nanorq_oti_scheme_specific(enc);

  nanorq *relay = nanorq_decoder_new(common, scheme);
  nanorq *dec = nanorq_decoder_new(common, scheme);
  CHECK(relay && dec, "tsnc large-k nodes");
  for (uint32_t i = 0; i < K; i++) {
    memset(coef, 0, K);
    coef[i] = 1;
    CHECK(nanorq_decoder_add_recoded_symbol(relay, src + i * T,
                                            nanorq_tag(0, i), coef,
                                            NULL) == NANORQ_SYM_ADDED,
          "relay add basis packet");
  }
  for (uint32_t esi = 0; esi < K + 20; esi++) {
    CHECK(nanorq_generate_recoded_symbol(relay, NULL, 0, esi, coef, pkt),
          "relay generate k>256 recode");
    CHECK(nanorq_decoder_add_recoded_symbol(dec, pkt, nanorq_tag(0, 1000 + esi),
                                            coef, dio) == NANORQ_SYM_ADDED,
          "dest add k>256 recode");
  }
  CHECK(nanorq_repair_block(dec, dio, 0), "tsnc k>256 relay repair");
  CHECK(memcmp(src, dec_buf, (size_t)K * T) == 0, "tsnc k>256 payload match");

  nanorq_free(dec);
  nanorq_free(relay);
  nanorq_free(enc);
  sio->destroy(sio);
  dio->destroy(dio);
  free(dec_buf);
  free(src);
  return true;
}

static bool high_overhead_burst_repair(void) {
  enum { K = 10, T = 32, OH = 250 };
  uint8_t src[K * T], dec_buf[K * T], pkt[T];
  for (size_t i = 0; i < sizeof(src); i++)
    src[i] = (uint8_t)(i * 13U + 7U);
  memset(dec_buf, 0, sizeof(dec_buf));
  struct ioctx *sio = ioctx_from_mem_ro(src, sizeof(src));
  struct ioctx *dio = ioctx_from_mem(dec_buf, sizeof(dec_buf));
  nanorq *enc = nanorq_encoder_new_ex(sizeof(src), T, K, 1, 1);
  CHECK(sio && dio && enc, "burst overhead setup");
  nanorq *dec = nanorq_decoder_new(nanorq_oti_common(enc),
                                   nanorq_oti_scheme_specific(enc));
  CHECK(dec, "burst overhead decoder");
  for (uint32_t esi = 1; esi < K; esi++)
    CHECK(nanorq_decoder_add_symbol(dec, src + esi * T, nanorq_tag(0, esi),
                                    dio) == NANORQ_SYM_ADDED,
          "burst add systematic");
  for (uint32_t esi = K; esi <= K + OH; esi++) {
    CHECK(nanorq_encode(enc, pkt, esi, 0, sio) == T, "burst encode repair");
    CHECK(nanorq_decoder_add_symbol(dec, pkt, nanorq_tag(0, esi), dio) ==
              NANORQ_SYM_ADDED,
          "burst add repair");
  }
  CHECK(nanorq_repair_block(dec, dio, 0), "burst repair with large overhead");
  CHECK(dec->encoders[0]->core.overhead == dec->encoders[0]->core.P.H + 8U,
        "burst repair uses the capped equation set");
  CHECK(memcmp(src, dec_buf, sizeof(src)) == 0, "burst repair payload match");
  nanorq_free(dec);
  nanorq_free(enc);
  sio->destroy(sio);
  dio->destroy(dio);
  return true;
}

static bool precalc_covers_unequal_blocks(void) {
  enum { T = 16, BYTES = 11 * T };
  uint8_t source[BYTES], decoded[BYTES] = {0}, packet[T];
  for (size_t i = 0; i < sizeof(source); i++)
    source[i] = (uint8_t)(i * 23U + 9U);
  struct ioctx *src_io = ioctx_from_mem_ro(source, sizeof(source));
  struct ioctx *dec_io = ioctx_from_mem(decoded, sizeof(decoded));
  nanorq *enc = nanorq_encoder_new_ex(sizeof(source), T, 6, 2, 1);
  CHECK(src_io && dec_io && enc && nanorq_blocks(enc) == 2,
        "precalc unequal setup");
  CHECK(nanorq_precalculate(enc), "precalc both KL and KS");
  CHECK(enc->precalc[0].core && enc->precalc[0].K == 6 &&
            enc->precalc[1].core && enc->precalc[1].K == 5,
        "both partition block sizes must be precalculated");
  nanorq *dec = nanorq_decoder_new(nanorq_oti_common(enc),
                                   nanorq_oti_scheme_specific(enc));
  CHECK(dec, "precalc unequal decoder");
  for (unsigned sbn = 0; sbn < 2; sbn++) {
    unsigned K = nanorq_block_symbols(enc, sbn);
    CHECK(nanorq_generate_symbols(enc, sbn, src_io), "generate via precalc");
    /* block scratch must remain unallocated when precalc cache hits */
    CHECK(enc->encoders[sbn]->prep_mem == NULL &&
              enc->encoders[sbn]->work_mem == NULL,
          "KS block must hit precalc cache without uncached GE");
    for (unsigned esi = 1; esi <= K; esi++) {
      CHECK(nanorq_encode(enc, packet, esi, sbn, src_io) == T,
            "encode unequal precalc");
      CHECK(nanorq_decoder_add_symbol(dec, packet, nanorq_tag(sbn, esi),
                                      dec_io) == NANORQ_SYM_ADDED,
            "add unequal precalc");
    }
    CHECK(nanorq_repair_block(dec, dec_io, sbn), "repair unequal precalc");
  }
  CHECK(memcmp(source, decoded, sizeof(source)) == 0,
        "unequal precalc roundtrip");
  nanorq_free(dec);
  nanorq_free(enc);
  dec_io->destroy(dec_io);
  src_io->destroy(src_io);
  return true;
}

static bool run_regressions(void) {
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
  CHECK(tsnc_relay_large_k(), "tsnc large-k relay regression");
  CHECK(high_overhead_burst_repair(), "high-overhead burst repair regression");
  CHECK(precalc_covers_unequal_blocks(), "unequal-block precalc regression");
  return true;
}

int main(void) {
  if (!run_regressions())
    return EXIT_FAILURE;
  puts("all API regressions passed");
  return EXIT_SUCCESS;
}
