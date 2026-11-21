/* portable fixture helper; exit 2 means valid but incomplete */
#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nanorq.h"

#define MAX_PAYLOAD_BYTES (128U * 1024U * 1024U)
#define MAX_FIXTURE_BYTES (256U * 1024U * 1024U)
#define MAX_PACKETS 1000000U

static const uint8_t magic[8] = {'N', 'R', 'Q', 'F', 'E', 'C', '1', 0};

static bool read_exact(FILE *file, void *buffer, size_t len) {
  if (fread(buffer, 1, len, file) == len)
    return true;
  fprintf(stderr, "truncated fixture or read error\n");
  return false;
}

static bool write_exact(FILE *file, const void *buffer, size_t len) {
  if (fwrite(buffer, 1, len, file) == len)
    return true;
  fprintf(stderr, "write error\n");
  return false;
}

static bool read_be(FILE *file, uint64_t *value, size_t len) {
  uint8_t bytes[8];
  if (!read_exact(file, bytes, len))
    return false;
  *value = 0;
  for (size_t i = 0; i < len; i++)
    *value = (*value << 8) | bytes[i];
  return true;
}

static bool write_be(FILE *file, uint64_t value, size_t len) {
  uint8_t bytes[8];
  for (size_t i = len; i > 0; i--) {
    bytes[i - 1] = (uint8_t)value;
    value >>= 8;
  }
  return write_exact(file, bytes, len);
}

static bool parse_number(const char *text, uint32_t max, uint32_t *value) {
  if (!text[0])
    return false;
  for (const char *p = text; *p; p++) {
    if (*p < '0' || *p > '9')
      return false;
  }
  errno = 0;
  char *end;
  unsigned long parsed = strtoul(text, &end, 10);
  if (errno || *end || parsed > max)
    return false;
  *value = (uint32_t)parsed;
  return true;
}

static void print_geometry(nanorq *rq) {
  uint32_t scheme = nanorq_oti_scheme_specific(rq);
  size_t blocks = nanorq_blocks(rq);
  printf("F=%zu T=%zu Z=%zu N=%u Al=%u K=", nanorq_transfer_length(rq),
         nanorq_symbol_size(rq), blocks, ((scheme >> 8) & 0xffff) + 1,
         scheme & 0xff);
  for (size_t sbn = 0; sbn < blocks; sbn++)
    printf("%s%zu", sbn ? "," : "", nanorq_block_symbols(rq, (uint8_t)sbn));
  printf("\n");
}

static bool bounded_fixture(uint64_t packets, size_t symbol_size) {
  return packets <= MAX_PACKETS &&
         packets <= (MAX_FIXTURE_BYTES - 24U) / (symbol_size + 8U);
}

static int encode(const char *input_path, const char *fixture_path,
                  uint32_t symbol_size, uint32_t requested_blocks,
                  uint32_t repairs) {
  int result = 1;
  nanorq *rq = NULL;
  FILE *fixture = NULL;
  uint8_t *symbol = NULL;
  struct ioctx *input = ioctx_from_file(input_path, IOCTX_MODE_READ);
  if (!input) {
    perror(input_path);
    return result;
  }
  size_t len = input->size(input);
  if (!len || len > MAX_PAYLOAD_BYTES) {
    fprintf(stderr, "payload must contain 1..%u bytes\n", MAX_PAYLOAD_BYTES);
    goto out;
  }
  if (requested_blocks) {
    if (symbol_size % 8 != 0) {
      fprintf(stderr,
              "explicit block geometry requires symbol size divisible by 8\n");
      goto out;
    }
    rq = nanorq_encoder_new_ex(len, (uint16_t)symbol_size, 0,
                               (uint16_t)requested_blocks, 8);
  } else {
    rq = nanorq_encoder_new(len, (uint16_t)symbol_size, 8);
  }
  if (!rq) {
    fprintf(stderr, "invalid encoder geometry or allocation failure\n");
    goto out;
  }
  if (requested_blocks && (nanorq_symbol_size(rq) != symbol_size ||
                           nanorq_blocks(rq) != requested_blocks)) {
    fprintf(stderr, "requested block geometry is not representable\n");
    goto out;
  }
  print_geometry(rq);
  symbol_size = (uint32_t)nanorq_symbol_size(rq);
  size_t blocks = nanorq_blocks(rq);
  uint64_t packets = 0;
  for (size_t sbn = 0; sbn < blocks; sbn++) {
    size_t sources = nanorq_block_symbols(rq, (uint8_t)sbn);
    if (repairs > (1U << 24) - sources) {
      fprintf(stderr, "repair count exceeds the ESI range\n");
      goto out;
    }
    packets += sources + repairs;
  }
  if (!bounded_fixture(packets, symbol_size)) {
    fprintf(stderr, "fixture exceeds packet or byte limit\n");
    goto out;
  }
  symbol = malloc(symbol_size);
  fixture = fopen(fixture_path, "wb");
  if (!symbol || !fixture) {
    fprintf(stderr, "unable to allocate symbol or open fixture output\n");
    goto out;
  }
  if (!write_exact(fixture, magic, sizeof(magic)) ||
      !write_be(fixture, nanorq_oti_common(rq), 8) ||
      !write_be(fixture, nanorq_oti_scheme_specific(rq), 4) ||
      !write_be(fixture, packets, 4))
    goto out;

  for (size_t sbn = 0; sbn < blocks; sbn++) {
    uint8_t block = (uint8_t)sbn;
    uint32_t sources = (uint32_t)nanorq_block_symbols(rq, block);
    for (uint32_t esi = 0; esi < sources + repairs; esi++) {
      if (nanorq_encode(rq, symbol, esi, block, input) != symbol_size) {
        fprintf(stderr, "failed to encode block %zu ESI %u\n", sbn, esi);
        goto out;
      }
      if (!write_be(fixture, nanorq_tag(block, esi), 4) ||
          !write_be(fixture, symbol_size, 4) ||
          !write_exact(fixture, symbol, symbol_size))
        goto out;
    }
    nanorq_encoder_cleanup(rq, block);
  }
  result = 0;
out:
  if (fixture && fclose(fixture) != 0) {
    fprintf(stderr, "failed to close fixture output\n");
    result = 1;
  }
  free(symbol);
  nanorq_free(rq);
  input->destroy(input);
  return result;
}

static int decode(const char *fixture_path, const char *output_path) {
  int result = 1;
  nanorq *rq = NULL;
  struct ioctx *output = NULL;
  uint8_t *decoded = NULL, *symbol = NULL;
  FILE *fixture = fopen(fixture_path, "rb");
  if (!fixture) {
    perror(fixture_path);
    return result;
  }
  uint8_t header_magic[8];
  uint64_t common, scheme, packets;
  if (!read_exact(fixture, header_magic, sizeof(header_magic)))
    goto out;
  if (memcmp(header_magic, magic, sizeof(magic))) {
    fprintf(stderr, "invalid fixture magic\n");
    goto out;
  }
  if (!read_be(fixture, &common, 8) || !read_be(fixture, &scheme, 4) ||
      !read_be(fixture, &packets, 4))
    goto out;
  uint64_t len = common >> 24;
  if (!len || len > MAX_PAYLOAD_BYTES) {
    fprintf(stderr, "invalid or oversized transfer length\n");
    goto out;
  }
  rq = nanorq_decoder_new(common, (uint32_t)scheme);
  if (!rq) {
    fprintf(stderr, "invalid OTI or allocation failure\n");
    goto out;
  }
  size_t symbol_size = nanorq_symbol_size(rq);
  size_t blocks = nanorq_blocks(rq);
  if (!bounded_fixture(packets, symbol_size)) {
    fprintf(stderr, "fixture exceeds packet or byte limit\n");
    goto out;
  }
  decoded = calloc((size_t)len, 1);
  symbol = malloc(symbol_size);
  if (!decoded || !symbol || !(output = ioctx_from_mem(decoded, (size_t)len))) {
    fprintf(stderr, "allocation failure\n");
    goto out;
  }
  for (uint64_t i = 0; i < packets; i++) {
    uint64_t tag, packet_size;
    if (!read_be(fixture, &tag, 4) || !read_be(fixture, &packet_size, 4))
      goto out;
    if (packet_size != symbol_size || (tag >> 24) >= blocks) {
      fprintf(stderr, "invalid symbol length or source block number\n");
      goto out;
    }
    if (!read_exact(fixture, symbol, symbol_size))
      goto out;
    if (nanorq_decoder_add_symbol(rq, symbol, (uint32_t)tag, output) ==
        NANORQ_SYM_ERR) {
      fprintf(stderr, "decoder rejected symbol tag %" PRIu64 "\n", tag);
      goto out;
    }
  }
  if (fgetc(fixture) != EOF || ferror(fixture)) {
    fprintf(stderr, "trailing fixture bytes or read error\n");
    goto out;
  }
  /* check complete framing even when early packets finish decoding */
  print_geometry(rq);
  for (size_t sbn = 0; sbn < blocks; sbn++) {
    if (!nanorq_repair_block(rq, output, (uint8_t)sbn)) {
      fprintf(stderr, "incomplete block %zu: missing=%zu repairs=%zu\n", sbn,
              nanorq_num_missing(rq, (uint8_t)sbn),
              nanorq_num_repair(rq, (uint8_t)sbn));
      result = 2;
      goto out;
    }
    nanorq_encoder_cleanup(rq, (uint8_t)sbn);
  }
  FILE *file = fopen(output_path, "wb");
  if (!file) {
    perror(output_path);
    goto out;
  }
  bool written = write_exact(file, decoded, (size_t)len);
  if (fclose(file) != 0) {
    fprintf(stderr, "failed to close decoded output\n");
    written = false;
  }
  result = written ? 0 : 1;
out:
  if (fclose(fixture) != 0 && result == 0)
    result = 1;
  if (output)
    output->destroy(output);
  free(symbol);
  free(decoded);
  nanorq_free(rq);
  return result;
}

int main(int argc, char **argv) {
  if (argc == 7 && !strcmp(argv[1], "encode")) {
    uint32_t symbol_size, blocks, repairs;
    if (!parse_number(argv[4], UINT16_MAX, &symbol_size) || !symbol_size ||
        !parse_number(argv[5], 255, &blocks) ||
        !parse_number(argv[6], MAX_PACKETS, &repairs)) {
      fprintf(stderr, "invalid symbol size, block count, or repair count\n");
      return 1;
    }
    return encode(argv[2], argv[3], symbol_size, blocks, repairs);
  }
  if (argc == 4 && !strcmp(argv[1], "decode"))
    return decode(argv[2], argv[3]);
  fprintf(stderr,
          "usage:\n"
          "  %s encode <payload> <fixture> <symbol-size> <blocks> <repairs>\n"
          "  %s decode <fixture> <output>\n"
          "blocks=0 selects default partitioning; exit 2 means incomplete\n",
          argv[0], argv[0]);
  return 1;
}
