#include "obl/oblas_common.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* independent reference for identical and disjoint ranges */
static void check(unsigned len, unsigned a, unsigned b) {
  uint8_t actual[12432], expected[12432];
  for (unsigned i = 0; i < sizeof(actual); i++)
    actual[i] = expected[i] = (uint8_t)(i * 73 + i / 13);
  for (unsigned i = 0; i < len; i++) {
    uint8_t value = expected[a + i];
    expected[a + i] = expected[b + i];
    expected[b + i] = value;
  }
  obl_swap(actual + a, actual + b, len);
  if (memcmp(actual, expected, sizeof(actual)) != 0) {
    fprintf(stderr, "swap mismatch: length=%u a=%u b=%u\n", len, a, b);
    exit(1);
  }
}

int main(void) {
  static const unsigned offsets[] = {0, 1, 7, 15, 31};
  static const unsigned gaps[] = {1, 3, 7, 15, 16, 31, 32, 63, 64};
  static const unsigned large[] = {255,  256,  257,  1279, 1280, 1281,
                                   1311, 1312, 1313, 2048, 4097};
  for (unsigned sample = 0; sample < 161 + sizeof(large) / sizeof(large[0]);
       sample++) {
    unsigned len = sample < 161 ? sample : large[sample - 161];
    for (unsigned i = 0; i < sizeof(offsets) / sizeof(offsets[0]); i++) {
      unsigned a = offsets[i];
      check(len, a, a);
      check(len, a, a + len);
      check(len, a + len, a);
      check(len, a, a + len + 33);
      for (unsigned j = 0; j < sizeof(gaps) / sizeof(gaps[0]); j++) {
        check(len, a, a + len + gaps[j]);
        check(len, a + len + gaps[j], a);
      }
    }
  }
  uint8_t byte = 0;
  obl_swap(NULL, NULL, 0);
  obl_swap(NULL, &byte, 0);
  obl_swap(&byte, NULL, 0);
  puts("row swap regressions passed");
  return 0;
}
