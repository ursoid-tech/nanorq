#include "params.h"
#include "rand.h"
#include "tuple.h"

_Static_assert(sizeof(J_gf2_delta) / sizeof(J_gf2_delta[0]) ==
                   sizeof(K_padded) / sizeof(K_padded[0]),
               "expected a GF(2) systematic index for every padded block size");

static int is_prime(u32 n) {
  if (n <= 1)
    return 0;
  if (n <= 3)
    return 1;
  if (n % 2 == 0 || n % 3 == 0)
    return 0;
  for (u32 i = 5; i * i <= n; i = i + 6)
    if (n % i == 0 || n % (i + 2) == 0)
      return 0;
  return 1;
}

static params make_params(u32 K, bool gf2_only) {
  u32 idx = 0;
  while (idx < K_padded_size && K > K_padded[idx])
    idx++;
  if (K == 0 || idx == K_padded_size)
    return (params){0};

  params P = {0};
  P.K = K;
  P.Kprime = K_padded[idx];
  P.J = J_K_padded[idx];
  P.S = S_H_W[idx][0];
  P.H = gf2_only ? 0 : S_H_W[idx][1];
  P.W = S_H_W[idx][2];
  P.L = P.Kprime + P.S + P.H;

  if (gf2_only) {
    /* Start at the RFC LT width, reserving ten PI columns where possible.
     * Small blocks start the prime search with at least four LT columns
     * beyond the S identity columns. */
    if (P.W > P.L - 10)
      P.W = P.L - 10;
    if (P.W < P.S + 4)
      P.W = P.S + 4;
    while (P.W > P.S + 2 && !is_prime(P.W))
      P.W--;

    /* This per-size offset selects a full-rank systematic matrix at H=0. */
    P.J = (P.J + J_gf2_delta[idx]) & 1023U;
  }

  P.P = P.L - P.W;
  P.U = P.P - P.H;
  P.B = P.W - P.S;
  P.P1 = P.P;
  while (!is_prime(P.P1))
    P.P1++;
  return P;
}

params params_init(u32 K) { return make_params(K, false); }

params params_init_gf2(u32 K) { return make_params(K, true); }

u32 params_set_idxs(params *P, u32 X, u32_vec *dst) {
  u32 num = 0;
  tuple t = gen_tuple(X, P);

  uv_push(*dst, t.b);
  num++;
  for (u32 j = 1; j < t.d; j++) {
    t.b = (t.b + t.a) % P->W;
    uv_push(*dst, t.b);
    num++;
  }
  while (t.b1 >= P->P)
    t.b1 = (t.b1 + t.a1) % P->P1;

  uv_push(*dst, P->W + t.b1);
  num++;
  for (u32 j = 1; j < t.d1; j++) {
    t.b1 = (t.b1 + t.a1) % P->P1;
    while (t.b1 >= P->P)
      t.b1 = (t.b1 + t.a1) % P->P1;
    uv_push(*dst, P->W + t.b1);
    num++;
  }
  return num;
}
