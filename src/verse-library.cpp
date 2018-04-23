#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <numeric>
#include <omp.h>
#include <queue>
#include <vector>

using namespace std;

#ifdef _WIN32
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C"
#endif
#define SIGMOID_BOUND 6.0
#define DEFAULT_ALIGN 128
#define MAX_CODE_LENGTH 64

#if defined(__AVX2__) || defined(__FMA__)
#define VECTORIZE 1
#define AVX_LOOP _Pragma("omp simd")
#else
#define AVX_LOOP
#endif

#ifndef UINT64_C
#define UINT64_C(c) (c##ULL)
#endif

typedef unsigned long long ull;
typedef unsigned char byte;

const int sigmoid_table_size = 1024;
float *sigmoid_table;
const float SIGMOID_RESOLUTION = sigmoid_table_size / (SIGMOID_BOUND * 2.0f);

static uint64_t rng_seed[2];

static inline uint64_t rotl(const uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

// http://xoroshiro.di.unimi.it/#shootout
uint64_t lrand() {
  const uint64_t s0 = rng_seed[0];
  uint64_t s1 = rng_seed[1];
  const uint64_t result = s0 + s1;
  s1 ^= s0;
  rng_seed[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
  rng_seed[1] = rotl(s1, 36);                   // c
  return result;
}

void set_rnd_gen(ull seed) {
  for (int i = 0; i < 2; i++) {
    ull z = seed += UINT64_C(0x9E3779B97F4A7C15);
    z = (z ^ z >> 30) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ z >> 27) * UINT64_C(0x94D049BB133111EB);
    rng_seed[i] = z ^ z >> 31;
  }
}

static inline double drand() {
  const union un {
    uint64_t i;
    double d;
  } a = {UINT64_C(0x3FF) << 52 | lrand() >> 12};
  return a.d - 1.0;
}

inline int irand(int min, int max) { return lrand() % (max - min) + min; }

inline int irand(int max) { return lrand() % max; }

inline void *aligned_malloc(size_t size, size_t align) {
#ifndef _MSC_VER
  void *result;
  if (posix_memalign(&result, align, size))
    result = 0;
#else
  void *result = _aligned_malloc(size, align);
#endif
  return result;
}

inline void aligned_free(void *ptr) {
#ifdef _MSC_VER
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

void init_sigmoid_table() {
  for (int k = 0; k != sigmoid_table_size; k++) {
    double x = 2 * SIGMOID_BOUND * k / sigmoid_table_size - SIGMOID_BOUND;
    sigmoid_table[k] = 1 / (1 + exp(-x));
  }
}

float FastSigmoid(float x) {
  if (x > SIGMOID_BOUND)
    return 1;
  else if (x < -SIGMOID_BOUND)
    return 0;
  int k = (x + SIGMOID_BOUND) * SIGMOID_RESOLUTION;
  return sigmoid_table[k];
}

inline int sample_neighbor(const int *offsets, const int *edges,
                           const int node) {
  if (offsets[node] == offsets[node + 1])
    return -1;
  return edges[irand(offsets[node], offsets[node + 1])];
}

inline int sample_rw(const int *offsets, const int *edges, const int node,
                     const double alpha) {
  int n2 = node;
  while (drand() < alpha) {
    int neighbor = sample_neighbor(offsets, edges, n2);
    if (neighbor == -1)
      return n2;
    n2 = neighbor;
  }
  return n2;
}

inline void update(float *w_s, float *w_t, const int label, const int n_hidden,
                   const float lr, const float bias) {
  float score = -bias;
  AVX_LOOP
  for (int c = 0; c < n_hidden; c++)
    score += w_s[c] * w_t[c];
  score = (label - FastSigmoid(score)) * lr;
  AVX_LOOP
  for (int c = 0; c < n_hidden; c++)
    w_t[c] += score * w_s[c];
  AVX_LOOP
  for (int c = 0; c < n_hidden; c++)
    w_s[c] += score * w_t[c];
}

inline void update(float *w_s, float *w_t, float *w_t_cache, const int label,
                   const int n_hidden, const float lr) {
  float score = 0;
  AVX_LOOP
  for (int c = 0; c < n_hidden; c++)
    score += w_s[c] * w_t[c];
  score = (label - FastSigmoid(score)) * lr;
  AVX_LOOP
  for (int c = 0; c < n_hidden; c++)
    w_t_cache[c] += score * w_s[c];
  AVX_LOOP
  for (int c = 0; c < n_hidden; c++)
    w_s[c] += score * w_t[c];
}

DLLEXPORT int verse_ppr_train(float *w0, int *offsets, int *edges,
                              int num_nodes, int num_edges, int n_hidden,
                              int steps, int n_neg_samples, float lr,
                              float alpha, int rng_seed, int n_threads) {
  if (rng_seed != 0)
    set_rnd_gen(rng_seed);
  else
    set_rnd_gen(time(nullptr));
  sigmoid_table = static_cast<float *>(
      aligned_malloc((sigmoid_table_size + 1) * sizeof(float), DEFAULT_ALIGN));
  init_sigmoid_table();
  ull step = 0;
#pragma omp parallel num_threads(n_threads)
  {
    const ull total_steps = steps * ull(num_nodes);
    const float nce_bias = log(num_nodes);
    const float nce_bias_neg = log(num_nodes / float(n_neg_samples));
    ull last_ncount = 0;
    ull ncount = 0;
#pragma omp barrier
    while (true) {
      if (ncount - last_ncount > 10000) {
        ull diff = ncount - last_ncount;
#pragma omp atomic
        step += diff;
        if (step > total_steps)
          break;
        last_ncount = ncount;
      }
      int n1 = irand(num_nodes);
      int n2 = sample_rw(offsets, edges, n1, alpha);
      update(&w0[n1 * n_hidden], &w0[n2 * n_hidden], 1, n_hidden, lr, nce_bias);
      for (int i = 0; i < n_neg_samples; i++) {
        int neg = irand(num_nodes);
        update(&w0[n1 * n_hidden], &w0[neg * n_hidden], 0, n_hidden, lr,
               nce_bias_neg);
      }
      ncount++;
    }
  }
  aligned_free(sigmoid_table);
  return 0;
}

DLLEXPORT int verse_neigh_train(float *w0, int *offsets, int *edges,
                                int num_nodes, int num_edges, int n_hidden,
                                int steps, int n_neg_samples, float lr,
                                int rng_seed, int n_threads) {
  if (rng_seed != 0)
    set_rnd_gen(rng_seed);
  else
    set_rnd_gen(time(nullptr));
  sigmoid_table = static_cast<float *>(
      aligned_malloc((sigmoid_table_size + 1) * sizeof(float), DEFAULT_ALIGN));
  init_sigmoid_table();
  ull step = 0;
#pragma omp parallel num_threads(n_threads)
  {
    const ull total_steps = steps * ull(num_nodes);
    const float nce_bias = log(num_nodes);
    const float nce_bias_neg = log(num_nodes / float(n_neg_samples));
    ull last_ncount = 0;
    ull ncount = 0;
#pragma omp barrier
    while (true) {
      if (ncount - last_ncount > 10000) {
        ull diff = ncount - last_ncount;
#pragma omp atomic
        step += diff;
        if (step > total_steps)
          break;
        last_ncount = ncount;
      }
      int n1 = irand(num_nodes);
      int n2 = sample_neighbor(offsets, edges, n1);
      if (n2 == -1)
        continue;
      update(&w0[n1 * n_hidden], &w0[n2 * n_hidden], 1, n_hidden, lr, nce_bias);
      for (int i = 0; i < n_neg_samples; i++) {
        int neg = irand(num_nodes);
        update(&w0[n1 * n_hidden], &w0[neg * n_hidden], 0, n_hidden, lr,
               nce_bias_neg);
      }
      ncount++;
    }
  }
  aligned_free(sigmoid_table);
  return 0;
}