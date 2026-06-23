/** libkeccak-tiny
 *
 * A single-file implementation of SHA-3 and SHAKE.
 *
 * Implementor: David Leon Gil
 * License: CC0, attribution kindly requested. Blame taken too,
 * but not liability.
 */
#define __STDC_WANT_LIB_EXT1__ 1

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/******** The Keccak-f[1600] permutation ********/

/*** Constants. ***/
__device__ static const uint8_t rho[24] = \
  { 1,  3,   6, 10, 15, 21,
    28, 36, 45, 55,  2, 14,
    27, 41, 56,  8, 25, 43,
    62, 18, 39, 61, 20, 44};
__device__ static const uint8_t pi[24] = \
  {10,  7, 11, 17, 18, 3,
    5, 16,  8, 21, 24, 4,
   15, 23, 19, 13, 12, 2,
   20, 14, 22,  9, 6,  1};
__device__ static const uint64_t RC[24] = \
  {1ULL, 0x8082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
   0x808bULL, 0x80000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
   0x8aULL, 0x88ULL, 0x80008009ULL, 0x8000000aULL,
   0x8000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
   0x8000000000008002ULL, 0x8000000000000080ULL, 0x800aULL, 0x800000008000000aULL,
   0x8000000080008081ULL, 0x8000000000008080ULL, 0x80000001ULL, 0x8000000080008008ULL};

/*** Helper macros to unroll the permutation. ***/
#define rol(x, s) (((x) << s) | ((x) >> (64 - s)))

/* Register-resident 3-input bitwise primitives for theta/chi.
 * On device each 64-bit op lowers to two lop3.b32 (one per 32-bit half):
 *   eor3(a,b,c) = a ^ b ^ c                 (LUT 0x96) — theta column mix
 *   bcax(a,b,c) = a ^ (~b & c)              (LUT 0xD2) — chi step
 * Bit-exact with the reference expressions (verified on RTX 3090, sm_86);
 * the host fallback below keeps non-device compilation passes valid. */
__device__ static __forceinline__ uint64_t eor3(uint64_t a, uint64_t b, uint64_t c) {
#if defined(__CUDA_ARCH__)
  uint32_t al = (uint32_t)a, ah = (uint32_t)(a >> 32);
  uint32_t bl = (uint32_t)b, bh = (uint32_t)(b >> 32);
  uint32_t cl = (uint32_t)c, ch = (uint32_t)(c >> 32);
  uint32_t rl, rh;
  asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(rl) : "r"(al), "r"(bl), "r"(cl));
  asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(rh) : "r"(ah), "r"(bh), "r"(ch));
  return ((uint64_t)rh << 32) | rl;
#else
  return a ^ b ^ c;
#endif
}
__device__ static __forceinline__ uint64_t bcax(uint64_t a, uint64_t b, uint64_t c) {
#if defined(__CUDA_ARCH__)
  uint32_t al = (uint32_t)a, ah = (uint32_t)(a >> 32);
  uint32_t bl = (uint32_t)b, bh = (uint32_t)(b >> 32);
  uint32_t cl = (uint32_t)c, ch = (uint32_t)(c >> 32);
  uint32_t rl, rh;
  asm("lop3.b32 %0, %1, %2, %3, 0xD2;" : "=r"(rl) : "r"(al), "r"(bl), "r"(cl));
  asm("lop3.b32 %0, %1, %2, %3, 0xD2;" : "=r"(rh) : "r"(ah), "r"(bh), "r"(ch));
  return ((uint64_t)rh << 32) | rl;
#else
  return a ^ (~b & c);
#endif
}
#define REPEAT6(e) e e e e e e
#define REPEAT24(e) REPEAT6(e e e e)
#define REPEAT5(e) e e e e e
#define FOR5(v, s, e) \
  v = 0;            \
  REPEAT5(e; v += s;)

/*** Keccak-f[1600] ***/
__device__ static inline void keccakf(void* state) {
  uint64_t* a = (uint64_t*)state;
  uint64_t b[5] = {0};
  uint64_t t = 0;
  uint8_t x, y;

  #pragma unroll
  for (int i = 0; i < 24; i++) {
    // Theta
    FOR5(x, 1,
         b[x] = 0;
         FOR5(y, 5,
              b[x] ^= a[x + y]; ))
    FOR5(x, 1,
         FOR5(y, 5,
              a[y + x] = eor3(a[y + x], b[(x + 4) % 5], rol(b[(x + 1) % 5], 1)); ))
    // Rho and pi
    t = a[1];
    x = 0;
    REPEAT24(b[0] = a[pi[x]];
             a[pi[x]] = rol(t, rho[x]);
             t = b[0];
             x++; )
    // Chi
    FOR5(y,
       5,
       FOR5(x, 1,
            b[x] = a[y + x];)
       FOR5(x, 1,
            a[y + x] = bcax(b[x], b[(x + 1) % 5], b[(x + 2) % 5]); ))
    // Iota
    a[0] ^= RC[i];
  }
}

/******** The FIPS202-defined functions. ********/

/*** Some helper macros. ***/
#define P keccakf
#define Plen 200


/** The sponge-based hash construction. **/
__device__ __forceinline__ static void hash(
                       const uint8_t initP[Plen],
                       uint8_t* out,
                       const uint8_t* in) {
  uint8_t a[Plen] = {0};

  #pragma unroll
  for (int i=0; i<10; i++) ((uint64_t *)a)[i] = ((uint64_t *)initP)[i] ^ ((uint64_t *)in)[i];
  #pragma unroll
  for (int i=10; i<25; i++) ((uint64_t *)a)[i] = ((uint64_t *)initP)[i];

  // Apply P
  P(a);
  // Squeeze output.
  #pragma unroll
  for (int i=0; i<4; i++) ((uint64_t *)out)[i] = ((uint64_t *)a)[i];

}

