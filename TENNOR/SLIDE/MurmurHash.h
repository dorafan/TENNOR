#ifndef MURMURHASH_H
#define MURMURHASH_H 1

#include <stdint.h>

#define MURMURHASH_VERSION "0.0.3"

#ifdef __cplusplus
extern "C" {
#endif






uint32_t MurmurHash (const char *, uint32_t, uint32_t);

#ifdef __cplusplus
}
#endif

#endif