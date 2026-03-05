/* ccitt.h */

/* Public declarations for ccitt.c */

#ifndef CCITT_H
#define CCITT_H

#include <stddef.h>
#include <stdint.h>

#ifdef _MSC_VER
#  include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#  include <sys/types.h>
#endif

#define CCITT_VERSION "2026.3.6"

#define CCITT_OK 0
#define CCITT_ERROR -1
#define CCITT_MEMORY_ERROR -2
#define CCITT_RUNTIME_ERROR -3
#define CCITT_VALUE_ERROR -4
#define CCITT_INPUT_CORRUPT -5
#define CCITT_OUTPUT_TOO_SMALL -6

void
ccitt_lut_init(
    void
);

ssize_t
ccitt_rle_decode_size(
    const uint8_t* src,
    const ssize_t srcsize,
    const ssize_t rowlen
);

ssize_t
ccitt_rle_decode(
    const uint8_t* src,
    const ssize_t srcsize,
    uint8_t* dst,
    const ssize_t dstsize,
    const ssize_t rowlen
);

ssize_t
ccitt_fax3_decode_size(
    const uint8_t* src,
    const ssize_t srcsize,
    const ssize_t rowlen,
    const int t4options
);

ssize_t
ccitt_fax3_decode(
    const uint8_t* src,
    const ssize_t srcsize,
    uint8_t* dst,
    const ssize_t dstsize,
    const ssize_t rowlen,
    const int t4options
);

ssize_t
ccitt_fax4_decode_size(
    const uint8_t* src,
    const ssize_t srcsize,
    const ssize_t rowlen
);

ssize_t
ccitt_fax4_decode(
    const uint8_t* src,
    const ssize_t srcsize,
    uint8_t* dst,
    const ssize_t dstsize,
    const ssize_t rowlen
);

#endif /* CCITT_H */
