/* pixarlog.h */

/* Public declarations for pixarlog.c */

#ifndef PIXARLOG_H
#define PIXARLOG_H

#include <stddef.h>
#include <stdint.h>

#ifdef _MSC_VER
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

#define PIXARLOG_VERSION "2026.3.2"

#define PIXARLOG_FMT_8BIT 0
#define PIXARLOG_FMT_8BITABGR 1
#define PIXARLOG_FMT_11BITLOG 2
#define PIXARLOG_FMT_12BITPICIO 3
#define PIXARLOG_FMT_16BIT 4
#define PIXARLOG_FMT_FLOAT 5

#define PIXARLOG_OK 0
#define PIXARLOG_ERROR -1
#define PIXARLOG_MEMORY_ERROR -2
#define PIXARLOG_VALUE_ERROR -3
#define PIXARLOG_OUTPUT_TOO_SMALL -4
#define PIXARLOG_ZLIB_ERROR -5

void
pixarlog_init(
    void
);

ssize_t
pixarlog_decode(
    const uint8_t* src,
    const ssize_t srcsize,
    uint8_t* dst,
    const ssize_t dstsize,
    const ssize_t width,
    const ssize_t stride,
    const int datafmt
);

ssize_t
pixarlog_encode(
    const uint8_t* src,
    const ssize_t srcsize,
    uint8_t* dst,
    const ssize_t dstsize,
    const ssize_t width,
    const ssize_t stride,
    const int datafmt,
    const int level
);

ssize_t
pixarlog_decode_raw(
    const uint8_t* src,
    const ssize_t srcsize,
    uint8_t* dst,
    const ssize_t dstsize,
    const ssize_t width,
    const ssize_t stride,
    const int datafmt
);

ssize_t
pixarlog_encode_raw(
    const uint8_t* src,
    const ssize_t srcsize,
    uint8_t* dst,
    const ssize_t dstsize,
    const ssize_t width,
    const ssize_t stride,
    const int datafmt
);

#endif /* PIXARLOG_H */
