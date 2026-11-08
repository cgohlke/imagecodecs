/* bmp.h */

/*
 * SPDX-License-Identifier: 0BSD
 * See bmp.c file for details.
 */

/* Public declarations for bmp.c */

#ifndef BMP_H
#define BMP_H

#include <stddef.h>
#include <stdint.h>

#ifdef _MSC_VER
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

#define BMP_VERSION "2026.11.11"

#define BMP_OK 0
#define BMP_ERROR -1
#define BMP_VALUE_ERROR -2
#define BMP_OUTPUT_TOO_SMALL -3
#define BMP_INVALID_HEADER -4
#define BMP_UNSUPPORTED_FORMAT -5
#define BMP_DATA_TRUNCATED -6
#define BMP_IMAGE_TOO_LARGE -7
#define BMP_INDEX_ERROR -8

#define BMP_BI_RGB 0
#define BMP_BI_RLE8 1
#define BMP_BI_RLE4 2
#define BMP_BI_BITFIELDS 3
#define BMP_BI_JPEG 4
#define BMP_BI_PNG 5

typedef struct bmp_info_t {
    ssize_t width;         /* image width in pixels */
    ssize_t height;        /* image height in pixels (absolute) */
    ssize_t samples;       /* decoded samples per pixel; 0 for JPEG/PNG */
    ssize_t rowstride;     /* bytes per encoded row */
    int topdown;           /* rows are stored top-down */
    int has_alpha;         /* bitfields contain an alpha mask */
    uint32_t filesize;     /* file size from file header */
    uint32_t offbits;      /* offset to pixel data */
    uint32_t header_size;  /* size of info header */
    uint32_t compression;  /* BMP_BI_* */
    uint32_t size_image;   /* size of pixel data */
    uint32_t clr_used;     /* number of palette entries */
    uint16_t planes;
    uint16_t bitcount;
    int32_t x_ppm;
    int32_t y_ppm;
    uint32_t masks[4];     /* red, green, blue, alpha bitfield masks */
    int shifts[4];
    int bits[4];
    uint8_t palette[768];  /* RGB palette, zero-filled after clr_used */
} bmp_info_t;

int
bmp_check(
    const uint8_t* src,
    const ssize_t srcsize
);

int
bmp_info(
    const uint8_t* src,
    const ssize_t srcsize,
    const int asrgb,
    bmp_info_t* info
);

ssize_t
bmp_decode(
    const uint8_t* src,
    const ssize_t srcsize,
    const bmp_info_t* info,
    uint8_t* dst,
    const ssize_t dstsize
);

ssize_t
bmp_encode_size(
    const ssize_t width,
    const ssize_t height,
    const ssize_t samples
);

ssize_t
bmp_encode(
    const uint8_t* src,
    const ssize_t width,
    const ssize_t height,
    const ssize_t samples,
    const int32_t ppm,
    uint8_t* dst,
    const ssize_t dstsize
);

#endif /* BMP_H */
