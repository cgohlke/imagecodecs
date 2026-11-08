/* pcx.h */

/*
 * SPDX-License-Identifier: 0BSD
 * See pcx.c file for details.
 */

/* Public declarations for pcx.c */

#ifndef PCX_H
#define PCX_H

#include <stddef.h>
#include <stdint.h>

#ifdef _MSC_VER
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

#define PCX_VERSION "2026.11.11"

#define PCX_OK 0
#define PCX_ERROR -1
#define PCX_VALUE_ERROR -2
#define PCX_OUTPUT_TOO_SMALL -3
#define PCX_INVALID_HEADER -4
#define PCX_UNSUPPORTED_FORMAT -5
#define PCX_DATA_TRUNCATED -6
#define PCX_IMAGE_TOO_LARGE -7
#define PCX_INDEX_ERROR -8
#define PCX_SHAPE_MISMATCH -9

#define PCX_DCX_MAXPAGES 1023

typedef struct pcx_info_t {
    ssize_t width;           /* image width in pixels */
    ssize_t height;          /* image height in pixels */
    ssize_t samples;         /* decoded samples per pixel */
    ssize_t bytes_per_line;  /* encoded bytes per scanline per plane */
    ssize_t frames;          /* number of pages in file */
    ssize_t index;           /* page index */
    ssize_t offset;          /* offset of page header */
    ssize_t end;             /* end of page data */
    int dcx;                 /* file is DCX container */
    int isbool;              /* decoded image is 1-bit monochrome */
    uint8_t version;
    uint8_t encoding;        /* 0: uncompressed, 1: RLE */
    uint8_t bits_per_pixel;  /* bits per pixel per plane */
    uint8_t nplanes;         /* number of color planes */
    uint16_t hdpi;
    uint16_t vdpi;
    uint16_t palette_type;
    uint8_t palette[768];    /* RGB palette */
} pcx_info_t;

int
pcx_check(
    const uint8_t* src,
    const ssize_t srcsize
);

int
pcx_info(
    const uint8_t* src,
    const ssize_t srcsize,
    const ssize_t index,
    pcx_info_t* info
);

ssize_t
pcx_decode(
    const uint8_t* src,
    const ssize_t srcsize,
    const ssize_t index,
    pcx_info_t* info,
    uint8_t* dst,
    const ssize_t dstsize
);

ssize_t
pcx_encode_size(
    const ssize_t width,
    const ssize_t height,
    const ssize_t samples,
    const ssize_t frames
);

ssize_t
pcx_encode(
    const uint8_t* src,
    const ssize_t width,
    const ssize_t height,
    const ssize_t samples,
    const ssize_t frames,
    uint8_t* dst,
    const ssize_t dstsize
);

#endif /* PCX_H */
