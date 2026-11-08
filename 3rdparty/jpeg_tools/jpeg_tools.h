/* jpeg_tools.h */

/*
 * SPDX-License-Identifier: 0BSD
 * See jpeg_tools.c file for details.
 */

/* Public declarations for jpeg_tools.c */

#ifndef JPEG_TOOLS_H
#define JPEG_TOOLS_H

#include <stddef.h>
#include <stdint.h>

#ifdef _MSC_VER
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

#define JPEG_TOOLS_VERSION "2026.11.11"

#define JPEG_TOOLS_OK 0
#define JPEG_TOOLS_ERROR -1
#define JPEG_TOOLS_VALUE_ERROR -2
#define JPEG_TOOLS_OUTPUT_TOO_SMALL -3
#define JPEG_TOOLS_INVALID_HEADER -4
#define JPEG_TOOLS_UNSUPPORTED_FORMAT -5
#define JPEG_TOOLS_DATA_TRUNCATED -6
#define JPEG_TOOLS_IMAGE_TOO_LARGE -7
#define JPEG_TOOLS_SHAPE_MISMATCH -8
#define JPEG_TOOLS_TILE_MISSING -9
#define JPEG_TOOLS_MEMORY_ERROR -10
#define JPEG_TOOLS_LIBJPEG_ERROR -11

#define JPEG_TOOLS_MSG_LENGTH 200  /* JMSG_LENGTH_MAX */

/* Tiles are arranged in row-major order. A tile is missing if its data
   pointer is NULL or its size is 0. If header is not NULL, tiles contain
   entropy-coded data only. */
typedef struct jpeg_tools_input_t {
    const uint8_t** data;
    const ssize_t* datasize;
    ssize_t rows;
    ssize_t columns;
    const uint8_t* header;
    ssize_t headersize;
    const uint8_t* tables;
    ssize_t tablessize;
    int colorspace;  /* J_COLOR_SPACE of components or 0 if unknown */
} jpeg_tools_input_t;

typedef struct jpeg_tools_options_t {
    ssize_t crop_top;     /* must be multiple of iMCU height */
    ssize_t crop_left;    /* must be multiple of iMCU width */
    ssize_t crop_height;  /* 0: to bottom of tile grid */
    ssize_t crop_width;   /* 0: to right of tile grid */
    int optimize;         /* -1: default, 0: no, 1: yes */
    int progressive;      /* -1: default, 0: sequential, 1: progressive */
    int arithmetic;       /* 0: Huffman, 1: arithmetic */
    int restart;          /* restart interval in MCU rows if > 0 */
    int grayscale;        /* discard chroma components if 1 */
    int copy;             /* markers, 0: none, 1: COM, 2: ICC, 3: all */
} jpeg_tools_options_t;

/* Merge JPEG header, tables, and tiles without decoding entropy-coded data.
   Return size of merged stream or negative error code.
   If dst is NULL, return required size. */
ssize_t
jpeg_tools_merge(
    const jpeg_tools_input_t* input,
    uint8_t* dst,
    const ssize_t dstsize
);

/* Losslessly transcode, crop, and mosaic JPEG tiles in DCT domain.
   *dst and *dstsize are handled like jpeg_mem_dest. On error, *dst is
   restored and a libjpeg message may be written to msg. */
int
jpeg_tools_transcode(
    const jpeg_tools_input_t* input,
    const jpeg_tools_options_t* options,
    uint8_t** dst,
    unsigned long* dstsize,
    char* msg
);

#endif /* JPEG_TOOLS_H */
