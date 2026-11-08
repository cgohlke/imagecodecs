/* tga.h */

/*
 * SPDX-License-Identifier: 0BSD
 * See tga.c file for details.
 */

/* Public declarations for tga.c */

#ifndef TGA_H
#define TGA_H

#include <stddef.h>
#include <stdint.h>

#ifdef _MSC_VER
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

#define TGA_VERSION "2026.11.11"

#define TGA_OK 0
#define TGA_ERROR -1
#define TGA_VALUE_ERROR -2
#define TGA_OUTPUT_TOO_SMALL -3
#define TGA_INVALID_HEADER -4
#define TGA_UNSUPPORTED_FORMAT -5
#define TGA_DATA_TRUNCATED -6
#define TGA_IMAGE_TOO_LARGE -7

#define TGA_TYPE_NONE 0
#define TGA_TYPE_COLORMAP 1
#define TGA_TYPE_TRUECOLOR 2
#define TGA_TYPE_GRAYSCALE 3
#define TGA_TYPE_RLE_COLORMAP 9
#define TGA_TYPE_RLE_TRUECOLOR 10
#define TGA_TYPE_RLE_GRAYSCALE 11

/* TGA 2.0 extension area; strings are NUL-terminated */
typedef struct tga_extension_t {
    int version;               /* 1 or 2 (has footer) */
    int valid;                 /* extension area present */
    char author[42];
    char comments[4][82];
    uint16_t month;
    uint16_t day;
    uint16_t year;
    uint16_t hour;
    uint16_t minute;
    uint16_t second;
    char job_name[42];
    uint16_t job_hours;
    uint16_t job_minutes;
    uint16_t job_seconds;
    char software[42];
    uint16_t software_version;
    uint32_t key_color;
    uint16_t aspect_numerator;
    uint16_t aspect_denominator;
    uint16_t gamma_numerator;
    uint16_t gamma_denominator;
    uint32_t color_correction_offset;
    uint32_t postage_stamp_offset;
    uint32_t scanline_offset;
    uint8_t software_letter;
    uint8_t attributes_type;
} tga_extension_t;

typedef struct tga_info_t {
    ssize_t width;             /* image width in pixels */
    ssize_t height;            /* image height in pixels */
    ssize_t samples;           /* decoded samples per pixel */
    ssize_t pixel_size;        /* encoded bytes per pixel */
    ssize_t pixel_offset;      /* offset to pixel data */
    ssize_t colormap_offset;   /* offset to colormap */
    ssize_t cmap_entry_size;   /* bytes per colormap entry */
    int alpha_bits;
    int topdown;
    int righttoleft;
    int rle;
    int colormapped;
    uint8_t id_length;
    uint8_t colormap_type;
    uint8_t image_type;        /* TGA_TYPE_* */
    uint8_t colormap_depth;
    uint8_t pixel_depth;
    uint8_t image_descriptor;
    uint16_t colormap_origin;
    uint16_t colormap_length;
    uint16_t x_origin;
    uint16_t y_origin;
    char image_id[256];
    tga_extension_t extension;
} tga_info_t;

int
tga_check(
    const uint8_t* src,
    const ssize_t srcsize
);

int
tga_info(
    const uint8_t* src,
    const ssize_t srcsize,
    tga_info_t* info
);

ssize_t
tga_decode(
    const uint8_t* src,
    const ssize_t srcsize,
    const tga_info_t* info,
    uint8_t* dst,
    const ssize_t dstsize
);

ssize_t
tga_encode_size(
    const ssize_t width,
    const ssize_t height,
    const ssize_t samples,
    const int rle
);

ssize_t
tga_encode(
    const uint8_t* src,
    const ssize_t width,
    const ssize_t height,
    const ssize_t samples,
    const int rle,
    uint8_t* dst,
    const ssize_t dstsize
);

#endif /* TGA_H */
