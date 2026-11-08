/* tga.c */

/* TGA (Truevision TARGA) image codec. */

/* Initially generated with assistance from Claude (Anthropic),
   then reviewed and adapted by Christoph Gohlke. */

/*
SPDX-License-Identifier: 0BSD

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
*/

#include "tga.h"

#include <string.h>

#define TGA_HEADER_SIZE 18
#define TGA_FOOTER_SIZE 26
#define TGA_EXTENSION_SIZE 495

static const char TGA_SIGNATURE[18] = "TRUEVISION-XFILE.";


static inline uint16_t
tga_u16(
    const uint8_t* p)
{
    return (uint16_t)(p[0] | ((uint16_t)p[1] << 8));
}


static inline uint32_t
tga_u32(
    const uint8_t* p)
{
    return (
        (uint32_t)p[0]
        | ((uint32_t)p[1] << 8)
        | ((uint32_t)p[2] << 16)
        | ((uint32_t)p[3] << 24)
    );
}


static inline void
tga_put_u16(
    uint8_t* p,
    const uint16_t v)
{
    p[0] = (uint8_t)(v & 0xFF);
    p[1] = (uint8_t)(v >> 8);
}


/* Copy at most size bytes up to first NUL; dst must hold size + 1 bytes. */
static void
tga_str(
    char* dst,
    const uint8_t* src,
    const ssize_t size)
{
    ssize_t i;
    for (i = 0; i < size && src[i] != 0; i++) {
        dst[i] = (char)src[i];
    }
    dst[i] = 0;
}


static int
tga_has_footer(
    const uint8_t* src,
    const ssize_t srcsize)
{
    return (
        srcsize >= TGA_HEADER_SIZE + TGA_FOOTER_SIZE
        && memcmp(src + srcsize - 18, TGA_SIGNATURE, 18) == 0
    );
}


/* Swap R and B of RGB(A); the conversion is its own inverse. */
static inline void
tga_pixel(
    uint8_t* dst,
    const uint8_t* src,
    const ssize_t samples)
{
    if (samples < 3) {
        dst[0] = src[0];
        if (samples == 2) {
            dst[1] = src[1];
        }
    }
    else {
        dst[0] = src[2];
        dst[1] = src[1];
        dst[2] = src[0];
        if (samples == 4) {
            dst[3] = src[3];
        }
    }
}


static inline void
tga_rgb555(
    uint8_t* dst,
    const uint8_t* src)
{
    const uint16_t v = tga_u16(src);
    const uint8_t r = (v >> 10) & 0x1F;
    const uint8_t g = (v >> 5) & 0x1F;
    const uint8_t b = v & 0x1F;
    dst[0] = (uint8_t)((r << 3) | (r >> 2));
    dst[1] = (uint8_t)((g << 3) | (g >> 2));
    dst[2] = (uint8_t)((b << 3) | (b >> 2));
}


int
tga_check(
    const uint8_t* src,
    const ssize_t srcsize)
{
    uint8_t t;
    if (src == NULL || srcsize < TGA_HEADER_SIZE) {
        return 0;
    }
    if (tga_has_footer(src, srcsize)) {
        return 1;
    }
    /* heuristic for TGA 1.0 files */
    t = src[2];
    if (src[1] > 1) {
        return 0;
    }
    if (t != 1 && t != 2 && t != 3 && t != 9 && t != 10 && t != 11) {
        return 0;
    }
    if (src[1] == 0 && (t == 1 || t == 9)) {
        return 0;
    }
    t = src[16];
    if (t != 8 && t != 15 && t != 16 && t != 24 && t != 32) {
        return 0;
    }
    if ((src[12] | src[13]) == 0 || (src[14] | src[15]) == 0) {
        return 0;
    }
    return -1;
}


static void
tga_extension(
    const uint8_t* src,
    const ssize_t srcsize,
    tga_extension_t* ext)
{
    const uint8_t* p;
    ssize_t offset;
    int k;

    ext->version = 1;
    if (!tga_has_footer(src, srcsize)) {
        return;
    }
    ext->version = 2;
    offset = (ssize_t)tga_u32(src + srcsize - TGA_FOOTER_SIZE);
    if (
        offset < TGA_HEADER_SIZE
        || offset > srcsize - TGA_FOOTER_SIZE - TGA_EXTENSION_SIZE
        || tga_u16(src + offset) < TGA_EXTENSION_SIZE)
    {
        return;
    }
    p = src + offset;
    ext->valid = 1;
    tga_str(ext->author, p + 2, 41);
    for (k = 0; k < 4; k++) {
        tga_str(ext->comments[k], p + 43 + k * 81, 81);
    }
    ext->month = tga_u16(p + 367);
    ext->day = tga_u16(p + 369);
    ext->year = tga_u16(p + 371);
    ext->hour = tga_u16(p + 373);
    ext->minute = tga_u16(p + 375);
    ext->second = tga_u16(p + 377);
    tga_str(ext->job_name, p + 379, 41);
    ext->job_hours = tga_u16(p + 420);
    ext->job_minutes = tga_u16(p + 422);
    ext->job_seconds = tga_u16(p + 424);
    tga_str(ext->software, p + 426, 41);
    ext->software_version = tga_u16(p + 467);
    ext->software_letter = p[469];
    ext->key_color = tga_u32(p + 470);
    ext->aspect_numerator = tga_u16(p + 474);
    ext->aspect_denominator = tga_u16(p + 476);
    ext->gamma_numerator = tga_u16(p + 478);
    ext->gamma_denominator = tga_u16(p + 480);
    ext->color_correction_offset = tga_u32(p + 482);
    ext->postage_stamp_offset = tga_u32(p + 486);
    ext->scanline_offset = tga_u32(p + 490);
    ext->attributes_type = p[494];
}


int
tga_info(
    const uint8_t* src,
    const ssize_t srcsize,
    tga_info_t* info)
{
    int64_t npixels;

    if (info == NULL) {
        return TGA_VALUE_ERROR;
    }
    memset(info, 0, sizeof(tga_info_t));

    if (src == NULL || srcsize < 0) {
        return TGA_VALUE_ERROR;
    }
    if (srcsize < TGA_HEADER_SIZE) {
        return TGA_DATA_TRUNCATED;
    }

    info->id_length = src[0];
    info->colormap_type = src[1];
    info->image_type = src[2];
    info->colormap_origin = tga_u16(src + 3);
    info->colormap_length = tga_u16(src + 5);
    info->colormap_depth = src[7];
    info->x_origin = tga_u16(src + 8);
    info->y_origin = tga_u16(src + 10);
    info->width = (ssize_t)tga_u16(src + 12);
    info->height = (ssize_t)tga_u16(src + 14);
    info->pixel_depth = src[16];
    info->image_descriptor = src[17];

    if (info->colormap_type > 1) {
        return TGA_UNSUPPORTED_FORMAT;
    }
    switch (info->image_type)
    {
        case TGA_TYPE_COLORMAP:
        case TGA_TYPE_TRUECOLOR:
        case TGA_TYPE_GRAYSCALE:
            break;
        case TGA_TYPE_RLE_COLORMAP:
        case TGA_TYPE_RLE_TRUECOLOR:
        case TGA_TYPE_RLE_GRAYSCALE:
            info->rle = 1;
            break;
        default:
            return TGA_UNSUPPORTED_FORMAT;
    }
    if (info->width == 0 || info->height == 0) {
        return TGA_INVALID_HEADER;
    }
    npixels = (int64_t)info->width * info->height;
    info->alpha_bits = info->image_descriptor & 0xF;
    info->righttoleft = (info->image_descriptor & 0x10) != 0;
    info->topdown = (info->image_descriptor & 0x20) != 0;
    info->colormapped = (
        info->image_type == TGA_TYPE_COLORMAP
        || info->image_type == TGA_TYPE_RLE_COLORMAP
    );
    if (info->colormapped && info->colormap_type != 1) {
        return TGA_INVALID_HEADER;
    }

    info->cmap_entry_size = (
        info->colormap_type == 1 ? ((ssize_t)info->colormap_depth + 7) / 8 : 0
    );
    info->colormap_offset = TGA_HEADER_SIZE + (ssize_t)info->id_length;
    info->pixel_offset = (
        info->colormap_offset
        + (ssize_t)info->colormap_length * info->cmap_entry_size
    );
    if (info->pixel_offset > srcsize) {
        return TGA_DATA_TRUNCATED;
    }

    if (
        info->image_type == TGA_TYPE_GRAYSCALE
        || info->image_type == TGA_TYPE_RLE_GRAYSCALE)
    {
        if (info->pixel_depth == 8) {
            info->samples = 1;
            info->pixel_size = 1;
        }
        else if (info->pixel_depth == 16 && info->alpha_bits == 8) {
            /* 8-bit luminance + 8-bit alpha */
            info->samples = 2;
            info->pixel_size = 2;
        }
        else {
            return TGA_UNSUPPORTED_FORMAT;
        }
    }
    else if (info->colormapped) {
        if (info->pixel_depth != 8) {
            return TGA_UNSUPPORTED_FORMAT;
        }
        switch (info->colormap_depth)
        {
            case 15:
            case 16:
            case 24:
                info->samples = 3;
                break;
            case 32:
                info->samples = 4;
                break;
            default:
                return TGA_UNSUPPORTED_FORMAT;
        }
        info->pixel_size = 1;
    }
    else {
        switch (info->pixel_depth)
        {
            case 15:
            case 16:
                info->samples = (
                    info->pixel_depth == 16 && info->alpha_bits == 1 ? 4 : 3
                );
                info->pixel_size = 2;
                break;
            case 24:
                info->samples = 3;
                info->pixel_size = 3;
                break;
            case 32:
                info->samples = 4;
                info->pixel_size = 4;
                break;
            default:
                return TGA_UNSUPPORTED_FORMAT;
        }
    }

    if (
        !info->rle
        && (int64_t)info->pixel_offset
        + (int64_t)npixels * info->pixel_size
        > (int64_t)srcsize)
    {
        return TGA_DATA_TRUNCATED;
    }
    if ((int64_t)npixels * info->samples > (int64_t)PTRDIFF_MAX) {
        return TGA_IMAGE_TOO_LARGE;
    }

    tga_str(info->image_id, src + TGA_HEADER_SIZE, info->id_length);
    tga_extension(src, srcsize, &info->extension);
    return TGA_OK;
}


/* Convert one encoded pixel to samples. */
static inline void
tga_convert(
    const tga_info_t* info,
    const uint8_t* palette,
    uint8_t* dst,
    const uint8_t* src)
{
    if (info->colormapped) {
        memcpy(dst, palette + (ssize_t)src[0] * 4, (size_t)info->samples);
    }
    else if (info->pixel_size == 2 && info->samples > 2) {
        tga_rgb555(dst, src);
        if (info->samples == 4) {
            dst[3] = (src[1] & 0x80) ? 255 : 0;
        }
    }
    else {
        tga_pixel(dst, src, info->samples);
    }
}


typedef struct tga_writer_t {
    const tga_info_t* info;
    uint8_t* dst;
    uint8_t* drow;
    ssize_t row;   /* file row */
    ssize_t col;   /* file column */
} tga_writer_t;


static inline void
tga_writer_row(
    tga_writer_t* w)
{
    const tga_info_t* info = w->info;
    const ssize_t row = info->topdown ? w->row : info->height - 1 - w->row;
    w->drow = w->dst + row * info->width * info->samples;
}


/* Return output location of next pixel in file order. */
static inline uint8_t*
tga_writer_next(
    tga_writer_t* w)
{
    const tga_info_t* info = w->info;
    uint8_t* d;
    const ssize_t col = info->righttoleft ? info->width - 1 - w->col : w->col;
    d = w->drow + col * info->samples;
    if (++w->col == info->width) {
        w->col = 0;
        if (++w->row < info->height) {
            tga_writer_row(w);
        }
    }
    return d;
}


ssize_t
tga_decode(
    const uint8_t* src,
    const ssize_t srcsize,
    const tga_info_t* info,
    uint8_t* dst,
    const ssize_t dstsize)
{
    const ssize_t samples = info->samples;
    const ssize_t pixel_size = info->pixel_size;
    const ssize_t npixels = info->width * info->height;
    uint8_t palette[1024];
    uint8_t pixel[4];
    const uint8_t* entry;
    tga_writer_t w;
    ssize_t i, k, count, total, cmap_index;
    uint8_t rle_hdr;

    if (src == NULL || dst == NULL || samples < 1 || pixel_size < 1) {
        return TGA_VALUE_ERROR;
    }
    if ((int64_t)info->width * info->height * samples > (int64_t)dstsize) {
        return TGA_OUTPUT_TOO_SMALL;
    }
    /* each RLE packet of at least 1 + pixel_size bytes yields <= 128 px */
    if (
        info->rle
        && (int64_t)npixels
        > (int64_t)(srcsize - info->pixel_offset) / (1 + pixel_size)
        * 128)
    {
        return TGA_DATA_TRUNCATED;
    }
    if (
        !info->rle
        && (int64_t)info->pixel_offset + (int64_t)npixels * pixel_size
        > (int64_t)srcsize)
    {
        return TGA_DATA_TRUNCATED;
    }

    if (info->colormapped) {
        memset(palette, 0, sizeof(palette));
        for (i = 0; i < 256; i++) {
            cmap_index = i - (ssize_t)info->colormap_origin;
            if (
                cmap_index < 0
                || cmap_index >= (ssize_t)info->colormap_length)
            {
                continue;
            }
            entry = (
                src + info->colormap_offset + cmap_index *
                info->cmap_entry_size
            );
            if (info->cmap_entry_size == 2) {
                tga_rgb555(palette + i * 4, entry);
            }
            else {
                tga_pixel(palette + i * 4, entry, info->cmap_entry_size);
            }
        }
    }

    w.info = info;
    w.dst = dst;
    w.row = 0;
    w.col = 0;
    tga_writer_row(&w);

    if (!info->rle) {
        const uint8_t* s = src + info->pixel_offset;
        if (pixel_size == samples && samples <= 2 && !info->righttoleft) {
            /* grayscale rows are stored as is */
            for (i = 0; i < info->height; i++) {
                memcpy(w.drow, s, (size_t)(info->width * samples));
                s += info->width * samples;
                w.row++;
                if (w.row < info->height) {
                    tga_writer_row(&w);
                }
            }
        }
        else {
            for (i = 0; i < npixels; i++) {
                tga_convert(info, palette, tga_writer_next(&w), s);
                s += pixel_size;
            }
        }
        return npixels * samples;
    }

    /* decode RLE packets directly into output */
    i = info->pixel_offset;
    total = 0;
    while (total < npixels && i < srcsize) {
        rle_hdr = src[i++];
        count = (rle_hdr & 0x7F) + 1;
        if (count > npixels - total) {
            count = npixels - total;
        }
        if (rle_hdr & 0x80) {
            if (i + pixel_size > srcsize) {
                break;
            }
            tga_convert(info, palette, pixel, src + i);
            for (k = 0; k < count; k++) {
                memcpy(tga_writer_next(&w), pixel, (size_t)samples);
            }
            i += pixel_size;
        }
        else {
            if (i + count * pixel_size > srcsize) {
                break;
            }
            for (k = 0; k < count; k++) {
                tga_convert(info, palette, tga_writer_next(&w), src + i);
                i += pixel_size;
            }
        }
        total += count;
    }
    if (total < npixels) {
        return TGA_DATA_TRUNCATED;
    }
    return npixels * samples;
}


ssize_t
tga_encode_size(
    const ssize_t width,
    const ssize_t height,
    const ssize_t samples,
    const int rle)
{
    int64_t size;
    if (
        width <= 0
        || width > UINT16_MAX
        || height <= 0
        || height > UINT16_MAX
        || samples < 1
        || samples > 4)
    {
        return TGA_VALUE_ERROR;
    }
    size = (
        TGA_HEADER_SIZE
        + TGA_FOOTER_SIZE
        + (int64_t)height * width * (rle ? samples + 1 : samples)
    );
    if (size > (int64_t)PTRDIFF_MAX) {
        return TGA_VALUE_ERROR;
    }
    return (ssize_t)size;
}


ssize_t
tga_encode(
    const uint8_t* src,
    const ssize_t width,
    const ssize_t height,
    const ssize_t samples,
    const int rle,
    uint8_t* dst,
    const ssize_t dstsize)
{
    const ssize_t rowsize = width * samples;
    const uint8_t* srow;
    const uint8_t* px;
    ssize_t minsize, dstindex, i, j, k, run, raw;

    minsize = tga_encode_size(width, height, samples, rle);
    if (minsize < 0) {
        return minsize;
    }
    if (src == NULL || dst == NULL) {
        return TGA_VALUE_ERROR;
    }
    if (
        dstsize < TGA_HEADER_SIZE + TGA_FOOTER_SIZE
        || (!rle && dstsize < minsize))
    {
        return TGA_OUTPUT_TOO_SMALL;
    }

    memset(dst, 0, TGA_HEADER_SIZE);
    if (samples <= 2) {
        dst[2] = rle ? TGA_TYPE_RLE_GRAYSCALE : TGA_TYPE_GRAYSCALE;
    }
    else {
        dst[2] = rle ? TGA_TYPE_RLE_TRUECOLOR : TGA_TYPE_TRUECOLOR;
    }
    tga_put_u16(dst + 12, (uint16_t)width);
    tga_put_u16(dst + 14, (uint16_t)height);
    dst[16] = (uint8_t)(samples * 8);
    /* bottom-up; bits 0-3: alpha channel depth */
    dst[17] = (samples == 2 || samples == 4) ? 8 : 0;
    dstindex = TGA_HEADER_SIZE;

    for (i = 0; i < height; i++) {
        srow = src + (height - 1 - i) * rowsize;
        if (!rle) {
            if (samples < 3) {
                memcpy(dst + dstindex, srow, (size_t)rowsize);
            }
            else {
                for (j = 0; j < rowsize; j += samples) {
                    tga_pixel(dst + dstindex + j, srow + j, samples);
                }
            }
            dstindex += rowsize;
            continue;
        }
        /* RLE packets do not cross scanlines */
        j = 0;
        while (j < width) {
            px = srow + j * samples;
            run = 1;
            while (
                run < 128
                && j + run < width
                && memcmp(px, px + run * samples, (size_t)samples) == 0)
            {
                run++;
            }
            if (run >= 2) {
                if (dstindex + 1 + samples + TGA_FOOTER_SIZE > dstsize) {
                    return TGA_OUTPUT_TOO_SMALL;
                }
                dst[dstindex] = (uint8_t)(0x80 | (run - 1));
                tga_pixel(dst + dstindex + 1, px, samples);
                dstindex += 1 + samples;
                j += run;
                continue;
            }
            raw = 1;
            while (raw < 128 && j + raw < width) {
                const uint8_t* nxt = px + raw * samples;
                if (
                    j + raw + 1 < width
                    && memcmp(nxt, nxt + samples, (size_t)samples) == 0)
                {
                    break;
                }
                raw++;
            }
            if (dstindex + 1 + raw * samples + TGA_FOOTER_SIZE > dstsize) {
                return TGA_OUTPUT_TOO_SMALL;
            }
            dst[dstindex++] = (uint8_t)(raw - 1);
            for (k = 0; k < raw; k++) {
                tga_pixel(dst + dstindex, px + k * samples, samples);
                dstindex += samples;
            }
            j += raw;
        }
    }

    /* TGA 2.0 footer without extension area and developer directory */
    memset(dst + dstindex, 0, 8);
    memcpy(dst + dstindex + 8, TGA_SIGNATURE, 18);
    return dstindex + TGA_FOOTER_SIZE;
}
