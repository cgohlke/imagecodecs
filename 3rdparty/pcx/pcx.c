/* pcx.c */

/* PCX and DCX image codec. */

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

#include "pcx.h"

#include <string.h>

#define PCX_MAGIC 0x0A
#define PCX_HEADER_SIZE 128
#define PCX_PALETTE_MARKER 0x0C
#define DCX_MAGIC 0x3ADE68B1u
#define DCX_TABLE_SIZE 1024


static inline uint16_t
pcx_u16(
    const uint8_t* p)
{
    return (uint16_t)(p[0] | ((uint16_t)p[1] << 8));
}


static inline uint32_t
pcx_u32(
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
pcx_put_u16(
    uint8_t* p,
    const uint16_t v)
{
    p[0] = (uint8_t)(v & 0xFF);
    p[1] = (uint8_t)(v >> 8);
}


static inline void
pcx_put_u32(
    uint8_t* p,
    const uint32_t v)
{
    p[0] = (uint8_t)(v & 0xFF);
    p[1] = (uint8_t)((v >> 8) & 0xFF);
    p[2] = (uint8_t)((v >> 16) & 0xFF);
    p[3] = (uint8_t)(v >> 24);
}


static int
pcx_is_dcx(
    const uint8_t* src,
    const ssize_t srcsize)
{
    return srcsize >= 4 && pcx_u32(src) == DCX_MAGIC;
}


/* Return number of pages in DCX page table. */
static ssize_t
pcx_dcx_pages(
    const uint8_t* src,
    const ssize_t srcsize)
{
    ssize_t n = 0;
    while (
        n < DCX_TABLE_SIZE
        && 4 + (n + 1) * 4 <= srcsize
        && pcx_u32(src + 4 + n * 4) != 0)
    {
        n++;
    }
    return n;
}


int
pcx_check(
    const uint8_t* src,
    const ssize_t srcsize)
{
    if (src == NULL || srcsize < PCX_HEADER_SIZE) {
        return 0;
    }
    if (pcx_is_dcx(src, srcsize)) {
        return 1;
    }
    return (
        src[0] == PCX_MAGIC
        && src[1] <= 5
        && src[2] <= 1
        && (src[3] == 1 || src[3] == 2 || src[3] == 4 || src[3] == 8)
        && src[65] >= 1
        && src[65] <= 4
    );
}


/* Parse PCX header at offset. */
static int
pcx_page_info(
    const uint8_t* src,
    const ssize_t srcsize,
    const ssize_t offset,
    ssize_t end,
    pcx_info_t* info)
{
    const uint8_t* hdr;
    const uint8_t* pal;
    ssize_t i;
    int is_grayscale, has_vga_palette;

    if (offset < 0 || offset > srcsize - PCX_HEADER_SIZE) {
        return PCX_DATA_TRUNCATED;
    }
    if (end < 0 || end > srcsize) {
        end = srcsize;
    }
    info->offset = offset;
    info->end = end;

    hdr = src + offset;
    if (hdr[0] != PCX_MAGIC) {
        return PCX_INVALID_HEADER;
    }
    info->version = hdr[1];
    info->encoding = hdr[2];
    info->bits_per_pixel = hdr[3];
    info->hdpi = pcx_u16(hdr + 12);
    info->vdpi = pcx_u16(hdr + 14);
    info->nplanes = hdr[65];
    info->bytes_per_line = (ssize_t)pcx_u16(hdr + 66);
    info->palette_type = pcx_u16(hdr + 68);

    if (info->encoding > 1) {
        return PCX_UNSUPPORTED_FORMAT;
    }
    info->width = (ssize_t)pcx_u16(hdr + 8) - (ssize_t)pcx_u16(hdr + 4) + 1;
    info->height = (ssize_t)pcx_u16(hdr + 10) - (ssize_t)pcx_u16(hdr + 6) + 1;
    if (
        info->width <= 0 || info->height <= 0 || info->bytes_per_line <= 0
    )
    {
        return PCX_INVALID_HEADER;
    }

    /* some fax files have bytes_per_line < ceil(width * bpp / 8) */
    switch (info->bits_per_pixel)
    {
        case 1:
            if (info->width > info->bytes_per_line * 8) {
                info->width = info->bytes_per_line * 8;
            }
            break;
        case 2:
            if (info->width > info->bytes_per_line * 4) {
                info->width = info->bytes_per_line * 4;
            }
            break;
        case 4:
            if (info->width > info->bytes_per_line * 2) {
                info->width = info->bytes_per_line * 2;
            }
            break;
        default:
            if (info->width > info->bytes_per_line) {
                info->width = info->bytes_per_line;
            }
    }

    memset(info->palette, 0, sizeof(info->palette));
    info->isbool = 0;

    if (info->bits_per_pixel == 8 && info->nplanes == 1) {
        /* paletted or grayscale */
        has_vga_palette = 0;
        is_grayscale = info->palette_type == 2;
        if (
            end - 769 >= offset + PCX_HEADER_SIZE
            && src[end - 769] == PCX_PALETTE_MARKER)
        {
            has_vga_palette = 1;
            memcpy(info->palette, src + end - 768, 768);
            if (!is_grayscale) {
                is_grayscale = 1;
                for (i = 0; i < 256; i++) {
                    pal = info->palette + i * 3;
                    if (pal[0] != i || pal[1] != i || pal[2] != i) {
                        is_grayscale = 0;
                        break;
                    }
                }
            }
        }
        info->samples = (is_grayscale || !has_vga_palette) ? 1 : 3;
    }
    else if (
        info->bits_per_pixel == 8
        && (info->nplanes == 3 || info->nplanes == 4))
    {
        info->samples = info->nplanes;
    }
    else if (info->bits_per_pixel == 1 && info->nplanes == 1) {
        info->samples = 1;
        info->isbool = 1;
    }
    else if (
        (info->bits_per_pixel == 1 && info->nplanes >= 2 && info->nplanes <= 4)
        || (
            (info->bits_per_pixel == 2 || info->bits_per_pixel == 4)
            && info->nplanes == 1
    ))
    {
        /* EGA palette */
        info->samples = 3;
        memcpy(info->palette, hdr + 16, 48);
    }
    else {
        return PCX_UNSUPPORTED_FORMAT;
    }
    return PCX_OK;
}


int
pcx_info(
    const uint8_t* src,
    const ssize_t srcsize,
    const ssize_t index,
    pcx_info_t* info)
{
    ssize_t idx = index < 0 ? 0 : index;
    ssize_t offset, end;

    if (info == NULL) {
        return PCX_VALUE_ERROR;
    }
    memset(info, 0, sizeof(pcx_info_t));

    if (src == NULL || srcsize < 0) {
        return PCX_VALUE_ERROR;
    }
    if (srcsize < PCX_HEADER_SIZE) {
        return PCX_DATA_TRUNCATED;
    }
    info->index = idx;
    if (!pcx_is_dcx(src, srcsize)) {
        info->frames = 1;
        if (idx != 0) {
            return PCX_INDEX_ERROR;
        }
        return pcx_page_info(src, srcsize, 0, srcsize, info);
    }

    info->dcx = 1;
    info->frames = pcx_dcx_pages(src, srcsize);
    if (info->frames == 0) {
        return PCX_INVALID_HEADER;
    }
    if (idx >= info->frames) {
        return PCX_INDEX_ERROR;
    }
    offset = (ssize_t)pcx_u32(src + 4 + idx * 4);
    if (idx + 1 < info->frames) {
        end = (ssize_t)pcx_u32(src + 4 + (idx + 1) * 4);
    }
    else {
        end = srcsize;
    }
    return pcx_page_info(src, srcsize, offset, end, info);
}


typedef struct pcx_reader_t {
    const uint8_t* src;
    ssize_t pos;
    ssize_t end;
    ssize_t count;
    int rle;
    uint8_t value;
} pcx_reader_t;


/* Return next decoded byte or 0 if data is exhausted. */
static inline uint8_t
pcx_read(
    pcx_reader_t* r)
{
    uint8_t b;
    if (r->count > 0) {
        r->count--;
        return r->value;
    }
    while (r->pos < r->end) {
        b = r->src[r->pos++];
        if (!r->rle || (b & 0xC0) != 0xC0) {
            return b;
        }
        if (r->pos >= r->end) {
            break;
        }
        r->value = r->src[r->pos++];
        r->count = b & 0x3F;
        if (r->count > 0) {
            r->count--;
            return r->value;
        }
    }
    r->pos = r->end;
    return 0;
}


/* Decode single page into dst. */
static void
pcx_decode_page(
    const uint8_t* src,
    const pcx_info_t* info,
    uint8_t* dst)
{
    const ssize_t width = info->width;
    const ssize_t samples = info->samples;
    const ssize_t bpl = info->bytes_per_line;
    const int bpp = info->bits_per_pixel;
    const int nplanes = info->nplanes;
    const uint8_t* palette = info->palette;
    pcx_reader_t r;
    uint8_t* drow;
    ssize_t row, j, x;
    int plane, k, ppb;
    uint8_t v, idx;

    r.src = src;
    r.pos = info->offset + PCX_HEADER_SIZE;
    r.end = info->end;
    r.count = 0;
    r.rle = info->encoding == 1;
    r.value = 0;
    ppb = bpp < 8 ? 8 / bpp : 1;

    for (row = 0; row < info->height; row++) {
        drow = dst + row * width * samples;
        /* runs do not continue across scanlines */
        r.count = 0;
        for (plane = 0; plane < nplanes; plane++) {
            for (j = 0; j < bpl; j++) {
                v = pcx_read(&r);
                if (bpp == 8) {
                    if (j >= width) {
                        continue;
                    }
                    if (nplanes > 1) {
                        drow[j * samples + plane] = v;
                    }
                    else if (samples == 1) {
                        drow[j] = v;
                    }
                    else {
                        memcpy(drow + j * 3, palette + v * 3, 3);
                    }
                }
                else if (bpp == 1) {
                    for (k = 0; k < 8; k++) {
                        x = j * 8 + k;
                        if (x >= width) {
                            break;
                        }
                        idx = (uint8_t)((v >> (7 - k)) & 1);
                        if (nplanes == 1) {
                            drow[x] = idx;
                        }
                        else if (plane == 0) {
                            drow[x * 3] = idx;
                        }
                        else {
                            /* accumulate palette index in first sample */
                            drow[x * 3] |= (uint8_t)(idx << plane);
                        }
                    }
                }
                else {
                    for (k = 0; k < ppb; k++) {
                        x = j * ppb + k;
                        if (x >= width) {
                            break;
                        }
                        idx = (uint8_t)(
                            (v >> (8 - bpp * (k + 1))) & ((1 << bpp) - 1)
                        );
                        memcpy(drow + x * 3, palette + idx * 3, 3);
                    }
                }
            }
        }
        if (bpp == 1 && nplanes > 1) {
            for (x = 0; x < width; x++) {
                idx = drow[x * 3];
                memcpy(drow + x * 3, palette + idx * 3, 3);
            }
        }
    }
}


ssize_t
pcx_decode(
    const uint8_t* src,
    const ssize_t srcsize,
    const ssize_t index,
    pcx_info_t* info,
    uint8_t* dst,
    const ssize_t dstsize)
{
    ssize_t pagesize, frames, width, height, samples, i;
    int ret, isbool;

    if (dst == NULL || info == NULL) {
        return PCX_VALUE_ERROR;
    }
    ret = pcx_info(src, srcsize, index, info);
    if (ret < 0) {
        return ret;
    }
    width = info->width;
    height = info->height;
    samples = info->samples;
    isbool = info->isbool;
    frames = index < 0 ? info->frames : 1;
    if ((int64_t)width * height * samples > (int64_t)PTRDIFF_MAX / frames) {
        return PCX_IMAGE_TOO_LARGE;
    }
    pagesize = width * height * samples;
    if (dstsize < pagesize * frames) {
        return PCX_OUTPUT_TOO_SMALL;
    }
    pcx_decode_page(src, info, dst);

    for (i = 1; i < frames; i++) {
        ret = pcx_info(src, srcsize, i, info);
        if (ret < 0) {
            return ret;
        }
        if (
            info->width != width
            || info->height != height
            || info->samples != samples
            || info->isbool != isbool)
        {
            return PCX_SHAPE_MISMATCH;
        }
        pcx_decode_page(src, info, dst + i * pagesize);
    }
    return pagesize * frames;
}


static int64_t
pcx_page_bounds(
    const ssize_t width,
    const ssize_t height,
    const ssize_t samples)
{
    const int64_t bytes_per_line = width + width % 2;
    return (
        PCX_HEADER_SIZE
        + (int64_t)height * samples * bytes_per_line * 2
        + (samples == 1 ? 769 : 0)
    );
}


ssize_t
pcx_encode_size(
    const ssize_t width,
    const ssize_t height,
    const ssize_t samples,
    const ssize_t frames)
{
    int64_t page;
    if (
        width <= 0
        || width > UINT16_MAX
        || height <= 0
        || height > UINT16_MAX
        || (samples != 1 && samples != 3 && samples != 4)
        || frames < 1
        || frames > PCX_DCX_MAXPAGES)
    {
        return PCX_VALUE_ERROR;
    }
    page = pcx_page_bounds(width, height, samples);
    if (page > ((int64_t)PTRDIFF_MAX - 8192) / frames) {
        return PCX_VALUE_ERROR;
    }
    if (frames == 1) {
        return (ssize_t)page;
    }
    return (ssize_t)(4 + (frames + 1) * 4 + frames * page);
}


/* Encode single PCX page. Return bytes written. */
static ssize_t
pcx_encode_page(
    const uint8_t* src,
    uint8_t* dst,
    const ssize_t dstsize,
    const ssize_t width,
    const ssize_t height,
    const ssize_t samples)
{
    const ssize_t bytes_per_line = width + width % 2;
    const uint8_t* srow;
    ssize_t dstindex, i, j, run;
    int plane;
    uint8_t this_byte, next_byte;

    if (dstsize < PCX_HEADER_SIZE) {
        return PCX_OUTPUT_TOO_SMALL;
    }
    memset(dst, 0, PCX_HEADER_SIZE);
    dst[0] = PCX_MAGIC;
    dst[1] = 5;  /* version */
    dst[2] = 1;  /* RLE */
    dst[3] = 8;  /* bits per pixel */
    pcx_put_u16(dst + 8, (uint16_t)(width - 1));
    pcx_put_u16(dst + 10, (uint16_t)(height - 1));
    pcx_put_u16(dst + 12, 72);
    pcx_put_u16(dst + 14, 72);
    dst[65] = (uint8_t)samples;
    pcx_put_u16(dst + 66, (uint16_t)bytes_per_line);
    pcx_put_u16(dst + 68, samples == 1 ? 2 : 1);
    dstindex = PCX_HEADER_SIZE;

    for (i = 0; i < height; i++) {
        srow = src + i * width * samples;
        for (plane = 0; plane < samples; plane++) {
            j = 0;
            while (j < bytes_per_line) {
                this_byte = j < width ? srow[j * samples + plane] : 0;
                run = 1;
                while (run < 63 && j + run < bytes_per_line) {
                    next_byte = (
                        j + run < width ? srow[(j + run) * samples + plane] : 0
                    );
                    if (next_byte != this_byte) {
                        break;
                    }
                    run++;
                }
                if (dstindex + 2 > dstsize) {
                    return PCX_OUTPUT_TOO_SMALL;
                }
                if (run > 1 || (this_byte & 0xC0) == 0xC0) {
                    dst[dstindex++] = (uint8_t)(0xC0 | run);
                }
                dst[dstindex++] = this_byte;
                j += run;
            }
        }
    }

    if (samples == 1) {
        /* identity grayscale VGA palette */
        if (dstindex + 769 > dstsize) {
            return PCX_OUTPUT_TOO_SMALL;
        }
        dst[dstindex++] = PCX_PALETTE_MARKER;
        for (j = 0; j < 256; j++) {
            dst[dstindex++] = (uint8_t)j;
            dst[dstindex++] = (uint8_t)j;
            dst[dstindex++] = (uint8_t)j;
        }
    }
    return dstindex;
}


ssize_t
pcx_encode(
    const uint8_t* src,
    const ssize_t width,
    const ssize_t height,
    const ssize_t samples,
    const ssize_t frames,
    uint8_t* dst,
    const ssize_t dstsize)
{
    const ssize_t framesize = width * height * samples;
    ssize_t size, pagesize, i;

    size = pcx_encode_size(width, height, samples, frames);
    if (size < 0) {
        return size;
    }
    if (src == NULL || dst == NULL || dstsize < 0) {
        return PCX_VALUE_ERROR;
    }
    if (frames == 1) {
        return pcx_encode_page(src, dst, dstsize, width, height, samples);
    }

    size = 4 + (frames + 1) * 4;
    if (dstsize < size) {
        return PCX_OUTPUT_TOO_SMALL;
    }
    pcx_put_u32(dst, DCX_MAGIC);
    for (i = 0; i < frames; i++) {
        if ((uint64_t)size > UINT32_MAX) {
            return PCX_VALUE_ERROR;
        }
        pcx_put_u32(dst + 4 + i * 4, (uint32_t)size);
        pagesize = pcx_encode_page(
            src + i * framesize,
            dst + size,
            dstsize - size,
            width,
            height,
            samples
        );
        if (pagesize < 0) {
            return pagesize;
        }
        size += pagesize;
    }
    pcx_put_u32(dst + 4 + frames * 4, 0);
    return size;
}
