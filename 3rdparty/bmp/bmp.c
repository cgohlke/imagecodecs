/* bmp.c */

/* BMP (Windows Bitmap) codec. */

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

#include "bmp.h"

#include <string.h>


static inline uint16_t
bmp_u16(
    const uint8_t* p)
{
    return (uint16_t)(p[0] | ((uint16_t)p[1] << 8));
}


static inline uint32_t
bmp_u32(
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
bmp_put_u16(
    uint8_t* p,
    const uint16_t v)
{
    p[0] = (uint8_t)(v & 0xFF);
    p[1] = (uint8_t)(v >> 8);
}


static inline void
bmp_put_u32(
    uint8_t* p,
    const uint32_t v)
{
    p[0] = (uint8_t)(v & 0xFF);
    p[1] = (uint8_t)((v >> 8) & 0xFF);
    p[2] = (uint8_t)((v >> 16) & 0xFF);
    p[3] = (uint8_t)(v >> 24);
}


static void
bmp_mask_shift(
    const uint32_t mask,
    int* shift,
    int* bits)
{
    uint32_t m = mask;
    int s = 0;
    int b = 0;
    if (m != 0) {
        while ((m & 1) == 0) {
            s++;
            m >>= 1;
        }
        while (m) {
            b += (int)(m & 1);
            m >>= 1;
        }
    }
    *shift = s;
    *bits = b;
}


/* Scale bitfield value to 8 bits. */
static inline uint8_t
bmp_scale(
    const uint32_t v,
    const int bits)
{
    uint64_t s;
    if (bits <= 0 || bits == 8) {
        return (uint8_t)v;
    }
    s = (uint64_t)v * 255 / ((1ull << bits) - 1);
    /* non-contiguous masks may exceed the maximum */
    return s > 255 ? 255 : (uint8_t)s;
}


int
bmp_check(
    const uint8_t* src,
    const ssize_t srcsize)
{
    return (src != NULL && srcsize > 26 && src[0] == 'B' && src[1] == 'M');
}


int
bmp_info(
    const uint8_t* src,
    const ssize_t srcsize,
    const int asrgb,
    bmp_info_t* info)
{
    const uint8_t* hdr;
    const uint8_t* pal;
    uint64_t height64, rowstride64, size64;
    ssize_t offset, palsize;
    uint32_t hs, i;
    int32_t width, height;
    int k;

    if (info == NULL) {
        return BMP_VALUE_ERROR;
    }
    memset(info, 0, sizeof(bmp_info_t));

    if (src == NULL || srcsize < 0) {
        return BMP_VALUE_ERROR;
    }
    if ((uint64_t)srcsize > UINT32_MAX) {
        return BMP_IMAGE_TOO_LARGE;
    }
    if (srcsize < 26) {
        return BMP_DATA_TRUNCATED;
    }
    if (src[0] != 'B' || src[1] != 'M') {
        return BMP_INVALID_HEADER;
    }
    info->filesize = bmp_u32(src + 2);
    info->offbits = bmp_u32(src + 10);
    if (info->filesize > (uint32_t)srcsize) {
        return BMP_DATA_TRUNCATED;
    }
    if (info->offbits >= (uint32_t)srcsize) {
        return BMP_INVALID_HEADER;
    }

    hs = bmp_u32(src + 14);
    if ((hs != 12 && hs < 16) || hs > 124 || (ssize_t)hs + 14 > srcsize) {
        return BMP_INVALID_HEADER;
    }
    info->header_size = hs;
    hdr = src + 14;
    palsize = 4;

#define BMP_HDR32(off) ((off) + 4 <= hs ? bmp_u32(hdr + (off)) : 0)

    if (hs == 12) {
        /* OS/2 BITMAPCOREHEADER */
        width = (int32_t)bmp_u16(hdr + 4);
        height = (int32_t)bmp_u16(hdr + 6);
        info->planes = bmp_u16(hdr + 8);
        info->bitcount = bmp_u16(hdr + 10);
        palsize = 3;
    }
    else {
        width = (int32_t)bmp_u32(hdr + 4);
        height = (int32_t)bmp_u32(hdr + 8);
        info->planes = bmp_u16(hdr + 12);
        info->bitcount = bmp_u16(hdr + 14);
        info->compression = BMP_HDR32(16);
        info->size_image = BMP_HDR32(20);
        info->x_ppm = (int32_t)BMP_HDR32(24);
        info->y_ppm = (int32_t)BMP_HDR32(28);
        info->clr_used = BMP_HDR32(32);
    }
    offset = 14 + (ssize_t)hs;
    info->topdown = height < 0;
    height64 = height < 0 ? (uint64_t)(-(int64_t)height) : (uint64_t)height;
    info->width = width;
    info->height = (ssize_t)height64;

    if (
        info->compression == BMP_BI_JPEG || info->compression == BMP_BI_PNG
    )
    {
        /* embedded image is decoded by caller */
        return BMP_OK;
    }

    if (width <= 0 || height == 0) {
        return BMP_INVALID_HEADER;
    }
    if (info->compression == BMP_BI_BITFIELDS) {
        if (info->bitcount != 16 && info->bitcount != 32) {
            return BMP_INVALID_HEADER;
        }
    }
    else if (info->compression > BMP_BI_RLE4) {
        return BMP_UNSUPPORTED_FORMAT;
    }
    if (info->compression == BMP_BI_RLE8 && info->bitcount != 8) {
        return BMP_INVALID_HEADER;
    }
    if (info->compression == BMP_BI_RLE4 && info->bitcount != 4) {
        return BMP_INVALID_HEADER;
    }
    switch (info->bitcount)
    {
        case 1:
        case 4:
        case 8:
        case 16:
        case 24:
        case 32:
            break;
        default:
            return BMP_UNSUPPORTED_FORMAT;
    }
    if (info->clr_used == 0 && info->bitcount < 16) {
        info->clr_used = 1u << info->bitcount;
    }
    if (info->clr_used > 256) {
        return BMP_UNSUPPORTED_FORMAT;
    }
    if (
        (uint64_t)info->clr_used * (uint64_t)palsize + (uint64_t)offset
        > (uint64_t)info->offbits)
    {
        return BMP_INVALID_HEADER;
    }

    rowstride64 = (((uint64_t)width * info->bitcount + 31) / 32) * 4;
    if (rowstride64 > UINT32_MAX) {
        return BMP_IMAGE_TOO_LARGE;
    }
    info->rowstride = (ssize_t)rowstride64;

    if (info->size_image == 0) {
        if (
            info->compression != BMP_BI_RGB
            && info->compression != BMP_BI_BITFIELDS)
        {
            return BMP_INVALID_HEADER;
        }
        size64 = height64 * rowstride64;
        if (size64 > UINT32_MAX) {
            return BMP_IMAGE_TOO_LARGE;
        }
        info->size_image = (uint32_t)size64;
    }
    if (
        (uint64_t)info->size_image + (uint64_t)info->offbits
        > (uint64_t)srcsize)
    {
        return BMP_DATA_TRUNCATED;
    }
    if (
        info->compression != BMP_BI_RLE8
        && info->compression != BMP_BI_RLE4
        && height64 * rowstride64 + info->offbits > (uint64_t)srcsize)
    {
        return BMP_DATA_TRUNCATED;
    }

    info->samples = 3;
    if (info->bitcount <= 8) {
        for (i = 0; i < info->clr_used; i++) {
            pal = src + offset + i * palsize;
            info->palette[i * 3] = pal[2];
            info->palette[i * 3 + 1] = pal[1];
            info->palette[i * 3 + 2] = pal[0];
        }
        if (asrgb < 0) {
            info->samples = 1;
            for (i = 0; i < info->clr_used; i++) {
                pal = info->palette + i * 3;
                if (pal[0] != i || pal[1] != i || pal[2] != i) {
                    info->samples = 3;
                    break;
                }
            }
        }
        else if (asrgb == 0) {
            info->samples = 1;
        }
    }
    else if (info->bitcount == 16 || info->bitcount == 32) {
        if (info->compression == BMP_BI_BITFIELDS) {
            if (hs >= 52) {
                info->masks[0] = BMP_HDR32(40);
                info->masks[1] = BMP_HDR32(44);
                info->masks[2] = BMP_HDR32(48);
                info->masks[3] = BMP_HDR32(52);
            }
            else if (offset + 12 <= (ssize_t)info->offbits) {
                /* masks follow BITMAPINFOHEADER */
                info->masks[0] = bmp_u32(src + offset);
                info->masks[1] = bmp_u32(src + offset + 4);
                info->masks[2] = bmp_u32(src + offset + 8);
                if (
                    info->bitcount == 32
                    && offset + 16 <= (ssize_t)info->offbits)
                {
                    info->masks[3] = bmp_u32(src + offset + 12);
                }
            }
        }
        else if (info->bitcount == 16) {
            /* RGB555 */
            info->masks[0] = 0x7C00;
            info->masks[1] = 0x03E0;
            info->masks[2] = 0x001F;
        }
        else {
            /* BGRX, high byte is padding */
            info->masks[0] = 0x00FF0000;
            info->masks[1] = 0x0000FF00;
            info->masks[2] = 0x000000FF;
        }
        for (k = 0; k < 4; k++) {
            bmp_mask_shift(info->masks[k], &info->shifts[k], &info->bits[k]);
        }
        info->has_alpha = info->masks[3] != 0;
        info->samples = info->has_alpha ? 4 : 3;
    }

#undef BMP_HDR32

    if (
        height64 * (uint64_t)width * (uint64_t)info->samples
        > (uint64_t)PTRDIFF_MAX)
    {
        return BMP_IMAGE_TOO_LARGE;
    }
    return BMP_OK;
}


static void
bmp_rle_put(
    const bmp_info_t* info,
    uint8_t* dst,
    const ssize_t x,
    const ssize_t y,
    const uint8_t index)
{
    const ssize_t row = info->topdown ? y : info->height - 1 - y;
    const ssize_t pos = row * info->width + x;
    if (info->samples == 3) {
        memcpy(dst + pos * 3, info->palette + (ssize_t)index * 3, 3);
    }
    else {
        dst[pos] = index;
    }
}


/* Decode BI_RLE8 or BI_RLE4 compressed image. */
static ssize_t
bmp_decode_rle(
    const uint8_t* src,
    const ssize_t srcsize,
    const bmp_info_t* info,
    uint8_t* dst)
{
    const ssize_t width = info->width;
    const ssize_t height = info->height;
    const int rle4 = info->compression == BMP_BI_RLE4;
    ssize_t s = (ssize_t)info->offbits;
    ssize_t limit = s + (ssize_t)info->size_image;
    ssize_t x = 0;
    ssize_t y = 0;
    ssize_t k, nbytes;
    uint8_t count, code, value;

    if (limit > srcsize) {
        limit = srcsize;
    }
    /* unset pixels default to palette index 0 */
    memset(dst, 0, (size_t)(height * width * info->samples));

    while (s + 1 < limit) {
        count = src[s];
        code = src[s + 1];
        s += 2;
        if (count > 0) {
            /* encoded run */
            for (k = 0; k < count; k++) {
                if (x >= width || y >= height) {
                    break;
                }
                if (rle4) {
                    value = (k & 1) ? (code & 0xF) : (code >> 4);
                }
                else {
                    value = code;
                }
                bmp_rle_put(info, dst, x, y, value);
                x++;
            }
        }
        else if (code == 0) {
            /* end of line */
            x = 0;
            y++;
        }
        else if (code == 1) {
            /* end of bitmap */
            break;
        }
        else if (code == 2) {
            /* delta */
            if (s + 1 >= limit) {
                break;
            }
            x += (ssize_t)src[s];
            y += (ssize_t)src[s + 1];
            s += 2;
        }
        else {
            /* absolute mode, word-aligned */
            nbytes = rle4 ? ((ssize_t)code + 1) / 2 : (ssize_t)code;
            for (k = 0; k < code; k++) {
                const ssize_t pos = s + (rle4 ? k / 2 : k);
                if (pos >= limit) {
                    break;
                }
                if (rle4) {
                    value = (k & 1) ? (src[pos] & 0xF) : (src[pos] >> 4);
                }
                else {
                    value = src[pos];
                }
                if (x < width && y < height) {
                    bmp_rle_put(info, dst, x, y, value);
                    x++;
                }
            }
            s += nbytes + (nbytes & 1);
        }
    }
    return height * width * info->samples;
}


ssize_t
bmp_decode(
    const uint8_t* src,
    const ssize_t srcsize,
    const bmp_info_t* info,
    uint8_t* dst,
    const ssize_t dstsize)
{
    const uint8_t* srow;
    uint8_t* drow;
    ssize_t width, height, samples, i, j;
    uint32_t pixel, index;

    if (src == NULL || dst == NULL || info == NULL) {
        return BMP_VALUE_ERROR;
    }
    width = info->width;
    height = info->height;
    samples = info->samples;
    if (samples == 0) {
        /* embedded JPEG or PNG */
        return BMP_UNSUPPORTED_FORMAT;
    }
    if (dstsize < height * width * samples) {
        return BMP_OUTPUT_TOO_SMALL;
    }
    if (
        info->compression == BMP_BI_RLE8 || info->compression == BMP_BI_RLE4
    )
    {
        return bmp_decode_rle(src, srcsize, info, dst);
    }
    if (
        (uint64_t)height * (uint64_t)info->rowstride + info->offbits
        > (uint64_t)srcsize)
    {
        return BMP_DATA_TRUNCATED;
    }

    for (i = 0; i < height; i++) {
        srow = src + info->offbits + i * info->rowstride;
        drow = dst + (info->topdown ? i : height - 1 - i) * width * samples;
        switch (info->bitcount)
        {
            case 1:
            case 4:
            case 8:
                for (j = 0; j < width; j++) {
                    if (info->bitcount == 8) {
                        index = srow[j];
                        if (samples == 1) {
                            drow[j] = (uint8_t)index;
                            continue;
                        }
                    }
                    else if (info->bitcount == 4) {
                        index = (j & 1) ? (srow[j >> 1] & 0xF)
                                        : (srow[j >> 1] >> 4);
                    }
                    else {
                        index = (srow[j >> 3] >> (7 - (j & 7))) & 1;
                    }
                    if (index >= info->clr_used) {
                        return BMP_INDEX_ERROR;
                    }
                    if (samples == 3) {
                        memcpy(drow + j * 3, info->palette + index * 3, 3);
                    }
                    else {
                        drow[j] = info->palette[index * 3];
                    }
                }
                break;
            case 24:
                for (j = 0; j < width; j++) {
                    drow[j * 3] = srow[j * 3 + 2];
                    drow[j * 3 + 1] = srow[j * 3 + 1];
                    drow[j * 3 + 2] = srow[j * 3];
                }
                break;
            case 16:
            case 32:
                for (j = 0; j < width; j++) {
                    if (info->bitcount == 16) {
                        pixel = bmp_u16(srow + j * 2);
                    }
                    else {
                        pixel = bmp_u32(srow + j * 4);
                    }
                    drow[0] = bmp_scale(
                        (pixel & info->masks[0]) >> info->shifts[0],
                        info->bits[0]
                    );
                    drow[1] = bmp_scale(
                        (pixel & info->masks[1]) >> info->shifts[1],
                        info->bits[1]
                    );
                    drow[2] = bmp_scale(
                        (pixel & info->masks[2]) >> info->shifts[2],
                        info->bits[2]
                    );
                    if (samples == 4) {
                        drow[3] = bmp_scale(
                            (pixel & info->masks[3]) >> info->shifts[3],
                            info->bits[3]
                        );
                    }
                    drow += samples;
                }
                break;
            default:
                return BMP_UNSUPPORTED_FORMAT;
        }
    }
    return height * width * samples;
}


static ssize_t
bmp_offbits(
    const ssize_t samples)
{
    return samples == 1 ? 1078 : (samples == 4 ? 122 : 54);
}


ssize_t
bmp_encode_size(
    const ssize_t width,
    const ssize_t height,
    const ssize_t samples)
{
    uint64_t size;
    if (
        width <= 0
        || height <= 0
        || width > INT32_MAX
        || height > INT32_MAX
        || (samples != 1 && samples != 3 && samples != 4))
    {
        return BMP_VALUE_ERROR;
    }
    size = (uint64_t)bmp_offbits(samples)
        + (uint64_t)height * (((uint64_t)width * samples + 3) & ~3ull);
    if (size > UINT32_MAX || size > (uint64_t)PTRDIFF_MAX) {
        return BMP_VALUE_ERROR;
    }
    return (ssize_t)size;
}


ssize_t
bmp_encode(
    const uint8_t* src,
    const ssize_t width,
    const ssize_t height,
    const ssize_t samples,
    const int32_t ppm,
    uint8_t* dst,
    const ssize_t dstsize)
{
    const ssize_t size = bmp_encode_size(width, height, samples);
    const ssize_t offbits = bmp_offbits(samples);
    const ssize_t rowsize = width * samples;
    const ssize_t rowstride = (rowsize + 3) & ~(ssize_t)3;
    const uint8_t* srow;
    uint8_t* hdr;
    uint8_t* drow;
    ssize_t i, j;

    if (size < 0) {
        return size;
    }
    if (src == NULL || dst == NULL) {
        return BMP_VALUE_ERROR;
    }
    if (dstsize < size) {
        return BMP_OUTPUT_TOO_SMALL;
    }

    memset(dst, 0, (size_t)offbits);
    dst[0] = 'B';
    dst[1] = 'M';
    bmp_put_u32(dst + 2, (uint32_t)size);
    bmp_put_u32(dst + 10, (uint32_t)offbits);

    hdr = dst + 14;
    bmp_put_u32(hdr, samples == 4 ? 108 : 40);
    bmp_put_u32(hdr + 4, (uint32_t)width);
    bmp_put_u32(hdr + 8, (uint32_t)height);
    bmp_put_u16(hdr + 12, 1);
    bmp_put_u16(hdr + 14, (uint16_t)(samples * 8));
    bmp_put_u32(hdr + 16, samples == 4 ? BMP_BI_BITFIELDS : BMP_BI_RGB);
    bmp_put_u32(hdr + 20, (uint32_t)(height * rowstride));
    bmp_put_u32(hdr + 24, (uint32_t)ppm);
    bmp_put_u32(hdr + 28, (uint32_t)ppm);
    if (samples == 4) {
        bmp_put_u32(hdr + 40, 0x00FF0000);
        bmp_put_u32(hdr + 44, 0x0000FF00);
        bmp_put_u32(hdr + 48, 0x000000FF);
        bmp_put_u32(hdr + 52, 0xFF000000);
    }
    else if (samples == 1) {
        /* identity grayscale palette */
        for (i = 0; i < 256; i++) {
            dst[54 + i * 4] = (uint8_t)i;
            dst[54 + i * 4 + 1] = (uint8_t)i;
            dst[54 + i * 4 + 2] = (uint8_t)i;
            dst[54 + i * 4 + 3] = 255;
        }
    }

    /* rows are stored bottom-up */
    for (i = 0; i < height; i++) {
        srow = src + i * rowsize;
        drow = dst + offbits + (height - 1 - i) * rowstride;
        if (samples == 1) {
            memcpy(drow, srow, (size_t)rowsize);
        }
        else {
            for (j = 0; j < rowsize; j += samples) {
                drow[j] = srow[j + 2];
                drow[j + 1] = srow[j + 1];
                drow[j + 2] = srow[j];
                if (samples == 4) {
                    drow[j + 3] = srow[j + 3];
                }
            }
        }
        if (rowstride > rowsize) {
            memset(drow + rowsize, 0, (size_t)(rowstride - rowsize));
        }
    }
    return size;
}
