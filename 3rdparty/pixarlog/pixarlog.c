/* pixarlog.c */

/* SPDX-License-Identifier: HPND-sell-variant */

/* Adapted from libtiff's tif_pixarlog.c by Christoph Gohlke */

/*
 * Original copyright and license from libtiff v4.7.1 tif_pixarlog.c:
 *
 * Copyright (c) 1996-1997 Sam Leffler
 * Copyright (c) 1996 Pixar
 *
 * Permission to use, copy, modify, distribute, and sell this software and
 * its documentation for any purpose is hereby granted without fee, provided
 * that (i) the above copyright notices and this permission notice appear in
 * all copies of the software and related documentation, and (ii) the names of
 * Pixar, Sam Leffler and Silicon Graphics may not be used in any advertising
 * or publicity relating to the software without the specific, prior written
 * permission of Pixar, Sam Leffler and Silicon Graphics.
 *
 * THE SOFTWARE IS PROVIDED "AS-IS" AND WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS, IMPLIED OR OTHERWISE, INCLUDING WITHOUT LIMITATION, ANY
 * WARRANTY OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 *
 * IN NO EVENT SHALL PIXAR, SAM LEFFLER OR SILICON GRAPHICS BE LIABLE FOR
 * ANY SPECIAL, INCIDENTAL, INDIRECT OR CONSEQUENTIAL DAMAGES OF ANY KIND,
 * OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER OR NOT ADVISED OF THE POSSIBILITY OF DAMAGE, AND ON ANY THEORY OF
 * LIABILITY, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THIS SOFTWARE.
 */

/*
 * Standalone PIXARLOG codec.
 *
 * Encoding and decoding of Pixar companded 11-bit log-encoded pixel data.
 *
 * The lookup table construction and horizontal accumulation/differencing
 * algorithms are based on libtiff's tif_pixarlog.c (v4.7.1),
 * contributed by Dan McCoy.
 *
 * The 11-bit companding scheme stores pixel values in a perceptually
 * uniform logarithmic space:
 *   - Codes 0 to ~249: linear region (values 0 to ~0.018)
 *   - Codes ~250 to 2047: logarithmic region (values ~0.018 to ~25.0)
 * The encoding is lossless for 8-bit input and slightly lossy for
 * 16-bit and float input. Code 1250 represents exactly 1.0.
 *
 * Compressed data format:
 *   encode: linear pixels -> 11-bit log + horizontal diff -> zlib compress
 *   decode: zlib decompress -> horizontal accumulate + log-to-linear
 *
 * The full-pipeline functions (pixarlog_decode/pixarlog_encode) handle
 * zlib inflate/deflate internally. The raw functions
 * (pixarlog_decode_raw/pixarlog_encode_raw) operate on already
 * decompressed uint16 data.
 */

#include "pixarlog.h"

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "zlib.h"


/* PixarLog constants (from libtiff v4.7.1 tif_pixarlog.c) */
#define TSIZE 2048  /* decode table size (11-bit token range) */
#define TSIZEP1 2049  /* plus one for boundary */
#define ONE 1250  /* token value representing 1.0 */
#define RATIO 1.004  /* nominal ratio for logarithmic region */
#define CODE_MASK 0x7ff  /* 11-bit mask */

/* Sizes for reverse lookup tables */
#define FROM14_SIZE 16384  /* 14-bit reverse table (16-bit input >> 2) */
#define FROM8_SIZE 256  /* 8-bit reverse table */
#define FROMLT2_MAX 28000  /* safe upper bound for float < 2.0 table */

/* Scale and clamp for 12-bit Picio format */
#define SCALE12 2048.0f
#define CLAMP12(t) (((t) < 3071) ? (uint16_t)(t) : 3071)


/*****************************************************************************/
/* Forward (decode) lookup tables: 11-bit code -> linear value */

static float pixarlog_ToLinearF[TSIZEP1];
static uint16_t pixarlog_ToLinear16[TSIZEP1];
static uint8_t pixarlog_ToLinear8[TSIZEP1];

/* Reverse (encode) lookup tables: linear value -> 11-bit code */
static uint16_t pixarlog_From8[FROM8_SIZE];
static uint16_t pixarlog_From14[FROM14_SIZE];
static uint16_t pixarlog_FromLT2[FROMLT2_MAX];
static int pixarlog_FromLT2_size;

/* Encode helper constants */
static float pixarlog_LogK1;
static float pixarlog_LogK2;
static float pixarlog_Fltsize;

/* Initialization flag */
static volatile int pixarlog_tables_ready = 0;


/*****************************************************************************/
/* Lookup table construction.
 *
 * Follows the exact same math as PixarLogMakeTables() in libtiff v4.7.1
 * tif_pixarlog.c to ensure bit-exact compatibility with libtiff output.
 */

void
pixarlog_init(void)
{
    int nlin, lt2size;
    int i, j;
    double b, c, linstep, v;

    if (pixarlog_tables_ready) {
        return;
    }

    /*
     * The 11-bit representation has two regions:
     *   [0, nlin):   linear, values 0 to ~0.018316 in steps of ~0.000073
     *   [nlin, 2048): logarithmic, constant ratio of RATIO between steps
     *
     * The two regions are continuous at the seam.
     */
    c = log(RATIO);
    nlin = (int)(1.0 / c);  /* ~250; must be integer */
    c = 1.0 / nlin;  /* recalculate for exact integer division */
    b = exp(-c * ONE);  /* scale factor: b * exp(c * ONE) = 1.0 */
    linstep = b * c * exp(1.0);

    pixarlog_LogK1 = (float)(1.0 / c);
    pixarlog_LogK2 = (float)(1.0 / b);
    lt2size = (int)(2.0 / linstep) + 1;
    if (lt2size > FROMLT2_MAX) {
        lt2size = FROMLT2_MAX;
    }
    pixarlog_FromLT2_size = lt2size;
    pixarlog_Fltsize = (float)(lt2size / 2);

    /* Build forward table: 11-bit code -> linear float */
    j = 0;
    for (i = 0; i < nlin; i++) {
        pixarlog_ToLinearF[j++] = (float)(i * linstep);
    }
    for (i = nlin; i < TSIZE; i++) {
        pixarlog_ToLinearF[j++] = (float)(b * exp(c * i));
    }
    pixarlog_ToLinearF[2048] = pixarlog_ToLinearF[2047];

    /* Derived integer tables */
    for (i = 0; i < TSIZEP1; i++) {
        v = pixarlog_ToLinearF[i] * 65535.0 + 0.5;
        pixarlog_ToLinear16[i] = (v > 65535.0) ? 65535 : (uint16_t)v;
        v = pixarlog_ToLinearF[i] * 255.0 + 0.5;
        pixarlog_ToLinear8[i] = (v > 255.0) ? 255 : (uint8_t)v;
    }

    /* Reverse table: float values < 2.0 -> 11-bit code */
    j = 0;
    for (i = 0; i < lt2size; i++) {
        if (
            (i * linstep) * (i * linstep)
            > pixarlog_ToLinearF[j] * pixarlog_ToLinearF[j + 1])
        {
            j++;
        }
        pixarlog_FromLT2[i] = (uint16_t)j;
    }

    /* Reverse table: 14-bit input (16-bit >> 2) -> 11-bit code */
    j = 0;
    for (i = 0; i < FROM14_SIZE; i++) {
        while (
            (i / 16383.0) * (i / 16383.0)
            > pixarlog_ToLinearF[j] * pixarlog_ToLinearF[j + 1])
        {
            j++;
        }
        pixarlog_From14[i] = (uint16_t)j;
    }

    /* Reverse table: 8-bit input -> 11-bit code */
    j = 0;
    for (i = 0; i < FROM8_SIZE; i++) {
        while (
            (i / 255.0) * (i / 255.0)
            > pixarlog_ToLinearF[j] * pixarlog_ToLinearF[j + 1])
        {
            j++;
        }
        pixarlog_From8[i] = (uint16_t)j;
    }

    pixarlog_tables_ready = 1;
}


/*****************************************************************************/
/* Horizontal accumulation (decode) functions.
 *
 * Undo horizontal differencing and convert 11-bit log codes to linear values.
 * Each channel is accumulated independently across the scanline.
 *
 * wp: input uint16 horizontally differenced 11-bit coded values
 * n:  total number of samples in the scanline (stride * imagewidth)
 * stride: number of channels per pixel
 * op: output buffer
 */

static void
pixarlog_accumulate_float(
    const uint16_t* wp,
    int n,
    int stride,
    float* op)
{
    unsigned int mask = CODE_MASK;
    int i;
    unsigned int cr, cg, cb, ca;

    if (n < stride) {
        return;
    }

    if (stride == 3) {
        cr = wp[0] & mask;
        cg = wp[1] & mask;
        cb = wp[2] & mask;
        op[0] = pixarlog_ToLinearF[cr];
        op[1] = pixarlog_ToLinearF[cg];
        op[2] = pixarlog_ToLinearF[cb];
        for (i = 3; i < n; i += 3) {
            cr = (cr + wp[i + 0]) & mask;
            cg = (cg + wp[i + 1]) & mask;
            cb = (cb + wp[i + 2]) & mask;
            op[i + 0] = pixarlog_ToLinearF[cr];
            op[i + 1] = pixarlog_ToLinearF[cg];
            op[i + 2] = pixarlog_ToLinearF[cb];
        }
    }
    else if (stride == 4) {
        cr = wp[0] & mask;
        cg = wp[1] & mask;
        cb = wp[2] & mask;
        ca = wp[3] & mask;
        op[0] = pixarlog_ToLinearF[cr];
        op[1] = pixarlog_ToLinearF[cg];
        op[2] = pixarlog_ToLinearF[cb];
        op[3] = pixarlog_ToLinearF[ca];
        for (i = 4; i < n; i += 4) {
            cr = (cr + wp[i + 0]) & mask;
            cg = (cg + wp[i + 1]) & mask;
            cb = (cb + wp[i + 2]) & mask;
            ca = (ca + wp[i + 3]) & mask;
            op[i + 0] = pixarlog_ToLinearF[cr];
            op[i + 1] = pixarlog_ToLinearF[cg];
            op[i + 2] = pixarlog_ToLinearF[cb];
            op[i + 3] = pixarlog_ToLinearF[ca];
        }
    }
    else {
        int c;
        unsigned int acc;

        for (c = 0; c < stride; c++) {
            acc = wp[c] & mask;
            op[c] = pixarlog_ToLinearF[acc];
            for (i = stride + c; i < n; i += stride) {
                acc = (acc + wp[i]) & mask;
                op[i] = pixarlog_ToLinearF[acc];
            }
        }
    }
}


static void
pixarlog_accumulate_16(
    const uint16_t* wp,
    int n,
    int stride,
    uint16_t* op)
{
    unsigned int mask = CODE_MASK;
    int i;
    unsigned int cr, cg, cb, ca;

    if (n < stride) {
        return;
    }

    if (stride == 3) {
        cr = wp[0] & mask;
        cg = wp[1] & mask;
        cb = wp[2] & mask;
        op[0] = pixarlog_ToLinear16[cr];
        op[1] = pixarlog_ToLinear16[cg];
        op[2] = pixarlog_ToLinear16[cb];
        for (i = 3; i < n; i += 3) {
            cr = (cr + wp[i + 0]) & mask;
            cg = (cg + wp[i + 1]) & mask;
            cb = (cb + wp[i + 2]) & mask;
            op[i + 0] = pixarlog_ToLinear16[cr];
            op[i + 1] = pixarlog_ToLinear16[cg];
            op[i + 2] = pixarlog_ToLinear16[cb];
        }
    }
    else if (stride == 4) {
        cr = wp[0] & mask;
        cg = wp[1] & mask;
        cb = wp[2] & mask;
        ca = wp[3] & mask;
        op[0] = pixarlog_ToLinear16[cr];
        op[1] = pixarlog_ToLinear16[cg];
        op[2] = pixarlog_ToLinear16[cb];
        op[3] = pixarlog_ToLinear16[ca];
        for (i = 4; i < n; i += 4) {
            cr = (cr + wp[i + 0]) & mask;
            cg = (cg + wp[i + 1]) & mask;
            cb = (cb + wp[i + 2]) & mask;
            ca = (ca + wp[i + 3]) & mask;
            op[i + 0] = pixarlog_ToLinear16[cr];
            op[i + 1] = pixarlog_ToLinear16[cg];
            op[i + 2] = pixarlog_ToLinear16[cb];
            op[i + 3] = pixarlog_ToLinear16[ca];
        }
    }
    else {
        int c;
        unsigned int acc;

        for (c = 0; c < stride; c++) {
            acc = wp[c] & mask;
            op[c] = pixarlog_ToLinear16[acc];
            for (i = stride + c; i < n; i += stride) {
                acc = (acc + wp[i]) & mask;
                op[i] = pixarlog_ToLinear16[acc];
            }
        }
    }
}


static void
pixarlog_accumulate_12(
    const uint16_t* wp,
    int n,
    int stride,
    int16_t* op)
{
    unsigned int mask = CODE_MASK;
    int i;
    unsigned int cr, cg, cb, ca;
    float t;

    if (n < stride) {
        return;
    }

    if (stride == 3) {
        cr = wp[0] & mask;
        cg = wp[1] & mask;
        cb = wp[2] & mask;
        t = pixarlog_ToLinearF[cr] * SCALE12;
        op[0] = (int16_t)CLAMP12(t);
        t = pixarlog_ToLinearF[cg] * SCALE12;
        op[1] = (int16_t)CLAMP12(t);
        t = pixarlog_ToLinearF[cb] * SCALE12;
        op[2] = (int16_t)CLAMP12(t);
        for (i = 3; i < n; i += 3) {
            cr = (cr + wp[i + 0]) & mask;
            cg = (cg + wp[i + 1]) & mask;
            cb = (cb + wp[i + 2]) & mask;
            t = pixarlog_ToLinearF[cr] * SCALE12;
            op[i + 0] = (int16_t)CLAMP12(t);
            t = pixarlog_ToLinearF[cg] * SCALE12;
            op[i + 1] = (int16_t)CLAMP12(t);
            t = pixarlog_ToLinearF[cb] * SCALE12;
            op[i + 2] = (int16_t)CLAMP12(t);
        }
    }
    else if (stride == 4) {
        cr = wp[0] & mask;
        cg = wp[1] & mask;
        cb = wp[2] & mask;
        ca = wp[3] & mask;
        t = pixarlog_ToLinearF[cr] * SCALE12;
        op[0] = (int16_t)CLAMP12(t);
        t = pixarlog_ToLinearF[cg] * SCALE12;
        op[1] = (int16_t)CLAMP12(t);
        t = pixarlog_ToLinearF[cb] * SCALE12;
        op[2] = (int16_t)CLAMP12(t);
        t = pixarlog_ToLinearF[ca] * SCALE12;
        op[3] = (int16_t)CLAMP12(t);
        for (i = 4; i < n; i += 4) {
            cr = (cr + wp[i + 0]) & mask;
            cg = (cg + wp[i + 1]) & mask;
            cb = (cb + wp[i + 2]) & mask;
            ca = (ca + wp[i + 3]) & mask;
            t = pixarlog_ToLinearF[cr] * SCALE12;
            op[i + 0] = (int16_t)CLAMP12(t);
            t = pixarlog_ToLinearF[cg] * SCALE12;
            op[i + 1] = (int16_t)CLAMP12(t);
            t = pixarlog_ToLinearF[cb] * SCALE12;
            op[i + 2] = (int16_t)CLAMP12(t);
            t = pixarlog_ToLinearF[ca] * SCALE12;
            op[i + 3] = (int16_t)CLAMP12(t);
        }
    }
    else {
        int c;
        unsigned int acc;

        for (c = 0; c < stride; c++) {
            acc = wp[c] & mask;
            t = pixarlog_ToLinearF[acc] * SCALE12;
            op[c] = (int16_t)CLAMP12(t);
            for (i = stride + c; i < n; i += stride) {
                acc = (acc + wp[i]) & mask;
                t = pixarlog_ToLinearF[acc] * SCALE12;
                op[i] = (int16_t)CLAMP12(t);
            }
        }
    }
}


/* Return raw 11-bit log codes with horizontal differencing undone. */
static void
pixarlog_accumulate_11(
    const uint16_t* wp,
    int n,
    int stride,
    uint16_t* op)
{
    unsigned int mask = CODE_MASK;
    int i;
    unsigned int cr, cg, cb, ca;

    if (n < stride) {
        return;
    }

    if (stride == 3) {
        cr = wp[0] & mask;
        cg = wp[1] & mask;
        cb = wp[2] & mask;
        op[0] = (uint16_t)cr;
        op[1] = (uint16_t)cg;
        op[2] = (uint16_t)cb;
        for (i = 3; i < n; i += 3) {
            cr = (cr + wp[i + 0]) & mask;
            cg = (cg + wp[i + 1]) & mask;
            cb = (cb + wp[i + 2]) & mask;
            op[i + 0] = (uint16_t)cr;
            op[i + 1] = (uint16_t)cg;
            op[i + 2] = (uint16_t)cb;
        }
    }
    else if (stride == 4) {
        cr = wp[0] & mask;
        cg = wp[1] & mask;
        cb = wp[2] & mask;
        ca = wp[3] & mask;
        op[0] = (uint16_t)cr;
        op[1] = (uint16_t)cg;
        op[2] = (uint16_t)cb;
        op[3] = (uint16_t)ca;
        for (i = 4; i < n; i += 4) {
            cr = (cr + wp[i + 0]) & mask;
            cg = (cg + wp[i + 1]) & mask;
            cb = (cb + wp[i + 2]) & mask;
            ca = (ca + wp[i + 3]) & mask;
            op[i + 0] = (uint16_t)cr;
            op[i + 1] = (uint16_t)cg;
            op[i + 2] = (uint16_t)cb;
            op[i + 3] = (uint16_t)ca;
        }
    }
    else {
        int c;
        unsigned int acc;

        for (c = 0; c < stride; c++) {
            acc = wp[c] & mask;
            op[c] = (uint16_t)acc;
            for (i = stride + c; i < n; i += stride) {
                acc = (acc + wp[i]) & mask;
                op[i] = (uint16_t)acc;
            }
        }
    }
}


static void
pixarlog_accumulate_8(
    const uint16_t* wp,
    int n,
    int stride,
    uint8_t* op)
{
    unsigned int mask = CODE_MASK;
    int i;
    unsigned int cr, cg, cb, ca;

    if (n < stride) {
        return;
    }

    if (stride == 3) {
        cr = wp[0] & mask;
        cg = wp[1] & mask;
        cb = wp[2] & mask;
        op[0] = pixarlog_ToLinear8[cr];
        op[1] = pixarlog_ToLinear8[cg];
        op[2] = pixarlog_ToLinear8[cb];
        for (i = 3; i < n; i += 3) {
            cr = (cr + wp[i + 0]) & mask;
            cg = (cg + wp[i + 1]) & mask;
            cb = (cb + wp[i + 2]) & mask;
            op[i + 0] = pixarlog_ToLinear8[cr];
            op[i + 1] = pixarlog_ToLinear8[cg];
            op[i + 2] = pixarlog_ToLinear8[cb];
        }
    }
    else if (stride == 4) {
        cr = wp[0] & mask;
        cg = wp[1] & mask;
        cb = wp[2] & mask;
        ca = wp[3] & mask;
        op[0] = pixarlog_ToLinear8[cr];
        op[1] = pixarlog_ToLinear8[cg];
        op[2] = pixarlog_ToLinear8[cb];
        op[3] = pixarlog_ToLinear8[ca];
        for (i = 4; i < n; i += 4) {
            cr = (cr + wp[i + 0]) & mask;
            cg = (cg + wp[i + 1]) & mask;
            cb = (cb + wp[i + 2]) & mask;
            ca = (ca + wp[i + 3]) & mask;
            op[i + 0] = pixarlog_ToLinear8[cr];
            op[i + 1] = pixarlog_ToLinear8[cg];
            op[i + 2] = pixarlog_ToLinear8[cb];
            op[i + 3] = pixarlog_ToLinear8[ca];
        }
    }
    else {
        int c;
        unsigned int acc;

        for (c = 0; c < stride; c++) {
            acc = wp[c] & mask;
            op[c] = pixarlog_ToLinear8[acc];
            for (i = stride + c; i < n; i += stride) {
                acc = (acc + wp[i]) & mask;
                op[i] = pixarlog_ToLinear8[acc];
            }
        }
    }
}


/* 8-bit output with RGB->BGR / RGBA->ABGR channel swap. */
static void
pixarlog_accumulate_8abgr(
    const uint16_t* wp,
    int n,
    int stride,
    uint8_t* op)
{
    unsigned int mask = CODE_MASK;
    int i;
    unsigned int cr, cg, cb, ca;

    if (n < stride) {
        return;
    }

    if (stride == 3) {
        /* RGB input -> 0BGR output (4 bytes per pixel) */
        cr = wp[0] & mask;
        cg = wp[1] & mask;
        cb = wp[2] & mask;
        op[0] = 0;
        op[1] = pixarlog_ToLinear8[cb];
        op[2] = pixarlog_ToLinear8[cg];
        op[3] = pixarlog_ToLinear8[cr];
        for (i = 3; i < n; i += 3) {
            cr = (cr + wp[i + 0]) & mask;
            cg = (cg + wp[i + 1]) & mask;
            cb = (cb + wp[i + 2]) & mask;
            op += 4;
            op[0] = 0;
            op[1] = pixarlog_ToLinear8[cb];
            op[2] = pixarlog_ToLinear8[cg];
            op[3] = pixarlog_ToLinear8[cr];
        }
    }
    else if (stride == 4) {
        /* RGBA input -> ABGR output */
        cr = wp[0] & mask;
        cg = wp[1] & mask;
        cb = wp[2] & mask;
        ca = wp[3] & mask;
        op[0] = pixarlog_ToLinear8[ca];
        op[1] = pixarlog_ToLinear8[cb];
        op[2] = pixarlog_ToLinear8[cg];
        op[3] = pixarlog_ToLinear8[cr];
        for (i = 4; i < n; i += 4) {
            cr = (cr + wp[i + 0]) & mask;
            cg = (cg + wp[i + 1]) & mask;
            cb = (cb + wp[i + 2]) & mask;
            ca = (ca + wp[i + 3]) & mask;
            op += 4;
            op[0] = pixarlog_ToLinear8[ca];
            op[1] = pixarlog_ToLinear8[cb];
            op[2] = pixarlog_ToLinear8[cg];
            op[3] = pixarlog_ToLinear8[cr];
        }
    }
    else {
        /* For other strides, no swap; equivalent to regular 8-bit */
        pixarlog_accumulate_8(wp, n, stride, op);
    }
}


/*****************************************************************************/
/* Horizontal differencing (encode) functions.
 *
 * Convert linear values to 11-bit log codes and apply horizontal differencing.
 */

/* Clamp and convert float to 11-bit log code. */
static inline uint16_t
pixarlog_float_to_code(
    float v,
    const uint16_t* FromLT2)
{
    if (!(v >= 0.0f)) {
        /* negative or NaN */
        return 0;
    }
    if (v < 2.0f) {
        return FromLT2[(int)(v * pixarlog_Fltsize)];
    }
    if (v > 24.2f) {
        return 2047;
    }
    return (uint16_t)(pixarlog_LogK1 * log(v * pixarlog_LogK2) + 0.5);
}


static void
pixarlog_difference_float(
    const float* ip,
    int n,
    int stride,
    uint16_t* wp)
{
    int mask = CODE_MASK;
    int i, c;
    int32_t prev, cur;

    /* first pixel stores codes, subsequent pixels differences */
    for (c = 0; c < stride && c < n; c++) {
        prev = (int32_t)pixarlog_float_to_code(ip[c], pixarlog_FromLT2);
        wp[c] = (uint16_t)prev;
        for (i = stride + c; i < n; i += stride) {
            cur = (int32_t)pixarlog_float_to_code(ip[i], pixarlog_FromLT2);
            wp[i] = (uint16_t)((cur - prev) & mask);
            prev = cur;
        }
    }
}


static void
pixarlog_difference_16(
    const uint16_t* ip,
    int n,
    int stride,
    uint16_t* wp)
{
    int mask = CODE_MASK;
    int i, c;
    int32_t prev, cur;

    for (c = 0; c < stride && c < n; c++) {
        prev = (int32_t)pixarlog_From14[ip[c] >> 2];
        wp[c] = (uint16_t)prev;
        for (i = stride + c; i < n; i += stride) {
            cur = (int32_t)pixarlog_From14[ip[i] >> 2];
            wp[i] = (uint16_t)((cur - prev) & mask);
            prev = cur;
        }
    }
}


static void
pixarlog_difference_8(
    const uint8_t* ip,
    int n,
    int stride,
    uint16_t* wp)
{
    int mask = CODE_MASK;
    int i, c;
    int32_t prev, cur;

    for (c = 0; c < stride && c < n; c++) {
        prev = (int32_t)pixarlog_From8[ip[c]];
        wp[c] = (uint16_t)prev;
        for (i = stride + c; i < n; i += stride) {
            cur = (int32_t)pixarlog_From8[ip[i]];
            wp[i] = (uint16_t)((cur - prev) & mask);
            prev = cur;
        }
    }
}


/*****************************************************************************/
/* Public API */

/* Return bytes per sample for a given output format. */
static int
pixarlog_sample_size(
    int datafmt)
{
    switch (datafmt)
    {
        case PIXARLOG_FMT_FLOAT:
            return (int)sizeof(float);
        case PIXARLOG_FMT_16BIT:
        case PIXARLOG_FMT_12BITPICIO:
        case PIXARLOG_FMT_11BITLOG:
            return (int)sizeof(uint16_t);
        case PIXARLOG_FMT_8BIT:
        case PIXARLOG_FMT_8BITABGR:
            return (int)sizeof(uint8_t);
        default:
            return 0;
    }
}


/* Swap bytes of uint16 values in place. */
static void
pixarlog_swab16(
    uint16_t* data,
    ssize_t size)
{
    ssize_t i;

    for (i = 0; i < size; i++) {
        data[i] = (uint16_t)((data[i] >> 8) | (data[i] << 8));
    }
}


/* Return number of samples per scanline or PIXARLOG_VALUE_ERROR. */
static ssize_t
pixarlog_line_length(
    const ssize_t width,
    const ssize_t stride)
{
    if (width <= 0 || stride <= 0 || stride > INT_MAX / width) {
        return PIXARLOG_VALUE_ERROR;
    }
    return stride * width;
}


ssize_t
pixarlog_decode_raw(
    const uint8_t* src,
    const ssize_t srcsize,
    uint8_t* dst,
    const ssize_t dstsize,
    const ssize_t width,
    const ssize_t stride,
    const int datafmt,
    const int byteswap)
{
    ssize_t nsamples;
    ssize_t llen;
    ssize_t nrows;
    ssize_t row;
    ssize_t out_sample_size;
    ssize_t out_row_bytes;
    ssize_t required_dstsize;
    uint16_t* tbuf = NULL;
    const uint16_t* wp;
    uint8_t* op;

    if (src == NULL || dst == NULL) {
        return PIXARLOG_VALUE_ERROR;
    }
    if (srcsize <= 0 || dstsize <= 0) {
        return PIXARLOG_VALUE_ERROR;
    }
    if (srcsize % (ssize_t)sizeof(uint16_t) != 0) {
        return PIXARLOG_VALUE_ERROR;
    }

    out_sample_size = pixarlog_sample_size(datafmt);
    if (out_sample_size == 0) {
        return PIXARLOG_VALUE_ERROR;
    }

    nsamples = srcsize / (ssize_t)sizeof(uint16_t);
    llen = pixarlog_line_length(width, stride);

    if (llen <= 0) {
        return PIXARLOG_VALUE_ERROR;
    }

    /* truncate if not aligned to scanline */
    nsamples -= nsamples % llen;
    if (nsamples <= 0) {
        return PIXARLOG_VALUE_ERROR;
    }

    nrows = nsamples / llen;

    /* Compute required output size */
    if (datafmt == PIXARLOG_FMT_8BITABGR && (stride == 3 || stride == 4)) {
        /* 8BITABGR produces 4 output bytes per pixel regardless */
        out_row_bytes = width * 4;
    }
    else {
        out_row_bytes = llen * out_sample_size;
    }
    /* decode only as many scanlines as fit in the output */
    if (nrows > dstsize / out_row_bytes) {
        nrows = dstsize / out_row_bytes;
        if (nrows <= 0) {
            return PIXARLOG_OUTPUT_TOO_SMALL;
        }
        nsamples = nrows * llen;
    }
    required_dstsize = nrows * out_row_bytes;

    if (byteswap || ((uintptr_t)src % sizeof(uint16_t)) != 0) {
        tbuf = (uint16_t *)malloc((size_t)nsamples * sizeof(uint16_t));
        if (tbuf == NULL) {
            return PIXARLOG_MEMORY_ERROR;
        }
        memcpy(tbuf, src, (size_t)nsamples * sizeof(uint16_t));
        if (byteswap) {
            pixarlog_swab16(tbuf, nsamples);
        }
        wp = tbuf;
    }
    else {
        wp = (const uint16_t *)src;
    }
    op = dst;

    for (row = 0; row < nrows; row++) {
        switch (datafmt)
        {
            case PIXARLOG_FMT_FLOAT:
                pixarlog_accumulate_float(
                    wp,
                    (int)llen,
                    (int)stride,
                    (float *)op
                );
                break;
            case PIXARLOG_FMT_16BIT:
                pixarlog_accumulate_16(
                    wp,
                    (int)llen,
                    (int)stride,
                    (uint16_t *)op
                );
                break;
            case PIXARLOG_FMT_12BITPICIO:
                pixarlog_accumulate_12(
                    wp,
                    (int)llen,
                    (int)stride,
                    (int16_t *)op
                );
                break;
            case PIXARLOG_FMT_11BITLOG:
                pixarlog_accumulate_11(
                    wp,
                    (int)llen,
                    (int)stride,
                    (uint16_t *)op
                );
                break;
            case PIXARLOG_FMT_8BIT:
                pixarlog_accumulate_8(wp, (int)llen, (int)stride, op);
                break;
            case PIXARLOG_FMT_8BITABGR:
                pixarlog_accumulate_8abgr(wp, (int)llen, (int)stride, op);
                break;
            default:
                /* unreachable: datafmt validated by sample size */
                break;
        }
        wp += llen;
        op += out_row_bytes;
    }

    free(tbuf);
    return required_dstsize;
}


ssize_t
pixarlog_encode_raw(
    const uint8_t* src,
    const ssize_t srcsize,
    uint8_t* dst,
    const ssize_t dstsize,
    const ssize_t width,
    const ssize_t stride,
    const int datafmt,
    const int byteswap)
{
    ssize_t nsamples;
    ssize_t llen;
    ssize_t nrows;
    ssize_t row;
    ssize_t in_sample_size;
    ssize_t out_bytes;
    const uint8_t* ip;
    uint16_t* wp;
    uint16_t* wp0;

    if (src == NULL || dst == NULL) {
        return PIXARLOG_VALUE_ERROR;
    }
    if (srcsize <= 0 || dstsize <= 0) {
        return PIXARLOG_VALUE_ERROR;
    }
    if (
        datafmt != PIXARLOG_FMT_FLOAT
        && datafmt != PIXARLOG_FMT_16BIT
        && datafmt != PIXARLOG_FMT_8BIT
    ) {
        return PIXARLOG_VALUE_ERROR;
    }

    in_sample_size = pixarlog_sample_size(datafmt);

    llen = pixarlog_line_length(width, stride);
    if (llen <= 0) {
        return PIXARLOG_VALUE_ERROR;
    }

    nsamples = srcsize / in_sample_size;

    /* truncate if not aligned */
    nsamples -= nsamples % llen;
    if (nsamples <= 0) {
        return PIXARLOG_VALUE_ERROR;
    }

    nrows = nsamples / llen;
    out_bytes = nsamples * (ssize_t)sizeof(uint16_t);

    if (dstsize < out_bytes) {
        return PIXARLOG_OUTPUT_TOO_SMALL;
    }

    ip = src;
    if (((uintptr_t)dst % sizeof(uint16_t)) != 0) {
        wp0 = (uint16_t *)malloc((size_t)out_bytes);
        if (wp0 == NULL) {
            return PIXARLOG_MEMORY_ERROR;
        }
    }
    else {
        wp0 = (uint16_t *)dst;
    }
    wp = wp0;

    for (row = 0; row < nrows; row++) {
        switch (datafmt)
        {
            case PIXARLOG_FMT_FLOAT:
                pixarlog_difference_float(
                    (const float *)ip,
                    (int)llen,
                    (int)stride,
                    wp
                );
                ip += llen * sizeof(float);
                break;
            case PIXARLOG_FMT_16BIT:
                pixarlog_difference_16(
                    (const uint16_t *)ip,
                    (int)llen,
                    (int)stride,
                    wp
                );
                ip += llen * sizeof(uint16_t);
                break;
            case PIXARLOG_FMT_8BIT:
                pixarlog_difference_8(ip, (int)llen, (int)stride, wp);
                ip += llen * sizeof(uint8_t);
                break;
            default:
                /* unreachable: datafmt validated above */
                break;
        }
        wp += llen;
    }

    if (byteswap) {
        pixarlog_swab16(wp0, nsamples);
    }
    if (wp0 != (uint16_t *)dst) {
        memcpy(dst, wp0, (size_t)out_bytes);
        free(wp0);
    }

    return out_bytes;
}


/*****************************************************************************/
/* Full-pipeline functions with zlib inflate/deflate. */


/* Inflate src into dst until dst is full or the stream ends.
 * Truncated streams are not an error.
 * Return number of bytes written or PIXARLOG_ZLIB_ERROR. */
static ssize_t
pixarlog_inflate(
    const uint8_t* src,
    ssize_t srcsize,
    uint8_t* dst,
    ssize_t dstsize)
{
    z_stream strm;
    ssize_t chunk;
    int zret;

    memset(&strm, 0, sizeof(strm));
    if (inflateInit(&strm) != Z_OK) {
        return PIXARLOG_ZLIB_ERROR;
    }
    strm.next_in = (Bytef *)src;
    strm.next_out = (Bytef *)dst;

    do {
        /* avail_in and avail_out are 32-bit */
        if (strm.avail_in == 0 && srcsize > 0) {
            chunk = ((size_t)srcsize > (size_t)UINT_MAX)
                ? (ssize_t)UINT_MAX : srcsize;
            strm.avail_in = (uInt)chunk;
            srcsize -= chunk;
        }
        if (strm.avail_out == 0 && dstsize > 0) {
            chunk = ((size_t)dstsize > (size_t)UINT_MAX)
                ? (ssize_t)UINT_MAX : dstsize;
            strm.avail_out = (uInt)chunk;
            dstsize -= chunk;
        }
        zret = inflate(&strm, Z_NO_FLUSH);
    } while (
        zret == Z_OK
        && (strm.avail_in > 0 || srcsize > 0)
        && (strm.avail_out > 0 || dstsize > 0)
    );

    inflateEnd(&strm);

    if (zret != Z_OK && zret != Z_STREAM_END && zret != Z_BUF_ERROR) {
        return PIXARLOG_ZLIB_ERROR;
    }
    return (ssize_t)((uint8_t *)strm.next_out - dst);
}


/* Deflate src into dst.
 * Return number of bytes written or negative error code. */
static ssize_t
pixarlog_deflate(
    const uint8_t* src,
    ssize_t srcsize,
    uint8_t* dst,
    ssize_t dstsize,
    const int level)
{
    z_stream strm;
    ssize_t chunk;
    int zret;

    memset(&strm, 0, sizeof(strm));
    if (deflateInit(&strm, level) != Z_OK) {
        return PIXARLOG_ZLIB_ERROR;
    }
    strm.next_in = (Bytef *)src;
    strm.next_out = (Bytef *)dst;

    do {
        if (strm.avail_in == 0 && srcsize > 0) {
            chunk = ((size_t)srcsize > (size_t)UINT_MAX)
                ? (ssize_t)UINT_MAX : srcsize;
            strm.avail_in = (uInt)chunk;
            srcsize -= chunk;
        }
        if (strm.avail_out == 0 && dstsize > 0) {
            chunk = ((size_t)dstsize > (size_t)UINT_MAX)
                ? (ssize_t)UINT_MAX : dstsize;
            strm.avail_out = (uInt)chunk;
            dstsize -= chunk;
        }
        zret = deflate(&strm, (srcsize > 0) ? Z_NO_FLUSH : Z_FINISH);
        if (zret == Z_STREAM_ERROR) {
            break;
        }
        if (zret != Z_STREAM_END && strm.avail_out == 0 && dstsize == 0) {
            zret = Z_BUF_ERROR;
            break;
        }
    } while (zret != Z_STREAM_END);

    deflateEnd(&strm);

    if (zret == Z_BUF_ERROR) {
        return PIXARLOG_OUTPUT_TOO_SMALL;
    }
    if (zret != Z_STREAM_END) {
        return PIXARLOG_ZLIB_ERROR;
    }
    return (ssize_t)((uint8_t *)strm.next_out - dst);
}


ssize_t
pixarlog_decode(
    const uint8_t* src,
    const ssize_t srcsize,
    uint8_t* dst,
    const ssize_t dstsize,
    const ssize_t width,
    const ssize_t stride,
    const int datafmt,
    const int byteswap)
{
    ssize_t out_sample_size;
    ssize_t nsamples;
    ssize_t llen;
    ssize_t tbuf_bytes;
    uint8_t* tbuf = NULL;
    ssize_t result;

    if (src == NULL || dst == NULL || srcsize <= 0 || dstsize <= 0) {
        return PIXARLOG_VALUE_ERROR;
    }

    out_sample_size = pixarlog_sample_size(datafmt);
    if (out_sample_size == 0) {
        return PIXARLOG_VALUE_ERROR;
    }

    llen = pixarlog_line_length(width, stride);
    if (llen <= 0) {
        return PIXARLOG_VALUE_ERROR;
    }

    /*
     * Compute number of uint16 samples for intermediate buffer.
     * For 8BITABGR with stride 3, each pixel produces 4 output bytes
     * from 3 input samples.
     */
    if (datafmt == PIXARLOG_FMT_8BITABGR && stride == 3) {
        /* 4 output bytes per pixel, 3 samples per pixel */
        ssize_t npixels = dstsize / 4;
        nsamples = npixels * 3;
    }
    else {
        nsamples = dstsize / out_sample_size;
    }

    /* align to scanline boundary */
    nsamples -= nsamples % llen;
    if (nsamples <= 0) {
        return PIXARLOG_VALUE_ERROR;
    }

    /* Allocate intermediate buffer for inflated uint16 data */
    tbuf_bytes = nsamples * (ssize_t)sizeof(uint16_t);
    tbuf = (uint8_t *)malloc((size_t)tbuf_bytes);
    if (tbuf == NULL) {
        return PIXARLOG_MEMORY_ERROR;
    }

    /* inflate as many scanlines as fit in the output */
    tbuf_bytes = pixarlog_inflate(src, srcsize, tbuf, tbuf_bytes);
    if (tbuf_bytes < 0) {
        free(tbuf);
        return tbuf_bytes;
    }
    tbuf_bytes -= tbuf_bytes % (ssize_t)sizeof(uint16_t);
    if (byteswap) {
        pixarlog_swab16(
            (uint16_t *)tbuf,
            tbuf_bytes / (ssize_t)sizeof(uint16_t)
        );
    }

    /* pixarlog_decode_raw validates remaining parameters */
    result = pixarlog_decode_raw(
        tbuf,
        tbuf_bytes,
        dst,
        dstsize,
        width,
        stride,
        datafmt,
        0
    );

    free(tbuf);
    return result;
}


ssize_t
pixarlog_encode(
    const uint8_t* src,
    const ssize_t srcsize,
    uint8_t* dst,
    const ssize_t dstsize,
    const ssize_t width,
    const ssize_t stride,
    const int datafmt,
    const int level,
    const int byteswap)
{
    ssize_t in_sample_size;
    ssize_t tbuf_bytes;
    uint8_t* tbuf = NULL;
    ssize_t raw_bytes;
    ssize_t result;

    if (dst == NULL || dstsize <= 0) {
        return PIXARLOG_VALUE_ERROR;
    }

    in_sample_size = pixarlog_sample_size(datafmt);
    if (in_sample_size == 0) {
        return PIXARLOG_VALUE_ERROR;
    }

    /* Allocate intermediate buffer for uint16 log-encoded data.
     * Size is upper bound; pixarlog_encode_raw validates and aligns. */
    tbuf_bytes = (srcsize / in_sample_size) * (ssize_t)sizeof(uint16_t);
    if (tbuf_bytes <= 0) {
        return PIXARLOG_VALUE_ERROR;
    }
    tbuf = (uint8_t *)malloc((size_t)tbuf_bytes);
    if (tbuf == NULL) {
        return PIXARLOG_MEMORY_ERROR;
    }

    /* Encode pixels to uint16 log codes (validates all parameters) */
    raw_bytes = pixarlog_encode_raw(
        src,
        srcsize,
        tbuf,
        tbuf_bytes,
        width,
        stride,
        datafmt,
        byteswap
    );
    if (raw_bytes < 0) {
        free(tbuf);
        return raw_bytes;  /* propagate error */
    }

    result = pixarlog_deflate(tbuf, raw_bytes, dst, dstsize, level);
    free(tbuf);
    return result;
}
