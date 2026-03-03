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

/* Maximum stride for stack-allocated accumulators.
 * Strides larger than this use heap allocation. */
#define MAX_STACK_STRIDE 64


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
pixarlog_init(
    void)
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
        /* Generic stride path using stack or heap accumulators */
        unsigned int acc_stack[MAX_STACK_STRIDE];
        unsigned int* acc = acc_stack;
        int c;

        if (stride > MAX_STACK_STRIDE) {
            acc = (unsigned int *)malloc(stride * sizeof(unsigned int));
            if (acc == NULL) {
                return;
            }
        }

        for (c = 0; c < stride; c++) {
            acc[c] = wp[c] & mask;
            op[c] = pixarlog_ToLinearF[acc[c]];
        }
        for (i = stride; i < n; i += stride) {
            for (c = 0; c < stride; c++) {
                acc[c] = (acc[c] + wp[i + c]) & mask;
                op[i + c] = pixarlog_ToLinearF[acc[c]];
            }
        }

        if (acc != acc_stack) {
            free(acc);
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
        unsigned int acc_stack[MAX_STACK_STRIDE];
        unsigned int* acc = acc_stack;
        int c;

        if (stride > MAX_STACK_STRIDE) {
            acc = (unsigned int *)malloc(stride * sizeof(unsigned int));
            if (acc == NULL) {
                return;
            }
        }

        for (c = 0; c < stride; c++) {
            acc[c] = wp[c] & mask;
            op[c] = pixarlog_ToLinear16[acc[c]];
        }
        for (i = stride; i < n; i += stride) {
            for (c = 0; c < stride; c++) {
                acc[c] = (acc[c] + wp[i + c]) & mask;
                op[i + c] = pixarlog_ToLinear16[acc[c]];
            }
        }

        if (acc != acc_stack) {
            free(acc);
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
        unsigned int acc_stack[MAX_STACK_STRIDE];
        unsigned int* acc = acc_stack;
        int c;

        if (stride > MAX_STACK_STRIDE) {
            acc = (unsigned int *)malloc(stride * sizeof(unsigned int));
            if (acc == NULL) {
                return;
            }
        }

        for (c = 0; c < stride; c++) {
            acc[c] = wp[c] & mask;
            t = pixarlog_ToLinearF[acc[c]] * SCALE12;
            op[c] = (int16_t)CLAMP12(t);
        }
        for (i = stride; i < n; i += stride) {
            for (c = 0; c < stride; c++) {
                acc[c] = (acc[c] + wp[i + c]) & mask;
                t = pixarlog_ToLinearF[acc[c]] * SCALE12;
                op[i + c] = (int16_t)CLAMP12(t);
            }
        }

        if (acc != acc_stack) {
            free(acc);
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
        unsigned int acc_stack[MAX_STACK_STRIDE];
        unsigned int* acc = acc_stack;
        int c;

        if (stride > MAX_STACK_STRIDE) {
            acc = (unsigned int *)malloc(stride * sizeof(unsigned int));
            if (acc == NULL) {
                return;
            }
        }

        for (c = 0; c < stride; c++) {
            acc[c] = wp[c] & mask;
            op[c] = (uint16_t)acc[c];
        }
        for (i = stride; i < n; i += stride) {
            for (c = 0; c < stride; c++) {
                acc[c] = (acc[c] + wp[i + c]) & mask;
                op[i + c] = (uint16_t)acc[c];
            }
        }

        if (acc != acc_stack) {
            free(acc);
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
        unsigned int acc_stack[MAX_STACK_STRIDE];
        unsigned int* acc = acc_stack;
        int c;

        if (stride > MAX_STACK_STRIDE) {
            acc = (unsigned int *)malloc(stride * sizeof(unsigned int));
            if (acc == NULL) {
                return;
            }
        }

        for (c = 0; c < stride; c++) {
            acc[c] = wp[c] & mask;
            op[c] = pixarlog_ToLinear8[acc[c]];
        }
        for (i = stride; i < n; i += stride) {
            for (c = 0; c < stride; c++) {
                acc[c] = (acc[c] + wp[i + c]) & mask;
                op[i + c] = pixarlog_ToLinear8[acc[c]];
            }
        }

        if (acc != acc_stack) {
            free(acc);
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
    if (v < 0.0f) {
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
    int32_t prev_stack[MAX_STACK_STRIDE];
    int32_t* prev = prev_stack;

    if (n < stride) {
        return;
    }

    if (stride > MAX_STACK_STRIDE) {
        prev = (int32_t *)malloc(stride * sizeof(int32_t));
        if (prev == NULL) {
            return;
        }
    }

    /* First pixel: store codes as-is */
    for (c = 0; c < stride; c++) {
        prev[c] = (int32_t)pixarlog_float_to_code(
            ip[c],
            pixarlog_FromLT2
        );
        wp[c] = (uint16_t)prev[c];
    }

    /* Subsequent pixels: store difference from previous */
    for (i = stride; i < n; i += stride) {
        for (c = 0; c < stride; c++) {
            int32_t cur = (int32_t)pixarlog_float_to_code(
                ip[i + c],
                pixarlog_FromLT2
            );
            wp[i + c] = (uint16_t)((cur - prev[c]) & mask);
            prev[c] = cur;
        }
    }

    if (prev != prev_stack) {
        free(prev);
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
    int32_t prev_stack[MAX_STACK_STRIDE];
    int32_t* prev = prev_stack;

    if (n < stride) {
        return;
    }

    if (stride > MAX_STACK_STRIDE) {
        prev = (int32_t *)malloc(stride * sizeof(int32_t));
        if (prev == NULL) {
            return;
        }
    }

    /* First pixel */
    for (c = 0; c < stride; c++) {
        prev[c] = (int32_t)pixarlog_From14[ip[c] >> 2];
        wp[c] = (uint16_t)prev[c];
    }

    /* Subsequent pixels */
    for (i = stride; i < n; i += stride) {
        for (c = 0; c < stride; c++) {
            int32_t cur = (int32_t)pixarlog_From14[ip[i + c] >> 2];
            wp[i + c] = (uint16_t)((cur - prev[c]) & mask);
            prev[c] = cur;
        }
    }

    if (prev != prev_stack) {
        free(prev);
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
    int32_t prev_stack[MAX_STACK_STRIDE];
    int32_t* prev = prev_stack;

    if (n < stride) {
        return;
    }

    if (stride > MAX_STACK_STRIDE) {
        prev = (int32_t *)malloc(stride * sizeof(int32_t));
        if (prev == NULL) {
            return;
        }
    }

    /* First pixel */
    for (c = 0; c < stride; c++) {
        prev[c] = (int32_t)pixarlog_From8[ip[c]];
        wp[c] = (uint16_t)prev[c];
    }

    /* Subsequent pixels */
    for (i = stride; i < n; i += stride) {
        for (c = 0; c < stride; c++) {
            int32_t cur = (int32_t)pixarlog_From8[ip[i + c]];
            wp[i + c] = (uint16_t)((cur - prev[c]) & mask);
            prev[c] = cur;
        }
    }

    if (prev != prev_stack) {
        free(prev);
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


ssize_t
pixarlog_decode_raw(
    const uint8_t* src,
    const ssize_t srcsize,
    uint8_t* dst,
    const ssize_t dstsize,
    const ssize_t width,
    const ssize_t stride,
    const int datafmt)
{
    ssize_t nsamples;
    ssize_t llen;
    ssize_t nrows;
    ssize_t row;
    ssize_t out_sample_size;
    ssize_t out_row_bytes;
    ssize_t required_dstsize;
    const uint16_t* wp;
    uint8_t* op;

    if (src == NULL || dst == NULL) {
        return PIXARLOG_VALUE_ERROR;
    }
    if (srcsize <= 0 || dstsize <= 0) {
        return PIXARLOG_VALUE_ERROR;
    }
    if (width <= 0 || stride <= 0) {
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
    llen = stride * width;

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
    required_dstsize = nrows * out_row_bytes;

    if (dstsize < required_dstsize) {
        return PIXARLOG_OUTPUT_TOO_SMALL;
    }

    wp = (const uint16_t *)src;
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
                return PIXARLOG_VALUE_ERROR;
        }
        wp += llen;
        op += out_row_bytes;
    }

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
    const int datafmt)
{
    ssize_t nsamples;
    ssize_t llen;
    ssize_t nrows;
    ssize_t row;
    ssize_t in_sample_size;
    ssize_t out_bytes;
    const uint8_t* ip;
    uint16_t* wp;

    if (src == NULL || dst == NULL) {
        return PIXARLOG_VALUE_ERROR;
    }
    if (srcsize <= 0 || dstsize <= 0) {
        return PIXARLOG_VALUE_ERROR;
    }
    if (width <= 0 || stride <= 0) {
        return PIXARLOG_VALUE_ERROR;
    }

    in_sample_size = pixarlog_sample_size(datafmt);
    if (in_sample_size == 0) {
        return PIXARLOG_VALUE_ERROR;
    }

    llen = stride * width;
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
    wp = (uint16_t *)dst;

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
                /* Encoding not supported for 11BITLOG, 12BITPICIO, 8BITABGR */
                return PIXARLOG_VALUE_ERROR;
        }
        wp += llen;
    }

    return out_bytes;
}


/*****************************************************************************/
/* Full-pipeline functions with zlib inflate/deflate. */


ssize_t
pixarlog_decode(
    const uint8_t* src,
    const ssize_t srcsize,
    uint8_t* dst,
    const ssize_t dstsize,
    const ssize_t width,
    const ssize_t stride,
    const int datafmt)
{
    ssize_t out_sample_size;
    ssize_t nsamples;
    ssize_t llen;
    ssize_t tbuf_bytes;
    uint8_t* tbuf = NULL;
    uLongf tbuf_len;
    int zret;
    ssize_t result;

    out_sample_size = pixarlog_sample_size(datafmt);
    if (out_sample_size == 0) {
        return PIXARLOG_VALUE_ERROR;
    }

    llen = stride * width;
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

    /* zlib uncompress */
    tbuf_len = (uLongf)tbuf_bytes;
    zret = uncompress(
        (Bytef *)tbuf,
        &tbuf_len,
        (const Bytef *)src,
        (uLong)srcsize
    );
    if (zret != Z_OK) {
        free(tbuf);
        return PIXARLOG_ZLIB_ERROR;
    }

    tbuf_bytes = (ssize_t)tbuf_len;

    /* Decode the inflated uint16 data to output pixels.
     * pixarlog_decode_raw validates remaining parameters. */
    result = pixarlog_decode_raw(
        tbuf,
        tbuf_bytes,
        dst,
        dstsize,
        width,
        stride,
        datafmt
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
    const int level)
{
    ssize_t in_sample_size;
    ssize_t tbuf_bytes;
    uint8_t* tbuf = NULL;
    ssize_t raw_bytes;
    uLongf dst_len;
    int zret;

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
        datafmt
    );
    if (raw_bytes < 0) {
        free(tbuf);
        return raw_bytes;  /* propagate error */
    }

    /* zlib compress */
    dst_len = (uLongf)dstsize;
    zret = compress2(
        (Bytef *)dst,
        &dst_len,
        (const Bytef *)tbuf,
        (uLong)raw_bytes,
        level
    );
    free(tbuf);

    if (zret != Z_OK) {
        if (zret == Z_BUF_ERROR) {
            return PIXARLOG_OUTPUT_TOO_SMALL;
        }
        return PIXARLOG_ZLIB_ERROR;
    }

    return (ssize_t)dst_len;
}
