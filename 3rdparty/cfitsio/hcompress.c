/* hcompress.c */
/*
  Copyright (Unpublished--all rights reserved under the copyright laws of
  the United States), U.S. Government as represented by the Administrator
  of the National Aeronautics and Space Administration.  No copyright is
  claimed in the United States under Title 17, U.S. Code.

  Permission to freely use, copy, modify, and distribute this software
  and its documentation without fee is hereby granted, provided that this
  copyright notice and disclaimer of warranty appears in all copies.

  DISCLAIMER:

  THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT ANY WARRANTY OF ANY KIND,
  EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO,
  ANY WARRANTY THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY
  IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
  PURPOSE, AND FREEDOM FROM INFRINGEMENT, AND ANY WARRANTY THAT THE
  DOCUMENTATION WILL CONFORM TO THE SOFTWARE, OR ANY WARRANTY THAT THE
  SOFTWARE WILL BE ERROR FREE.  IN NO EVENT SHALL NASA BE LIABLE FOR ANY
  DAMAGES, INCLUDING, BUT NOT LIMITED TO, DIRECT, INDIRECT, SPECIAL OR
  CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM, OR IN ANY WAY
  CONNECTED WITH THIS SOFTWARE, WHETHER OR NOT BASED UPON WARRANTY,
  CONTRACT, TORT , OR OTHERWISE, WHETHER OR NOT INJURY WAS SUSTAINED BY
  PERSONS OR PROPERTY OR OTHERWISE, AND WHETHER OR NOT LOSS WAS SUSTAINED
  FROM, OR AROSE OUT OF THE RESULTS OF, OR USE OF, THE SOFTWARE OR
  SERVICES PROVIDED HEREUNDER.
*/
/*
  Modifications applied by Christoph Gohlke:

  - Lint code.
  - Return error codes.
  - Remove CFITSIO dependencies (ffpmsg, FFLOCK/FFUNLOCK, fitsio2.h).
  - Replace LONGLONG with long long.
  - Rename public functions from fits_h* to hcomp_*.
  - Combine fits_hcompress.c and fits_hdecompress.c into single file.
  - Replace static globals with context structs for thread safety.
  - Add na (allocation size) parameter and overflow checks in decode.
  - Add (size_t) cast in qtree_decode malloc.
  - Fix memory leak in qtree_encode error path.
  - Add bounds check in output_nnybble to prevent buffer overflow.
  - Add nbin (input size) parameter and bounds checks in decode functions.
  - Add integer overflow check for nx*ny in encode functions.
  - Replace float-based log2n with integer-only ilog2n.
  - Validate positive dimensions in compress functions.
  - Return early on doencode/doencode64 error in encode functions.

*/
/*
  The following routines to apply the H-compress compression algorithm
  to a 2-D FITS image were written by R. White at the STScI and were
  obtained from the STScI at http://www.stsci.edu/software/hcompress.html

  The compress source is a concatenation of:
    htrans.c, digitize.c, encode.c, qwrite.c, doencode.c,
    bit_output.c, qtree_encode.c

  The decompress source is a concatenation of:
    hinv.c, hsmooth.c, undigitize.c, decode.c, dodecode.c,
    qtree_decode.c, qread.c, bit_input.c
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>

#include "hcompress.h"

#ifndef min
#define min(a,b) (((a)<(b))?(a):(b))
#endif
#ifndef max
#define max(a,b) (((a)>(b))?(a):(b))
#endif

/* Compute ceil(log2(n)) using integer arithmetic. */
static int
ilog2n(
    int n)
{
    int log2n = 0;
    unsigned int v = 1;
    while (v < (unsigned int)n) {
        v <<= 1;
        log2n++;
    }
    return log2n;
}

/* Thread-safe contexts replacing former static globals */

typedef struct {
    long noutchar;
    long noutmax;
    long long bitcount;
    int buffer2;
    int bits_to_go2;
    int bitbuffer;
    int bits_to_go3;
} hcomp_cctx;

typedef struct {
    long nextchar;
    long ninmax;
    int buffer2_in;
    int bits_to_go_in;
} hcomp_dctx;

/* ======================================================================== */
/*  COMPRESS routines                                                       */
/* ======================================================================== */

static int htrans(
    int a[],
    int nx,
    int ny
);
static void digitize(
    int a[],
    int nx,
    int ny,
    int scale
);
static int encode(
    hcomp_cctx *ctx,
    char *outfile,
    long *nlen,
    int a[],
    int nx,
    int ny,
    int scale
);
static void shuffle(
    int a[],
    int n,
    int n2,
    int tmp[]
);

static int htrans64(
    long long a[],
    int nx,
    int ny
);
static void digitize64(
    long long a[],
    int nx,
    int ny,
    int scale
);
static int encode64(
    hcomp_cctx *ctx,
    char *outfile,
    long *nlen,
    long long a[],
    int nx,
    int ny,
    int scale
);
static void shuffle64(
    long long a[],
    int n,
    int n2,
    long long tmp[]
);

static void writeint(
    hcomp_cctx *ctx,
    char *outfile,
    int a
);
static void writelonglong(
    hcomp_cctx *ctx,
    char *outfile,
    long long a
);
static int doencode(
    hcomp_cctx *ctx,
    char *outfile,
    int a[],
    int nx,
    int ny,
    unsigned char nbitplanes[3]
);
static int doencode64(
    hcomp_cctx *ctx,
    char *outfile,
    long long a[],
    int nx,
    int ny,
    unsigned char nbitplanes[3]
);
static int qwrite(
    hcomp_cctx *ctx,
    char *file,
    char buffer[],
    int n
);

static int qtree_encode(
    hcomp_cctx *ctx,
    char *outfile,
    int a[],
    int n,
    int nqx,
    int nqy,
    int nbitplanes
);
static int qtree_encode64(
    hcomp_cctx *ctx,
    char *outfile,
    long long a[],
    int n,
    int nqx,
    int nqy,
    int nbitplanes
);
static void start_outputing_bits(
    hcomp_cctx *ctx
);
static void done_outputing_bits(
    hcomp_cctx *ctx,
    char *outfile
);
static void output_nbits(
    hcomp_cctx *ctx,
    char *outfile,
    int bits,
    int n
);
static void output_nybble(
    hcomp_cctx *ctx,
    char *outfile,
    int bits
);
static void output_nnybble(
    hcomp_cctx *ctx,
    char *outfile,
    int n,
    unsigned char array[]
);

static void qtree_onebit(
    int a[],
    int n,
    int nx,
    int ny,
    unsigned char b[],
    int bit
);
static void qtree_onebit64(
    long long a[],
    int n,
    int nx,
    int ny,
    unsigned char b[],
    int bit
);
static void qtree_reduce(
    unsigned char a[],
    int n,
    int nx,
    int ny,
    unsigned char b[]
);
static int bufcopy(
    hcomp_cctx *ctx,
    unsigned char a[],
    int n,
    unsigned char buffer[],
    int *b,
    int bmax
);
static void write_bdirect(
    hcomp_cctx *ctx,
    char *outfile,
    int a[],
    int n,
    int nqx,
    int nqy,
    unsigned char scratch[],
    int bit
);
static void write_bdirect64(
    hcomp_cctx *ctx,
    char *outfile,
    long long a[],
    int n,
    int nqx,
    int nqy,
    unsigned char scratch[],
    int bit
);

static void output_nybble(
    hcomp_cctx *ctx,
    char *outfile,
    int bits
);
static void output_nnybble(
    hcomp_cctx *ctx,
    char *outfile,
    int n,
    unsigned char array[]
);

#define output_huffman(ctx,outfile,c) \
    output_nbits(ctx,outfile,code[c],ncode[c])

/* ======================================================================== */
/*  DECOMPRESS forward declarations                                         */
/* ======================================================================== */

static int decode(
    hcomp_dctx *ctx,
    unsigned char *infile,
    int *a,
    int na,
    int *nx,
    int *ny,
    int *scale
);
static int decode64(
    hcomp_dctx *ctx,
    unsigned char *infile,
    long long *a,
    int na,
    int *nx,
    int *ny,
    int *scale
);
static int hinv(
    int a[],
    int nx,
    int ny,
    int smooth,
    int scale
);
static int hinv64(
    long long a[],
    int nx,
    int ny,
    int smooth,
    int scale
);
static void undigitize(
    int a[],
    int nx,
    int ny,
    int scale
);
static void undigitize64(
    long long a[],
    int nx,
    int ny,
    int scale
);
static void unshuffle(
    int a[],
    int n,
    int n2,
    int tmp[]
);
static void unshuffle64(
    long long a[],
    int n,
    int n2,
    long long tmp[]
);
static void hsmooth(
    int a[],
    int nxtop,
    int nytop,
    int ny,
    int scale
);
static void hsmooth64(
    long long a[],
    int nxtop,
    int nytop,
    int ny,
    int scale
);
static void qread(
    hcomp_dctx *ctx,
    unsigned char *infile,
    char *a,
    int n
);
static int readint(
    hcomp_dctx *ctx,
    unsigned char *infile
);
static long long readlonglong(
    hcomp_dctx *ctx,
    unsigned char *infile
);
static int dodecode(
    hcomp_dctx *ctx,
    unsigned char *infile,
    int a[],
    int nx,
    int ny,
    unsigned char nbitplanes[3]
);
static int dodecode64(
    hcomp_dctx *ctx,
    unsigned char *infile,
    long long a[],
    int nx,
    int ny,
    unsigned char nbitplanes[3]
);
static int qtree_decode(
    hcomp_dctx *ctx,
    unsigned char *infile,
    int a[],
    int n,
    int nqx,
    int nqy,
    int nbitplanes
);
static int qtree_decode64(
    hcomp_dctx *ctx,
    unsigned char *infile,
    long long a[],
    int n,
    int nqx,
    int nqy,
    int nbitplanes
);
static void start_inputing_bits(
    hcomp_dctx *ctx
);
static int input_bit(
    hcomp_dctx *ctx,
    unsigned char *infile
);
static int input_nbits(
    hcomp_dctx *ctx,
    unsigned char *infile,
    int n
);
static int input_nybble(
    hcomp_dctx *ctx,
    unsigned char *infile
);
static int input_nnybble(
    hcomp_dctx *ctx,
    unsigned char *infile,
    int n,
    unsigned char *array
);

static void qtree_expand(
    hcomp_dctx *ctx,
    unsigned char *infile,
    unsigned char a[],
    int nx,
    int ny,
    unsigned char b[]
);
static void qtree_bitins(
    unsigned char a[],
    int nx,
    int ny,
    int b[],
    int n,
    int bit
);
static void qtree_bitins64(
    unsigned char a[],
    int nx,
    int ny,
    long long b[],
    int n,
    int bit
);
static void qtree_copy(
    unsigned char a[],
    int nx,
    int ny,
    unsigned char b[],
    int n
);
static void read_bdirect(
    hcomp_dctx *ctx,
    unsigned char *infile,
    int a[],
    int n,
    int nqx,
    int nqy,
    unsigned char scratch[],
    int bit
);
static void read_bdirect64(
    hcomp_dctx *ctx,
    unsigned char *infile,
    long long a[],
    int n,
    int nqx,
    int nqy,
    unsigned char scratch[],
    int bit
);
static int input_huffman(
    hcomp_dctx *ctx,
    unsigned char *infile
);

/* ======================================================================== */
/*  PUBLIC API                                                              */
/* ======================================================================== */

int hcomp_compress(
    int *a,
    int ny,
    int nx,
    int scale,
    char *output,
    long *nbytes,
    int *status)
{
    int stat;
    hcomp_cctx ctx;

    if (*status > 0) return(*status);

    if (nx <= 0 || ny <= 0) {
        *status = HCOMP_ERROR;
        return(*status);
    }
    if (nx > INT_MAX / ny) {
        *status = HCOMP_ERROR_OVERFLOW;
        return(*status);
    }

    stat = htrans(a, nx, ny);
    if (stat) {
        *status = stat;
        return(*status);
    }

    digitize(a, nx, ny, scale);

    ctx.noutmax = *nbytes;
    *nbytes = 0;

    stat = encode(&ctx, output, nbytes, a, nx, ny, scale);

    *status = stat;
    return(*status);
}

int hcomp_compress64(
    long long *a,
    int ny,
    int nx,
    int scale,
    char *output,
    long *nbytes,
    int *status)
{
    int stat;
    hcomp_cctx ctx;

    if (*status > 0) return(*status);

    if (nx <= 0 || ny <= 0) {
        *status = HCOMP_ERROR;
        return(*status);
    }
    if (nx > INT_MAX / ny) {
        *status = HCOMP_ERROR_OVERFLOW;
        return(*status);
    }

    stat = htrans64(a, nx, ny);
    if (stat) {
        *status = stat;
        return(*status);
    }

    digitize64(a, nx, ny, scale);

    ctx.noutmax = *nbytes;
    *nbytes = 0;

    stat = encode64(&ctx, output, nbytes, a, nx, ny, scale);

    *status = stat;
    return(*status);
}

int hcomp_decompress(
    unsigned char *input,
    int nbin,
    int smooth,
    int *a,
    int na,
    int *ny,
    int *nx,
    int *scale,
    int *status)
{
    int stat;
    hcomp_dctx ctx;

    if (*status > 0) return(*status);

    ctx.nextchar = 0;
    ctx.ninmax = nbin;
    stat = decode(&ctx, input, a, na, nx, ny, scale);
    *status = stat;
    if (stat) return(*status);

    undigitize(a, *nx, *ny, *scale);

    stat = hinv(a, *nx, *ny, smooth, *scale);
    *status = stat;

    return(*status);
}

int hcomp_decompress64(
    unsigned char *input,
    int nbin,
    int smooth,
    long long *a,
    int na,
    int *ny,
    int *nx,
    int *scale,
    int *status)
{
    int stat;
    hcomp_dctx ctx;

    if (*status > 0) return(*status);

    ctx.nextchar = 0;
    ctx.ninmax = nbin;
    stat = decode64(&ctx, input, a, na, nx, ny, scale);
    *status = stat;
    if (stat) return(*status);

    undigitize64(a, *nx, *ny, *scale);

    stat = hinv64(a, *nx, *ny, smooth, *scale);
    *status = stat;

    return(*status);
}

/* ======================================================================== */
/*  htrans.c   H-transform of NX x NY integer image                        */
/*  Programmer: R. White         Date: 11 May 1992                          */
/* ======================================================================== */

static int
htrans(
    int a[],
    int nx,
    int ny)
{
    int nmax, log2n, h0, hx, hy, hc, nxtop, nytop, i, j, k;
    int oddx, oddy;
    int shift, mask, mask2, prnd, prnd2, nrnd2;
    int s10, s00;
    int *tmp;

    nmax = (nx > ny) ? nx : ny;
    log2n = ilog2n(nmax);

    tmp = (int *)malloc(((nmax + 1) / 2) * sizeof(int));
    if (tmp == (int *)NULL) {
        return HCOMP_ERROR_MEMORY;
    }

    shift = 0;
    mask = -2;
    mask2 = mask << 1;
    prnd = 1;
    prnd2 = prnd << 1;
    nrnd2 = prnd2 - 1;

    nxtop = nx;
    nytop = ny;

    for (k = 0; k < log2n; k++) {
        oddx = nxtop % 2;
        oddy = nytop % 2;
        for (i = 0; i < nxtop - oddx; i += 2) {
            s00 = i * ny;
            s10 = s00 + ny;
            for (j = 0; j < nytop - oddy; j += 2) {
                h0 = (a[s10 + 1] + a[s10] + a[s00 + 1] + a[s00]) >> shift;
                hx = (a[s10 + 1] + a[s10] - a[s00 + 1] - a[s00]) >> shift;
                hy = (a[s10 + 1] - a[s10] + a[s00 + 1] - a[s00]) >> shift;
                hc = (a[s10 + 1] - a[s10] - a[s00 + 1] + a[s00]) >> shift;

                a[s10 + 1] = hc;
                a[s10] = ((hx >= 0) ? (hx + prnd) : hx) & mask;
                a[s00 + 1] = ((hy >= 0) ? (hy + prnd) : hy) & mask;
                a[s00] = ((h0 >= 0) ? (h0 + prnd2) : (h0 + nrnd2)) & mask2;
                s00 += 2;
                s10 += 2;
            }
            if (oddy) {
                h0 = (a[s10] + a[s00]) << (1 - shift);
                hx = (a[s10] - a[s00]) << (1 - shift);
                a[s10] = ((hx >= 0) ? (hx + prnd) : hx) & mask;
                a[s00] = ((h0 >= 0) ? (h0 + prnd2) : (h0 + nrnd2)) & mask2;
                s00 += 1;
                s10 += 1;
            }
        }
        if (oddx) {
            s00 = i * ny;
            for (j = 0; j < nytop - oddy; j += 2) {
                h0 = (a[s00 + 1] + a[s00]) << (1 - shift);
                hy = (a[s00 + 1] - a[s00]) << (1 - shift);
                a[s00 + 1] = ((hy >= 0) ? (hy + prnd) : hy) & mask;
                a[s00] = ((h0 >= 0) ? (h0 + prnd2) : (h0 + nrnd2)) & mask2;
                s00 += 2;
            }
            if (oddy) {
                h0 = a[s00] << (2 - shift);
                a[s00] = ((h0 >= 0) ? (h0 + prnd2) : (h0 + nrnd2)) & mask2;
            }
        }
        for (i = 0; i < nxtop; i++) {
            shuffle(&a[ny * i], nytop, 1, tmp);
        }
        for (j = 0; j < nytop; j++) {
            shuffle(&a[j], nxtop, ny, tmp);
        }
        nxtop = (nxtop + 1) >> 1;
        nytop = (nytop + 1) >> 1;
        shift = 1;
        mask = mask2;
        prnd = prnd2;
        mask2 = mask2 << 1;
        prnd2 = prnd2 << 1;
        nrnd2 = prnd2 - 1;
    }
    free(tmp);
    return 0;
}

static int
htrans64(
    long long a[],
    int nx,
    int ny)
{
    int nmax, log2n, nxtop, nytop, i, j, k;
    int oddx, oddy;
    int shift;
    int s10, s00;
    long long h0, hx, hy, hc, prnd, prnd2, nrnd2, mask, mask2;
    long long *tmp;

    nmax = (nx > ny) ? nx : ny;
    log2n = ilog2n(nmax);

    tmp = (long long *)malloc(((nmax + 1) / 2) * sizeof(long long));
    if (tmp == (long long *)NULL) {
        return HCOMP_ERROR_MEMORY;
    }

    shift = 0;
    mask = (long long)-2;
    mask2 = mask << 1;
    prnd = (long long)1;
    prnd2 = prnd << 1;
    nrnd2 = prnd2 - 1;

    nxtop = nx;
    nytop = ny;

    for (k = 0; k < log2n; k++) {
        oddx = nxtop % 2;
        oddy = nytop % 2;
        for (i = 0; i < nxtop - oddx; i += 2) {
            s00 = i * ny;
            s10 = s00 + ny;
            for (j = 0; j < nytop - oddy; j += 2) {
                h0 = (a[s10 + 1] + a[s10] + a[s00 + 1] + a[s00]) >> shift;
                hx = (a[s10 + 1] + a[s10] - a[s00 + 1] - a[s00]) >> shift;
                hy = (a[s10 + 1] - a[s10] + a[s00 + 1] - a[s00]) >> shift;
                hc = (a[s10 + 1] - a[s10] - a[s00 + 1] + a[s00]) >> shift;

                a[s10 + 1] = hc;
                a[s10] = ((hx >= 0) ? (hx + prnd) : hx) & mask;
                a[s00 + 1] = ((hy >= 0) ? (hy + prnd) : hy) & mask;
                a[s00] = ((h0 >= 0) ? (h0 + prnd2) : (h0 + nrnd2)) & mask2;
                s00 += 2;
                s10 += 2;
            }
            if (oddy) {
                h0 = (a[s10] + a[s00]) << (1 - shift);
                hx = (a[s10] - a[s00]) << (1 - shift);
                a[s10] = ((hx >= 0) ? (hx + prnd) : hx) & mask;
                a[s00] = ((h0 >= 0) ? (h0 + prnd2) : (h0 + nrnd2)) & mask2;
                s00 += 1;
                s10 += 1;
            }
        }
        if (oddx) {
            s00 = i * ny;
            for (j = 0; j < nytop - oddy; j += 2) {
                h0 = (a[s00 + 1] + a[s00]) << (1 - shift);
                hy = (a[s00 + 1] - a[s00]) << (1 - shift);
                a[s00 + 1] = ((hy >= 0) ? (hy + prnd) : hy) & mask;
                a[s00] = ((h0 >= 0) ? (h0 + prnd2) : (h0 + nrnd2)) & mask2;
                s00 += 2;
            }
            if (oddy) {
                h0 = a[s00] << (2 - shift);
                a[s00] = ((h0 >= 0) ? (h0 + prnd2) : (h0 + nrnd2)) & mask2;
            }
        }
        for (i = 0; i < nxtop; i++) {
            shuffle64(&a[ny * i], nytop, 1, tmp);
        }
        for (j = 0; j < nytop; j++) {
            shuffle64(&a[j], nxtop, ny, tmp);
        }
        nxtop = (nxtop + 1) >> 1;
        nytop = (nytop + 1) >> 1;
        shift = 1;
        mask = mask2;
        prnd = prnd2;
        mask2 = mask2 << 1;
        prnd2 = prnd2 << 1;
        nrnd2 = prnd2 - 1;
    }
    free(tmp);
    return 0;
}

/* ======================================================================== */

static void
shuffle(
    int a[],
    int n,
    int n2,
    int tmp[])
{
    int i;
    int *p1, *p2, *pt;

    pt = tmp;
    p1 = &a[n2];
    for (i = 1; i < n; i += 2) {
        *pt = *p1;
        pt += 1;
        p1 += (n2 + n2);
    }
    p1 = &a[n2];
    p2 = &a[n2 + n2];
    for (i = 2; i < n; i += 2) {
        *p1 = *p2;
        p1 += n2;
        p2 += (n2 + n2);
    }
    pt = tmp;
    for (i = 1; i < n; i += 2) {
        *p1 = *pt;
        p1 += n2;
        pt += 1;
    }
}

static void
shuffle64(
    long long a[],
    int n,
    int n2,
    long long tmp[])
{
    int i;
    long long *p1, *p2, *pt;

    pt = tmp;
    p1 = &a[n2];
    for (i = 1; i < n; i += 2) {
        *pt = *p1;
        pt += 1;
        p1 += (n2 + n2);
    }
    p1 = &a[n2];
    p2 = &a[n2 + n2];
    for (i = 2; i < n; i += 2) {
        *p1 = *p2;
        p1 += n2;
        p2 += (n2 + n2);
    }
    pt = tmp;
    for (i = 1; i < n; i += 2) {
        *p1 = *pt;
        p1 += n2;
        pt += 1;
    }
}

/* ======================================================================== */
/*  digitize.c  digitize H-transform                                        */
/* ======================================================================== */

static void
digitize(
    int a[],
    int nx,
    int ny,
    int scale)
{
    int d, *p;

    if (scale <= 1) return;
    d = (scale + 1) / 2 - 1;
    for (p = a; p <= &a[nx * ny - 1]; p++)
        *p = ((*p > 0) ? (*p + d) : (*p - d)) / scale;
}

static void
digitize64(
    long long a[],
    int nx,
    int ny,
    int scale)
{
    long long d, *p, scale64;

    if (scale <= 1) return;
    d = (scale + 1) / 2 - 1;
    scale64 = scale;

    for (p = a; p <= &a[nx * ny - 1]; p++)
        *p = ((*p > 0) ? (*p + d) : (*p - d)) / scale64;
}

/* ======================================================================== */
/*  encode.c    encode H-transform and write to outfile                     */
/* ======================================================================== */

static char code_magic[2] = { (char)0xDD, (char)0x99 };

static int
encode(
    hcomp_cctx *ctx,
    char *outfile,
    long *nlength,
    int a[],
    int nx,
    int ny,
    int scale)
{
    int nel, nx2, ny2, i, j, k, q, vmax[3], nsign, bits_to_go;
    unsigned char nbitplanes[3];
    unsigned char *signbits;
    int stat;

    ctx->noutchar = 0;
    nel = nx * ny;

    qwrite(ctx, outfile, code_magic, sizeof(code_magic));
    writeint(ctx, outfile, nx);
    writeint(ctx, outfile, ny);
    writeint(ctx, outfile, scale);
    writelonglong(ctx, outfile, (long long)a[0]);

    a[0] = 0;

    signbits = (unsigned char *)calloc(1, (nel + 7) / 8);
    if (signbits == (unsigned char *)NULL) {
        return HCOMP_ERROR_MEMORY;
    }
    nsign = 0;
    bits_to_go = 8;
    for (i = 0; i < nel; i++) {
        if (a[i] > 0) {
            signbits[nsign] <<= 1;
            bits_to_go -= 1;
        }
        else if (a[i] < 0) {
            signbits[nsign] <<= 1;
            signbits[nsign] |= 1;
            bits_to_go -= 1;
            a[i] = -a[i];
        }
        if (bits_to_go == 0) {
            bits_to_go = 8;
            nsign += 1;
        }
    }
    if (bits_to_go != 8) {
        signbits[nsign] <<= bits_to_go;
        nsign += 1;
    }

    for (q = 0; q < 3; q++) {
        vmax[q] = 0;
    }
    nx2 = (nx + 1) / 2;
    ny2 = (ny + 1) / 2;
    j = 0;
    k = 0;
    for (i = 0; i < nel; i++) {
        q = (j >= ny2) + (k >= nx2);
        if (vmax[q] < a[i]) vmax[q] = a[i];
        if (++j >= ny) {
            j = 0;
            k += 1;
        }
    }

    for (q = 0; q < 3; q++) {
        for (
            nbitplanes[q] = 0; vmax[q] > 0;
            vmax[q] = vmax[q] >> 1, nbitplanes[q]++)
            ;
    }

    if (0 == qwrite(ctx, outfile, (char *)nbitplanes, sizeof(nbitplanes))) {
        *nlength = ctx->noutchar;
        free(signbits);
        return HCOMP_ERROR_OVERFLOW;
    }

    stat = doencode(ctx, outfile, a, nx, ny, nbitplanes);
    if (stat) {
        free(signbits);
        *nlength = ctx->noutchar;
        return stat;
    }

    if (nsign > 0) {
        if (0 == qwrite(ctx, outfile, (char *)signbits, nsign)) {
            free(signbits);
            *nlength = ctx->noutchar;
            return HCOMP_ERROR_OVERFLOW;
        }
    }

    free(signbits);
    *nlength = ctx->noutchar;
    if (ctx->noutchar >= ctx->noutmax) {
        return HCOMP_ERROR_OVERFLOW;
    }

    return stat;
}

static int
encode64(
    hcomp_cctx *ctx,
    char *outfile,
    long *nlength,
    long long a[],
    int nx,
    int ny,
    int scale)
{
    int nel, nx2, ny2, i, j, k, q, nsign, bits_to_go;
    long long vmax[3];
    unsigned char nbitplanes[3];
    unsigned char *signbits;
    int stat;

    ctx->noutchar = 0;
    nel = nx * ny;

    qwrite(ctx, outfile, code_magic, sizeof(code_magic));
    writeint(ctx, outfile, nx);
    writeint(ctx, outfile, ny);
    writeint(ctx, outfile, scale);
    writelonglong(ctx, outfile, a[0]);

    a[0] = 0;

    signbits = (unsigned char *)calloc(1, (nel + 7) / 8);
    if (signbits == (unsigned char *)NULL) {
        return HCOMP_ERROR_MEMORY;
    }
    nsign = 0;
    bits_to_go = 8;
    for (i = 0; i < nel; i++) {
        if (a[i] > 0) {
            signbits[nsign] <<= 1;
            bits_to_go -= 1;
        }
        else if (a[i] < 0) {
            signbits[nsign] <<= 1;
            signbits[nsign] |= 1;
            bits_to_go -= 1;
            a[i] = -a[i];
        }
        if (bits_to_go == 0) {
            bits_to_go = 8;
            nsign += 1;
        }
    }
    if (bits_to_go != 8) {
        signbits[nsign] <<= bits_to_go;
        nsign += 1;
    }

    for (q = 0; q < 3; q++) {
        vmax[q] = 0;
    }
    nx2 = (nx + 1) / 2;
    ny2 = (ny + 1) / 2;
    j = 0;
    k = 0;
    for (i = 0; i < nel; i++) {
        q = (j >= ny2) + (k >= nx2);
        if (vmax[q] < a[i]) vmax[q] = a[i];
        if (++j >= ny) {
            j = 0;
            k += 1;
        }
    }

    for (q = 0; q < 3; q++) {
        for (
            nbitplanes[q] = 0; vmax[q] > 0;
            vmax[q] = vmax[q] >> 1, nbitplanes[q]++)
            ;
    }

    if (0 == qwrite(ctx, outfile, (char *)nbitplanes, sizeof(nbitplanes))) {
        *nlength = ctx->noutchar;
        free(signbits);
        return HCOMP_ERROR_OVERFLOW;
    }

    stat = doencode64(ctx, outfile, a, nx, ny, nbitplanes);
    if (stat) {
        free(signbits);
        *nlength = ctx->noutchar;
        return stat;
    }

    if (nsign > 0) {
        if (0 == qwrite(ctx, outfile, (char *)signbits, nsign)) {
            free(signbits);
            *nlength = ctx->noutchar;
            return HCOMP_ERROR_OVERFLOW;
        }
    }

    free(signbits);
    *nlength = ctx->noutchar;
    if (ctx->noutchar >= ctx->noutmax) {
        return HCOMP_ERROR_OVERFLOW;
    }

    return stat;
}

/* ======================================================================== */
/*  qwrite.c   Write binary data                                            */
/* ======================================================================== */

static void
writeint(
    hcomp_cctx *ctx,
    char *outfile,
    int a)
{
    int i;
    unsigned char b[4];

    for (i = 3; i >= 0; i--) {
        b[i] = a & 0x000000ff;
        a >>= 8;
    }
    for (i = 0; i < 4; i++)
        qwrite(ctx, outfile, (char *)&b[i], 1);
}

static void
writelonglong(
    hcomp_cctx *ctx,
    char *outfile,
    long long a)
{
    int i;
    unsigned char b[8];

    for (i = 7; i >= 0; i--) {
        b[i] = (unsigned char)(a & 0x000000ff);
        a >>= 8;
    }
    for (i = 0; i < 8; i++)
        qwrite(ctx, outfile, (char *)&b[i], 1);
}

static int
qwrite(
    hcomp_cctx *ctx,
    char *file,
    char buffer[],
    int n)
{
    if (ctx->noutchar + n > ctx->noutmax) return 0;
    memcpy(&file[ctx->noutchar], buffer, n);
    ctx->noutchar += n;
    return n;
}

/* ======================================================================== */
/*  doencode.c  Encode 2-D array and write stream                          */
/* ======================================================================== */

static int
doencode(
    hcomp_cctx *ctx,
    char *outfile,
    int a[],
    int nx,
    int ny,
    unsigned char nbitplanes[3])
{
    int nx2, ny2, stat;

    nx2 = (nx + 1) / 2;
    ny2 = (ny + 1) / 2;

    start_outputing_bits(ctx);

    stat = qtree_encode(ctx, outfile, &a[0], ny, nx2, ny2, nbitplanes[0]);
    if (!stat)
        stat = qtree_encode(
            ctx,
            outfile,
            &a[ny2],
            ny,
            nx2,
            ny / 2,
            nbitplanes[1]
        );
    if (!stat)
        stat = qtree_encode(
            ctx,
            outfile,
            &a[ny * nx2],
            ny,
            nx / 2,
            ny2,
            nbitplanes[1]
        );
    if (!stat)
        stat = qtree_encode(
            ctx,
            outfile,
            &a[ny * nx2 + ny2],
            ny,
            nx / 2,
            ny / 2,
            nbitplanes[2]
        );

    output_nybble(ctx, outfile, 0);
    done_outputing_bits(ctx, outfile);

    return stat;
}

static int
doencode64(
    hcomp_cctx *ctx,
    char *outfile,
    long long a[],
    int nx,
    int ny,
    unsigned char nbitplanes[3])
{
    int nx2, ny2, stat;

    nx2 = (nx + 1) / 2;
    ny2 = (ny + 1) / 2;

    start_outputing_bits(ctx);

    stat = qtree_encode64(ctx, outfile, &a[0], ny, nx2, ny2, nbitplanes[0]);
    if (!stat)
        stat = qtree_encode64(
            ctx,
            outfile,
            &a[ny2],
            ny,
            nx2,
            ny / 2,
            nbitplanes[1]
        );
    if (!stat)
        stat = qtree_encode64(
            ctx,
            outfile,
            &a[ny * nx2],
            ny,
            nx / 2,
            ny2,
            nbitplanes[1]
        );
    if (!stat)
        stat = qtree_encode64(
            ctx,
            outfile,
            &a[ny * nx2 + ny2],
            ny,
            nx / 2,
            ny / 2,
            nbitplanes[2]
        );

    output_nybble(ctx, outfile, 0);
    done_outputing_bits(ctx, outfile);

    return stat;
}

/* ======================================================================== */
/*  bit_output.c   Bit output routines                                      */
/* ======================================================================== */

static void
start_outputing_bits(
    hcomp_cctx *ctx)
{
    ctx->buffer2 = 0;
    ctx->bits_to_go2 = 8;
    ctx->bitcount = 0;
}

static void
output_nbits(
    hcomp_cctx *ctx,
    char *outfile,
    int bits,
    int n)
{
    static int mask[9] = { 0, 1, 3, 7, 15, 31, 63, 127, 255 };

    ctx->buffer2 <<= n;
    ctx->buffer2 |= (bits & (*(mask + n)));
    ctx->bits_to_go2 -= n;
    if (ctx->bits_to_go2 <= 0) {
        outfile[ctx->noutchar] = ((ctx->buffer2 >>
            (-ctx->bits_to_go2)) & 0xff);
        if (ctx->noutchar < ctx->noutmax) ctx->noutchar++;
        ctx->bits_to_go2 += 8;
    }
    ctx->bitcount += n;
}

static void
output_nybble(
    hcomp_cctx *ctx,
    char *outfile,
    int bits)
{
    ctx->buffer2 = (ctx->buffer2 << 4) | (bits & 15);
    ctx->bits_to_go2 -= 4;
    if (ctx->bits_to_go2 <= 0) {
        outfile[ctx->noutchar] = ((ctx->buffer2 >>
            (-ctx->bits_to_go2)) & 0xff);
        if (ctx->noutchar < ctx->noutmax) ctx->noutchar++;
        ctx->bits_to_go2 += 8;
    }
    ctx->bitcount += 4;
}

static void
output_nnybble(
    hcomp_cctx *ctx,
    char *outfile,
    int n,
    unsigned char array[])
{
    int ii, jj, kk = 0, shift;

    if (n == 1) {
        output_nybble(ctx, outfile, (int)array[0]);
        return;
    }

    if (ctx->bits_to_go2 <= 4) {
        output_nybble(ctx, outfile, array[0]);
        kk++;
        if (n == 2) {
            output_nybble(ctx, outfile, (int)array[1]);
            return;
        }
    }

    shift = 8 - ctx->bits_to_go2;
    jj = (n - kk) / 2;

    if (ctx->bits_to_go2 == 8) {
        ctx->buffer2 = 0;
        for (ii = 0; ii < jj; ii++) {
            if (ctx->noutchar < ctx->noutmax) {
                outfile[ctx->noutchar] =
                    ((array[kk] & 15) << 4) | (array[kk + 1] & 15);
            }
            kk += 2;
            ctx->noutchar++;
        }
    }
    else {
        for (ii = 0; ii < jj; ii++) {
            ctx->buffer2 = (ctx->buffer2 << 8) | ((array[kk] & 15) << 4) |
                (array[kk + 1] & 15);
            kk += 2;
            if (ctx->noutchar < ctx->noutmax) {
                outfile[ctx->noutchar] = ((ctx->buffer2 >> shift) & 0xff);
            }
            ctx->noutchar++;
        }
    }

    ctx->bitcount += (8 * (ii - 1));

    if (kk != n)
        output_nybble(ctx, outfile, (int)array[n - 1]);

    return;
}

static void
done_outputing_bits(
    hcomp_cctx *ctx,
    char *outfile)
{
    if (ctx->bits_to_go2 < 8) {
        outfile[ctx->noutchar] = (ctx->buffer2 << ctx->bits_to_go2);
        if (ctx->noutchar < ctx->noutmax) ctx->noutchar++;
        ctx->bitcount += ctx->bits_to_go2;
    }
}

/* ======================================================================== */
/*  qtree_encode.c  Encode values using binary quadtree coding              */
/* ======================================================================== */

static int code[16] = {
    0x3e, 0x00, 0x01, 0x08, 0x02, 0x09, 0x1a, 0x1b,
    0x03, 0x1c, 0x0a, 0x1d, 0x0b, 0x1e, 0x3f, 0x0c
};
static int ncode[16] = {
    6, 3, 3, 4, 3, 4, 5, 5,
    3, 5, 4, 5, 4, 5, 6, 4
};

static int
qtree_encode(
    hcomp_cctx *ctx,
    char *outfile,
    int a[],
    int n,
    int nqx,
    int nqy,
    int nbitplanes)
{
    int log2n, i, k, bit, b, bmax, nqmax, nqx2, nqy2, nx, ny;
    unsigned char *scratch, *buffer;

    nqmax = (nqx > nqy) ? nqx : nqy;
    log2n = ilog2n(nqmax);

    nqx2 = (nqx + 1) / 2;
    nqy2 = (nqy + 1) / 2;
    bmax = (nqx2 * nqy2 + 1) / 2;

    scratch = (unsigned char *)malloc(2 * bmax);
    if (scratch == (unsigned char *)NULL) {
        return HCOMP_ERROR_MEMORY;
    }
    buffer = (unsigned char *)malloc(bmax);
    if (buffer == (unsigned char *)NULL) {
        free(scratch);
        return HCOMP_ERROR_MEMORY;
    }

    for (bit = nbitplanes - 1; bit >= 0; bit--) {
        b = 0;
        ctx->bitbuffer = 0;
        ctx->bits_to_go3 = 0;

        qtree_onebit(a, n, nqx, nqy, scratch, bit);
        nx = (nqx + 1) >> 1;
        ny = (nqy + 1) >> 1;

        if (bufcopy(ctx, scratch, nx * ny, buffer, &b, bmax)) {
            write_bdirect(ctx, outfile, a, n, nqx, nqy, scratch, bit);
            goto bitplane_done;
        }

        for (k = 1; k < log2n; k++) {
            qtree_reduce(scratch, ny, nx, ny, scratch);
            nx = (nx + 1) >> 1;
            ny = (ny + 1) >> 1;
            if (bufcopy(ctx, scratch, nx * ny, buffer, &b, bmax)) {
                write_bdirect(ctx, outfile, a, n, nqx, nqy, scratch, bit);
                goto bitplane_done;
            }
        }

        output_nybble(ctx, outfile, 0xF);
        if (b == 0) {
            if (ctx->bits_to_go3 > 0) {
                output_nbits(
                    ctx,
                    outfile,
                    ctx->bitbuffer & ((1 << ctx->bits_to_go3) - 1),
                    ctx->bits_to_go3
                );
            }
            else {
                output_huffman(ctx, outfile, 0);
            }
        }
        else {
            if (ctx->bits_to_go3 > 0) {
                output_nbits(
                    ctx,
                    outfile,
                    ctx->bitbuffer & ((1 << ctx->bits_to_go3) - 1),
                    ctx->bits_to_go3
                );
            }
            for (i = b - 1; i >= 0; i--) {
                output_nbits(ctx, outfile, buffer[i], 8);
            }
        }
bitplane_done:;
    }
    free(buffer);
    free(scratch);
    return 0;
}

static int
qtree_encode64(
    hcomp_cctx *ctx,
    char *outfile,
    long long a[],
    int n,
    int nqx,
    int nqy,
    int nbitplanes)
{
    int log2n, i, k, bit, b, nqmax, nqx2, nqy2, nx, ny;
    int bmax;
    unsigned char *scratch, *buffer;

    nqmax = (nqx > nqy) ? nqx : nqy;
    log2n = ilog2n(nqmax);

    nqx2 = (nqx + 1) / 2;
    nqy2 = (nqy + 1) / 2;
    bmax = ((nqx2) * (nqy2) + 1) / 2;

    scratch = (unsigned char *)malloc(2 * bmax);
    if (scratch == (unsigned char *)NULL) {
        return HCOMP_ERROR_MEMORY;
    }
    buffer = (unsigned char *)malloc(bmax);
    if (buffer == (unsigned char *)NULL) {
        free(scratch);
        return HCOMP_ERROR_MEMORY;
    }

    for (bit = nbitplanes - 1; bit >= 0; bit--) {
        b = 0;
        ctx->bitbuffer = 0;
        ctx->bits_to_go3 = 0;

        qtree_onebit64(a, n, nqx, nqy, scratch, bit);
        nx = (nqx + 1) >> 1;
        ny = (nqy + 1) >> 1;

        if (bufcopy(ctx, scratch, nx * ny, buffer, &b, bmax)) {
            write_bdirect64(ctx, outfile, a, n, nqx, nqy, scratch, bit);
            goto bitplane_done;
        }

        for (k = 1; k < log2n; k++) {
            qtree_reduce(scratch, ny, nx, ny, scratch);
            nx = (nx + 1) >> 1;
            ny = (ny + 1) >> 1;
            if (bufcopy(ctx, scratch, nx * ny, buffer, &b, bmax)) {
                write_bdirect64(ctx, outfile, a, n, nqx, nqy, scratch, bit);
                goto bitplane_done;
            }
        }

        output_nybble(ctx, outfile, 0xF);
        if (b == 0) {
            if (ctx->bits_to_go3 > 0) {
                output_nbits(
                    ctx,
                    outfile,
                    ctx->bitbuffer & ((1 << ctx->bits_to_go3) - 1),
                    ctx->bits_to_go3
                );
            }
            else {
                output_huffman(ctx, outfile, 0);
            }
        }
        else {
            if (ctx->bits_to_go3 > 0) {
                output_nbits(
                    ctx,
                    outfile,
                    ctx->bitbuffer & ((1 << ctx->bits_to_go3) - 1),
                    ctx->bits_to_go3
                );
            }
            for (i = b - 1; i >= 0; i--) {
                output_nbits(ctx, outfile, buffer[i], 8);
            }
        }
bitplane_done:;
    }
    free(buffer);
    free(scratch);
    return 0;
}

static int
bufcopy(
    hcomp_cctx *ctx,
    unsigned char a[],
    int n,
    unsigned char buffer[],
    int *b,
    int bmax)
{
    int i;

    for (i = 0; i < n; i++) {
        if (a[i] != 0) {
            ctx->bitbuffer |= code[a[i]] << ctx->bits_to_go3;
            ctx->bits_to_go3 += ncode[a[i]];
            if (ctx->bits_to_go3 >= 8) {
                buffer[*b] = ctx->bitbuffer & 0xFF;
                *b += 1;
                if (*b >= bmax) return 1;
                ctx->bitbuffer >>= 8;
                ctx->bits_to_go3 -= 8;
            }
        }
    }
    return 0;
}

static void
qtree_onebit(
    int a[],
    int n,
    int nx,
    int ny,
    unsigned char b[],
    int bit)
{
    int i, j, k;
    int b0, b1, b2, b3;
    int s10, s00;

    b0 = 1 << bit;
    b1 = b0 << 1;
    b2 = b0 << 2;
    b3 = b0 << 3;
    k = 0;
    for (i = 0; i < nx - 1; i += 2) {
        s00 = n * i;
        s10 = s00 + n;
        for (j = 0; j < ny - 1; j += 2) {
            b[k] = (((a[s10 + 1] & b0) |
                ((a[s10] << 1) & b1) |
                ((a[s00 + 1] << 2) & b2) |
                ((a[s00] << 3) & b3)) >>
                bit);
            k += 1;
            s00 += 2;
            s10 += 2;
        }
        if (j < ny) {
            b[k] = ((((a[s10] << 1) & b1) |
                ((a[s00] << 3) & b3)) >>
                bit);
            k += 1;
        }
    }
    if (i < nx) {
        s00 = n * i;
        for (j = 0; j < ny - 1; j += 2) {
            b[k] = ((((a[s00 + 1] << 2) & b2) |
                ((a[s00] << 3) & b3)) >>
                bit);
            k += 1;
            s00 += 2;
        }
        if (j < ny) {
            b[k] = (((a[s00] << 3) & b3) >> bit);
            k += 1;
        }
    }
}

static void
qtree_onebit64(
    long long a[],
    int n,
    int nx,
    int ny,
    unsigned char b[],
    int bit)
{
    int i, j, k;
    long long b0, b1, b2, b3;
    int s10, s00;

    b0 = ((long long)1) << bit;
    b1 = b0 << 1;
    b2 = b0 << 2;
    b3 = b0 << 3;
    k = 0;
    for (i = 0; i < nx - 1; i += 2) {
        s00 = n * i;
        s10 = s00 + n;
        for (j = 0; j < ny - 1; j += 2) {
            b[k] = (unsigned char)(((a[s10 + 1] & b0) |
                ((a[s10] << 1) & b1) |
                ((a[s00 + 1] << 2) & b2) |
                ((a[s00] << 3) & b3)) >>
                bit);
            k += 1;
            s00 += 2;
            s10 += 2;
        }
        if (j < ny) {
            b[k] = (unsigned char)((((a[s10] << 1) & b1) |
                ((a[s00] << 3) & b3)) >>
                bit);
            k += 1;
        }
    }
    if (i < nx) {
        s00 = n * i;
        for (j = 0; j < ny - 1; j += 2) {
            b[k] = (unsigned char)((((a[s00 + 1] << 2) & b2) |
                ((a[s00] << 3) & b3)) >>
                bit);
            k += 1;
            s00 += 2;
        }
        if (j < ny) {
            b[k] = (unsigned char)(((a[s00] << 3) & b3) >> bit);
            k += 1;
        }
    }
}

static void
qtree_reduce(
    unsigned char a[],
    int n,
    int nx,
    int ny,
    unsigned char b[])
{
    int i, j, k;
    int s10, s00;

    k = 0;
    for (i = 0; i < nx - 1; i += 2) {
        s00 = n * i;
        s10 = s00 + n;
        for (j = 0; j < ny - 1; j += 2) {
            b[k] = (a[s10 + 1] != 0) |
                ((a[s10] != 0) << 1) |
                ((a[s00 + 1] != 0) << 2) |
                ((a[s00] != 0) << 3);
            k += 1;
            s00 += 2;
            s10 += 2;
        }
        if (j < ny) {
            b[k] = ((a[s10] != 0) << 1) |
                ((a[s00] != 0) << 3);
            k += 1;
        }
    }
    if (i < nx) {
        s00 = n * i;
        for (j = 0; j < ny - 1; j += 2) {
            b[k] = ((a[s00 + 1] != 0) << 2) |
                ((a[s00] != 0) << 3);
            k += 1;
            s00 += 2;
        }
        if (j < ny) {
            b[k] = ((a[s00] != 0) << 3);
            k += 1;
        }
    }
}

static void
write_bdirect(
    hcomp_cctx *ctx,
    char *outfile,
    int a[],
    int n,
    int nqx,
    int nqy,
    unsigned char scratch[],
    int bit)
{
    output_nybble(ctx, outfile, 0x0);
    qtree_onebit(a, n, nqx, nqy, scratch, bit);
    output_nnybble(ctx, outfile, ((nqx + 1) / 2) * ((nqy + 1) / 2), scratch);
}

static void
write_bdirect64(
    hcomp_cctx *ctx,
    char *outfile,
    long long a[],
    int n,
    int nqx,
    int nqy,
    unsigned char scratch[],
    int bit)
{
    output_nybble(ctx, outfile, 0x0);
    qtree_onebit64(a, n, nqx, nqy, scratch, bit);
    output_nnybble(ctx, outfile, ((nqx + 1) / 2) * ((nqy + 1) / 2), scratch);
}

/* ======================================================================== */
/*  DECOMPRESS routines                                                     */
/* ======================================================================== */

/* ======================================================================== */
/*  decode.c    read codes from infile and construct array                  */
/* ======================================================================== */

static int
decode(
    hcomp_dctx *ctx,
    unsigned char *infile,
    int *a,
    int na,
    int *nx,
    int *ny,
    int *scale)
{
    long long sumall;
    int stat;
    unsigned char nbitplanes[3];
    char tmagic[2];

    qread(ctx, infile, tmagic, sizeof(tmagic));
    if (memcmp(tmagic, code_magic, sizeof(code_magic)) != 0) {
        return HCOMP_ERROR_FORMAT;
    }
    *nx = readint(ctx, infile);
    *ny = readint(ctx, infile);
    *scale = readint(ctx, infile);

    if ((*nx) > INT_MAX / (*ny)) {
        return HCOMP_ERROR_OVERFLOW;
    }
    if ((*nx) * (*ny) > na) {
        return HCOMP_ERROR_FORMAT;
    }

    sumall = readlonglong(ctx, infile);

    qread(ctx, infile, (char *)nbitplanes, sizeof(nbitplanes));

    stat = dodecode(ctx, infile, a, *nx, *ny, nbitplanes);

    a[0] = (int)sumall;
    return stat;
}

static int
decode64(
    hcomp_dctx *ctx,
    unsigned char *infile,
    long long *a,
    int na,
    int *nx,
    int *ny,
    int *scale)
{
    int stat;
    long long sumall;
    unsigned char nbitplanes[3];
    char tmagic[2];

    qread(ctx, infile, tmagic, sizeof(tmagic));
    if (memcmp(tmagic, code_magic, sizeof(code_magic)) != 0) {
        return HCOMP_ERROR_FORMAT;
    }
    *nx = readint(ctx, infile);
    *ny = readint(ctx, infile);
    *scale = readint(ctx, infile);

    if ((*nx) > INT_MAX / (*ny)) {
        return HCOMP_ERROR_OVERFLOW;
    }
    if ((*nx) * (*ny) > na) {
        return HCOMP_ERROR_FORMAT;
    }

    sumall = readlonglong(ctx, infile);

    qread(ctx, infile, (char *)nbitplanes, sizeof(nbitplanes));

    stat = dodecode64(ctx, infile, a, *nx, *ny, nbitplanes);

    a[0] = sumall;
    return stat;
}

/* ======================================================================== */
/*  hinv.c   Inverse H-transform of NX x NY integer image                  */
/* ======================================================================== */

static int
hinv(
    int a[],
    int nx,
    int ny,
    int smooth,
    int scale)
{
    int nmax, log2n, i, j, k;
    int nxtop, nytop, nxf, nyf, c;
    int oddx, oddy;
    int shift, bit0, bit1, bit2, mask0, mask1, mask2,
        prnd0, prnd1, prnd2, nrnd0, nrnd1, nrnd2, lowbit0, lowbit1;
    int h0, hx, hy, hc;
    int s10, s00;
    int *tmp;

    nmax = (nx > ny) ? nx : ny;
    log2n = ilog2n(nmax);

    tmp = (int *)malloc(((nmax + 1) / 2) * sizeof(int));
    if (tmp == (int *)NULL) {
        return HCOMP_ERROR_MEMORY;
    }

    shift = 1;
    bit0 = 1 << (log2n - 1);
    bit1 = bit0 << 1;
    bit2 = bit0 << 2;
    mask0 = -bit0;
    mask1 = mask0 << 1;
    mask2 = mask0 << 2;
    prnd0 = bit0 >> 1;
    prnd1 = bit1 >> 1;
    prnd2 = bit2 >> 1;
    nrnd0 = prnd0 - 1;
    nrnd1 = prnd1 - 1;
    nrnd2 = prnd2 - 1;

    a[0] = (a[0] + ((a[0] >= 0) ? prnd2 : nrnd2)) & mask2;

    nxtop = 1;
    nytop = 1;
    nxf = nx;
    nyf = ny;
    c = 1 << log2n;
    for (k = log2n - 1; k >= 0; k--) {
        c = c >> 1;
        nxtop = nxtop << 1;
        nytop = nytop << 1;
        if (nxf <= c) {
            nxtop -= 1;
        }
        else {
            nxf -= c;
        }
        if (nyf <= c) {
            nytop -= 1;
        }
        else {
            nyf -= c;
        }

        if (k == 0) {
            nrnd0 = 0;
            shift = 2;
        }

        for (i = 0; i < nxtop; i++) {
            unshuffle(&a[ny * i], nytop, 1, tmp);
        }
        for (j = 0; j < nytop; j++) {
            unshuffle(&a[j], nxtop, ny, tmp);
        }

        if (smooth) hsmooth(a, nxtop, nytop, ny, scale);

        oddx = nxtop % 2;
        oddy = nytop % 2;
        for (i = 0; i < nxtop - oddx; i += 2) {
            s00 = ny * i;
            s10 = s00 + ny;
            for (j = 0; j < nytop - oddy; j += 2) {
                h0 = a[s00];
                hx = a[s10];
                hy = a[s00 + 1];
                hc = a[s10 + 1];

                hx = (hx + ((hx >= 0) ? prnd1 : nrnd1)) & mask1;
                hy = (hy + ((hy >= 0) ? prnd1 : nrnd1)) & mask1;
                hc = (hc + ((hc >= 0) ? prnd0 : nrnd0)) & mask0;

                lowbit0 = hc & bit0;
                hx = (hx >= 0) ? (hx - lowbit0) : (hx + lowbit0);
                hy = (hy >= 0) ? (hy - lowbit0) : (hy + lowbit0);

                lowbit1 = (hc ^ hx ^ hy) & bit1;
                h0 = (h0 >= 0)
                         ? (h0 + lowbit0 - lowbit1)
                         : (h0 + ((lowbit0 == 0)
                                      ? lowbit1
                                      : (lowbit0 - lowbit1)));

                a[s10 + 1] = (h0 + hx + hy + hc) >> shift;
                a[s10] = (h0 + hx - hy - hc) >> shift;
                a[s00 + 1] = (h0 - hx + hy - hc) >> shift;
                a[s00] = (h0 - hx - hy + hc) >> shift;
                s00 += 2;
                s10 += 2;
            }
            if (oddy) {
                h0 = a[s00];
                hx = a[s10];
                hx = ((hx >= 0) ? (hx + prnd1) : (hx + nrnd1)) & mask1;
                lowbit1 = hx & bit1;
                h0 = (h0 >= 0) ? (h0 - lowbit1) : (h0 + lowbit1);
                a[s10] = (h0 + hx) >> shift;
                a[s00] = (h0 - hx) >> shift;
            }
        }
        if (oddx) {
            s00 = ny * i;
            for (j = 0; j < nytop - oddy; j += 2) {
                h0 = a[s00];
                hy = a[s00 + 1];
                hy = ((hy >= 0) ? (hy + prnd1) : (hy + nrnd1)) & mask1;
                lowbit1 = hy & bit1;
                h0 = (h0 >= 0) ? (h0 - lowbit1) : (h0 + lowbit1);
                a[s00 + 1] = (h0 + hy) >> shift;
                a[s00] = (h0 - hy) >> shift;
                s00 += 2;
            }
            if (oddy) {
                h0 = a[s00];
                a[s00] = h0 >> shift;
            }
        }

        bit2 = bit1;
        bit1 = bit0;
        bit0 = bit0 >> 1;
        mask1 = mask0;
        mask0 = mask0 >> 1;
        prnd1 = prnd0;
        prnd0 = prnd0 >> 1;
        nrnd1 = nrnd0;
        nrnd0 = prnd0 - 1;
    }
    free(tmp);
    return 0;
}

static int
hinv64(
    long long a[],
    int nx,
    int ny,
    int smooth,
    int scale)
{
    int nmax, log2n, i, j, k;
    int nxtop, nytop, nxf, nyf, c;
    int oddx, oddy;
    int shift;
    long long mask0, mask1, mask2, prnd0, prnd1, prnd2, bit0, bit1, bit2;
    long long nrnd0, nrnd1, nrnd2, lowbit0, lowbit1;
    long long h0, hx, hy, hc;
    int s10, s00;
    long long *tmp;

    nmax = (nx > ny) ? nx : ny;
    log2n = ilog2n(nmax);

    tmp = (long long *)malloc(((nmax + 1) / 2) * sizeof(long long));
    if (tmp == (long long *)NULL) {
        return HCOMP_ERROR_MEMORY;
    }

    shift = 1;
    bit0 = ((long long)1) << (log2n - 1);
    bit1 = bit0 << 1;
    bit2 = bit0 << 2;
    mask0 = -bit0;
    mask1 = mask0 << 1;
    mask2 = mask0 << 2;
    prnd0 = bit0 >> 1;
    prnd1 = bit1 >> 1;
    prnd2 = bit2 >> 1;
    nrnd0 = prnd0 - 1;
    nrnd1 = prnd1 - 1;
    nrnd2 = prnd2 - 1;

    a[0] = (a[0] + ((a[0] >= 0) ? prnd2 : nrnd2)) & mask2;

    nxtop = 1;
    nytop = 1;
    nxf = nx;
    nyf = ny;
    c = 1 << log2n;
    for (k = log2n - 1; k >= 0; k--) {
        c = c >> 1;
        nxtop = nxtop << 1;
        nytop = nytop << 1;
        if (nxf <= c) {
            nxtop -= 1;
        }
        else {
            nxf -= c;
        }
        if (nyf <= c) {
            nytop -= 1;
        }
        else {
            nyf -= c;
        }

        if (k == 0) {
            nrnd0 = 0;
            shift = 2;
        }

        for (i = 0; i < nxtop; i++) {
            unshuffle64(&a[ny * i], nytop, 1, tmp);
        }
        for (j = 0; j < nytop; j++) {
            unshuffle64(&a[j], nxtop, ny, tmp);
        }

        if (smooth) hsmooth64(a, nxtop, nytop, ny, scale);

        oddx = nxtop % 2;
        oddy = nytop % 2;
        for (i = 0; i < nxtop - oddx; i += 2) {
            s00 = ny * i;
            s10 = s00 + ny;
            for (j = 0; j < nytop - oddy; j += 2) {
                h0 = a[s00];
                hx = a[s10];
                hy = a[s00 + 1];
                hc = a[s10 + 1];

                hx = (hx + ((hx >= 0) ? prnd1 : nrnd1)) & mask1;
                hy = (hy + ((hy >= 0) ? prnd1 : nrnd1)) & mask1;
                hc = (hc + ((hc >= 0) ? prnd0 : nrnd0)) & mask0;

                lowbit0 = hc & bit0;
                hx = (hx >= 0) ? (hx - lowbit0) : (hx + lowbit0);
                hy = (hy >= 0) ? (hy - lowbit0) : (hy + lowbit0);

                lowbit1 = (hc ^ hx ^ hy) & bit1;
                h0 = (h0 >= 0)
                         ? (h0 + lowbit0 - lowbit1)
                         : (h0 + ((lowbit0 == 0)
                                      ? lowbit1
                                      : (lowbit0 - lowbit1)));

                a[s10 + 1] = (h0 + hx + hy + hc) >> shift;
                a[s10] = (h0 + hx - hy - hc) >> shift;
                a[s00 + 1] = (h0 - hx + hy - hc) >> shift;
                a[s00] = (h0 - hx - hy + hc) >> shift;
                s00 += 2;
                s10 += 2;
            }
            if (oddy) {
                h0 = a[s00];
                hx = a[s10];
                hx = ((hx >= 0) ? (hx + prnd1) : (hx + nrnd1)) & mask1;
                lowbit1 = hx & bit1;
                h0 = (h0 >= 0) ? (h0 - lowbit1) : (h0 + lowbit1);
                a[s10] = (h0 + hx) >> shift;
                a[s00] = (h0 - hx) >> shift;
            }
        }
        if (oddx) {
            s00 = ny * i;
            for (j = 0; j < nytop - oddy; j += 2) {
                h0 = a[s00];
                hy = a[s00 + 1];
                hy = ((hy >= 0) ? (hy + prnd1) : (hy + nrnd1)) & mask1;
                lowbit1 = hy & bit1;
                h0 = (h0 >= 0) ? (h0 - lowbit1) : (h0 + lowbit1);
                a[s00 + 1] = (h0 + hy) >> shift;
                a[s00] = (h0 - hy) >> shift;
                s00 += 2;
            }
            if (oddy) {
                h0 = a[s00];
                a[s00] = h0 >> shift;
            }
        }

        bit2 = bit1;
        bit1 = bit0;
        bit0 = bit0 >> 1;
        mask1 = mask0;
        mask0 = mask0 >> 1;
        prnd1 = prnd0;
        prnd0 = prnd0 >> 1;
        nrnd1 = nrnd0;
        nrnd0 = prnd0 - 1;
    }
    free(tmp);
    return 0;
}

/* ======================================================================== */
/*  unshuffle                                                               */
/* ======================================================================== */

static void
unshuffle(
    int a[],
    int n,
    int n2,
    int tmp[])
{
    int i;
    int nhalf;
    int *p1, *p2, *pt;

    nhalf = (n + 1) >> 1;
    pt = tmp;
    p1 = &a[n2 * nhalf];
    for (i = nhalf; i < n; i++) {
        *pt = *p1;
        p1 += n2;
        pt += 1;
    }
    p2 = &a[n2 * (nhalf - 1)];
    p1 = &a[(n2 * (nhalf - 1)) << 1];
    for (i = nhalf - 1; i >= 0; i--) {
        *p1 = *p2;
        p2 -= n2;
        p1 -= (n2 + n2);
    }
    pt = tmp;
    p1 = &a[n2];
    for (i = 1; i < n; i += 2) {
        *p1 = *pt;
        p1 += (n2 + n2);
        pt += 1;
    }
}

static void
unshuffle64(
    long long a[],
    int n,
    int n2,
    long long tmp[])
{
    int i;
    int nhalf;
    long long *p1, *p2, *pt;

    nhalf = (n + 1) >> 1;
    pt = tmp;
    p1 = &a[n2 * nhalf];
    for (i = nhalf; i < n; i++) {
        *pt = *p1;
        p1 += n2;
        pt += 1;
    }
    p2 = &a[n2 * (nhalf - 1)];
    p1 = &a[(n2 * (nhalf - 1)) << 1];
    for (i = nhalf - 1; i >= 0; i--) {
        *p1 = *p2;
        p2 -= n2;
        p1 -= (n2 + n2);
    }
    pt = tmp;
    p1 = &a[n2];
    for (i = 1; i < n; i += 2) {
        *p1 = *pt;
        p1 += (n2 + n2);
        pt += 1;
    }
}

/* ======================================================================== */
/*  hsmooth.c   Smooth H-transform image by adjusting coefficients          */
/* ======================================================================== */

static void
hsmooth(
    int a[],
    int nxtop,
    int nytop,
    int ny,
    int scale)
{
    int i, j;
    int ny2, s10, s00, diff, dmax, dmin, s, smax;
    int hm, h0, hp, hmm, hpm, hmp, hpp, hx2, hy2;
    int m1, m2;

    smax = (scale >> 1);
    if (smax <= 0) return;
    ny2 = ny << 1;

    /* Adjust x difference hx */
    for (i = 2; i < nxtop - 2; i += 2) {
        s00 = ny * i;
        s10 = s00 + ny;
        for (j = 0; j < nytop; j += 2) {
            hm = a[s00 - ny2];
            h0 = a[s00];
            hp = a[s00 + ny2];
            diff = hp - hm;
            dmax = max(min((hp - h0), (h0 - hm)), 0) << 2;
            dmin = min(max((hp - h0), (h0 - hm)), 0) << 2;
            if (dmin < dmax) {
                diff = max(min(diff, dmax), dmin);
                s = diff - (a[s10] << 3);
                s = (s >= 0) ? (s >> 3) : ((s + 7) >> 3);
                s = max(min(s, smax), -smax);
                a[s10] = a[s10] + s;
            }
            s00 += 2;
            s10 += 2;
        }
    }

    /* Adjust y difference hy */
    for (i = 0; i < nxtop; i += 2) {
        s00 = ny * i + 2;
        s10 = s00 + ny;
        for (j = 2; j < nytop - 2; j += 2) {
            hm = a[s00 - 2];
            h0 = a[s00];
            hp = a[s00 + 2];
            diff = hp - hm;
            dmax = max(min((hp - h0), (h0 - hm)), 0) << 2;
            dmin = min(max((hp - h0), (h0 - hm)), 0) << 2;
            if (dmin < dmax) {
                diff = max(min(diff, dmax), dmin);
                s = diff - (a[s00 + 1] << 3);
                s = (s >= 0) ? (s >> 3) : ((s + 7) >> 3);
                s = max(min(s, smax), -smax);
                a[s00 + 1] = a[s00 + 1] + s;
            }
            s00 += 2;
            s10 += 2;
        }
    }

    /* Adjust curvature difference hc */
    for (i = 2; i < nxtop - 2; i += 2) {
        s00 = ny * i + 2;
        s10 = s00 + ny;
        for (j = 2; j < nytop - 2; j += 2) {
            hmm = a[s00 - ny2 - 2];
            hpm = a[s00 + ny2 - 2];
            hmp = a[s00 - ny2 + 2];
            hpp = a[s00 + ny2 + 2];
            h0 = a[s00];
            diff = hpp + hmm - hmp - hpm;
            hx2 = a[s10] << 1;
            hy2 = a[s00 + 1] << 1;
            m1 = min(
                max(hpp - h0, 0) - hx2 - hy2,
                max(h0 - hpm, 0) + hx2 - hy2
            );
            m2 = min(
                max(h0 - hmp, 0) - hx2 + hy2,
                max(hmm - h0, 0) + hx2 + hy2
            );
            dmax = min(m1, m2) << 4;
            m1 = max(
                min(hpp - h0, 0) - hx2 - hy2,
                min(h0 - hpm, 0) + hx2 - hy2
            );
            m2 = max(
                min(h0 - hmp, 0) - hx2 + hy2,
                min(hmm - h0, 0) + hx2 + hy2
            );
            dmin = max(m1, m2) << 4;
            if (dmin < dmax) {
                diff = max(min(diff, dmax), dmin);
                s = diff - (a[s10 + 1] << 6);
                s = (s >= 0) ? (s >> 6) : ((s + 63) >> 6);
                s = max(min(s, smax), -smax);
                a[s10 + 1] = a[s10 + 1] + s;
            }
            s00 += 2;
            s10 += 2;
        }
    }
}

static void
hsmooth64(
    long long a[],
    int nxtop,
    int nytop,
    int ny,
    int scale)
{
    int i, j;
    int ny2, s10, s00;
    long long hm, h0, hp, hmm, hpm, hmp, hpp, hx2, hy2;
    long long diff, dmax, dmin, s, smax, m1, m2;

    smax = (scale >> 1);
    if (smax <= 0) return;
    ny2 = ny << 1;

    for (i = 2; i < nxtop - 2; i += 2) {
        s00 = ny * i;
        s10 = s00 + ny;
        for (j = 0; j < nytop; j += 2) {
            hm = a[s00 - ny2];
            h0 = a[s00];
            hp = a[s00 + ny2];
            diff = hp - hm;
            dmax = max(min((hp - h0), (h0 - hm)), 0) << 2;
            dmin = min(max((hp - h0), (h0 - hm)), 0) << 2;
            if (dmin < dmax) {
                diff = max(min(diff, dmax), dmin);
                s = diff - (a[s10] << 3);
                s = (s >= 0) ? (s >> 3) : ((s + 7) >> 3);
                s = max(min(s, smax), -smax);
                a[s10] = a[s10] + s;
            }
            s00 += 2;
            s10 += 2;
        }
    }

    for (i = 0; i < nxtop; i += 2) {
        s00 = ny * i + 2;
        s10 = s00 + ny;
        for (j = 2; j < nytop - 2; j += 2) {
            hm = a[s00 - 2];
            h0 = a[s00];
            hp = a[s00 + 2];
            diff = hp - hm;
            dmax = max(min((hp - h0), (h0 - hm)), 0) << 2;
            dmin = min(max((hp - h0), (h0 - hm)), 0) << 2;
            if (dmin < dmax) {
                diff = max(min(diff, dmax), dmin);
                s = diff - (a[s00 + 1] << 3);
                s = (s >= 0) ? (s >> 3) : ((s + 7) >> 3);
                s = max(min(s, smax), -smax);
                a[s00 + 1] = a[s00 + 1] + s;
            }
            s00 += 2;
            s10 += 2;
        }
    }

    for (i = 2; i < nxtop - 2; i += 2) {
        s00 = ny * i + 2;
        s10 = s00 + ny;
        for (j = 2; j < nytop - 2; j += 2) {
            hmm = a[s00 - ny2 - 2];
            hpm = a[s00 + ny2 - 2];
            hmp = a[s00 - ny2 + 2];
            hpp = a[s00 + ny2 + 2];
            h0 = a[s00];
            diff = hpp + hmm - hmp - hpm;
            hx2 = a[s10] << 1;
            hy2 = a[s00 + 1] << 1;
            m1 = min(
                max(hpp - h0, 0) - hx2 - hy2,
                max(h0 - hpm, 0) + hx2 - hy2
            );
            m2 = min(
                max(h0 - hmp, 0) - hx2 + hy2,
                max(hmm - h0, 0) + hx2 + hy2
            );
            dmax = min(m1, m2) << 4;
            m1 = max(
                min(hpp - h0, 0) - hx2 - hy2,
                min(h0 - hpm, 0) + hx2 - hy2
            );
            m2 = max(
                min(h0 - hmp, 0) - hx2 + hy2,
                min(hmm - h0, 0) + hx2 + hy2
            );
            dmin = max(m1, m2) << 4;
            if (dmin < dmax) {
                diff = max(min(diff, dmax), dmin);
                s = diff - (a[s10 + 1] << 6);
                s = (s >= 0) ? (s >> 6) : ((s + 63) >> 6);
                s = max(min(s, smax), -smax);
                a[s10 + 1] = a[s10 + 1] + s;
            }
            s00 += 2;
            s10 += 2;
        }
    }
}

/* ======================================================================== */
/*  undigitize.c    undigitize H-transform                                  */
/* ======================================================================== */

static void
undigitize(
    int a[],
    int nx,
    int ny,
    int scale)
{
    int *p;

    if (scale <= 1) return;
    for (p = a; p <= &a[nx * ny - 1]; p++)
        *p = (*p) * scale;
}

static void
undigitize64(
    long long a[],
    int nx,
    int ny,
    int scale)
{
    long long *p, scale64;

    if (scale <= 1) return;
    scale64 = (long long)scale;

    for (p = a; p <= &a[nx * ny - 1]; p++)
        *p = (*p) * scale64;
}

/* ======================================================================== */
/*  dodecode.c  Decode stream of characters and return array                */
/* ======================================================================== */

static int
dodecode(
    hcomp_dctx *ctx,
    unsigned char *infile,
    int a[],
    int nx,
    int ny,
    unsigned char nbitplanes[3])
{
    int i, nel, nx2, ny2, stat;

    nel = nx * ny;
    nx2 = (nx + 1) / 2;
    ny2 = (ny + 1) / 2;

    for (i = 0; i < nel; i++)
        a[i] = 0;

    start_inputing_bits(ctx);

    stat = qtree_decode(ctx, infile, &a[0], ny, nx2, ny2, nbitplanes[0]);
    if (stat) return stat;

    stat = qtree_decode(ctx, infile, &a[ny2], ny, nx2, ny / 2, nbitplanes[1]);
    if (stat) return stat;

    stat = qtree_decode(
        ctx,
        infile,
        &a[ny * nx2],
        ny,
        nx / 2,
        ny2,
        nbitplanes[1]
    );
    if (stat) return stat;

    stat = qtree_decode(
        ctx,
        infile,
        &a[ny * nx2 + ny2],
        ny,
        nx / 2,
        ny / 2,
        nbitplanes[2]
    );
    if (stat) return stat;

    if (input_nybble(ctx, infile) != 0) {
        return HCOMP_ERROR_FORMAT;
    }

    start_inputing_bits(ctx);
    for (i = 0; i < nel; i++) {
        if (a[i]) {
            if (input_bit(ctx, infile)) a[i] = -a[i];
        }
    }
    return 0;
}

static int
dodecode64(
    hcomp_dctx *ctx,
    unsigned char *infile,
    long long a[],
    int nx,
    int ny,
    unsigned char nbitplanes[3])
{
    int i, nel, nx2, ny2, stat;

    nel = nx * ny;
    nx2 = (nx + 1) / 2;
    ny2 = (ny + 1) / 2;

    for (i = 0; i < nel; i++)
        a[i] = 0;

    start_inputing_bits(ctx);

    stat = qtree_decode64(ctx, infile, &a[0], ny, nx2, ny2, nbitplanes[0]);
    if (stat) return stat;

    stat = qtree_decode64(
        ctx,
        infile,
        &a[ny2],
        ny,
        nx2,
        ny / 2,
        nbitplanes[1]
    );
    if (stat) return stat;

    stat = qtree_decode64(
        ctx,
        infile,
        &a[ny * nx2],
        ny,
        nx / 2,
        ny2,
        nbitplanes[1]
    );
    if (stat) return stat;

    stat = qtree_decode64(
        ctx,
        infile,
        &a[ny * nx2 + ny2],
        ny,
        nx / 2,
        ny / 2,
        nbitplanes[2]
    );
    if (stat) return stat;

    if (input_nybble(ctx, infile) != 0) {
        return HCOMP_ERROR_FORMAT;
    }

    start_inputing_bits(ctx);
    for (i = 0; i < nel; i++) {
        if (a[i]) {
            if (input_bit(ctx, infile) != 0) a[i] = -a[i];
        }
    }
    return 0;
}

/* ======================================================================== */
/*  qtree_decode.c  Quadtree decoding                                       */
/* ======================================================================== */

static int
qtree_decode(
    hcomp_dctx *ctx,
    unsigned char *infile,
    int a[],
    int n,
    int nqx,
    int nqy,
    int nbitplanes)
{
    int log2n, k, bit, b, nqmax;
    int nx, ny, nfx, nfy, c;
    int nqx2, nqy2;
    unsigned char *scratch;

    nqmax = (nqx > nqy) ? nqx : nqy;
    log2n = ilog2n(nqmax);

    nqx2 = (nqx + 1) / 2;
    nqy2 = (nqy + 1) / 2;
    scratch = (unsigned char *)malloc((size_t)nqx2 * nqy2);
    if (scratch == (unsigned char *)NULL) {
        return HCOMP_ERROR_MEMORY;
    }

    for (bit = nbitplanes - 1; bit >= 0; bit--) {
        b = input_nybble(ctx, infile);

        if (b == 0) {
            read_bdirect(ctx, infile, a, n, nqx, nqy, scratch, bit);
        }
        else if (b != 0xf) {
            free(scratch);
            return HCOMP_ERROR_FORMAT;
        }
        else {
            scratch[0] = input_huffman(ctx, infile);

            nx = 1;
            ny = 1;
            nfx = nqx;
            nfy = nqy;
            c = 1 << log2n;
            for (k = 1; k < log2n; k++) {
                c = c >> 1;
                nx = nx << 1;
                ny = ny << 1;
                if (nfx <= c) {
                    nx -= 1;
                }
                else {
                    nfx -= c;
                }
                if (nfy <= c) {
                    ny -= 1;
                }
                else {
                    nfy -= c;
                }
                qtree_expand(ctx, infile, scratch, nx, ny, scratch);
            }
            qtree_bitins(scratch, nqx, nqy, a, n, bit);
        }
    }
    free(scratch);
    return 0;
}

static int
qtree_decode64(
    hcomp_dctx *ctx,
    unsigned char *infile,
    long long a[],
    int n,
    int nqx,
    int nqy,
    int nbitplanes)
{
    int log2n, k, bit, b, nqmax;
    int nx, ny, nfx, nfy, c;
    int nqx2, nqy2;
    unsigned char *scratch;

    nqmax = (nqx > nqy) ? nqx : nqy;
    log2n = ilog2n(nqmax);

    nqx2 = (nqx + 1) / 2;
    nqy2 = (nqy + 1) / 2;
    scratch = (unsigned char *)malloc((size_t)nqx2 * nqy2);
    if (scratch == (unsigned char *)NULL) {
        return HCOMP_ERROR_MEMORY;
    }

    for (bit = nbitplanes - 1; bit >= 0; bit--) {
        b = input_nybble(ctx, infile);

        if (b == 0) {
            read_bdirect64(ctx, infile, a, n, nqx, nqy, scratch, bit);
        }
        else if (b != 0xf) {
            free(scratch);
            return HCOMP_ERROR_FORMAT;
        }
        else {
            scratch[0] = input_huffman(ctx, infile);

            nx = 1;
            ny = 1;
            nfx = nqx;
            nfy = nqy;
            c = 1 << log2n;
            for (k = 1; k < log2n; k++) {
                c = c >> 1;
                nx = nx << 1;
                ny = ny << 1;
                if (nfx <= c) {
                    nx -= 1;
                }
                else {
                    nfx -= c;
                }
                if (nfy <= c) {
                    ny -= 1;
                }
                else {
                    nfy -= c;
                }
                qtree_expand(ctx, infile, scratch, nx, ny, scratch);
            }
            qtree_bitins64(scratch, nqx, nqy, a, n, bit);
        }
    }
    free(scratch);
    return 0;
}

static void
qtree_expand(
    hcomp_dctx *ctx,
    unsigned char *infile,
    unsigned char a[],
    int nx,
    int ny,
    unsigned char b[])
{
    int i;

    qtree_copy(a, nx, ny, b, ny);

    for (i = nx * ny - 1; i >= 0; i--) {
        if (b[i]) b[i] = input_huffman(ctx, infile);
    }
}

static void
qtree_copy(
    unsigned char a[],
    int nx,
    int ny,
    unsigned char b[],
    int n)
{
    int i, j, k, nx2, ny2;
    int s00, s10;

    nx2 = (nx + 1) / 2;
    ny2 = (ny + 1) / 2;
    k = ny2 * (nx2 - 1) + ny2 - 1;
    for (i = nx2 - 1; i >= 0; i--) {
        s00 = 2 * (n * i + ny2 - 1);
        for (j = ny2 - 1; j >= 0; j--) {
            b[s00] = a[k];
            k -= 1;
            s00 -= 2;
        }
    }

    for (i = 0; i < nx - 1; i += 2) {
        s00 = n * i;
        s10 = s00 + n;
        for (j = 0; j < ny - 1; j += 2) {
            switch (b[s00])
            {
                case (0):
                    b[s10 + 1] = 0; b[s10] = 0; b[s00 + 1] = 0; b[s00] = 0;
                    break;
                case (1):
                    b[s10 + 1] = 1; b[s10] = 0; b[s00 + 1] = 0; b[s00] = 0;
                    break;
                case (2):
                    b[s10 + 1] = 0; b[s10] = 1; b[s00 + 1] = 0; b[s00] = 0;
                    break;
                case (3):
                    b[s10 + 1] = 1; b[s10] = 1; b[s00 + 1] = 0; b[s00] = 0;
                    break;
                case (4):
                    b[s10 + 1] = 0; b[s10] = 0; b[s00 + 1] = 1; b[s00] = 0;
                    break;
                case (5):
                    b[s10 + 1] = 1; b[s10] = 0; b[s00 + 1] = 1; b[s00] = 0;
                    break;
                case (6):
                    b[s10 + 1] = 0; b[s10] = 1; b[s00 + 1] = 1; b[s00] = 0;
                    break;
                case (7):
                    b[s10 + 1] = 1; b[s10] = 1; b[s00 + 1] = 1; b[s00] = 0;
                    break;
                case (8):
                    b[s10 + 1] = 0; b[s10] = 0; b[s00 + 1] = 0; b[s00] = 1;
                    break;
                case (9):
                    b[s10 + 1] = 1; b[s10] = 0; b[s00 + 1] = 0; b[s00] = 1;
                    break;
                case (10):
                    b[s10 + 1] = 0; b[s10] = 1; b[s00 + 1] = 0; b[s00] = 1;
                    break;
                case (11):
                    b[s10 + 1] = 1; b[s10] = 1; b[s00 + 1] = 0; b[s00] = 1;
                    break;
                case (12):
                    b[s10 + 1] = 0; b[s10] = 0; b[s00 + 1] = 1; b[s00] = 1;
                    break;
                case (13):
                    b[s10 + 1] = 1; b[s10] = 0; b[s00 + 1] = 1; b[s00] = 1;
                    break;
                case (14):
                    b[s10 + 1] = 0; b[s10] = 1; b[s00 + 1] = 1; b[s00] = 1;
                    break;
                case (15):
                    b[s10 + 1] = 1; b[s10] = 1; b[s00 + 1] = 1; b[s00] = 1;
                    break;
            }
            s00 += 2;
            s10 += 2;
        }
        if (j < ny) {
            b[s10] = (b[s00] >> 1) & 1;
            b[s00] = (b[s00] >> 3) & 1;
        }
    }
    if (i < nx) {
        s00 = n * i;
        for (j = 0; j < ny - 1; j += 2) {
            b[s00 + 1] = (b[s00] >> 2) & 1;
            b[s00] = (b[s00] >> 3) & 1;
            s00 += 2;
        }
        if (j < ny) {
            b[s00] = (b[s00] >> 3) & 1;
        }
    }
}

static void
qtree_bitins(
    unsigned char a[],
    int nx,
    int ny,
    int b[],
    int n,
    int bit)
{
    int i, j, k;
    int s00;
    int plane_val;

    plane_val = 1 << bit;

    k = 0;
    for (i = 0; i < nx - 1; i += 2) {
        s00 = n * i;
        for (j = 0; j < ny - 1; j += 2) {
            switch (a[k])
            {
                case (0):  break;
                case (1):  b[s00 + n + 1] |= plane_val; break;
                case (2):  b[s00 + n] |= plane_val; break;
                case (3):  b[s00 + n + 1] |= plane_val;
                    b[s00 + n] |= plane_val; break;
                case (4):  b[s00 + 1] |= plane_val; break;
                case (5):  b[s00 + n + 1] |= plane_val;
                    b[s00 + 1] |= plane_val; break;
                case (6):  b[s00 + n] |= plane_val; b[s00 + 1] |= plane_val;
                    break;
                case (7):  b[s00 + n + 1] |= plane_val;
                    b[s00 + n] |= plane_val; b[s00 + 1] |= plane_val; break;
                case (8):  b[s00] |= plane_val; break;
                case (9):  b[s00 + n + 1] |= plane_val; b[s00] |= plane_val;
                    break;
                case (10): b[s00 + n] |= plane_val; b[s00] |= plane_val; break;
                case (11): b[s00 + n + 1] |= plane_val;
                    b[s00 + n] |= plane_val; b[s00] |= plane_val; break;
                case (12): b[s00 + 1] |= plane_val; b[s00] |= plane_val; break;
                case (13): b[s00 + n + 1] |= plane_val;
                    b[s00 + 1] |= plane_val; b[s00] |= plane_val; break;
                case (14): b[s00 + n] |= plane_val; b[s00 + 1] |= plane_val;
                    b[s00] |= plane_val; break;
                case (15): b[s00 + n + 1] |= plane_val;
                    b[s00 + n] |= plane_val; b[s00 + 1] |= plane_val;
                    b[s00] |= plane_val; break;
            }
            s00 += 2;
            k += 1;
        }
        if (j < ny) {
            switch (a[k])
            {
                case (0):  break;
                case (2):  b[s00 + n] |= plane_val; break;
                case (3):  b[s00 + n] |= plane_val; break;
                case (6):  b[s00 + n] |= plane_val; break;
                case (7):  b[s00 + n] |= plane_val; break;
                case (8):  b[s00] |= plane_val; break;
                case (9):  b[s00] |= plane_val; break;
                case (10): b[s00 + n] |= plane_val; b[s00] |= plane_val; break;
                case (11): b[s00 + n] |= plane_val; b[s00] |= plane_val; break;
                case (12): b[s00] |= plane_val; break;
                case (13): b[s00] |= plane_val; break;
                case (14): b[s00 + n] |= plane_val; b[s00] |= plane_val; break;
                case (15): b[s00 + n] |= plane_val; b[s00] |= plane_val; break;
                default:  break;
            }
            k += 1;
        }
    }
    if (i < nx) {
        s00 = n * i;
        for (j = 0; j < ny - 1; j += 2) {
            switch (a[k])
            {
                case (0):  break;
                case (4):  b[s00 + 1] |= plane_val; break;
                case (5):  b[s00 + 1] |= plane_val; break;
                case (6):  b[s00 + 1] |= plane_val; break;
                case (7):  b[s00 + 1] |= plane_val; break;
                case (8):  b[s00] |= plane_val; break;
                case (9):  b[s00] |= plane_val; break;
                case (10): b[s00] |= plane_val; break;
                case (11): b[s00] |= plane_val; break;
                case (12): b[s00 + 1] |= plane_val; b[s00] |= plane_val; break;
                case (13): b[s00 + 1] |= plane_val; b[s00] |= plane_val; break;
                case (14): b[s00 + 1] |= plane_val; b[s00] |= plane_val; break;
                case (15): b[s00 + 1] |= plane_val; b[s00] |= plane_val; break;
                default:  break;
            }
            s00 += 2;
            k += 1;
        }
        if (j < ny) {
            switch (a[k])
            {
                case (8): case (9): case (10): case (11):
                case (12): case (13): case (14): case (15):
                    b[s00] |= plane_val; break;
                default: break;
            }
            k += 1;
        }
    }
}

static void
qtree_bitins64(
    unsigned char a[],
    int nx,
    int ny,
    long long b[],
    int n,
    int bit)
{
    int i, j, k;
    int s00;
    long long plane_val;

    plane_val = ((long long)1) << bit;

    k = 0;
    for (i = 0; i < nx - 1; i += 2) {
        s00 = n * i;
        for (j = 0; j < ny - 1; j += 2) {
            switch (a[k])
            {
                case (0):  break;
                case (1):  b[s00 + n + 1] |= plane_val; break;
                case (2):  b[s00 + n] |= plane_val; break;
                case (3):  b[s00 + n + 1] |= plane_val;
                    b[s00 + n] |= plane_val; break;
                case (4):  b[s00 + 1] |= plane_val; break;
                case (5):  b[s00 + n + 1] |= plane_val;
                    b[s00 + 1] |= plane_val; break;
                case (6):  b[s00 + n] |= plane_val; b[s00 + 1] |= plane_val;
                    break;
                case (7):  b[s00 + n + 1] |= plane_val;
                    b[s00 + n] |= plane_val; b[s00 + 1] |= plane_val; break;
                case (8):  b[s00] |= plane_val; break;
                case (9):  b[s00 + n + 1] |= plane_val; b[s00] |= plane_val;
                    break;
                case (10): b[s00 + n] |= plane_val; b[s00] |= plane_val; break;
                case (11): b[s00 + n + 1] |= plane_val;
                    b[s00 + n] |= plane_val; b[s00] |= plane_val; break;
                case (12): b[s00 + 1] |= plane_val; b[s00] |= plane_val; break;
                case (13): b[s00 + n + 1] |= plane_val;
                    b[s00 + 1] |= plane_val; b[s00] |= plane_val; break;
                case (14): b[s00 + n] |= plane_val; b[s00 + 1] |= plane_val;
                    b[s00] |= plane_val; break;
                case (15): b[s00 + n + 1] |= plane_val;
                    b[s00 + n] |= plane_val; b[s00 + 1] |= plane_val;
                    b[s00] |= plane_val; break;
            }
            s00 += 2;
            k += 1;
        }
        if (j < ny) {
            switch (a[k])
            {
                case (0):  break;
                case (2):  b[s00 + n] |= plane_val; break;
                case (3):  b[s00 + n] |= plane_val; break;
                case (6):  b[s00 + n] |= plane_val; break;
                case (7):  b[s00 + n] |= plane_val; break;
                case (8):  b[s00] |= plane_val; break;
                case (9):  b[s00] |= plane_val; break;
                case (10): b[s00 + n] |= plane_val; b[s00] |= plane_val; break;
                case (11): b[s00 + n] |= plane_val; b[s00] |= plane_val; break;
                case (12): b[s00] |= plane_val; break;
                case (13): b[s00] |= plane_val; break;
                case (14): b[s00 + n] |= plane_val; b[s00] |= plane_val; break;
                case (15): b[s00 + n] |= plane_val; b[s00] |= plane_val; break;
                default:  break;
            }
            k += 1;
        }
    }
    if (i < nx) {
        s00 = n * i;
        for (j = 0; j < ny - 1; j += 2) {
            switch (a[k])
            {
                case (0):  break;
                case (4):  b[s00 + 1] |= plane_val; break;
                case (5):  b[s00 + 1] |= plane_val; break;
                case (6):  b[s00 + 1] |= plane_val; break;
                case (7):  b[s00 + 1] |= plane_val; break;
                case (8):  b[s00] |= plane_val; break;
                case (9):  b[s00] |= plane_val; break;
                case (10): b[s00] |= plane_val; break;
                case (11): b[s00] |= plane_val; break;
                case (12): b[s00 + 1] |= plane_val; b[s00] |= plane_val; break;
                case (13): b[s00 + 1] |= plane_val; b[s00] |= plane_val; break;
                case (14): b[s00 + 1] |= plane_val; b[s00] |= plane_val; break;
                case (15): b[s00 + 1] |= plane_val; b[s00] |= plane_val; break;
                default:  break;
            }
            s00 += 2;
            k += 1;
        }
        if (j < ny) {
            switch (a[k])
            {
                case (8): case (9): case (10): case (11):
                case (12): case (13): case (14): case (15):
                    b[s00] |= plane_val; break;
                default: break;
            }
            k += 1;
        }
    }
}

static void
read_bdirect(
    hcomp_dctx *ctx,
    unsigned char *infile,
    int a[],
    int n,
    int nqx,
    int nqy,
    unsigned char scratch[],
    int bit)
{
    input_nnybble(ctx, infile, ((nqx + 1) / 2) * ((nqy + 1) / 2), scratch);
    qtree_bitins(scratch, nqx, nqy, a, n, bit);
}

static void
read_bdirect64(
    hcomp_dctx *ctx,
    unsigned char *infile,
    long long a[],
    int n,
    int nqx,
    int nqy,
    unsigned char scratch[],
    int bit)
{
    input_nnybble(ctx, infile, ((nqx + 1) / 2) * ((nqy + 1) / 2), scratch);
    qtree_bitins64(scratch, nqx, nqy, a, n, bit);
}

static int
input_huffman(
    hcomp_dctx *ctx,
    unsigned char *infile)
{
    int c;

    c = input_nbits(ctx, infile, 3);
    if (c < 4) {
        return (1 << c);
    }

    c = input_bit(ctx, infile) | (c << 1);
    if (c < 13) {
        switch (c)
        {
            case 8:  return 3;
            case 9:  return 5;
            case 10: return 10;
            case 11: return 12;
            case 12: return 15;
        }
    }

    c = input_bit(ctx, infile) | (c << 1);
    if (c < 31) {
        switch (c)
        {
            case 26: return 6;
            case 27: return 7;
            case 28: return 9;
            case 29: return 11;
            case 30: return 13;
        }
    }

    c = input_bit(ctx, infile) | (c << 1);
    if (c == 62) {
        return 0;
    }
    else {
        return 14;
    }
}

/* ======================================================================== */
/*  qread.c   Read binary data                                              */
/* ======================================================================== */

static int
readint(
    hcomp_dctx *ctx,
    unsigned char *infile)
{
    int a, i;
    unsigned char b[4];

    for (i = 0; i < 4; i++)
        qread(ctx, infile, (char *)&b[i], 1);
    a = b[0];
    for (i = 1; i < 4; i++)
        a = (a << 8) + b[i];
    return a;
}

static long long
readlonglong(
    hcomp_dctx *ctx,
    unsigned char *infile)
{
    int i;
    long long a;
    unsigned char b[8];

    for (i = 0; i < 8; i++)
        qread(ctx, infile, (char *)&b[i], 1);
    a = b[0];
    for (i = 1; i < 8; i++)
        a = (a << 8) + b[i];
    return a;
}

static void
qread(
    hcomp_dctx *ctx,
    unsigned char *file,
    char buffer[],
    int n)
{
    long avail = ctx->ninmax - ctx->nextchar;
    if (avail < n) {
        if (avail > 0) {
            memcpy(buffer, &file[ctx->nextchar], avail);
            memset(buffer + avail, 0, n - avail);
        }
        else {
            memset(buffer, 0, n);
        }
        ctx->nextchar = ctx->ninmax;
        return;
    }
    memcpy(buffer, &file[ctx->nextchar], n);
    ctx->nextchar += n;
}

/* ======================================================================== */
/*  bit_input.c   Bit input routines                                        */
/* ======================================================================== */

static void
start_inputing_bits(
    hcomp_dctx *ctx)
{
    ctx->bits_to_go_in = 0;
}

static int
input_bit(
    hcomp_dctx *ctx,
    unsigned char *infile)
{
    if (ctx->bits_to_go_in == 0) {
        if (ctx->nextchar < ctx->ninmax) {
            ctx->buffer2_in = infile[ctx->nextchar];
            ctx->nextchar++;
        }
        else {
            ctx->buffer2_in = 0;
        }
        ctx->bits_to_go_in = 8;
    }
    ctx->bits_to_go_in -= 1;
    return ((ctx->buffer2_in >> ctx->bits_to_go_in) & 1);
}

static int
input_nbits(
    hcomp_dctx *ctx,
    unsigned char *infile,
    int n)
{
    static int mask[9] = { 0, 1, 3, 7, 15, 31, 63, 127, 255 };

    if (ctx->bits_to_go_in < n) {
        if (ctx->nextchar < ctx->ninmax) {
            ctx->buffer2_in =
                (ctx->buffer2_in << 8) | (int)infile[ctx->nextchar];
            ctx->nextchar++;
        }
        else {
            ctx->buffer2_in = ctx->buffer2_in << 8;
        }
        ctx->bits_to_go_in += 8;
    }
    ctx->bits_to_go_in -= n;
    return ((ctx->buffer2_in >> ctx->bits_to_go_in) & (*(mask + n)));
}

static int
input_nybble(
    hcomp_dctx *ctx,
    unsigned char *infile)
{
    if (ctx->bits_to_go_in < 4) {
        if (ctx->nextchar < ctx->ninmax) {
            ctx->buffer2_in =
                (ctx->buffer2_in << 8) | (int)infile[ctx->nextchar];
            ctx->nextchar++;
        }
        else {
            ctx->buffer2_in = ctx->buffer2_in << 8;
        }
        ctx->bits_to_go_in += 8;
    }
    ctx->bits_to_go_in -= 4;
    return ((ctx->buffer2_in >> ctx->bits_to_go_in) & 15);
}

static int
input_nnybble(
    hcomp_dctx *ctx,
    unsigned char *infile,
    int n,
    unsigned char array[])
{
    int ii, kk, shift1, shift2;

    if (n == 1) {
        array[0] = input_nybble(ctx, infile);
        return 0;
    }

    if (ctx->bits_to_go_in == 8) {
        ctx->nextchar--;
        ctx->bits_to_go_in = 0;
    }

    shift1 = ctx->bits_to_go_in + 4;
    shift2 = ctx->bits_to_go_in;
    kk = 0;

    if (ctx->bits_to_go_in == 0) {
        for (ii = 0; ii < n / 2; ii++) {
            if (ctx->nextchar < ctx->ninmax) {
                ctx->buffer2_in = (ctx->buffer2_in <<
                        8) | (int)infile[ctx->nextchar];
                ctx->nextchar++;
            }
            else {
                ctx->buffer2_in = ctx->buffer2_in << 8;
            }
            array[kk] = (int)((ctx->buffer2_in >> 4) & 15);
            array[kk + 1] = (int)((ctx->buffer2_in) & 15);
            kk += 2;
        }
    }
    else {
        for (ii = 0; ii < n / 2; ii++) {
            if (ctx->nextchar < ctx->ninmax) {
                ctx->buffer2_in = (ctx->buffer2_in <<
                        8) | (int)infile[ctx->nextchar];
                ctx->nextchar++;
            }
            else {
                ctx->buffer2_in = ctx->buffer2_in << 8;
            }
            array[kk] = (int)((ctx->buffer2_in >> shift1) & 15);
            array[kk + 1] = (int)((ctx->buffer2_in >> shift2) & 15);
            kk += 2;
        }
    }

    if (ii * 2 != n) {
        array[n - 1] = input_nybble(ctx, infile);
    }

    return ((ctx->buffer2_in >> ctx->bits_to_go_in) & 15);
}
