/* pliocomp.c */
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
  IRAF PLIO (pixel list I/O) line-list compression.

  The original prototype code was provided by Doug Tody, NRAO, for
  performing conversion between pixel arrays and line lists.
  The compression technique is used in IRAF.
  Translated from the SPP version using xc -f, f2c.  8Sep99 DCT.

  Forked from cfitsio 4.6.3.
  https://heasarc.gsfc.nasa.gov/fitsio/

  Modifications applied by Christoph Gohlke:

  - Rewrite f2c-generated code as structured C.
  - Use 0-based array indexing.
  - Remove xs (start index) parameter; always encode from index 0.
  - Add maxout parameter and bounds checks to prevent buffer overflow.
  - Add nsrc parameter and bounds checks to prevent buffer over-read.
  - Return error codes.
  - Lint code.

*/

#include "pliocomp.h"

#include <stdlib.h>

/*
  PLIO line-list encoding format:

  The encoded line list is an array of short (16-bit) values.
  The first 7 shorts form a header:

    [0]  unused
    [1]  header size: 7 (offset of first opcode)
    [2]  version marker: -100
    [3]  total length lo (filled at end)
    [4]  total length hi (filled at end)
    [5]  unused
    [6]  unused

  After the header, each short encodes an opcode (upper 4 bits) and
  a data value (lower 12 bits, range 0-4095).

  Opcode encoding (upper nibble, after Fortran-to-C adjustment):

    0 (0x0)  zero-run:  data = number of zero pixels (up to 4095)
    1 (0x1)  high-value set: value = next_word * 4096 + data (2 words)
    2 (0x2)  positive delta:  pv += data
    3 (0x3)  negative delta:  pv -= data
    4 (0x4)  data-run:  data = number of pixels at current value pv
    5 (0x5)  zero-run + single: data-1 zeros then one pixel of pv
    6 (0x6)  positive delta + single pixel: pv += data, emit one pixel
    7 (0x7)  negative delta + single pixel: pv -= data, emit one pixel

  Runs longer than 4095 are split across multiple words.
  Values larger than 4095 use opcode 1 (two-word high-value encoding).
  Delta values within +/-4095 use opcodes 2/3 (or 6/7 for single pixels).
*/

/* Write one short to the output buffer with bounds check. */
#define PLIO_EMIT(dst, op, maxout, value) \
    do { \
        if ((op) >= (maxout)) { \
            return PLIO_ERROR_OVERFLOW; \
        } \
        (dst)[(op)++] = (short)(value); \
    } while (0)


/* Encode a pixel array into a PLIO line list.

   src:     input pixel array (non-negative integers, max 24-bit)
   npix:    number of pixels in src
   dst:     output line list (array of short)
   maxout:  maximum number of shorts in dst
   nout:    receives the number of shorts written

   Returns PLIO_OK on success, or an error code.
*/
int
plio_encode(
    const int *src,
    int npix,
    short *dst,
    int maxout,
    int *nout)
{
    int ip;       /* current pixel index */
    int op;       /* output index (in shorts) */
    int pv;       /* current pixel value (clamped to >= 0) */
    int nv = 0;   /* next pixel value */
    int hi;       /* last emitted high value (for delta encoding) */
    int dv;       /* delta from hi to pv */
    int x1;       /* start of current nonzero run */
    int iz;       /* start of current zero run */
    int np;       /* length of nonzero run */
    int nz;       /* length of zero run */
    int v;

    if (src == NULL || dst == NULL || nout == NULL) {
        return PLIO_ERROR;
    }

    *nout = 0;

    if (npix <= 0) {
        return PLIO_OK;
    }

    if (maxout < PLIO_HEADER_SIZE + 1) {
        return PLIO_ERROR_OVERFLOW;
    }

    /* Write header (0-based layout matching cfitsio's 1-based convention).
       [0] = unused
       [1] = header size (7)
       [2] = version marker (-100)
       [3] = total length lo (filled at end)
       [4] = total length hi (filled at end)
       [5] = unused
       [6] = unused
       First opcode at index 7. */
    dst[0] = 0;
    dst[1] = PLIO_HEADER_SIZE;  /* header size */
    dst[2] = -100;  /* version marker */
    dst[3] = 0;
    dst[4] = 0;
    dst[5] = 0;
    dst[6] = 0;
    op = PLIO_HEADER_SIZE;

    pv = src[0] > 0 ? src[0] : 0;
    if (pv > PLIO_MAX_VALUE) {
        return PLIO_ERROR;
    }
    x1 = 0;
    iz = 0;
    hi = 1;

    for (ip = 0; ip < npix; ip++) {
        /* Determine next value. */
        if (ip < npix - 1) {
            nv = src[ip + 1] > 0 ? src[ip + 1] : 0;
            if (nv > PLIO_MAX_VALUE) {
                return PLIO_ERROR;
            }
            if (nv == pv) {
                /* Still in a run of the same value; continue. */
                continue;
            }
            if (pv == 0) {
                /* Transition from zero to nonzero; update run start. */
                pv = nv;
                x1 = ip + 1;
                continue;
            }
            /* pv != 0 && nv != pv: emit the accumulated run. */
        }
        else {
            /* Last pixel. */
            if (pv == 0) {
                x1 = npix;
            }
            /* Fall through to emit. */
        }

        /* Emit the current segment: nz zeros followed by np pixels of pv. */
        np = ip - x1 + 1;
        nz = x1 - iz;

        /* Emit value-change opcodes if pv > 0. */
        if (pv > 0) {
            dv = pv - hi;
            if (dv != 0) {
                hi = pv;
                if (abs(dv) > 4095) {
                    /* High-value encoding: opcode 1 + continuation word. */
                    PLIO_EMIT(dst, op, maxout, (pv & 4095) + 4096);
                    PLIO_EMIT(dst, op, maxout, pv / 4096);
                }
                else if (dv < 0) {
                    /* Negative delta: opcode 3. */
                    PLIO_EMIT(dst, op, maxout, -dv + 12288);

                    /* Optimization: single nonzero pixel with no zero run
                       preceding it - combine delta + pixel into one word
                       (opcode 7). Only valid for single-word deltas. */
                    if (np == 1 && nz == 0) {
                        v = dst[op - 1];
                        dst[op - 1] = (short)(v | 16384);
                        goto done_segment;
                    }
                }
                else {
                    /* Positive delta: opcode 2. */
                    PLIO_EMIT(dst, op, maxout, dv + 8192);

                    /* Optimization: single nonzero pixel with no zero run
                       preceding it - combine delta + pixel into one word
                       (opcode 6). Only valid for single-word deltas. */
                    if (np == 1 && nz == 0) {
                        v = dst[op - 1];
                        dst[op - 1] = (short)(v | 16384);
                        goto done_segment;
                    }
                }
            }
        }

        /* Emit zero-run words (opcode 0, up to 4095 per word). */
        if (nz > 0) {
            while (nz > 0) {
                int chunk = nz < 4095 ? nz : 4095;
                PLIO_EMIT(dst, op, maxout, chunk);
                nz -= chunk;
            }

            /* Optimization: if single nonzero pixel after a zero run,
               combine the last zero-run word with a data marker
               (opcode 5). */
            if (np == 1 && pv > 0) {
                dst[op - 1] = (short)(dst[op - 1] + 20481);
                goto done_segment;
            }
        }

        /* Emit data-run words (opcode 4, up to 4095 per word). */
        while (np > 0) {
            int chunk = np < 4095 ? np : 4095;
            PLIO_EMIT(dst, op, maxout, chunk + 16384);
            np -= chunk;
        }

done_segment:
        x1 = ip + 1;
        iz = x1;
        pv = nv;
    }

    /* Store total length in header.
       The stored value uses cfitsio's 1-based convention:
       op (0-based one-past-end) equals the 1-based last-used index. */
    dst[3] = (short)(op % 32768);
    dst[4] = (short)(op / 32768);

    *nout = op;
    return PLIO_OK;
}


/* Decode a PLIO line list into a pixel array.

   src:   input line list (array of short)
   nsrc:  number of shorts in src
   dst:   output pixel array
   npix:  number of pixels to decode

   Returns PLIO_OK on success, or an error code.
*/
int
plio_decode(
    const short *src,
    int nsrc,
    int *dst,
    int npix)
{
    int ip;       /* index into src (line list) */
    int op;       /* index into dst (pixel array) */
    int x1;       /* current position in logical pixel space (1-based) */
    int x2;       /* end of current range */
    int xe;       /* last pixel index (npix, 1-based) */
    int pv;       /* current pixel value */
    int opcode;   /* upper 4 bits of current word */
    int data;     /* lower 12 bits of current word */
    int lllen;    /* total length of line list */
    int llfirst;  /* index of first opcode in line list */
    int skipwd;   /* flag: skip next word (used by high-value opcode) */
    int i1, i2, np, otop, i;

    if (src == NULL || dst == NULL) {
        return PLIO_ERROR;
    }

    if (npix <= 0) {
        return PLIO_OK;
    }

    if (nsrc < PLIO_HEADER_SIZE) {
        return PLIO_ERROR_FORMAT;
    }

    /* Parse header (0-based array indexing).

       Old format (cfitsio 1-based src[3] > 0, i.e., 0-based src[2] > 0):
         src[2] = total length, first opcode at index 3.

       New format (0-based src[2] == -100):
         src[1] = header size (offset to first opcode)
         src[2] = -100 (version marker)
         src[3] + src[4] * 32768 = total length (1-based last index)
         First opcode at 0-based index src[1]. */
    if (src[2] > 0) {
        /* Old format: stored length is 1-based; convert to 0-based. */
        lllen = src[2] - 1;
        llfirst = 3;
    }
    else {
        /* New format (version -100).
           The stored length is in cfitsio's 1-based convention.
           Convert to 0-based last index for our loop. */
        lllen = ((int)src[3] & 0x7FFF) + ((int)src[4] << 15) - 1;
        llfirst = src[1];
        if (llfirst < PLIO_HEADER_SIZE || llfirst > nsrc) {
            return PLIO_ERROR_FORMAT;
        }
    }

    if (lllen <= 0) {
        return PLIO_OK;
    }

    /* Validate that the declared length does not exceed the buffer.
       lllen is now the 0-based index of the last used element,
       so we need lllen < nsrc. */
    if (lllen >= nsrc) {
        return PLIO_ERROR_FORMAT;
    }

    /* xe is the last 1-based pixel index we need to output. */
    xe = npix;

    skipwd = 0;
    op = 0;
    x1 = 1;
    pv = 1;

    for (ip = llfirst; ip <= lllen; ip++) {
        if (skipwd) {
            skipwd = 0;
            continue;
        }

        if (ip < 0 || ip >= nsrc) {
            return PLIO_ERROR_FORMAT;
        }

        opcode = src[ip] / 4096;
        data = src[ip] & 4095;

        switch (opcode)
        {
            case 0: /* zero-run */
            case 4: /* data-run (repeat pv) */
            case 5: /* zero-run + single nonzero at end */
                x2 = x1 + data - 1;
                i1 = x1 > 1 ? x1 : 1;
                i2 = x2 < xe ? x2 : xe;
                np = i2 - i1 + 1;
                if (np > 0) {
                    if (op + np > npix) {
                        return PLIO_ERROR_OVERFLOW;
                    }
                    otop = op + np;
                    if (opcode == 4) {
                        /* Fill with current value. */
                        for (i = op; i < otop; i++) {
                            dst[i] = pv;
                        }
                    }
                    else {
                        /* Fill with zeros. */
                        for (i = op; i < otop; i++) {
                            dst[i] = 0;
                        }
                        /* Opcode 5: set last pixel to pv. */
                        if (opcode == 5 && i2 == x2) {
                            dst[otop - 1] = pv;
                        }
                    }
                    op = otop;
                }
                x1 = x2 + 1;
                break;

            case 1: /* high-value set (two words) */
                if (ip + 1 >= nsrc) {
                    return PLIO_ERROR_FORMAT;
                }
                pv = ((int)src[ip + 1] << 12) + data;
                skipwd = 1;
                break;

            case 2: /* positive delta */
                pv += data;
                break;

            case 3: /* negative delta */
                pv -= data;
                break;

            case 6: /* positive delta + single pixel */
                pv += data;
                if (x1 >= 1 && x1 <= xe) {
                    if (op >= npix) {
                        return PLIO_ERROR_OVERFLOW;
                    }
                    dst[op++] = pv;
                }
                x1++;
                break;

            case 7: /* negative delta + single pixel */
                pv -= data;
                if (x1 >= 1 && x1 <= xe) {
                    if (op >= npix) {
                        return PLIO_ERROR_OVERFLOW;
                    }
                    dst[op++] = pv;
                }
                x1++;
                break;

            default:
                /* Unknown opcode; ignore. */
                break;
        }

        if (x1 > xe) {
            break;
        }
    }

    /* Zero-fill any remaining pixels. */
    for (i = op; i < npix; i++) {
        dst[i] = 0;
    }

    return PLIO_OK;
}
