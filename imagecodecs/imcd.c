/* imcd.c */

/*
Copyright (c) 2008-2021, Christoph Gohlke.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

#include "imcd.h"

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fenv.h>

#ifndef M_PI
#define M_PI (3.1415926535897932384626433832795)
#endif
#define M_2PI (6.283185307179586476925286766559)

#define MAX_(a, b) (((a) > (b)) ? (a) : (b))
#define MIN_(a, b) (((a) < (b)) ? (a) : (b))


/******************************** Swap Bytes *********************************/

#define SWAP2BYTES(x) \
  ((((x) >> 8) & 0x00FF) | (((x) & 0x00FF) << 8))

#define SWAP4BYTES(x) \
  ((((x) >> 24) & 0x00FF) | (((x)&0x00FF) << 24) | \
   (((x) >> 8 ) & 0xFF00) | (((x)&0xFF00) << 8))

#define SWAP8BYTES(x) \
  ((((x) >> 56) & 0x00000000000000FF) | (((x) >> 40) & 0x000000000000FF00) | \
   (((x) >> 24) & 0x0000000000FF0000) | (((x) >> 8)  & 0x00000000FF000000) | \
   (((x) << 8)  & 0x000000FF00000000) | (((x) << 24) & 0x0000FF0000000000) | \
   (((x) << 40) & 0x00FF000000000000) | (((x) << 56) & 0xFF00000000000000))


/* Inplace swap bytes. */
void imcd_swapbytes(
    void* src,
    const ssize_t srcsize,
    const ssize_t itemsize)
{
    uint16_t* d16;
    uint32_t* d32;
    uint64_t* d64;
    ssize_t i;
    switch (itemsize)
    {
        case 2: {
            d16 = (uint16_t*)src;
            for (i = 0; i < srcsize; i++) {
                *d16 = SWAP2BYTES(*d16);
                d16++;
            }
            break;
        }
        case 4: {
            d32 = (uint32_t*)src;
            for (i = 0; i < srcsize; i++) {
                *d32 = SWAP4BYTES(*d32);
                d32++;
            }
            break;
        }
        case 8: {
            d64 = (uint64_t*)src;
            for (i = 0; i < srcsize; i++) {
                *d64 = SWAP8BYTES(*d64);
                d64++;
            }
            break;
        }
    }
}


/* Return mask for number of bits. */
uint8_t imcd_bitmask(const int bps)
{
    uint8_t result = 0;
    uint8_t power = 1;
    int i;
    for (i = 0; i < bps; i++) {
        result += power;
        power *= 2;
    }
    return result << (8 - bps);
}


/********************************** Delta ************************************/

#define DELTA(dtype)  \
{  \
    if (decode) {  \
        dtype sum = *(dtype*)srcptr;  \
        if (inplace && (stride == sizeof(dtype))  \
                    && (srcstride == sizeof(dtype))) {  \
            /* decode contiguous in-place */  \
            dtype* qsrc = (dtype*)srcptr;  \
            for (i = 1; i < size; i++) {  \
                sum += qsrc[i];  \
                qsrc[i] = sum;  \
            }  \
        }  \
        else {  \
            *(dtype*)dstptr = sum;  \
            for (i = 1; i < size; i++) {  \
                dstptr += stride;  \
                srcptr += srcstride;  \
                sum += *(dtype*)srcptr;  \
                *(dtype*)dstptr = sum;  \
            }  \
        }  \
    }  \
    else {  \
        dtype src0, src1;  \
        src0 = *(dtype*)srcptr;  \
        *(dtype*)dstptr = src0;  \
        for (i = 1; i < size; i++) {  \
            dstptr += stride;  \
            srcptr += srcstride;  \
            src1 = *(dtype*)srcptr;  \
            *(dtype*)dstptr = src1 - src0;  \
            src0 = src1;  \
        }  \
    }  \
}


/* Encode or decode integer differences. */
ssize_t imcd_delta(
    void* src,
    const ssize_t srcsize,
    const ssize_t srcstride,
    void* dst,
    const ssize_t dstsize,
    const ssize_t dststride,
    const ssize_t itemsize,
    const int decode)
{
    const int inplace = (dst == NULL) || (dst == src);
    const ssize_t size = inplace ? srcsize : MIN_(srcsize, dstsize);
    const ssize_t stride = inplace ? srcstride : dststride;
    char* srcptr = (char*)src;
    char* dstptr = inplace ? (char*)src : (char*)dst;
    ssize_t i;

    if ((srcsize < 0) || (dstsize < 0)) {
        return IMCD_VALUE_ERROR;
    }

    if (size <= 0) {
        return 0;
    }

    switch (itemsize)
    {
        case 1:
            DELTA(uint8_t);
            break;
        case 2:
            DELTA(uint16_t);
            break;
        case 4:
            DELTA(uint32_t);
            break;
        case 8:
            DELTA(uint64_t);
            break;
        default:
            return IMCD_VALUE_ERROR;
    }
    return size;
}


/* Encode or decode integer or floating-point differences. */
ssize_t imcd_diff(
    void* src,
    const ssize_t srcsize,
    const ssize_t srcstride,
    void* dst,
    const ssize_t dstsize,
    const ssize_t dststride,
    const ssize_t itemsize,
    const char itemtype,
    const int decode)
{
    const int inplace = (dst == NULL) || (dst == src);
    const ssize_t size = inplace ? srcsize : MIN_(srcsize, dstsize);
    const ssize_t stride = inplace ? srcstride : dststride;
    char* srcptr = (char*)src;
    char* dstptr = inplace ? (char*)src : (char*)dst;
    ssize_t i;

    if ((srcsize < 0) || (dstsize < 0)) {
        return IMCD_VALUE_ERROR;
    }

    if (size <= 0) {
        return 0;
    }

    switch (itemtype)
    {
        case 'B':
        case 'u':
        {
            switch (itemsize)
            {
                case 1:
                    DELTA(uint8_t);
                    break;
                case 2:
                    DELTA(uint16_t);
                    break;
                case 4:
                    DELTA(uint32_t);
                    break;
                case 8:
                    DELTA(uint64_t);
                    break;
                default:
                    return IMCD_VALUE_ERROR;
            }
            break;
        }
        case 'b':
        case 'i':
        {
            switch (itemsize)
            {
                case 1:
                    DELTA(int8_t);
                    break;
                case 2:
                    DELTA(int16_t);
                    break;
                case 4:
                    DELTA(int32_t);
                    break;
                case 8:
                    DELTA(int64_t);
                    break;
                default:
                    return IMCD_VALUE_ERROR;
            }
            break;
        }
        case 'f':
        {
            switch (itemsize)
            {
                case 4:
                    DELTA(float);
                    break;
                case 8:
                    DELTA(double);
                    break;
                default:
                    return IMCD_VALUE_ERROR;
            }
            break;
        }
    }
    return size;
}

/********************************* XOR Delta *********************************/

#define XOR(dtype)  \
{  \
    dtype prev, t;  \
    if (decode) {  \
        if (inplace && (stride == sizeof(dtype))  \
                    && (srcstride == sizeof(dtype))) {  \
            /* decode contiguous in-place */  \
            dtype* psrc = (dtype*)srcptr;  \
            prev = psrc[0];  \
            for (i = 1; i < size; i++) {  \
                prev = psrc[i] ^= prev;  \
            }  \
        }  \
        else {  \
            *(dtype*)dstptr = prev = (*(dtype*)srcptr);  \
            for (i = 1; i < size; i++) {  \
                dstptr += stride;  \
                srcptr += srcstride;  \
                *(dtype*)dstptr = prev = (*(dtype*)srcptr) ^ prev;  \
            }  \
        }  \
    }  \
    else {  \
        *(dtype*)dstptr = prev = (*(dtype*)srcptr);  \
        for (i = 1; i < size; i++) {  \
            dstptr += stride;  \
            srcptr += srcstride;  \
            t = (*(dtype*)srcptr);  \
            *(dtype*)dstptr = t ^ prev;  \
            prev = t;  \
        }  \
    }  \
}


/* Encode or decode XOR. */
ssize_t imcd_xor(
    void* src,
    const ssize_t srcsize,
    const ssize_t srcstride,
    void* dst,
    const ssize_t dstsize,
    const ssize_t dststride,
    const ssize_t itemsize,
    const int decode)
{
    const int inplace = (dst == NULL) || (dst == src);
    const ssize_t size = inplace ? srcsize : MIN_(srcsize, dstsize);
    const ssize_t stride = inplace ? srcstride : dststride;
    char* srcptr = (char*)src;
    char* dstptr = inplace ? (char*)src : (char*)dst;
    ssize_t i;

    if ((srcsize < 0) || (dstsize < 0)) {
        return IMCD_VALUE_ERROR;
    }

    if (size <= 0) {
        return 0;
    }

    switch (itemsize)
    {
        case 1:
            XOR(uint8_t);
            break;
        case 2:
            XOR(uint16_t);
            break;
        case 4:
            XOR(uint32_t);
            break;
        case 8:
            XOR(uint64_t);
            break;
        default:
            return IMCD_VALUE_ERROR;
    }
    return size;
}


/**************************** Bitorder Reversal ******************************/

/* http://graphics.stanford.edu/~seander/bithacks.html#ReverseByteWith64Bits */

#define REVERSE_BITS(b) ((uint8_t)(((((uint8_t)b) * 0x80200802ULL) \
                                   & 0x0884422110ULL) * 0x0101010101ULL >> 32))


/* Reverse bitorder of all bytes in string. */
ssize_t imcd_bitorder(
    uint8_t* src,
    const ssize_t srcsize,
    const ssize_t srcstride,
    const ssize_t itemsize,
    uint8_t* dst,
    const ssize_t dstsize,
    const ssize_t dststride)
{
    ssize_t i, j;

    if ((srcsize < 0) || (dstsize < 0) || (itemsize <= 0) ||
        (srcsize % itemsize))
    {
        return IMCD_VALUE_ERROR;
    }
    if (srcsize == 0) {
        return 0;
    }

    if ((dst == NULL) || (dst == src)) {
        /* inplace */
        if (srcstride == itemsize) {
            /* contiguous inplace */
            for (i = 0; i < srcsize; i++) {
                src[i] = REVERSE_BITS(src[i]);
            }
        }
        else {
            /* strided inplace */
            for (j = 0; j < srcsize / itemsize; j++) {
                for (i = 0; i < itemsize; i++) {
                    src[i] = REVERSE_BITS(src[i]);
                }
                src += srcstride;
            }
        }
        return srcsize;
    }
    else {
        /* not inplace */
        const ssize_t size = MIN_(srcsize, dstsize);
        if (size == 0) {
            return 0;
        }
        if ((srcstride == itemsize) && (dststride == itemsize)) {
            /* contiguous, not inplace */
            for (i = 0; i < size; i++) {
                dst[i] = REVERSE_BITS(src[i]);
            }
        }
        else {
            /* strided, not inplace */
            for (j = 0; j < size / itemsize; j++) {
                for (i = 0; i < itemsize; i++) {
                    dst[i] = REVERSE_BITS(src[i]);
                }
                src += srcstride;
                dst += dststride;
            }
        }
        return size;
    }
}


/************************* Floating Point Predictor **************************/

/* TIFF Technical Note 3. April 8, 2005. */

/* Encode or decode floating point predictor. */
ssize_t imcd_floatpred(
    void* src,
    const ssize_t srcsize,  /* size in bytes */
    const ssize_t srcstride,
    void* dst,
    const ssize_t dstsize,  /* size in bytes */
    const ssize_t dststride,
    const ssize_t itemsize,
    const ssize_t samples,
    const char byteorder,
    const int decode)
{
    uint8_t* srcptr = (uint8_t*)src;
    uint8_t* dstptr = (uint8_t*)dst;
    const ssize_t size = itemsize > 0 ? MIN_(srcsize, dstsize) / itemsize : -1;
    ssize_t i, j;

    if ((src == NULL) || (dst == NULL) || (src == dst) ||
        (srcsize < 0) || (dstsize < 0) || (samples < 1) ||
        (size <= 0) || (size % samples) ||
        ((itemsize != 2) && (itemsize != 3) &&
         (itemsize != 4) && (itemsize != 8)))
    {
        return IMCD_VALUE_ERROR;
    }

    if (decode) {
        /* decode */
        if ((srcstride != itemsize) || (dststride % itemsize)) {
            return IMCD_VALUE_ERROR;
        }
        /* TODO: do not temporarily modify src */
        /* undo byte differencing; separate for interleaved samples */
        for (i = samples; i < size*itemsize; i++) {
            srcptr[i] += srcptr[i - samples];
        }
        /* restore byte order into dst buffer */
        if (byteorder != '>') {
            /* little endian */
            for (i = 0; i < size; i++) {
                for (j = 0; j < itemsize; j++) {
                    *(dstptr + dststride * i + j) =
                        srcptr[(itemsize - j - 1) * size + i];
                }
            }
        }
        else {
            /* big endian */
            for (i = 0; i < size; i++) {
                for (j = 0; j < itemsize; j++) {
                    *(dstptr + dststride * i + j) = srcptr[j * size + i];
                }
            }
        }
        /* restore byte differencing in src */
        for (i = size * itemsize - 1; i >= samples; i--) {
            srcptr[i] -= srcptr[i - samples];
        }
    }
    else {
        /* encode */
        if ((dststride != itemsize) || (srcstride % itemsize)) {
            return IMCD_VALUE_ERROR;
        }
        /* reorder src bytes into dst */
        if (byteorder != '>') {
            /* little endian */
            for (i = 0; i < size; i++) {
                for (j = 0; j < itemsize; j++) {
                    dstptr[(itemsize - j - 1) * size + i] =
                        *(srcptr + srcstride * i + j);
                }
            }
        }
        else {
            /* big endian */
            for (i = 0; i < size; i++) {
                for (j = 0; j < itemsize; j++) {
                    dstptr[j * size + i] = *(srcptr + srcstride * i + j);
                }
            }
        }
        /* byte differencing; separate for interleaved samples */
        for (i = size * itemsize - 1; i >= samples; i--) {
            dstptr[i] -= dstptr[i - samples];
        }
    }
    return size;
}


/********************************* PackBits **********************************/

/* Section 9: PackBits Compression. TIFF Revision 6.0 Final. June 3, 1992 */
/* Apple Technical Note TN1023. Understanding PackBits. Feb 1, 1996 */

/* Return length of uncompressed sequence. */
ssize_t imcd_packbits_size(
    const uint8_t* src,
    const ssize_t srcsize)
{
    uint8_t* srcptr = (uint8_t*)src;
    const uint8_t* srcend = (srcptr == NULL) ? NULL : srcptr + srcsize;
    ssize_t dstsize = 0;
    ssize_t n;

    if ((srcptr == NULL) || (srcsize < 0)) {
        return IMCD_VALUE_ERROR;
    }
    if (srcsize == 0) {
        return 0;
    }

    while (srcptr < srcend) {
        n = (ssize_t)(*srcptr++) + 1;
        if (n < 129) {
            /* literal */
            if (srcptr + n > srcend)
                n = (ssize_t)(srcend - srcptr);
            srcptr += n;
            dstsize += n;
        }
        else if (n > 129) {
            /* replicate */
            srcptr++;
            dstsize += 258 - n;
        }
    }
    return dstsize;
}


/* Decode PackBits. */
ssize_t imcd_packbits_decode(
    const uint8_t* src,
    const ssize_t srcsize,
    uint8_t* dst,
    const ssize_t dstsize)
{
    uint8_t* srcptr = (uint8_t*)src;
    uint8_t* dstptr = dst;
    const uint8_t* srcend = (srcptr == NULL) ? NULL : srcptr + srcsize;
    ssize_t n = 0;
    ssize_t dstlen = 0;

    if ((srcptr == NULL) || (srcsize < 0) ||
        (dstptr == NULL) || (dstsize < 0))
    {
        return IMCD_VALUE_ERROR;
    }
    if ((srcsize == 0) || (dstsize == 0)) {
        return 0;
    }

    while ((srcptr < srcend) && (dstlen < dstsize)) {
        n = (ssize_t)(*srcptr++) + 1;
        if (n < 129) {
            /* literal */
            if (srcptr + n > srcend)
                n = (ssize_t)(srcend - srcptr);
            if (dstlen + n > dstsize) {
                n = dstsize - dstlen;
            }
            dstlen += n;
            while (n--) {
                *dstptr++ = *srcptr++;
            }
        }
        else if (n > 129) {
            /* replicate */
            const uint8_t e = *srcptr++;
            n = 258 - n;
            if (dstlen + n > dstsize) {
                n = dstsize - dstlen;
            }
            dstlen += n;
            while (n--) {
                *dstptr++ = e;
            }
        }
        /* else if (n == 129) {NOP} */
    }
    return dstlen;
}


/* Encode PackBits. */
ssize_t imcd_packbits_encode(
    const uint8_t* src,
    const ssize_t srcsize,
    uint8_t* dst,
    const ssize_t dstsize)
{
    uint8_t* srcptr = (uint8_t*)src;
    uint8_t* dstptr = dst;
    uint8_t* srcend = NULL;
    uint8_t* dstend = NULL;
    ssize_t dstlen = 0;
    int replicate = 0;
    int literal = 0;
    uint8_t value, v;

    if ((srcptr == NULL) || (srcsize < 0) ||
        (dstptr == NULL) || (dstsize < 0))
    {
        return IMCD_VALUE_ERROR;
    }
    if ((srcsize == 0) || (dstsize == 0)) {
        return 0;
    }

    srcend = srcptr + srcsize;
    dstend = dstptr + dstsize - 1;
    replicate = 0;
    literal = 0;
    value = *srcptr;

    while ((srcptr < srcend) && (dstptr < dstend))
    {
        v = *srcptr++;
        /* printf("%i %i %i %i\n", (int)value, (int)v, literal, replicate); */
        if (value == v) {
            if (literal == 1) {
                literal = 0;
                replicate = 1;
            }
            else if (literal > 1) {
                srcptr = srcptr - literal - 1;
                literal--;
                if (dstptr + literal > dstend) {
                    literal = (int)(dstend - dstptr);
                    literal = MIN_(0, literal);
                }
                *dstptr++ = (uint8_t)(literal - 1);
                while (literal--) {
                    *dstptr++ = *srcptr++;
                }
                srcptr++;
                srcptr++;
                literal = 0;
                replicate = 1;
            }
            replicate++;
        }
        else {
            if (replicate == 1) {
                replicate = 0;
                literal = 1;
            }
            else if (replicate > 1) {
                *dstptr++ = (uint8_t)(257 - replicate);
                *dstptr++ = value;
                replicate = 0;
                literal = 0;
            }
            literal++;
        }
        if (replicate == 128) {
            *dstptr++ = (uint8_t)(257 - replicate);
            *dstptr++ = value;
            replicate = 0;
            literal = 0;
        }
        else if (literal == 128) {
            srcptr = srcptr - literal;
            if (dstptr + literal > dstend) {
                literal = (int)(dstend - dstptr);
                literal = MIN_(0, literal);
            }
            *dstptr++ = (uint8_t)(literal - 1);
            while (literal--) {
                *dstptr++ = *srcptr++;
            }
            replicate = 0;
            literal = 0;
        }
        value = v;
    }
    if (replicate == 1) {
        replicate = 0;
        literal = 1;
    }
    if (literal > 0) {
        srcptr = srcptr - literal;
        if (dstptr + literal > dstend) {
            literal = (int)(dstend - dstptr);
            literal = MIN_(0, literal);
        }
        *dstptr++ = (uint8_t)(literal - 1);
        while (literal--) {
            *dstptr++ = *srcptr++;
        }
    }
    else if (replicate > 0) {
        *dstptr++ = (uint8_t)(257 - replicate);
        *dstptr++ = value;
    }

    dstlen = (ssize_t)(dstptr - dst);
    /* NOP
    while (dstptr < dstend+1) {
        *dstptr++ = 128;
    }
    */
    return dstlen;
}


/***************************** Packed Integers *******************************/

typedef union {
    uint8_t b[2];
    uint16_t i;
} u_uint16_t;

typedef union {
    uint8_t b[4];
    uint32_t i;
} u_uint32_t;

typedef union {
    uint8_t b[8];
    uint64_t i;
} u_uint64_t;


/* Unpack sequence of packed 1-32 bit integers.

Input src array should be padded to the next 16, 32 or 64-bit boundary
if itemsize not in (1, 2, 4, 8, 16, 24, 32, 64).

*/
ssize_t imcd_packints_decode(
    const uint8_t* src,
    const ssize_t srcsize,  /** size of src in bytes */
    uint8_t* dst,  /** buffer to store unpacked items */
    const ssize_t dstsize,   /** number of items to unpack */
    const int bps  /** number of bits in integer */
    )
{
    ssize_t i, j, k;
    ssize_t itemsize;
    uint8_t value;

    if (srcsize == 0) {
        return 0;
    }

    /* Input validation is done in wrapper function */
    itemsize = (ssize_t)(ceil(bps / 8.0));
    itemsize = itemsize < 3 ? itemsize : itemsize > 4 ? 8 : 4;
    switch (bps)
    {
        case 8:
        case 16:
        case 32:
        case 64:
            memcpy(dst, src, dstsize * itemsize);
            return dstsize;
        case 1:
            for (i = 0, j = 0; i < dstsize/8; i++) {
                value = src[i];
                dst[j++] = (value & (uint8_t)(128)) >> 7;
                dst[j++] = (value & (uint8_t)(64)) >> 6;
                dst[j++] = (value & (uint8_t)(32)) >> 5;
                dst[j++] = (value & (uint8_t)(16)) >> 4;
                dst[j++] = (value & (uint8_t)(8)) >> 3;
                dst[j++] = (value & (uint8_t)(4)) >> 2;
                dst[j++] = (value & (uint8_t)(2)) >> 1;
                dst[j++] = (value & (uint8_t)(1));
            }
            if (dstsize % 8) {
                value = src[i];
                switch (dstsize % 8)
                {
                    case 7: dst[j+6] = (value & (uint8_t)(2)) >> 1;
                    case 6: dst[j+5] = (value & (uint8_t)(4)) >> 2;
                    case 5: dst[j+4] = (value & (uint8_t)(8)) >> 3;
                    case 4: dst[j+3] = (value & (uint8_t)(16)) >> 4;
                    case 3: dst[j+2] = (value & (uint8_t)(32)) >> 5;
                    case 2: dst[j+1] = (value & (uint8_t)(64)) >> 6;
                    case 1: dst[j] = (value & (uint8_t)(128)) >> 7;
                }
            }
            return dstsize;
        case 2:
            for (i = 0, j = 0; i < dstsize/4; i++) {
                value = src[i];
                dst[j++] = (value & (uint8_t)(192)) >> 6;
                dst[j++] = (value & (uint8_t)(48)) >> 4;
                dst[j++] = (value & (uint8_t)(12)) >> 2;
                dst[j++] = (value & (uint8_t)(3));
            }
            if (dstsize % 4) {
                value = src[i];
                switch (dstsize % 4)
                {
                    case 3: dst[j+2] = (value & (uint8_t)(12)) >> 2;
                    case 2: dst[j+1] = (value & (uint8_t)(48)) >> 4;
                    case 1: dst[j] = (value & (uint8_t)(192)) >> 6;
                }
            }
            return dstsize;
        case 4:
            for (i = 0, j = 0; i < dstsize/2; i++) {
                value = src[i];
                dst[j++] = (value & (uint8_t)(240)) >> 4;
                dst[j++] = (value & (uint8_t)(15));
            }
            if (dstsize % 2) {
                value = src[i];
                dst[j] = (value & (uint8_t)(240)) >> 4;
            }
            return dstsize;
        case 24:
            j = k = 0;
            for (i = 0; i < dstsize; i++) {
                dst[j++] = 0;
                dst[j++] = src[k++];
                dst[j++] = src[k++];
                dst[j++] = src[k++];
            }
            return dstsize;
    }
    /* 3, 5, 6, 7 */
    if (bps < 8) {
        int shr = 16;
        u_uint16_t value, mask, tmp;
        j = k = 0;
        value.b[IMCD_MSB] = src[j++];
        value.b[IMCD_LSB] = src[j++];
        mask.b[IMCD_MSB] = imcd_bitmask(bps);
        mask.b[IMCD_LSB] = 0;
        for (i = 0; i < dstsize; i++) {
            shr -= bps;
            tmp.i = (value.i & mask.i) >> shr;
            dst[k++] = tmp.b[IMCD_LSB];
            if (shr < bps) {
                value.b[IMCD_MSB] = value.b[IMCD_LSB];
                value.b[IMCD_LSB] = src[j++];
                mask.i <<= 8 - bps;
                shr += 8;
            }
            else {
                mask.i >>= bps;
            }
        }
        return dstsize;
    }
    /* 9, 10, 11, 12, 13, 14, 15 */
    if (bps < 16) {
        int shr = 32;
        u_uint32_t value, mask, tmp;
        mask.i = 0;
        j = k = 0;
#if IMCD_MSB
        for (i = 3; i >= 0; i--) {
            value.b[i] = src[j++];
        }
        mask.b[3] = 0xFF;
        mask.b[2] = imcd_bitmask(bps-8);
        for (i = 0; i < dstsize; i++) {
            shr -= bps;
            tmp.i = (value.i & mask.i) >> shr;
            dst[k++] = tmp.b[0]; /* swap bytes */
            dst[k++] = tmp.b[1];
            if (shr < bps) {
                value.b[3] = value.b[1];
                value.b[2] = value.b[0];
                value.b[1] = j < srcsize ? src[j++] : 0;
                value.b[0] = j < srcsize ? src[j++] : 0;
                mask.i <<= 16 - bps;
                shr += 16;
            }
            else {
                mask.i >>= bps;
            }
        }
#else
    /* not implemented */
#endif
        return dstsize;
    }
    /* 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31 */
    if (bps < 32) {
        int shr = 64;
        u_uint64_t value, mask, tmp;
        mask.i = 0;
        j = k = 0;
#if IMCD_MSB
        for (i = 7; i >= 0; i--) {
            value.b[i] = src[j++];
        }
        mask.b[7] = 0xFF;
        mask.b[6] = 0xFF;
        mask.b[5] = bps > 23 ? 0xFF : imcd_bitmask(bps - 16);
        mask.b[4] = bps < 24 ? 0x00 : imcd_bitmask(bps - 24);
        for (i = 0; i < dstsize; i++) {
            shr -= bps;
            tmp.i = (value.i & mask.i) >> shr;
            dst[k++] = tmp.b[0]; /* swap bytes */
            dst[k++] = tmp.b[1];
            dst[k++] = tmp.b[2];
            dst[k++] = tmp.b[3];
            if (shr < bps) {
                value.b[7] = value.b[3];
                value.b[6] = value.b[2];
                value.b[5] = value.b[1];
                value.b[4] = value.b[0];
                value.b[3] = j < srcsize ? src[j++] : 0;
                value.b[2] = j < srcsize ? src[j++] : 0;
                value.b[1] = j < srcsize ? src[j++] : 0;
                value.b[0] = j < srcsize ? src[j++] : 0;
                mask.i <<= 32 - bps;
                shr += 32;
            }
            else {
                mask.i >>= bps;
            }
        }
#else
    /* Not implemented */
#endif
        return dstsize;
    }
    return IMCD_ERROR;
}


/* Pack sequence of 1-32 bit integers.

*/
ssize_t imcd_packints_encode(
    const uint8_t* src,
    const ssize_t srcsize,  /** size of src in bytes */
    uint8_t* dst,  /** buffer to store packed items */
    const ssize_t dstsize,   /** number of items to pack */
    const int bps  /** number of bits in integer */
    )
{
    if (srcsize == 0) {
        return 0;
    }

    /* Input validation is done in wrapper function */

    return IMCD_NOTIMPLEMENTED_ERROR;
}


/********************************** Float24 **********************************/

/*
Adobe Photoshop(r) TIFF Technical Note 3. April 8, 2005.
24 bit floating point numbers have 1 sign bit, 7 exponent bits (biased by 64),
and 16 mantissa bits. The interpretation of the sign, exponent and mantissa
is analogous to IEEE-754 floating-point numbers. The 24 bit floating point
format supports normalized and denormalized numbers, infinities and NANs
(Not A Number).
*/

ssize_t imcd_float24_decode(
    const uint8_t* src,
    const ssize_t srcsize,
    uint8_t* dst,
    const char byteorder)
{
    /* input validation is done in wrapper function */

    ssize_t i;
    uint8_t s0, s1, s2, f0, f1, f2, f3;
    uint8_t sign = 0;
    uint8_t exponent = 0;
    uint32_t mantissa = 0;

    if (srcsize < 3) {
        return 0;
    }

    for (i = 0; i < srcsize; i += 3) {
        if (byteorder == '<') {
            /* from little endian */
            s2 = *src++;
            s1 = *src++;
            s0 = *src++;
        }
        else {
            /* from big endian */
            s0 = *src++;
            s1 = *src++;
            s2 = *src++;
        }

        sign = s0 & 0x80;
        exponent = s0 & 0x7F;
#if IMCD_MSB
        mantissa = s1;
        mantissa <<= 8;
        mantissa |= s2;
#else
        return IMCD_NOTIMPLEMENTED_ERROR;
#endif

        f0 = sign;
        if ((exponent == 0) & (mantissa == 0)) {
            /* +/- Zero */
            f1 = 0;
            f2 = 0;
            f3 = 0;
        }
        else if (exponent == 0x7F) {
            /* +/- INF or quiet NaN */
            f0 |= 0x7F;
            f1 = (mantissa == 0) ? 0x80 : 0xC0;  /* TODO: signaling NaN ? */
            f2 = 0;
            f3 = 0;
        }
        else {
            if (exponent == 0) {
                /* denormal/subnormal -> normalized */
                int shift = -1;
                do {
                    /* shift mantissa until lead bit overflows into exponent */
                    shift++;
                    mantissa <<= 1;
                } while ((mantissa & 0x10000) == 0);
                s2 = mantissa & 0xFF;
                s1 = (mantissa >> 8) & 0xFF;
                exponent = exponent - 63 + 127 - shift;  /* change bias */
            }
            else {
                /* normalized */
                exponent = exponent - 63 + 127;  /* change bias */
            }
            f0 |= exponent >> 1;
            f1 = ((exponent & 0x01) << 7) | ((s1 & 0xFE) >> 1);
            f2 = ((s1 & 0x01) << 7) | ((s2 & 0xFE) >> 1);
            f3 = (s2 & 0x01) << 7;
        }
#if IMCD_MSB
        /* to little endian */
        *dst++ = f3;
        *dst++ = f2;
        *dst++ = f1;
        *dst++ = f0;
#else
        /* to big endian */
        *dst++ = f0;
        *dst++ = f1;
        *dst++ = f2;
        *dst++ = f3;
#endif
    }
    return (srcsize / 3) * 3;
}


ssize_t imcd_float24_encode(
    const uint8_t* src,
    const ssize_t srcsize,
    uint8_t* dst,
    const char byteorder,
    int rounding)
{
    /* input validation is done in wrapper function */

    ssize_t i;
    uint8_t s0, s1, s2, s3, f0, f1, f2;
    uint8_t sign;
    uint8_t roundbit;
    uint32_t mantissa;
    int32_t exponent;

    if (srcsize < 4) {
        return 0;
    }

    if (rounding < 0) {
        rounding = fegetround();
    }
    if (
        (rounding != FE_TONEAREST) &&
        (rounding != FE_TOWARDZERO) &&
        (rounding != FE_UPWARD) &&
        (rounding != FE_DOWNWARD)
    ) {
        rounding = FE_TONEAREST;
    }

    for (i = 0; i < srcsize; i += 4) {
#if IMCD_MSB
        /* from little endian */
        s3 = *src++;
        s2 = *src++;
        s1 = *src++;
        s0 = *src++;
#else
        /* from big endian */
        s0 = *src++;
        s1 = *src++;
        s2 = *src++;
        s3 = *src++;
#endif
        sign = s0 & 0x80;
        exponent = ((s0 & 0x7F) << 1) | ((s1 & 0x80) >> 7);
#if IMCD_MSB
        mantissa = s1 & 0x7F;
        mantissa <<= 8;
        mantissa |= s2;
        mantissa <<= 8;
        mantissa |= s3;
#else
        return IMCD_NOTIMPLEMENTED_ERROR;
#endif
        f0 = sign;
        if ((exponent == 0) & (mantissa == 0)) {
            /* Zero */
            f1 = 0;
            f2 = 0;
        }
        else if (exponent == 0xFF) {
            /* INF or quiet NaN */
            f0 |= 0x7F;
            f1 = (mantissa == 0) ? 0x00 : 0x80;  /* TODO: signaling NaN ? */
            f2 = 0;
        }
        else if (exponent == 0) {
            /* from denormal/subnormal */
            if (
                ((rounding == FE_DOWNWARD) && sign) ||
                ((rounding == FE_UPWARD) && (!sign))
            ) {
                /* to smallest denormal */
                f1 = 0;
                f2 = 0x01;
            }
            else {
                /* to Zero */
                f1 = 0;
                f2 = 0;
            }
        }
        else {
            /* from normalized */
            roundbit = 0;
            exponent = exponent - 127 + 63;  /* change bias */
            if (exponent >= 0x7F) {
                /* to INF */
                f0 |= 0x7F;
                f1 = 0;
                f2 = 0;
            }
            else if (exponent <= 0) {
                /* to denormal; exponent is 0 */
                int32_t n = 8 - exponent;  /* number bits to shift mantissa */
                mantissa |= 0x800000;  /* add 24th mantissa bit */
                if (n < 32) {
                    /* TODO: this case is not well tested ! */
                    f2 = (mantissa >> n) & 0xFF;
                    f1 = (mantissa >> (n + 8)) & 0xFF;
                    if (n <= 24) {
                        uint32_t mask = 0xFFFFFFu & ~((0xFFFFFFu >> n) << n);
                        uint32_t firstbit = (mask >> (n - 1)) << (n - 1);
                        uint32_t trail = mantissa & mask;
                        roundbit = trail && (
                            ((rounding == FE_TONEAREST) &&
                                ((trail > firstbit) ||
                                ((trail == firstbit) && (f2 & 0x01)))) ||
                            ((rounding == FE_DOWNWARD) && sign) ||
                            ((rounding == FE_UPWARD) && !sign));
                    }
                    else {
                        /* rounding depending on 7 trailing mantissa bits */
                        roundbit = (s3 &= 0x7F) && (
                            ((rounding == FE_DOWNWARD) && sign) ||
                            ((rounding == FE_UPWARD) && !sign));
                    }
                }
                else {
                    f1 = 0x00;
                    f2 = 0x00;
                }
            }
            else {
                /* to normalized */
                uint32_t trail = s3 & 0x7F;  /* 7 trailing mantissa bits */
                f0 |= exponent & 0x7F;
                f1 = ((s1 & 0x7F) << 1) | ((s2 & 0x80) >> 7);
                f2 = ((s2 & 0x7F) << 1) | ((s3 & 0x80) >> 7);
                roundbit = trail && (
                    ((rounding == FE_TONEAREST) &&
                        ((trail > 0x40) || ((trail == 0x40) && (s3 & 0x80)))
                    ) ||
                    ((rounding == FE_DOWNWARD) && sign) ||
                    ((rounding == FE_UPWARD) && !sign));
            }
            if (roundbit) {
                if (f2 == 0xFF) {
                    f2 = 0x00;
                    if (f1 == 0xFF) {
                        f1 = 0x00;
                        /* exponent += 1;  overflow to exponent ? */
                    }
                    else {
                        f1 += 1;
                    }
                }
                else {
                    f2 += 1;
                }
            }
        }
        if (byteorder == '<') {
            /* to little endian */
            *dst++ = f2;
            *dst++ = f1;
            *dst++ = f0;
        } else {
            /* to big endian */
            *dst++ = f0;
            *dst++ = f1;
            *dst++ = f2;
        }
    }
    return (srcsize / 4) * 4;
}


/************************************ LZW ************************************/

/* Section 13: LZW Compression. TIFF Revision 6.0 Final. June 3, 1992 */

/* LZW table size is 4098 + 1024 buffer for old style */
#define LZW_TABLESIZE 5120
#define LZW_BUFFERSIZE 65536  /* 64 KB */
#define LZW_CLEAR 256
#define LZW_EOI 257


/* Allocate buffer or re-allocate in multiples of LZW_BUFFERSIZE */
ssize_t _lzw_alloc_buffer(
    imcd_lzw_handle_t* handle,
    ssize_t buffersize)
{
    if (handle == NULL) {
        return IMCD_VALUE_ERROR;
    }

    if (buffersize <= 0) {
        /* free buffer */
        free(handle->buffer);
        handle->buffer = NULL;
        handle->buffersize = 0;
        return 0;
    }

    if (handle->buffer == NULL) {
        /* allocate buffer */
        handle->buffer = (uint8_t*)malloc(buffersize);
    }
    else {
        /* reallocate buffer */
        buffersize = (((buffersize-1) / LZW_BUFFERSIZE) + 1) * LZW_BUFFERSIZE;
        handle->buffer = (uint8_t*)realloc((void*)handle->buffer, buffersize);
    }

    if (handle->buffer == NULL) {
        return IMCD_MEMORY_ERROR;
    }
    handle->buffersize = buffersize;
    return buffersize;
}


/* Allocate LZW handle. */
imcd_lzw_handle_t* imcd_lzw_new(ssize_t buffersize)
{
    /* TODO: check alignment of structs */
    imcd_lzw_handle_t* handle = NULL;
    ssize_t size = (
        sizeof(imcd_lzw_handle_t) + sizeof(imcd_lzw_table_t) * LZW_TABLESIZE
    );

    handle = (imcd_lzw_handle_t*)malloc(size);
    if (handle == NULL) {
        return NULL;
    }
    handle->table = (
        imcd_lzw_table_t*)((char*)handle + sizeof(imcd_lzw_handle_t)
    );
    handle->buffer = NULL;
    handle->buffersize = 0;

    if (_lzw_alloc_buffer(handle, buffersize) < 0) {
        imcd_lzw_del(handle);
        return NULL;
    }
    return handle;
}


/* Free LZW handle. */
void imcd_lzw_del(imcd_lzw_handle_t* handle)
{
    free(handle->buffer);
    free(handle);
}


/* MSB: TIFF and PDF */
#define LZW_GET_NEXT_CODE_MSB \
{  \
    if ((bitcount + bitw) <= srcbitsize)  \
    {  \
        const uint8_t* bytes = (uint8_t*)((void*)(src + (bitcount >> 3)));  \
        code = (uint32_t) bytes[0];  \
        code <<= 8;  \
        code |= (uint32_t) bytes[1];  \
        code <<= 8;  \
        if ((bitcount + 24) <= srcbitsize)  \
            code |= (uint32_t) bytes[2];  \
        code <<= 8;  \
        code <<= (uint32_t)(bitcount % 8);  \
        code &= mask;  \
        code >>= shr;  \
        bitcount += bitw;  \
    }  \
    else {code = LZW_EOI;}  \
}


/* LSB: GIF and old-style TIFF LZW */
#define LZW_GET_NEXT_CODE_LSB \
{  \
    if ((bitcount + bitw) <= srcbitsize)  \
    {  \
        const uint8_t* bytes = (uint8_t*)((void*)(src + (bitcount >> 3)));  \
        code = 0;  \
        if ((bitcount + 24) <= srcbitsize)  \
            code = bytes[2];  \
        code <<= 8;  \
        code |= bytes[1];  \
        code <<= 8;  \
        code |= bytes[0];  \
        code >>= (uint32_t)(bitcount % 8);  \
        code &= mask;  \
        bitcount += bitw;  \
    }  \
    else {code = LZW_EOI;}  \
}


/* Return 1 if compressed string begins with a CLEAR code */
int imcd_lzw_check(const uint8_t* src, const ssize_t size)
{
    if (src == NULL) {
        return 0;
    }
    if (size == 0) {
        return 1;
    }
    if (size < 2) {
        return IMCD_LZW_INVALID;
    }
    if ((*src == 0) && (*(src + 1) & 1)) {
        return 1;
    }
    if ((*src != 128) || ((*(src + 1) & 128))) {
        return 0;
    }
    return 1;
}


/* Return length of decompressed LZW string and initialize buffer. */
ssize_t imcd_lzw_decode_size(
    imcd_lzw_handle_t* handle,
    const uint8_t* src,
    const ssize_t srcsize)
{
    imcd_lzw_table_t* table;
    uint32_t tablesize = 258;
    uint32_t code = 0;
    uint32_t oldcode = 0;
    uint32_t shr = 23;
    uint32_t mask = 4286578688u;
    uint64_t bitw = 9;
    uint64_t bitcount = 0;
    ssize_t dstsize = 0;
    ssize_t buffer_size = 0;
    ssize_t buffersize = 0;
    const uint64_t srcbitsize = srcsize * 8;  /* size in bits */
    ssize_t i;
    int msb = 1;  /* bit ordering of codes */

    if ((handle == NULL) || (src == NULL) || (srcsize < 0)) {
        return IMCD_VALUE_ERROR;
    }
    if (srcsize == 0) {
        return 0;
    }
    if (srcsize < 2) {
        return IMCD_LZW_INVALID;
    }
    table = handle->table;

    if ((*src == 0) && (*(src + 1) & 1)) {
        msb = 0;
        mask = 511;
    }
    else if ((*src != 128) || ((*(src + 1) & 128))) {
        /* compressed string must begin with CLEAR code */
        return IMCD_LZW_INVALID;
    }

    for (i = 0; i < LZW_TABLESIZE; i++) {
        table[i].len = 1;
    }

    while (1) {

        if (msb) {
            LZW_GET_NEXT_CODE_MSB
        }
        else {
            LZW_GET_NEXT_CODE_LSB
        }

        if (code == LZW_EOI) break;

        if (code == LZW_CLEAR) {
            /* initialize table and switch to 9-bit */
            tablesize = 258;
            bitw = 9;
            shr = 23;

            if (buffersize > buffer_size)
                buffer_size = buffersize;
            buffersize = 0;

            if (msb) {
                mask = 4286578688u;
                do { LZW_GET_NEXT_CODE_MSB } while (code == LZW_CLEAR);
            }
            else {
                mask = 511;
                do { LZW_GET_NEXT_CODE_LSB } while (code == LZW_CLEAR);
            }

            if (code == LZW_EOI) break;

            dstsize++;
            oldcode = code;
            continue;
        }

        if (tablesize >= LZW_TABLESIZE) {
            return IMCD_LZW_TABLE_TOO_SMALL;
        }

        if (code < tablesize) {
            /* code is in table */
            dstsize += table[code].len;
            buffersize += table[oldcode].len + 1;
        }
        else if (code > tablesize) {
            /* return dstsize; */
            return IMCD_LZW_CORRUPT;
        }
        else {
            /* code is not in table */
            dstsize += table[oldcode].len + 1;
        }
        table[tablesize++].len = table[oldcode].len + 1;

        /* increase bit-width if necessary */
        if (msb) {
            /* early change */
            switch (tablesize)
            {
                case 511:
                    bitw = 10; shr = 22; mask = 4290772992u;
                    break;
                case 1023:
                    bitw = 11; shr = 21; mask = 4292870144u;
                    break;
                case 2047:
                    bitw = 12; shr = 20; mask = 4293918720u;
            }
        }
        else {
            /* late change */
            switch (tablesize)
            {
                case 512:
                    bitw = 10; mask = 1023;
                    break;
                case 1024:
                    bitw = 11; mask = 2047;
                    break;
                case 2048:
                    bitw = 12; mask = 4095;
                    break;
                /* continue with 12-bit for tablesize >= 4096 */
            }
        }

        oldcode = code;
    }

    if (buffersize > buffer_size) {
        buffer_size = buffersize;
    }

    if (buffer_size > handle->buffersize) {
        if (_lzw_alloc_buffer(handle, buffer_size) < 0) {
            return IMCD_MEMORY_ERROR;
        }
    }
    return dstsize;
}


/* Decode LZW compressed string. */
ssize_t imcd_lzw_decode(
    imcd_lzw_handle_t* handle,
    const uint8_t* src,
    const ssize_t srcsize,
    uint8_t* dst,
    const ssize_t dstsize)
{
    imcd_lzw_table_t* table;
    uint8_t* buffer;
    uint32_t tablesize = 258;
    uint32_t code = 0;
    uint32_t oldcode = 0;
    uint32_t shr = 23;
    uint32_t mask = 4286578688u;
    uint64_t bitw = 9;
    uint64_t bitcount = 0;
    ssize_t buffersize = 0;
    ssize_t remaining = dstsize;
    const uint64_t srcbitsize = srcsize * 8;  /* size in bits */
    ssize_t i;
    int msb = 1;  /* bit ordering of codes */

    if ((handle == NULL) ||
        (src == NULL) || (srcsize < 0) ||
        (dst == NULL) || (dstsize < 0)) {
        return IMCD_VALUE_ERROR;
    }
    if ((srcsize == 0) || (dstsize == 0)) {
        return 0;
    }
    if (srcsize < 2) {
        return IMCD_LZW_INVALID;
    }

    table = handle->table;
    buffer = handle->buffer;

    if ((*src == 0) && (*(src + 1) & 1)) {
        msb = 0;
        mask = 511;
    }
    else if ((*src != 128) || ((*(src + 1) & 128))) {
        /* compressed string must begin with CLEAR code */
        return IMCD_LZW_INVALID;
    }

    for (i = 0; i < LZW_TABLESIZE; i++) {
        table[i].len = 1;
    }

    if (handle->buffersize == 0) {
        if (_lzw_alloc_buffer(handle, LZW_BUFFERSIZE) < 0) {
            return IMCD_MEMORY_ERROR;
        }
    }
    buffersize = handle->buffersize;

    while (remaining > 0) {

        if (msb) {
            LZW_GET_NEXT_CODE_MSB
        }
        else {
            LZW_GET_NEXT_CODE_LSB
        }

        if (code == LZW_EOI) break;

        if (code == LZW_CLEAR) {
            /* initialize table and switch to 9-bit */
            tablesize = 258;
            bitw = 9;
            shr = 23;

            buffer = handle->buffer;
            buffersize = handle->buffersize;

            if (msb) {
                mask = 4286578688u;
                do { LZW_GET_NEXT_CODE_MSB } while (code == LZW_CLEAR);
            }
            else {
                mask = 511;
                do { LZW_GET_NEXT_CODE_LSB } while (code == LZW_CLEAR);
            }

            if (code == LZW_EOI) break;

            remaining--;

            *dst++ = code;
            oldcode = code;
            continue;
        }

        if (tablesize >= LZW_TABLESIZE) {
            return IMCD_LZW_TABLE_TOO_SMALL;
        }

        if (code < tablesize) {
            /* code is in table */
            buffersize -= table[oldcode].len + 1;
            if (buffersize < 0) {
                /* reallocate buffer */
                const uint8_t* oldbuffer = handle->buffer;
                const ssize_t bufferlen = buffer - oldbuffer;

                if (
                    _lzw_alloc_buffer(handle, handle->buffersize - buffersize)
                    <= 0
                    )
                {
                    return IMCD_MEMORY_ERROR;
                }
                if (handle->buffer != oldbuffer) {
                    /* correct pointers */
                    uint32_t i;
                    const ssize_t bufferdiff = handle->buffer - oldbuffer;

                    for (i = 256; i < tablesize; i++) {
                        if ((table[i].buf >= oldbuffer) &&
                            (table[i].buf < buffer))
                        {
                            table[i].buf += bufferdiff;
                        }
                    }
                }
                buffersize =
                    handle->buffersize - bufferlen - table[oldcode].len - 1;
                buffer = handle->buffer + bufferlen;
            }

            /* decompressed.append(table[code]) */
            if (code < 256) {
                remaining--;
                *dst++ = code;
            }
            else {
                uint8_t* pstr = table[code].buf;
                ssize_t len = table[code].len;
                if (len > remaining) {
                    len = remaining;
                }
                remaining -= len;
                for (i = 0; i < len; i++) {
                    *dst++ = *pstr++;
                }
            }
            /* table.append(table[oldcode] + table[code][0]) */
            table[tablesize].buf = buffer;
            if (oldcode < 256) {
                *buffer++ = oldcode;
            }
            else {
                uint8_t* pstr = table[oldcode].buf;
                for (i = 0; i < table[oldcode].len; i++) {
                    *buffer++ = *pstr++;
                }
            }
            if (code < 256) {
                *buffer++ = code;
            }
            else {
                *buffer++ = table[code].buf[0];
            }
        }
        else if (code > tablesize) {
            /* return dstsize - remaining; */
            return IMCD_LZW_CORRUPT;
        }
        else {
            /* code is not in table */
            /* outstring = table[oldcode] + table[oldcode][0] */
            /* decompressed.append(outstring) */
            /* table.append(outstring) */
            table[tablesize].buf = dst;
            if (oldcode < 256) {
                remaining--;
                *dst++ = oldcode;
                if (--remaining < 0) break;
                *dst++ = oldcode;
            }
            else {
                uint8_t* pstr = table[oldcode].buf;
                ssize_t len = table[oldcode].len;
                if (len > remaining) {
                    len = remaining;
                }
                remaining -= len;
                for (i = 0; i < len; i++) {
                    *dst++ = *pstr++;
                }
                if (--remaining < 0) break;
                *dst++ = table[oldcode].buf[0];
            }
        }
        table[tablesize++].len = table[oldcode].len + 1;
        oldcode = code;

        /* increase bit-width if necessary */
        if (msb) {
            /* early change */
            switch (tablesize)
            {
                case 511:
                    bitw = 10; shr = 22; mask = 4290772992u;
                    break;
                case 1023:
                    bitw = 11; shr = 21; mask = 4292870144u;
                    break;
                case 2047:
                    bitw = 12; shr = 20; mask = 4293918720u;
            }
        }
        else {
            /* late change */
            switch (tablesize)
            {
                case 512:
                    bitw = 10; mask = 1023;
                    break;
                case 1024:
                    bitw = 11; mask = 2047;
                    break;
                case 2048:
                    bitw = 12; mask = 4095;
                    break;
            }
        }
    }

    return dstsize;
}
