/* imagecodecs/imcd.c */

/*
Copyright (c) 2008-2026, Christoph Gohlke.
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
#include <stdbool.h>
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

/* http://graphics.stanford.edu/~seander/bithacks.html#ByteSwapObvious

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
*/

#ifdef _MSC_VER
#  include <intrin.h>
#  define SWAP2BYTES(x) _byteswap_ushort(x)
#  define SWAP4BYTES(x) _byteswap_ulong(x)
#  define SWAP8BYTES(x) _byteswap_uint64(x)
#else
#  define SWAP2BYTES(x) __builtin_bswap16(x)
#  define SWAP4BYTES(x) __builtin_bswap32(x)
#  define SWAP8BYTES(x) __builtin_bswap64(x)
#endif


/* Inplace swap bytes. */
void imcd_swapbytes(
    void* src,
    const ssize_t srcsize,  /* number items, not bytes */
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


/* Return mask for number of bits (MSB-aligned, e.g. bps=3 -> 0xE0). */
uint8_t imcd_bitmask(
    const int bps)
{
    if ((bps <= 0) || (bps > 8)) {
        return 0;
    }
    return (uint8_t)(0xFFu << (8 - bps));
}

uint16_t imcd_bitmask2(
    const int bps)
{
    if ((bps < 0) || (bps > 16)) {
        return 0;
    }
    /* 0xFFFFu is unsigned int (>=32 bits). right-shifting by 0..16 is safe */
    return (uint16_t)(0xFFFFu >> (16 - bps));
}


/********************************** Delta ************************************/

#define DELTA_FLOAT(dtype) \
{ \
    if (decode) { \
        dtype sum = *(dtype*)srcptr; \
        if (inplace && (stride == sizeof(dtype)) \
                    && (srcstride == sizeof(dtype))) { \
        /* decode contiguous in-place */ \
            dtype* qsrc = (dtype*)srcptr; \
            for (i = 1; i < size; i++) { \
                sum += qsrc[i]; \
                qsrc[i] = sum; \
            } \
        } \
        else { \
            *(dtype*)dstptr = sum; \
            for (i = 1; i < size; i++) { \
                dstptr += stride; \
                srcptr += srcstride; \
                sum += *(dtype*)srcptr; \
                *(dtype*)dstptr = sum; \
            } \
        } \
    } \
    else { \
        if (!inplace && (stride == sizeof(dtype)) \
                     && (srcstride == sizeof(dtype))) { \
        /* encode contiguous not-in-place: typed indexing enables  \
           auto-vectorization of dst[i] = src[i] - src[i-1] */ \
            const dtype* qsrc = (const dtype*)srcptr; \
            dtype* qdst = (dtype*)dstptr; \
            qdst[0] = qsrc[0]; \
            for (i = 1; i < size; i++) { \
                qdst[i] = qsrc[i] - qsrc[i - 1]; \
            } \
        } \
        else if (inplace && (stride == sizeof(dtype)) \
                         && (srcstride == sizeof(dtype))) { \
        /* encode contiguous in-place */ \
            dtype* qsrc = (dtype*)srcptr; \
            for (i = size - 1; i > 0; i--) { \
                qsrc[i] = qsrc[i] - qsrc[i - 1]; \
            } \
        } \
        else { \
            dtype src0, src1; \
            src0 = *(dtype*)srcptr; \
            *(dtype*)dstptr = src0; \
            for (i = 1; i < size; i++) { \
                dstptr += stride; \
                srcptr += srcstride; \
                src1 = *(dtype*)srcptr; \
                *(dtype*)dstptr = src1 - src0; \
                src0 = src1; \
            } \
        } \
    } \
}

#define DELTA_INT(dtype) \
{ \
    if (decode) { \
        if (inplace && (stride == sizeof(dtype)) \
                    && (srcstride == sizeof(dtype))) { \
        /* decode contiguous in-place */ \
            dtype* qsrc = (dtype*)srcptr; \
            uint64_t acc = (uint64_t)qsrc[0]; \
            for (i = 1; i < size - 1; i += 2) { \
                acc += (uint64_t)qsrc[i]; \
                qsrc[i] = (dtype)acc; \
                acc += (uint64_t)qsrc[i + 1]; \
                qsrc[i + 1] = (dtype)acc; \
            } \
            if (i < size) { \
                acc += (uint64_t)qsrc[i]; \
                qsrc[i] = (dtype)acc; \
            } \
        } \
        else { \
            dtype sum = *(dtype*)srcptr; \
            *(dtype*)dstptr = sum; \
            for (i = 1; i < size; i++) { \
                dstptr += stride; \
                srcptr += srcstride; \
                sum += *(dtype*)srcptr; \
                *(dtype*)dstptr = sum; \
            } \
        } \
    } \
    else { \
        if (!inplace && (stride == sizeof(dtype)) \
                     && (srcstride == sizeof(dtype))) { \
        /* encode contiguous not-in-place: typed indexing enables  \
           auto-vectorization of dst[i] = src[i] - src[i-1] */ \
            const dtype* qsrc = (const dtype*)srcptr; \
            dtype* qdst = (dtype*)dstptr; \
            qdst[0] = qsrc[0]; \
            for (i = 1; i < size; i++) { \
                qdst[i] = qsrc[i] - qsrc[i - 1]; \
            } \
        } \
        else if (inplace && (stride == sizeof(dtype)) \
                         && (srcstride == sizeof(dtype))) { \
        /* encode contiguous in-place */ \
            dtype* qsrc = (dtype*)srcptr; \
            for (i = size - 1; i > 0; i--) { \
                qsrc[i] = qsrc[i] - qsrc[i - 1]; \
            } \
        } \
        else { \
            dtype src0, src1; \
            src0 = *(dtype*)srcptr; \
            *(dtype*)dstptr = src0; \
            for (i = 1; i < size; i++) { \
                dstptr += stride; \
                srcptr += srcstride; \
                src1 = *(dtype*)srcptr; \
                *(dtype*)dstptr = src1 - src0; \
                src0 = src1; \
            } \
        } \
    } \
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
    const bool decode)
{
    const bool inplace = (dst == NULL) || (dst == src);
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
            DELTA_INT(uint8_t);
            break;
        case 2:
            DELTA_INT(uint16_t);
            break;
        case 4:
            DELTA_INT(uint32_t);
            break;
        case 8:
            DELTA_INT(uint64_t);
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
    const bool decode)
{
    const bool inplace = (dst == NULL) || (dst == src);
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
                    DELTA_INT(uint8_t);
                    break;
                case 2:
                    DELTA_INT(uint16_t);
                    break;
                case 4:
                    DELTA_INT(uint32_t);
                    break;
                case 8:
                    DELTA_INT(uint64_t);
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
                    DELTA_INT(int8_t);
                    break;
                case 2:
                    DELTA_INT(int16_t);
                    break;
                case 4:
                    DELTA_INT(int32_t);
                    break;
                case 8:
                    DELTA_INT(int64_t);
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
                    DELTA_FLOAT(float);
                    break;
                case 8:
                    DELTA_FLOAT(double);
                    break;
                default:
                    return IMCD_VALUE_ERROR;
            }
            break;
        }
        default:
            return IMCD_VALUE_ERROR;
    }
    return size;
}

/********************************* XOR Delta *********************************/

#define XOR(dtype) \
{ \
    dtype prev, t; \
    if (decode) { \
        if (inplace && (stride == sizeof(dtype)) \
                    && (srcstride == sizeof(dtype))) { \
        /* decode contiguous in-place */ \
            dtype* psrc = (dtype*)srcptr; \
            uint64_t acc = (uint64_t)psrc[0]; \
            for (i = 1; i < size - 1; i += 2) { \
                acc ^= (uint64_t)psrc[i]; \
                psrc[i] = (dtype)acc; \
                acc ^= (uint64_t)psrc[i + 1]; \
                psrc[i + 1] = (dtype)acc; \
            } \
            if (i < size) { \
                acc ^= (uint64_t)psrc[i]; psrc[i] = (dtype)acc; \
            } \
        } \
        else { \
            *(dtype*)dstptr = prev = (*(dtype*)srcptr); \
            for (i = 1; i < size; i++) { \
                dstptr += stride; \
                srcptr += srcstride; \
                *(dtype*)dstptr = prev = (*(dtype*)srcptr) ^ prev; \
            } \
        } \
    } \
    else { \
        if (!inplace && (stride == sizeof(dtype)) \
                     && (srcstride == sizeof(dtype))) { \
        /* encode contiguous not-in-place: typed indexing enables  \
           auto-vectorization of dst[i] = src[i] ^ src[i-1] */ \
            const dtype* qsrc = (const dtype*)srcptr; \
            dtype* qdst = (dtype*)dstptr; \
            qdst[0] = qsrc[0]; \
            for (i = 1; i < size; i++) { \
                qdst[i] = qsrc[i] ^ qsrc[i - 1]; \
            } \
        } \
        else if (inplace && (stride == sizeof(dtype)) \
                         && (srcstride == sizeof(dtype))) { \
        /* encode contiguous in-place */ \
            dtype* qsrc = (dtype*)srcptr; \
            for (i = size - 1; i > 0; i--) { \
                qsrc[i] = qsrc[i] ^ qsrc[i - 1]; \
            } \
        } \
        else { \
            *(dtype*)dstptr = prev = (*(dtype*)srcptr); \
            for (i = 1; i < size; i++) { \
                dstptr += stride; \
                srcptr += srcstride; \
                t = (*(dtype*)srcptr); \
                *(dtype*)dstptr = t ^ prev; \
                prev = t; \
            } \
        } \
    } \
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
    const bool decode)
{
    const bool inplace = (dst == NULL) || (dst == src);
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

/*
#define REVERSE_BITS(b) ((uint8_t)(((((uint8_t)b) * 0x80200802ULL) \
                                   & 0x0884422110ULL) * 0x0101010101ULL >> 32))
*/

#define REVERSE_BITS(b) (imcd_bitreverse_lut[(uint8_t)(b)])

static const uint8_t imcd_bitreverse_lut[256] = {
    0x00, 0x80, 0x40, 0xC0, 0x20, 0xA0, 0x60, 0xE0,
    0x10, 0x90, 0x50, 0xD0, 0x30, 0xB0, 0x70, 0xF0,
    0x08, 0x88, 0x48, 0xC8, 0x28, 0xA8, 0x68, 0xE8,
    0x18, 0x98, 0x58, 0xD8, 0x38, 0xB8, 0x78, 0xF8,
    0x04, 0x84, 0x44, 0xC4, 0x24, 0xA4, 0x64, 0xE4,
    0x14, 0x94, 0x54, 0xD4, 0x34, 0xB4, 0x74, 0xF4,
    0x0C, 0x8C, 0x4C, 0xCC, 0x2C, 0xAC, 0x6C, 0xEC,
    0x1C, 0x9C, 0x5C, 0xDC, 0x3C, 0xBC, 0x7C, 0xFC,
    0x02, 0x82, 0x42, 0xC2, 0x22, 0xA2, 0x62, 0xE2,
    0x12, 0x92, 0x52, 0xD2, 0x32, 0xB2, 0x72, 0xF2,
    0x0A, 0x8A, 0x4A, 0xCA, 0x2A, 0xAA, 0x6A, 0xEA,
    0x1A, 0x9A, 0x5A, 0xDA, 0x3A, 0xBA, 0x7A, 0xFA,
    0x06, 0x86, 0x46, 0xC6, 0x26, 0xA6, 0x66, 0xE6,
    0x16, 0x96, 0x56, 0xD6, 0x36, 0xB6, 0x76, 0xF6,
    0x0E, 0x8E, 0x4E, 0xCE, 0x2E, 0xAE, 0x6E, 0xEE,
    0x1E, 0x9E, 0x5E, 0xDE, 0x3E, 0xBE, 0x7E, 0xFE,
    0x01, 0x81, 0x41, 0xC1, 0x21, 0xA1, 0x61, 0xE1,
    0x11, 0x91, 0x51, 0xD1, 0x31, 0xB1, 0x71, 0xF1,
    0x09, 0x89, 0x49, 0xC9, 0x29, 0xA9, 0x69, 0xE9,
    0x19, 0x99, 0x59, 0xD9, 0x39, 0xB9, 0x79, 0xF9,
    0x05, 0x85, 0x45, 0xC5, 0x25, 0xA5, 0x65, 0xE5,
    0x15, 0x95, 0x55, 0xD5, 0x35, 0xB5, 0x75, 0xF5,
    0x0D, 0x8D, 0x4D, 0xCD, 0x2D, 0xAD, 0x6D, 0xED,
    0x1D, 0x9D, 0x5D, 0xDD, 0x3D, 0xBD, 0x7D, 0xFD,
    0x03, 0x83, 0x43, 0xC3, 0x23, 0xA3, 0x63, 0xE3,
    0x13, 0x93, 0x53, 0xD3, 0x33, 0xB3, 0x73, 0xF3,
    0x0B, 0x8B, 0x4B, 0xCB, 0x2B, 0xAB, 0x6B, 0xEB,
    0x1B, 0x9B, 0x5B, 0xDB, 0x3B, 0xBB, 0x7B, 0xFB,
    0x07, 0x87, 0x47, 0xC7, 0x27, 0xA7, 0x67, 0xE7,
    0x17, 0x97, 0x57, 0xD7, 0x37, 0xB7, 0x77, 0xF7,
    0x0F, 0x8F, 0x4F, 0xCF, 0x2F, 0xAF, 0x6F, 0xEF,
    0x1F, 0x9F, 0x5F, 0xDF, 0x3F, 0xBF, 0x7F, 0xFF,
};


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

    if (
        (srcsize < 0) || (dstsize < 0) || (itemsize <= 0) ||
        (srcsize % itemsize) || (dstsize % itemsize))
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

/* Encode or decode byteshuffle or floating point predictor. */
ssize_t imcd_byteshuffle(
    void* src,
    const ssize_t srcsize,  /* size in bytes */
    const ssize_t srcstride,
    void* dst,
    const ssize_t dstsize,  /* size in bytes */
    const ssize_t dststride,
    const ssize_t itemsize,
    const ssize_t samples,
    const char byteorder,
    const bool delta,
    const bool decode)
{
    uint8_t* srcptr = (uint8_t*)src;
    uint8_t* dstptr = (uint8_t*)dst;
    const ssize_t size = itemsize > 0 ? MIN_(srcsize, dstsize) / itemsize : -1;
    ssize_t i, j;

    if (
        (src == NULL) || (dst == NULL) || (src == dst) ||
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
        if (delta) {
            /* undo byte differencing; separate for interleaved samples */
            for (i = samples; i < size * itemsize; i++) {
                srcptr[i] += srcptr[i - samples];
            }
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
        if (delta) {
            /* restore byte differencing in src */
            for (i = size * itemsize - 1; i >= samples; i--) {
                srcptr[i] -= srcptr[i - samples];
            }
        }
    }
    else {
        /* encode */
        if (
            (dststride != itemsize) || (srcstride < 0) ||
            (srcstride % itemsize))
        {
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
        if (delta) {
            /* byte differencing; separate for interleaved samples */
            for (i = size * itemsize - 1; i >= samples; i--) {
                dstptr[i] -= dstptr[i - samples];
            }
        }
    }
    return size;
}


/********************************* PackBits **********************************/

/* Section 9: PackBits Compression. TIFF Revision 6.0 Final. 1992

TIFF compression type 32773, a simple byte-oriented run-length scheme.

Apple Technical Note TN1023. Understanding PackBits. Feb 1, 1996

*/

/* Return length of uncompressed PackBits. */
ssize_t imcd_packbits_decode_size(
    const uint8_t* src,
    const ssize_t srcsize)
{
    uint8_t* srcptr = (uint8_t*)src;
    const uint8_t* srcend = srcptr + srcsize;
    ssize_t dstsize = 0;
    ssize_t n;

    if ((srcptr == NULL) || (srcsize < 0)) {
        return IMCD_VALUE_ERROR;
    }
    while (srcptr < srcend) {
        n = (ssize_t)(*srcptr++) + 1;
        if ((n == 1) && (srcptr == srcend)) {
            /* DICOM pad byte */
            break;
        }
        if (n < 129) {
            /* literal */
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
    const ssize_t dstsize,
    const ssize_t dststride)
{
    uint8_t* srcptr = (uint8_t*)src;
    uint8_t* dstptr = dst;
    const uint8_t* srcend = srcptr + srcsize;
    const uint8_t* dstend = dstptr + dstsize;
    uint8_t e;
    ssize_t n;

    if (
        (srcptr == NULL) || (srcsize < 0) || (dstptr == NULL) ||
        (dstsize < 0))
    {
        return IMCD_VALUE_ERROR;
    }

    if (dststride == 1) {
        /* fast path: contiguous output */
        while (srcptr < srcend) {
            n = (ssize_t)(*srcptr++) + 1;
            if ((n == 1) && (srcptr == srcend)) {
                /* DICOM pad byte */
                break;
            }
            if (n < 129) {
                /* literal */
                if (srcptr + n > srcend) {
                    return IMCD_INPUT_CORRUPT;
                }
                if (dstptr + n > dstend) {
                    return IMCD_OUTPUT_TOO_SMALL;
                }
                memcpy(dstptr, srcptr, n);
                srcptr += n;
                dstptr += n;
            }
            else if (n > 129) {
                /* replicate */
                n = 258 - n;
                if (srcptr >= srcend) {
                    return IMCD_INPUT_CORRUPT;
                }
                if (dstptr + n > dstend) {
                    return IMCD_OUTPUT_TOO_SMALL;
                }
                memset(dstptr, *srcptr++, n);
                dstptr += n;
            }
            /* else if (n == 129) {NOP} */
        }
    }
    else {
        /* strided output */
        while (srcptr < srcend) {
            n = (ssize_t)(*srcptr++) + 1;
            if ((n == 1) && (srcptr == srcend)) {
                /* DICOM pad byte */
                break;
            }
            if (n < 129) {
                /* literal */
                if (srcptr + n > srcend) {
                    return IMCD_INPUT_CORRUPT;
                }
                if (dstptr + (n - 1) * dststride >= dstend) {
                    return IMCD_OUTPUT_TOO_SMALL;
                }
                while (n--) {
                    *dstptr = *srcptr++;
                    dstptr += dststride;
                }
            }
            else if (n > 129) {
                /* replicate */
                n = 258 - n;
                if (srcptr >= srcend) {
                    return IMCD_INPUT_CORRUPT;
                }
                if (dstptr + (n - 1) * dststride >= dstend) {
                    return IMCD_OUTPUT_TOO_SMALL;
                }
                e = *srcptr++;
                while (n--) {
                    *dstptr = e;
                    dstptr += dststride;
                }
            }
            /* else if (n == 129) {NOP} */
        }
    }
    return ((ssize_t)(dstptr - dst)) / dststride;
}

/* Return maximum length of PackBits compressed sequence. */
ssize_t imcd_packbits_encode_size(
    const ssize_t srcsize)
{
    return srcsize + (srcsize + 127) / 128;
}


/* Return pointer to next replicate run. */
inline uint8_t* _packbits_next_replicate(
    uint8_t* srcptr,
    const uint8_t* srcend)
{
    while (++srcptr < srcend) {
        if (srcptr[-1] == srcptr[0]) {
            return srcptr - 1;
        }
    }
    return NULL;  /* no replicate */
}


/* Return length of replicate run. */
inline ssize_t _packbits_replicate_length(
    const uint8_t* src,
    const uint8_t* srcend)
{
    uint8_t* srcptr = (uint8_t*)src;
    const uint8_t value = (srcptr < srcend) ? *srcptr : 0;

    while ((++srcptr < srcend) && (value == *srcptr)) {
        ;
    }
    return (ssize_t)(srcptr - src);
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
    uint8_t* dupptr = NULL;
    const ssize_t maxdst = (srcsize + (srcsize + 127) / 128);
    const uint8_t* srcend = srcptr + srcsize;
    const uint8_t* dstend = dstptr + MIN_(dstsize, maxdst) - 1;
    ssize_t replicate, literal;

    if (
        (srcptr == NULL) || (srcsize < 0) || (dstptr == NULL) ||
        (dstsize < 0))
    {
        return IMCD_VALUE_ERROR;
    }
    if ((srcsize == 0) || (dstsize == 0)) {
        return 0;
    }

    while (srcptr < srcend) {
        /* limit lookahead to one literal packet (max 128 bytes) */
        const uint8_t* lkend = srcptr + 129 < srcend ? srcptr + 129 : srcend;
        dupptr = _packbits_next_replicate(srcptr, lkend);
        if (srcptr == dupptr) {
            /* replicate */
            replicate = _packbits_replicate_length(dupptr, srcend);
            replicate = MIN_(128, replicate);
            if (dstptr < dstend) {
                *dstptr++ = (uint8_t)((int)257 - (int)replicate);
                *dstptr++ = *srcptr;
            }
            else {
                dstptr = NULL;
                break;
            }
            srcptr += replicate;
            continue;
        }
        if (dupptr == NULL) {
            /* no more replicate runs found within lookahead */
            literal = srcend - srcptr;
        }
        else {
            /* try skip next replicate run < 3 */
            replicate = _packbits_replicate_length(dupptr, srcend);
            if (replicate < 3) {
                uint8_t* nextsrc = dupptr + replicate;
                uint8_t* nextdup = _packbits_next_replicate(nextsrc, lkend);
                /* discard 2-byte run? */
                dupptr = (nextdup > nextsrc) ? nextdup : dupptr;
            }
            literal = dupptr - srcptr;
        }
        literal = MIN_(128, literal);
        if (dstptr + literal < dstend) {
            *dstptr++ = (uint8_t)(literal - 1);
            while (literal--) {
                *dstptr++ = *srcptr++;
            }
            /* memcpy(dstptr, srcptr, literal); */
            /* dstptr += literal; */
            /* srcptr += literal; */
        }
        else {
            dstptr = NULL;
            break;
        }
    }

    if (dstptr == NULL) {
        if (dstsize < maxdst) {
            return IMCD_OUTPUT_TOO_SMALL;
        }
        /* encoding exceeded maximum literal-only length */
        /* re-encode with only literal packets */
        srcptr = (uint8_t*)src;
        dstptr = dst;
        while (srcptr < srcend) {
            literal = MIN_(128, srcend - srcptr);
            *dstptr++ = (uint8_t)(literal - 1);
            while (literal--) {
                *dstptr++ = *srcptr++;
            }
            /* memcpy(dstptr, srcptr, literal); */
            /* dstptr += literal; */
            /* srcptr += literal; */
        }
    }
    return (ssize_t)(dstptr - dst);
}


/***************************** Packed Integers *******************************/

/* Section 8: Compression. TIFF Revision 6.0 Final. 1992

TIFF compression scheme 1: No compression, but pack data into bytes as
tightly as possible leaving no unused bits except at the end of a row.

*/

/* Unpack sequence of packed 1-64 bit integers.

Input src array should be padded to the next 16, 32 or 64-bit boundary
if itemsize not in {1, 2, 4, 8, 16, 24, 32, 64}.

*/
ssize_t imcd_packints_decode(
    const uint8_t* src,
    const ssize_t srcsize,  /** size of src in bytes */
    uint8_t* dst,  /** buffer to store unpacked items */
    const ssize_t items,  /** number of items to unpack */
    const int bps  /** number of bits in integer */
)
{
    ssize_t i, j, k;
    ssize_t itemsize;
    uint8_t value;

    if ((srcsize == 0) || (items == 0)) {
        return 0;
    }

    /* input validation is done in wrapper function */
    itemsize = (ssize_t)(ceil(bps / 8.0));
    itemsize = itemsize < 3 ? itemsize : itemsize > 4 ? 8 : 4;
    if (items < 0 || items > (ssize_t)((SIZE_MAX >> 1) / (size_t)itemsize)) {
        return IMCD_VALUE_ERROR;
    }
    switch (bps)
    {
        case 8:
        case 16:
        case 32:
        case 64:
            memcpy(dst, src, (size_t)(items * itemsize));
            return items;
        case 1:
            for (i = 0; i < items / 8; i++) {
                value = src[i];
                dst[8 * i + 0] = (value & (uint8_t)(128)) >> 7;
                dst[8 * i + 1] = (value & (uint8_t)(64)) >> 6;
                dst[8 * i + 2] = (value & (uint8_t)(32)) >> 5;
                dst[8 * i + 3] = (value & (uint8_t)(16)) >> 4;
                dst[8 * i + 4] = (value & (uint8_t)(8)) >> 3;
                dst[8 * i + 5] = (value & (uint8_t)(4)) >> 2;
                dst[8 * i + 6] = (value & (uint8_t)(2)) >> 1;
                dst[8 * i + 7] = (value & (uint8_t)(1));
            }
            if (items % 8) {
                j = (items / 8) * 8;
                value = src[items / 8];
                switch (items % 8)
                {
                    case 7: dst[j + 6] = (value & (uint8_t)(2)) >> 1;
                    case 6: dst[j + 5] = (value & (uint8_t)(4)) >> 2;
                    case 5: dst[j + 4] = (value & (uint8_t)(8)) >> 3;
                    case 4: dst[j + 3] = (value & (uint8_t)(16)) >> 4;
                    case 3: dst[j + 2] = (value & (uint8_t)(32)) >> 5;
                    case 2: dst[j + 1] = (value & (uint8_t)(64)) >> 6;
                    case 1: dst[j]   = (value & (uint8_t)(128)) >> 7;
                }
            }
            return items;
        case 2:
            for (i = 0; i < items / 4; i++) {
                value = src[i];
                dst[4 * i + 0] = (value & (uint8_t)(192)) >> 6;
                dst[4 * i + 1] = (value & (uint8_t)(48)) >> 4;
                dst[4 * i + 2] = (value & (uint8_t)(12)) >> 2;
                dst[4 * i + 3] = (value & (uint8_t)(3));
            }
            if (items % 4) {
                j = (items / 4) * 4;
                value = src[items / 4];
                switch (items % 4)
                {
                    case 3: dst[j + 2] = (value & (uint8_t)(12)) >> 2;
                    case 2: dst[j + 1] = (value & (uint8_t)(48)) >> 4;
                    case 1: dst[j]   = (value & (uint8_t)(192)) >> 6;
                }
            }
            return items;
        case 4:
            for (i = 0; i < items / 2; i++) {
                value = src[i];
                dst[2 * i + 0] = (value & (uint8_t)(240)) >> 4;
                dst[2 * i + 1] = (value & (uint8_t)(15));
            }
            if (items % 2) {
                dst[items - 1] = (src[items / 2] & (uint8_t)(240)) >> 4;
            }
            return items;
        case 24:
            for (i = 0; i < items; i++) {
                dst[4 * i + 0] = src[3 * i + 2]; /* LSB */
                dst[4 * i + 1] = src[3 * i + 1];
                dst[4 * i + 2] = src[3 * i + 0]; /* MSB */
                dst[4 * i + 3] = 0;
            }
            return items;
    }
    /* 3, 5, 6, 7 */
    if (bps < 8) {
        const uint32_t mask8 = ((uint32_t)1 << bps) - 1;
        uint32_t acc = 0;
        int nacc = 0;
        j = k = 0;
        for (i = 0; i < items; i++) {
            while (nacc < bps) {
                acc = (acc << 8) | (j < srcsize ? src[j++] : 0);
                nacc += 8;
            }
            nacc -= bps;
            dst[k++] = (uint8_t)((acc >> nacc) & mask8);
        }
        return items;
    }
    /* 9, 10, 11, 12, 13, 14, 15 */
    if (bps < 16) {
        const uint64_t mask16 = ((uint64_t)1 << bps) - 1;
        uint64_t acc = 0;
        uint16_t v16;
        int nacc = 0;
        j = k = 0;
        for (i = 0; i < items; i++) {
            while (nacc < bps) {
                acc = (acc << 8) | (j < srcsize ? src[j++] : 0);
                nacc += 8;
            }
            nacc -= bps;
            v16 = (uint16_t)((acc >> nacc) & mask16);
            memcpy(dst + k, &v16, 2);
            k += 2;
        }
        return items;
    }
    /* 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31 */
    if (bps < 32) {
        const uint64_t mask32 = ((uint64_t)1 << bps) - 1;
        uint64_t acc = 0;
        uint32_t v32;
        int nacc = 0;
        j = k = 0;
        for (i = 0; i < items; i++) {
            while (nacc < bps) {
                acc = (acc << 8) | (j < srcsize ? src[j++] : 0);
                nacc += 8;
            }
            nacc -= bps;
            v32 = (uint32_t)((acc >> nacc) & mask32);
            memcpy(dst + k, &v32, 4);
            k += 4;
        }
        return items;
    }
    return IMCD_ERROR;
}


/* Pack sequence of 1-64 bit integers. */
ssize_t imcd_packints_encode(
    const uint8_t* src,
    const ssize_t srcsize,  /** size of src in bytes */
    uint8_t* dst,  /** buffer to store packed items */
    const ssize_t items,  /** number of items to pack */
    const int bps  /** number of bits in integer */
)
{
    ssize_t i, j, k;
    ssize_t itemsize;
    uint8_t value;

    if ((src == NULL) || (dst == NULL) || (srcsize < 0) || (items <= 0)) {
        return IMCD_VALUE_ERROR;
    }
    if (srcsize == 0) {
        return 0;
    }

    itemsize = (ssize_t)(ceil(bps / 8.0));
    itemsize = itemsize < 3 ? itemsize : itemsize > 4 ? 8 : 4;
    if (items > (ssize_t)((SIZE_MAX >> 1) / (size_t)itemsize)) {
        return IMCD_VALUE_ERROR;
    }

    switch (bps)
    {
        case 8:
        case 16:
        case 32:
        case 64:
            memcpy(dst, src, (size_t)(items * itemsize));
            return items;
        case 1:
            for (i = 0, j = 0; j < items / 8; j++) {
                value = 0;
                value |= (src[i++] << 7) & (uint8_t)(128);
                value |= (src[i++] << 6) & (uint8_t)(64);
                value |= (src[i++] << 5) & (uint8_t)(32);
                value |= (src[i++] << 4) & (uint8_t)(16);
                value |= (src[i++] << 3) & (uint8_t)(8);
                value |= (src[i++] << 2) & (uint8_t)(4);
                value |= (src[i++] << 1) & (uint8_t)(2);
                value |= (src[i++] << 0) & (uint8_t)(1);
                dst[j] = value;
            }
            if (items % 8) {
                int r = items % 8, m;
                value = 0;
                for (m = 0; m < r; m++) {
                    value |= (uint8_t)((src[i++] & 1) << (7 - m));
                }
                dst[j] = value;
            }
            return items;
    }
    /* 2, 3, 4, 5, 6, 7 */
    if (bps < 8) {
        uint16_t val = 0;
        uint8_t mask = (uint8_t)((1 << bps) - 1);
        int fill = 0;
        k = j = 0;
        for (i = 0; i < items; i++) {
            val |= (uint16_t)(src[k++] & mask) << (16 - fill - bps);
            fill += bps;
            while (fill >= 8) {
                dst[j++] = (uint8_t)(val >> 8);
                val <<= 8;
                fill -= 8;
            }
        }
        if (fill > 0) {
            dst[j] = (uint8_t)(val >> 8);
        }
        return items;
    }
    /* 9, 10, 11, 12, 13, 14, 15 */
    if (bps < 16) {
        uint32_t val = 0;
        uint16_t mask = imcd_bitmask2(bps);
        int fill = 0;
        k = j = 0;
        for (i = 0; i < items; i++) {
            uint32_t item = (uint32_t)src[k] | ((uint32_t)src[k + 1] << 8);
            k += 2;
            val |= (uint32_t)(item & mask) << (32 - fill - bps);
            fill += bps;
            while (fill >= 8) {
                dst[j++] = (uint8_t)(val >> 24);
                val <<= 8;
                fill -= 8;
            }
        }
        if (fill > 0) {
            dst[j] = (uint8_t)(val >> 24);
        }
        return items;
    }
    /* 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 */
    if (bps < 32) {
        uint64_t val = 0;
        uint64_t mask = ((uint64_t)1 << bps) - 1;
        int fill = 0;
        k = j = 0;
        for (i = 0; i < items; i++) {
            uint64_t item = (uint64_t)src[k] | ((uint64_t)src[k + 1] << 8)
                | ((uint64_t)src[k + 2] << 16) | ((uint64_t)src[k + 3] << 24);
            k += 4;
            val |= (item & mask) << (64 - fill - bps);
            fill += bps;
            while (fill >= 8) {
                dst[j++] = (uint8_t)(val >> 56);
                val <<= 8;
                fill -= 8;
            }
        }
        if (fill > 0) {
            dst[j] = (uint8_t)(val >> 56);
        }
        return items;
    }
    return IMCD_ERROR;
}


/****************** Packed Integers (GenICam "p" format) *********************/

/* GenICam Pixel Format Naming Convention (PFNC) 2.1.

Mono10p, Mono12p, Mono14p: Little-endian continuous bitstream.
Pixel values are packed LSB-first into consecutive bytes with no padding.

For example, Mono12p packs 2 pixels into 3 bytes:
  Byte[0] = P0[7:0]
  Byte[1] = P0[11:8] | (P1[3:0] << 4)
  Byte[2] = P1[11:4]

*/


/* Unpack GenICam "p" format packed integers (LE continuous bitstream). */
ssize_t imcd_packints_decode_lsb(
    const uint8_t* src,
    const ssize_t srcsize,  /** size of src in bytes */
    uint8_t* dst,  /** buffer to store unpacked items as uint16 LE */
    const ssize_t items,  /** number of items to unpack */
    const int bps  /** number of bits per sample (10, 12, or 14) */
)
{
    ssize_t i;
    ssize_t bit_offset, byte_offset;
    int bit_shift;
    uint32_t value;
    uint32_t mask;

    if ((srcsize == 0) || (items == 0)) {
        return 0;
    }
    if ((src == NULL) || (dst == NULL) || (srcsize < 0) || (items < 0)) {
        return IMCD_VALUE_ERROR;
    }
    if ((bps < 9) || (bps > 14)) {
        return IMCD_VALUE_ERROR;
    }
    if (items > (ssize_t)(SIZE_MAX >> 1) / (ssize_t)bps) {
        return IMCD_VALUE_ERROR;
    }

    mask = ((uint32_t)1 << bps) - 1;

    for (i = 0; i < items; i++) {
        bit_offset = i * bps;
        byte_offset = bit_offset / 8;
        bit_shift = (int)(bit_offset % 8);

        if (byte_offset + 2 < srcsize) {
            value = (uint32_t)src[byte_offset]
                | ((uint32_t)src[byte_offset + 1] << 8)
                | ((uint32_t)src[byte_offset + 2] << 16);
        }
        else if (byte_offset + 1 < srcsize) {
            value = (uint32_t)src[byte_offset]
                | ((uint32_t)src[byte_offset + 1] << 8);
        }
        else if (byte_offset < srcsize) {
            value = (uint32_t)src[byte_offset];
        }
        else {
            break;
        }
        value = (value >> bit_shift) & mask;

        /* store as uint16 LE */
        dst[2 * i] = (uint8_t)(value & 0xFF);
        dst[2 * i + 1] = (uint8_t)((value >> 8) & 0xFF);
    }
    return i;
}


/* Pack integers into GenICam "p" format (LE continuous bitstream). */
ssize_t imcd_packints_encode_lsb(
    const uint8_t* src,
    const ssize_t srcsize,  /** size of src in bytes */
    uint8_t* dst,  /** buffer to store packed items */
    const ssize_t items,  /** number of items to pack */
    const int bps  /** number of bits per sample (10, 12, or 14) */
)
{
    ssize_t i;
    ssize_t bit_offset, byte_offset;
    ssize_t dstsize;
    int bit_shift;
    uint32_t value;
    uint32_t mask;

    if ((srcsize == 0) || (items == 0)) {
        return 0;
    }
    if ((src == NULL) || (dst == NULL) || (srcsize < 0) || (items < 0)) {
        return IMCD_VALUE_ERROR;
    }
    if ((bps < 9) || (bps > 14)) {
        return IMCD_VALUE_ERROR;
    }
    if (items > (ssize_t)(SIZE_MAX >> 1) / (ssize_t)bps) {
        return IMCD_VALUE_ERROR;
    }

    mask = ((uint32_t)1 << bps) - 1;
    dstsize = (items * (ssize_t)bps + 7) / 8;

    /* zero destination */
    memset(dst, 0, (size_t)dstsize);

    for (i = 0; i < items; i++) {
        /* read uint16 LE */
        value = (uint32_t)src[2 * i] | ((uint32_t)src[2 * i + 1] << 8);
        value &= mask;

        bit_offset = i * bps;
        byte_offset = bit_offset / 8;
        bit_shift = (int)(bit_offset % 8);

        /* OR into destination bytes */
        dst[byte_offset] |= (uint8_t)((value << bit_shift) & 0xFF);
        if (byte_offset + 1 < dstsize) {
            dst[byte_offset + 1] |=
                (uint8_t)(((value << bit_shift) >> 8) & 0xFF);
        }
        if ((bit_shift + bps > 16) && (byte_offset + 2 < dstsize)) {
            dst[byte_offset + 2] |=
                (uint8_t)(((value << bit_shift) >> 16) & 0xFF);
        }
    }
    return items;
}


/************** Packed Integers (GigE Vision legacy "Packed") ****************/

/* GigE Vision 1.x Packed format.

Mono10Packed, Mono12Packed: MSB-first paired pixel format.
Each pair of pixels is stored in 3 bytes, with the MSBs in the first
and third byte, and the LSBs of both pixels sharing the middle byte.

Mono12Packed: 2 pixels in 3 bytes (0 unused bits):
  Byte[0] = P0[11:4]
  Byte[1] = P0[3:0] | (P1[3:0] << 4)
  Byte[2] = P1[11:4]

Mono10Packed: 2 pixels in 3 bytes (4 unused bits):
  Byte[0] = P0[9:2]
  Byte[1] = P0[1:0] | (P1[1:0] << 4)
  Byte[2] = P1[9:2]
  Bits 3:2 and 7:6 of Byte[1] are zero/unused.

*/


/* Unpack GigE Vision legacy "Packed" format integers. */
ssize_t imcd_packints_decode_msb(
    const uint8_t* src,
    const ssize_t srcsize,  /** size of src in bytes */
    uint8_t* dst,  /** buffer to store unpacked items as uint16 LE */
    const ssize_t items,  /** number of items to unpack (must be even) */
    const int bps  /** number of bits per sample (10 or 12) */
)
{
    ssize_t npairs;
    ssize_t i;
    uint32_t byte0, byte1, byte2;
    uint32_t p0, p1;
    int lsb;  /* number of LSB bits per pixel in the middle byte */

    if ((srcsize == 0) || (items == 0)) {
        return 0;
    }
    if ((src == NULL) || (dst == NULL) || (srcsize < 0) || (items < 0)) {
        return IMCD_VALUE_ERROR;
    }
    if ((bps != 10) && (bps != 12)) {
        return IMCD_VALUE_ERROR;
    }
    if (items % 2) {
        return IMCD_VALUE_ERROR;
    }

    lsb = bps - 8;
    npairs = items / 2;

    for (i = 0; i < npairs; i++) {
        if (3 * i + 2 >= srcsize) {
            break;
        }
        byte0 = (uint32_t)src[3 * i];
        byte1 = (uint32_t)src[3 * i + 1];
        byte2 = (uint32_t)src[3 * i + 2];

        /* MSBs are in byte0/byte2, LSBs share byte1 */
        p0 = (byte0 << lsb) | (byte1 & (((uint32_t)1 << lsb) - 1));
        p1 = (byte2 << lsb) | ((byte1 >> 4) & (((uint32_t)1 << lsb) - 1));

        /* store as uint16 LE */
        dst[4 * i + 0] = (uint8_t)(p0 & 0xFF);
        dst[4 * i + 1] = (uint8_t)((p0 >> 8) & 0xFF);
        dst[4 * i + 2] = (uint8_t)(p1 & 0xFF);
        dst[4 * i + 3] = (uint8_t)((p1 >> 8) & 0xFF);
    }
    return i * 2;
}


/* Pack integers into GigE Vision legacy "Packed" format. */
ssize_t imcd_packints_encode_msb(
    const uint8_t* src,
    const ssize_t srcsize,  /** size of src in bytes */
    uint8_t* dst,  /** buffer to store packed items */
    const ssize_t items,  /** number of items to pack (must be even) */
    const int bps  /** number of bits per sample (10 or 12) */
)
{
    ssize_t npairs;
    ssize_t i;
    uint32_t p0, p1;
    int lsb;  /* number of LSB bits per pixel in the middle byte */
    uint32_t lsb_mask;

    if ((srcsize == 0) || (items == 0)) {
        return 0;
    }
    if ((src == NULL) || (dst == NULL) || (srcsize < 0) || (items < 0)) {
        return IMCD_VALUE_ERROR;
    }
    if ((bps != 10) && (bps != 12)) {
        return IMCD_VALUE_ERROR;
    }
    if (items % 2) {
        return IMCD_VALUE_ERROR;
    }

    lsb = bps - 8;
    lsb_mask = ((uint32_t)1 << lsb) - 1;
    npairs = items / 2;

    for (i = 0; i < npairs; i++) {
        /* read uint16 LE */
        p0 = (uint32_t)src[4 * i + 0] | ((uint32_t)src[4 * i + 1] << 8);
        p1 = (uint32_t)src[4 * i + 2] | ((uint32_t)src[4 * i + 3] << 8);

        dst[3 * i + 0] = (uint8_t)(p0 >> lsb);
        dst[3 * i + 1] = (uint8_t)((p0 & lsb_mask) | ((p1 & lsb_mask) << 4));
        dst[3 * i + 2] = (uint8_t)(p1 >> lsb);
    }
    return items;
}


/********************************** Float24 **********************************/

/* Adobe Photoshop(r) TIFF Technical Note 3. April 8, 2005.

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
    const int islittle)
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
        if (islittle) {
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
        if ((exponent == 0) && (mantissa == 0)) {
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
                /* change bias */
                exponent = exponent - 63 + 127 - (uint8_t) shift;
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
    const int islittle,
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
        (rounding != FE_DOWNWARD))
    {
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
        if ((exponent == 0) && (mantissa == 0)) {
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
                ((rounding == FE_UPWARD) && (!sign)))
            {
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
        if (islittle) {
            /* to little endian */
            *dst++ = f2;
            *dst++ = f1;
            *dst++ = f0;
        }
        else {
            /* to big endian */
            *dst++ = f0;
            *dst++ = f1;
            *dst++ = f2;
        }
    }
    return (srcsize / 4) * 4;
}


/********************************* Bfloat16 **********************************/

/* https://en.wikipedia.org/wiki/Bfloat16_floating-point_format */

ssize_t imcd_bfloat16_decode(
    const uint8_t* src,
    const ssize_t srcsize,
    uint8_t* dst,
    const int islittle)
{
    /* input validation is done in wrapper function */

    ssize_t i;
    uint8_t f0, f1;

    if (srcsize < 2) {
        return 0;
    }

    for (i = 0; i < srcsize; i += 2) {
        if (islittle) {
            /* from little endian */
            f1 = *src++;
            f0 = *src++;
        }
        else {
            /* from big endian */
            f0 = *src++;
            f1 = *src++;
        }

#if IMCD_MSB
        /* to little endian */
        *dst++ = 0;
        *dst++ = 0;
        *dst++ = f1;
        *dst++ = f0;
#else
        /* to big endian */
        *dst++ = f0;
        *dst++ = f1;
        *dst++ = 0;
        *dst++ = 0;
#endif
    }
    return (srcsize / 2) * 2;
}


ssize_t imcd_bfloat16_encode(
    const uint8_t* src,
    const ssize_t srcsize,
    uint8_t* dst,
    const int islittle,
    int rounding)
{
    /* input validation is done in wrapper function */

    ssize_t i;
    uint8_t f0, f1;
    uint32_t uvalue;
    float fvalue;

    if (srcsize < 4) {
        return 0;
    }

    if (rounding < 0) {
        rounding = fegetround();
    }

    for (i = 0; i < srcsize; i += 4) {
        memcpy(&fvalue, &src[i], 4);

        if (isnan(fvalue)) {
            f0 = signbit(fvalue) ? 0xFF : 0x7F;
            f1 = 0xC0;
        }
        else {
            memcpy(&uvalue, &fvalue, sizeof(uvalue));

            switch (rounding)
            {
                case FE_TOWARDZERO:
                    /* truncate: discard lower 16 bits unchanged */
                    break;
                case FE_UPWARD:
                    /* toward +inf: round up if positive */
                    /* and low bits non-zero */
                    if ((uvalue & 0xFFFF) && !(uvalue & 0x80000000)) {
                        uvalue += 0x10000;
                    }
                    break;
                case FE_DOWNWARD:
                    /* toward -inf: round up magnitude if negative */
                    /* and low bits non-zero */
                    if ((uvalue & 0xFFFF) && (uvalue & 0x80000000)) {
                        uvalue += 0x10000;
                    }
                    break;
                case FE_TONEAREST:
                default:
                    /* round half to nearest even */
                    uvalue += 0x7FFF + ((uvalue >> 16) & 1);
                    break;
            }
            f1 = (uvalue >> 16) & 0xFF;
            f0 = (uvalue >> 24) & 0xFF;
        }
        if (islittle) {
            *dst++ = f1;
            *dst++ = f0;
        }
        else {
            *dst++ = f0;
            *dst++ = f1;
        }
    }
    return (srcsize / 4) * 4;
}


/*********************** Electron Event Representation ***********************/

/* EER file format documentation 3.0. Section 4. by M. Leichsenring. March 2023

The Electron Event Representation uses a variable length bitstream to encode
detected electron events with sub pixel information.

*/

ssize_t
imcd_eer_decode(
    const uint8_t* src,
    const ssize_t srcsize,
    uint8_t* dst,
    const ssize_t height,
    const ssize_t width,
    const uint32_t skipbits,
    const uint32_t horzbits,
    const uint32_t vertbits,
    const uint32_t superres)
{
    const ssize_t dstsize = height * width;
    const ssize_t nbits = skipbits + horzbits + vertbits;
    const ssize_t srcbits = srcsize * 8 - nbits;
    const uint32_t skipmask = imcd_bitmask2(skipbits);
    const uint32_t horzmask = imcd_bitmask2(horzbits);
    const uint32_t vertmask = imcd_bitmask2(vertbits);
    const uint32_t horzshift = MAX_(0, (ssize_t)horzbits - (ssize_t)superres);
    const uint32_t vertshift = MAX_(0, (ssize_t)vertbits - (ssize_t)superres);
    const ssize_t horzsize = (ssize_t)((horzmask + 1) >> horzshift);
    const ssize_t vertsize = (ssize_t)((vertmask + 1) >> vertshift);
    const ssize_t width2 = width / horzsize;
    const ssize_t pixelsize = dstsize / (horzsize * vertsize);
    ssize_t bitindex = 0;
    ssize_t dstindex = 0;
    ssize_t pixelindex = 0;
    ssize_t events = 0;
    uint32_t word = 0;
    uint32_t skip = 0;
    ssize_t v, h;

    if (
        (src == NULL) || (srcsize % 2) || (dst == NULL) ||
        (height < 1) || (width < 1) || (nbits > 16) || (nbits <= 8) ||
        (skipbits < 4) || (horzbits < 1) || (vertbits < 1))
    {
        return IMCD_VALUE_ERROR;
    }
    if (height > (ssize_t)(SIZE_MAX >> 1) / width) {
        return IMCD_VALUE_ERROR;
    }

    if ((superres > 0) && ((width % horzsize) || (height % vertsize))) {
        return IMCD_VALUE_ERROR;
    }

    while (bitindex < srcbits) {
        word = 0;
        v = bitindex / 8;
        h = srcsize - v;
        memcpy(&word, src + v, h > 3 ? 4 : h);
        word >>= bitindex % 8;
        skip = word & skipmask;
        pixelindex += (ssize_t)skip;
        if (pixelindex == pixelsize) {
            break;
        }
        if (pixelindex > pixelsize) {
            return IMCD_OUTPUT_TOO_SMALL;
        }
        if (skip == skipmask) {
            bitindex += skipbits;
            continue;
        }
        if (superres == 0) {
            dstindex = pixelindex;
        }
        else {
            word >>= skipbits;
            h = (ssize_t)(((word & horzmask) ^ horzbits) >> horzshift);
            h += (pixelindex % width2) * horzsize;

            word >>= horzbits;
            v = (ssize_t)(((word & vertmask) ^ vertbits) >> vertshift);
            v += (pixelindex / width2) * vertsize;

            dstindex = v * width + h;
            if (dstindex >= dstsize) {
                return IMCD_OUTPUT_TOO_SMALL;
            }
        }
        dst[dstindex] += 1;
        pixelindex++;
        bitindex += nbits;
        events++;
    }
    return events;
}


ssize_t
imcd_eer_decode_u2(
    const uint8_t* src,
    const ssize_t srcsize,
    uint16_t* dst,
    const ssize_t height,
    const ssize_t width,
    const uint32_t skipbits,
    const uint32_t horzbits,
    const uint32_t vertbits,
    const uint32_t superres)
{
    const ssize_t dstsize = height * width;
    const ssize_t nbits = skipbits + horzbits + vertbits;
    const ssize_t srcbits = srcsize * 8 - nbits;
    const uint32_t skipmask = imcd_bitmask2(skipbits);
    const uint32_t horzmask = imcd_bitmask2(horzbits);
    const uint32_t vertmask = imcd_bitmask2(vertbits);
    const uint32_t horzshift = MAX_(0, (ssize_t)horzbits - (ssize_t)superres);
    const uint32_t vertshift = MAX_(0, (ssize_t)vertbits - (ssize_t)superres);
    const ssize_t horzsize = (ssize_t)((horzmask + 1) >> horzshift);
    const ssize_t vertsize = (ssize_t)((vertmask + 1) >> vertshift);
    const ssize_t width2 = width / horzsize;
    const ssize_t pixelsize = dstsize / (horzsize * vertsize);
    ssize_t bitindex = 0;
    ssize_t dstindex = 0;
    ssize_t pixelindex = 0;
    ssize_t events = 0;
    uint32_t word = 0;
    uint32_t skip = 0;
    ssize_t v, h;

    if (
        (src == NULL) || (srcsize % 2) || (dst == NULL) ||
        (height < 1) || (width < 1) || (nbits > 16) || (nbits <= 8) ||
        (skipbits < 4) || (horzbits < 1) || (vertbits < 1))
    {
        return IMCD_VALUE_ERROR;
    }
    if (height > (ssize_t)(SIZE_MAX >> 1) / width) {
        return IMCD_VALUE_ERROR;
    }
    if ((superres > 0) && ((width % horzsize) || (height % vertsize))) {
        return IMCD_VALUE_ERROR;
    }

    while (bitindex < srcbits) {
        word = 0;
        v = bitindex / 8;
        h = srcsize - v;
        memcpy(&word, src + v, h > 3 ? 4 : h);
        word >>= bitindex % 8;
        skip = word & skipmask;
        pixelindex += (ssize_t)skip;
        if (pixelindex == pixelsize) {
            break;
        }
        if (pixelindex > pixelsize) {
            return IMCD_OUTPUT_TOO_SMALL;
        }
        if (skip == skipmask) {
            bitindex += skipbits;
            continue;
        }
        if (superres == 0) {
            dstindex = pixelindex;
        }
        else {
            word >>= skipbits;
            h = (ssize_t)(((word & horzmask) ^ horzbits) >> horzshift);
            h += (pixelindex % width2) * horzsize;

            word >>= horzbits;
            v = (ssize_t)(((word & vertmask) ^ vertbits) >> vertshift);
            v += (pixelindex / width2) * vertsize;

            dstindex = v * width + h;
            if (dstindex >= dstsize) {
                return IMCD_OUTPUT_TOO_SMALL;
            }
        }
        dst[dstindex] += 1;
        pixelindex++;
        bitindex += nbits;
        events++;
    }
    return events;
}

/************************************ LZW ************************************/

/* Section 13: LZW Compression. TIFF Revision 6.0 Final. 1992

TIFF compression scheme 5, an adaptive compression scheme for raster images.

*/

/* LZW table size is 4098 + 1024 buffer for old style */
#define LZW_TABLESIZE 5120
#define LZW_CLEAR 256
#define LZW_EOI 257
#define LZW_FIRST 258
#define LZW_HASH_SIZE 8192  /* power of two */
#define LZW_HASH_MASK (LZW_HASH_SIZE - 1)
#define LZW_CHECK_GAP 10000  /* input bytes between compression ratio checks */

static const uint8_t LZW_LITERALS[256 + 8] = {
    0, 1, 2, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63,
    64, 65, 66, 67, 68, 69, 70, 71,
    72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 83, 84, 85, 86, 87,
    88, 89, 90, 91, 92, 93, 94, 95,
    96, 97, 98, 99, 100, 101, 102, 103,
    104, 105, 106, 107, 108, 109, 110, 111,
    112, 113, 114, 115, 116, 117, 118, 119,
    120, 121, 122, 123, 124, 125, 126, 127,
    128, 129, 130, 131, 132, 133, 134, 135,
    136, 137, 138, 139, 140, 141, 142, 143,
    144, 145, 146, 147, 148, 149, 150, 151,
    152, 153, 154, 155, 156, 157, 158, 159,
    160, 161, 162, 163, 164, 165, 166, 167,
    168, 169, 170, 171, 172, 173, 174, 175,
    176, 177, 178, 179, 180, 181, 182, 183,
    184, 185, 186, 187, 188, 189, 190, 191,
    192, 193, 194, 195, 196, 197, 198, 199,
    200, 201, 202, 203, 204, 205, 206, 207,
    208, 209, 210, 211, 212, 213, 214, 215,
    216, 217, 218, 219, 220, 221, 222, 223,
    224, 225, 226, 227, 228, 229, 230, 231,
    232, 233, 234, 235, 236, 237, 238, 239,
    240, 241, 242, 243, 244, 245, 246, 247,
    248, 249, 250, 251, 252, 253, 254, 255,
    0, 0, 0, 0, 0, 0, 0, 0  /* padding bytes for safe 8-byte reads */
};

struct imcd_lzw_handle {
    /* pointer into output buffer or literals */
    const uint8_t* string[LZW_TABLESIZE];
    uint16_t length[LZW_TABLESIZE];
};

/* Allocate LZW handle. */
imcd_lzw_handle_t* imcd_lzw_new(void)
{
    return (imcd_lzw_handle_t*)malloc(sizeof(imcd_lzw_handle_t));
}


/* Free LZW handle. */
void imcd_lzw_del(
    imcd_lzw_handle_t* handle)
{
    free(handle);
}


/* MSB: TIFF and PDF */
/* bitbuf is left-aligned: valid bits in top bitcnt positions */
#define LZW_REFILL_MSB \
{ \
    if (bitpos + 8 <= srcsize) { \
        uint64_t c; \
        memcpy(&c, src + bitpos, 8); \
        bitbuf |= SWAP8BYTES(c) >> bitcnt; \
        bitpos += (63 - bitcnt) >> 3; \
        bitcnt |= 56; \
    } \
    else { \
        while (bitcnt <= 56 && bitpos < srcsize) { \
            bitbuf |= (uint64_t)src[bitpos++] << (56 - bitcnt); \
            bitcnt += 8; \
        } \
    } \
}

#define LZW_GET_NEXT_CODE_MSB \
{ \
    if (bitcnt < (int)bitw) { LZW_REFILL_MSB } \
    if (bitcnt >= (int)bitw) { \
        code = (uint32_t)(bitbuf >> (64 - (int)bitw)) \
            & (((uint32_t)1 << (int)bitw) - 1u); \
        bitbuf <<= (int)bitw; \
        bitcnt -= (int)bitw; \
    } \
    else { code = LZW_EOI; } \
}


/* LSB: GIF and old-style TIFF LZW */
/* bitbuf is right-aligned: valid bits in bottom bitcnt positions */
#define LZW_REFILL_LSB \
{ \
    if (bitpos + 8 <= srcsize) { \
        uint64_t c; \
        memcpy(&c, src + bitpos, 8); \
        bitbuf |= c << bitcnt; \
        bitpos += (63 - bitcnt) >> 3; \
        bitcnt |= 56; \
    } \
    else { \
        while (bitcnt <= 56 && bitpos < srcsize) { \
            bitbuf |= ((uint64_t)src[bitpos++]) << bitcnt; \
            bitcnt += 8; \
        } \
    } \
}

#define LZW_GET_NEXT_CODE_LSB \
{ \
    if (bitcnt < (int)bitw) { LZW_REFILL_LSB } \
    if (bitcnt >= (int)bitw) { \
        code = (uint32_t)(bitbuf & (((uint32_t)1 << (int)bitw) - 1u)); \
        bitbuf >>= (int)bitw; \
        bitcnt -= (int)bitw; \
    } \
    else { code = LZW_EOI; } \
}

/* Record previous output string as the new table entry. */
#define LZW_ADD_ENTRY \
    if (oldcode >= 0) { \
        str[tablesize] = prev_out; \
        len[tablesize] = (uint16_t)(prev_len + 1); \
        tablesize++; \
    }

/* Emit string corresponding to current code to output buffer. */
#define LZW_EMIT_CODE \
    { \
        const uint8_t* s = str[code]; \
        uint32_t n = len[code]; \
        uint32_t avail = (uint32_t)(dstend - dst); \
        if (n > avail) n = avail; \
        if ((n <= 8) && (avail >= 8)) { \
            uint64_t t; \
            memcpy(&t, s, 8); \
            memcpy(dst, &t, 8); \
        } \
        else if ((n <= 16) && (avail >= 16)) { \
            uint64_t t0, t1; \
            memcpy(&t0, s, 8); \
            memcpy(&t1, s + 8, 8); \
            memcpy(dst, &t0, 8); \
            memcpy(dst + 8, &t1, 8); \
        } \
        else { \
            memcpy(dst, s, n); \
        } \
        prev_out = dst; \
        prev_len = n; \
        dst += n; \
    }


/* Return 1 if compressed string begins with a CLEAR code. */
int imcd_lzw_check(
    const uint8_t* src,
    const ssize_t size)
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


/* Return length of decompressed LZW string. */
ssize_t imcd_lzw_decode_size(
    imcd_lzw_handle_t* handle,
    const uint8_t* src,
    const ssize_t srcsize)
{
    uint16_t len[LZW_TABLESIZE];
    uint32_t tablesize = 258;
    uint32_t code = 0;
    int32_t oldcode = -1;
    ssize_t dstsize = 0;
    ssize_t bitpos = 0;
    uint64_t bitbuf = 0;
    uint64_t bitw = 9;
    int32_t bitcnt = 0;
    uint32_t i;
    bool msb = true;

    if ((handle == NULL) || (src == NULL) || (srcsize < 0)) {
        return IMCD_VALUE_ERROR;
    }
    if (srcsize == 0) {
        return 0;
    }
    if (srcsize < 2) {
        return IMCD_LZW_INVALID;
    }

    if ((*src == 0) && (*(src + 1) & 1)) {
        msb = false;
    }
    else if ((*src != 128) || ((*(src + 1) & 128))) {
        return IMCD_LZW_INVALID;
    }

    for (i = 0; i < 258; i++) {
        len[i] = 1;
    }

    if (msb) {
        uint32_t maxcode = 510;  /* early change at 511, 1023, 2047 */
        while (1) {
            LZW_GET_NEXT_CODE_MSB
            if ((code != LZW_EOI) & (code != LZW_CLEAR)) {
                /* most common: literals (0-255) and compound (>=258) */
                if (tablesize >= LZW_TABLESIZE) {
                    return IMCD_LZW_TABLE_TOO_SMALL;
                }
                if (code > tablesize) {
                    return IMCD_LZW_CORRUPT;
                }
                if (code == tablesize) {
                    /* K-w-K */
                    if (oldcode < 0) {
                        return IMCD_LZW_CORRUPT;
                    }
                    dstsize += (ssize_t)len[(uint32_t)oldcode] + 1;
                }
                else {
                    /* literals: len=1; compound: actual */
                    dstsize += len[code];
                }
                if (oldcode >= 0) {
                    len[tablesize++] = len[(uint32_t)oldcode] + 1;
                }
                oldcode = (int32_t)code;
            }
            else if (code == LZW_CLEAR) {
                tablesize = 258;
                bitw = 9;
                maxcode = 510;
                oldcode = -1;
            }
            else {
                break;  /* EOI */
            }
            /* early-change bit-width update */
            if (tablesize > maxcode) {
                bitw++;
                maxcode = (bitw < 12) ? ((maxcode << 1) | 2) : LZW_TABLESIZE;
            }
        }
    }
    else {
        uint32_t maxcode = 511;  /* late change at 512, 1024, 2048 */
        while (1) {
            LZW_GET_NEXT_CODE_LSB
            if ((code != LZW_EOI) & (code != LZW_CLEAR)) {
                /* most common: literals (0-255) and compound (>=258) */
                if (tablesize >= LZW_TABLESIZE) {
                    return IMCD_LZW_TABLE_TOO_SMALL;
                }
                if (code > tablesize) {
                    return IMCD_LZW_CORRUPT;
                }
                if (code == tablesize) {
                    /* K-w-K */
                    if (oldcode < 0) {
                        return IMCD_LZW_CORRUPT;
                    }
                    dstsize += (ssize_t)len[(uint32_t)oldcode] + 1;
                }
                else {
                    /* literals: len=1; compound: actual */
                    dstsize += len[code];
                }
                if (oldcode >= 0) {
                    len[tablesize++] = len[(uint32_t)oldcode] + 1;
                }
                oldcode = (int32_t)code;
            }
            else if (code == LZW_CLEAR) {
                tablesize = 258;
                bitw = 9;
                maxcode = 511;
                oldcode = -1;
            }
            else {
                break;  /* EOI */
            }
            /* late-change bit-width update */
            if (tablesize > maxcode) {
                bitw++;
                maxcode = (bitw < 12) ? ((maxcode << 1) | 1) : LZW_TABLESIZE;
            }
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
    const uint8_t** const str = handle->string;
    uint16_t* const len = handle->length;
    const uint8_t* dstin = dst;
    uint8_t* const dstend = dst + dstsize;
    uint32_t tablesize = 258;
    uint32_t code = 0;
    int32_t oldcode = -1;
    ssize_t bitpos = 0;
    uint64_t bitbuf = 0;
    uint64_t bitw = 9;
    int32_t bitcnt = 0;
    bool msb = true;
    uint32_t i;
    uint8_t* prev_out = dst;
    uint32_t prev_len = 0;

    if (
        (handle == NULL) ||
        (src == NULL) || (srcsize < 0) ||
        (dst == NULL) || (dstsize < 0))
    {
        return IMCD_VALUE_ERROR;
    }
    if ((srcsize == 0) || (dstsize == 0)) {
        return 0;
    }
    if (srcsize < 2) {
        return IMCD_LZW_INVALID;
    }

    if ((*src == 0) && (*(src + 1) & 1)) {
        msb = false;
    }
    else if ((*src != 128) || ((*(src + 1) & 128))) {
        /* compressed string must begin with CLEAR code */
        return IMCD_LZW_INVALID;
    }

    /* initialize literal entries 0..255 */
    for (i = 0; i < 256; i++) {
        str[i] = LZW_LITERALS + i;
        len[i] = 1;
    }

    if (msb) {
        /* MSB: TIFF and PDF */
        uint32_t maxcode = 510;  /* early change at at 511, 1022, 2046 */
        while (dst < dstend) {
            LZW_GET_NEXT_CODE_MSB
            if ((code != LZW_EOI) & (code != LZW_CLEAR)) {
                /* most common: literals (0-255) and compound (>=258) */
                if (tablesize >= LZW_TABLESIZE) {
                    return IMCD_LZW_TABLE_TOO_SMALL;
                }
                if (code > tablesize) {
                    return IMCD_LZW_CORRUPT;
                }
                if (code == tablesize) {
                    /* K-w-K */
                    if (oldcode < 0) {
                        return IMCD_LZW_CORRUPT;
                    }
                    LZW_ADD_ENTRY
                        oldcode = (int32_t)code;
                    {
                        uint32_t n = prev_len + 1;
                        uint32_t j;
                        if (n >
                            (uint32_t)(dstend - dst))
                            n = (uint32_t)(dstend - dst);
                        for (j = 0; j < n; j++) dst[j] = prev_out[j];
                        prev_out = dst;
                        prev_len = n;
                        dst += n;
                    }
                }
                else {
                    LZW_ADD_ENTRY
                        oldcode = (int32_t)code;
                    LZW_EMIT_CODE
                }
            }
            else if (code == LZW_CLEAR) {
                tablesize = 258;
                bitw = 9;
                maxcode = 510;
                oldcode = -1;
            }
            else {
                break;  /* EOI */
            }
            /* early-change bit-width update */
            if (tablesize > maxcode) {
                bitw++;
                maxcode = (bitw < 12) ? ((maxcode << 1) | 2) : LZW_TABLESIZE;
            }
            if (dst < dstend) continue;
        }
    }
    else {
        /* LSB: GIF and old-style TIFF LZW */
        uint32_t maxcode = 511;  /* late-change at 512, 1024, 2048 */
        while (dst < dstend) {
            LZW_GET_NEXT_CODE_LSB
            if ((code != LZW_EOI) & (code != LZW_CLEAR)) {
                /* most common: literals (0-255) and compound (>=258) */
                if (tablesize >= LZW_TABLESIZE) {
                    return IMCD_LZW_TABLE_TOO_SMALL;
                }
                if (code > tablesize) {
                    return IMCD_LZW_CORRUPT;
                }
                if (code == tablesize) {
                    if (oldcode < 0) {
                        return IMCD_LZW_CORRUPT;
                    }
                    LZW_ADD_ENTRY
                        oldcode = (int32_t)code;
                    {
                        uint32_t n = prev_len + 1;
                        uint32_t j;
                        if (n >
                            (uint32_t)(dstend - dst)) n = (uint32_t)(dstend -
                            dst);
                        for (j = 0; j < n; j++) dst[j] = prev_out[j];
                        prev_out = dst;
                        prev_len = n;
                        dst += n;
                    }
                }
                else {
                    LZW_ADD_ENTRY
                        oldcode = (int32_t)code;
                    LZW_EMIT_CODE
                }
            }
            else if (code == LZW_CLEAR) {
                tablesize = 258;
                bitw = 9;
                maxcode = 511;
                oldcode = -1;
            }
            else {
                break;  /* EOI */
            }
            /* late-change bit-width update */
            if (tablesize > maxcode) {
                bitw++;
                maxcode = (bitw < 12) ? ((maxcode << 1) | 1) : LZW_TABLESIZE;
            }
            if (dst < dstend) continue;
        }
    }

    return (ssize_t)(dst - dstin);
}


/* Return maximum length of LZW compressed sequence. */
ssize_t imcd_lzw_encode_size(
    const ssize_t srcsize)
{
    return (srcsize * 141) / 100 + 3;
}


#define LZW_WRITE_DST \
{ \
    if (dstindex >= dstsize) { \
        dstindex = IMCD_OUTPUT_TOO_SMALL; \
        goto DONE; \
    } \
    dst[dstindex++] = (uint8_t)(dstbyte >> bitc); \
}


/* Encode LZW. */
ssize_t imcd_lzw_encode(
    const uint8_t* src,
    const ssize_t srcsize,
    uint8_t* dst,
    const ssize_t dstsize)
{
    ssize_t dstindex = 0;
    ssize_t srcindex = 0;
    uint32_t tag = 0;
    uint32_t gen = 1;  /* 0 means stale, starts at 1 */
    uint32_t* hash_keys = NULL;
    int* hash_values = NULL;
    int hashkey = 0;
    int hashcode = 0;
    int nextcode = LZW_FIRST;
    int dstbyte = LZW_CLEAR;
    int bitw = 9;  /* current number of bits in code */
    int bitc = 1;  /* used bits */
    int omega = 0;
    int k = 0;
    /* srcindex at last CLEAR */
    ssize_t incount_base = 0;
    /* dstindex at last CLEAR (1 for initial CLEAR byte) */
    ssize_t outcount_base = 1;
    /* next srcindex at which to check ratio */
    ssize_t checkpoint = LZW_CHECK_GAP;
    /* compression ratio at last check; not yet set */
    ssize_t enc_ratio = 0;

    if ((src == NULL) || (srcsize < 0) || (dst == NULL) || (dstsize < 0)) {
        return IMCD_VALUE_ERROR;
    }
    if (dstsize < 3) {
        return IMCD_OUTPUT_TOO_SMALL;
    }

    /* write CLEAR code */
    bitc = 1;
    dstbyte = LZW_CLEAR;
    LZW_WRITE_DST;

    if (srcsize < 1) {
        /* write EOI */
        dstbyte = ((dstbyte << bitw) | LZW_EOI) << 8;
        bitc += bitw;
        LZW_WRITE_DST;
        bitc -= 8;
        LZW_WRITE_DST;
        return dstindex;
    }

    /* allocate hash table */
    hash_keys = calloc(LZW_HASH_SIZE * 2, sizeof(uint32_t));
    if (hash_keys == NULL) {
        return IMCD_MEMORY_ERROR;
    }
    hash_values = (int*)(hash_keys + LZW_HASH_SIZE);

    omega = src[0] & 0xff;

    for (srcindex = 1; srcindex < srcsize; srcindex++) {
        k = src[srcindex] & 0xff;
        hashkey = (omega << 8) | k;
        hashcode = (int)((hashkey * 2654435761u) >> 19);
        tag = (gen << 20) | (uint32_t)hashkey;

        while (
            hash_keys[hashcode] != tag
            && (hash_keys[hashcode] >> 20) == gen)
        {
            hashcode = (hashcode + 1) & LZW_HASH_MASK;
        }
        if (hash_keys[hashcode] == tag) {
            /* Omega+K in table */
            omega = hash_values[hashcode];
            goto OUTER;
        }

        /* Omega+K not in table */
        /* add entry to table */
        hash_keys[hashcode] = tag;
        hash_values[hashcode] = nextcode++;

        /* write last code */
        dstbyte = (dstbyte << bitw) | omega;
        bitc += bitw - 8;
        LZW_WRITE_DST;
        if (bitc >= 8) {
            bitc -= 8;
            LZW_WRITE_DST;
        }

        omega = k;

        /* Bit-width change (TIFF/MSB style, TIFF6 spec p.60):
           although the ADD step precedes WRITE here, the observable behavior
           is identical: the code written when entry 511 is added uses 9-bit,
           and only the following code uses 10-bit.
           The MSB decoder switches on tablesize==511 (i.e. after adding entry
           510, which lags the encoder by 1 due to the classic LZW decode lag),
           so both sides agree on the transition point.
        */
        if (nextcode == 4096) {
            /* table full: write CLEAR and reset */
            dstbyte = (dstbyte << bitw) | LZW_CLEAR;
            bitc += bitw - 8;
            LZW_WRITE_DST;
            if (bitc >= 8) {
                bitc -= 8;
                LZW_WRITE_DST;
            }
            if (++gen == (1u << 12)) {
                memset(hash_keys, 0, sizeof(uint32_t) * LZW_HASH_SIZE);
                gen = 1;
            }
            nextcode = LZW_FIRST;
            bitw = 9;
            incount_base = srcindex;
            outcount_base = dstindex;
            enc_ratio = 0;
            checkpoint = srcindex + LZW_CHECK_GAP;
        }
        else {
            switch (nextcode)
            {
                case 512:  bitw = 10; break;
                case 1024: bitw = 11; break;
                case 2048: bitw = 12; break;
            }
            /* if compression ratio is degrading, reset the table
               so codes stay short and match patterns in the new context */
            if (srcindex >= checkpoint) {
                ssize_t incount = srcindex - incount_base;
                ssize_t outcount = dstindex - outcount_base;
                checkpoint = srcindex + LZW_CHECK_GAP;
                if (outcount > 0) {
                    ssize_t ratio;
                    if (incount > 0x007fffff) {
                        ratio = outcount >> 8;
                        ratio = (ratio == 0) ? 0x7fffffff : incount / ratio;
                    }
                    else {
                        ratio = (incount << 8) / outcount;
                    }
                    if (ratio <= enc_ratio) {
                        /* degrading: write CLEAR and reset */
                        dstbyte = (dstbyte << bitw) | LZW_CLEAR;
                        bitc += bitw - 8;
                        LZW_WRITE_DST;
                        if (bitc >= 8) {
                            bitc -= 8;
                            LZW_WRITE_DST;
                        }
                        if (++gen == (1u << 12)) {
                            memset(
                                hash_keys,
                                0,
                                sizeof(uint32_t) * LZW_HASH_SIZE
                            );
                            gen = 1;
                        }
                        nextcode = LZW_FIRST;
                        bitw = 9;
                        incount_base = srcindex;
                        outcount_base = dstindex;
                        enc_ratio = 0;
                    }
                    else {
                        enc_ratio = ratio;
                    }
                }
            }
        }
OUTER:  ;
    }

    /* write Omega code */
    dstbyte = (dstbyte << bitw) | omega;
    bitc += bitw - 8;
    LZW_WRITE_DST;
    if (bitc >= 8) {
        bitc -= 8;
        LZW_WRITE_DST;
    }

    /* write EOI */
    switch (nextcode)
    {
        case 511:
            bitw = 10;
            break;
        case 1023:
            bitw = 11;
            break;
        case 2047:
            bitw = 12;
            break;
    }
    dstbyte = ((dstbyte << bitw) | LZW_EOI) << 8;
    bitc += bitw;
    LZW_WRITE_DST;
    if (bitc >= 8) {
        bitc -= 8;
        LZW_WRITE_DST;
        if (bitc >= 8) {
            bitc -= 8;
            LZW_WRITE_DST;
        }
    }

DONE:
    free(hash_keys);

    return dstindex;
}


/********************************* Utilities *********************************/

/* Search for byte sequence needle in src.
   Return offset of first match, or -1 if not found.
   Return 0 if needlelen is 0.
*/

ssize_t imcd_memsearch(
    const char* src,
    const ssize_t srclen,
    const char* needle,
    const ssize_t needlelen)
{
    ssize_t i, j, k;

    if ((src == NULL) || (needle == NULL) || (srclen < 0) || (needlelen < 0)) {
        return IMCD_VALUE_ERROR;
    }
    if (needlelen == 0) {
        return 0;
    }
    if (needlelen > srclen) {
        return -1;
    }
    for (i = 0; i <= srclen - needlelen; i++) {
        if (src[i] == needle[0]) {
            int found = 1;
            for (j = 0; j < needlelen; j++) {
                k = i + j;
                if (needle[j] != src[k]) {
                    found = 0;
                    break;
                }
            }
            if (found) {
                return i;
            }
        }
    }
    return -1;
}


/* Search for byte sequence needle in src up to srclen bytes.
   Stop at null character in src.
   Return offset of first match, or -1 if not found.
   Return 0 if needlelen is 0.
*/

ssize_t imcd_strsearch(
    const char* src,
    const ssize_t srclen,
    const char* needle,
    const ssize_t needlelen)
{
    ssize_t i, j, k;

    if ((src == NULL) || (needle == NULL) || (srclen < 0) || (needlelen < 0)) {
        return IMCD_VALUE_ERROR;
    }
    if (needlelen == 0) {
        return 0;
    }
    if (needlelen > srclen) {
        return -1;
    }
    for (i = 0; i <= srclen - needlelen; i++) {
        if (src[i] == '\0') {
            return -1;
        }
        if (src[i] == needle[0]) {
            int found = 1;
            for (j = 0; j < needlelen; j++) {
                k = i + j;
                if ((src[k] == '\0') || (needle[j] != src[k])) {
                    found = 0;
                    break;
                }
            }
            if (found) {
                return i;
            }
        }
    }
    return -1;
}
