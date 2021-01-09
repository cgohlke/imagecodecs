/* imcd.h */

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

#ifndef IMCD_H
#define IMCD_H

#define IMCD_VERSION "2021.1.8"

#include <stdint.h>

#ifndef HAVE_SSIZE_T
#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#define HAVE_SSIZE_T 1
#else
#include <sys/types.h>
#endif
#endif

#ifndef SSIZE_MAX
#define SSIZE_MAX INTPTR_MAX
#define SSIZE_MIN INTPTR_MIN
#endif


/* Endianness */

#ifndef IMCD_MSB
#define IMCD_MSB 1
#endif

#if IMCD_MSB
#define IMCD_LSB 0
#define IMCD_BOC '<'
#else
#define IMCD_LSB 1
#define IMCD_BOC '>'
#endif

#if IMCD_LSB
#error Big-endian platforms not supported
#endif


/* Error codes */

#define IMCD_OK 0
#define IMCD_ERROR -1
#define IMCD_MEMORY_ERROR -2
#define IMCD_RUNTIME_ERROR -3
#define IMCD_NOTIMPLEMENTED_ERROR -4
#define IMCD_VALUE_ERROR -5

#define IMCD_LZW_INVALID -10
#define IMCD_LZW_NOTIMPLEMENTED -11
#define IMCD_LZW_BUFFER_TOO_SMALL -12
#define IMCD_LZW_TABLE_TOO_SMALL -13
#define IMCD_LZW_CORRUPT -14

/* Function declarations */

void imcd_swapbytes(
    void* src,
    const ssize_t srcsize,
    const ssize_t itemsize
);


unsigned char imcd_bitmask(
    const int bps
);


ssize_t imcd_packints_decode(
    const uint8_t* src,
    const ssize_t srcsize,
    uint8_t* dst,
    const ssize_t dstsize,
    const int bps
);


ssize_t imcd_packints_encode(
    const uint8_t* src,
    const ssize_t srcsize,
    uint8_t* dst,
    const ssize_t dstsize,
    const int bps
);


ssize_t imcd_packbits_size(
    const uint8_t* src,
    const ssize_t srcsize
);


ssize_t imcd_packbits_decode(
    const uint8_t* src,
    const ssize_t srcsize,
    uint8_t* dst,
    const ssize_t dstsize
);


ssize_t imcd_packbits_encode(
    const uint8_t* src,
    const ssize_t srcsize,
    uint8_t* dst,
    const ssize_t dstsize
);


ssize_t imcd_delta(
    void* src,
    const ssize_t srcsize,
    const ssize_t srcstride,
    void* dst,
    const ssize_t dstsize,
    const ssize_t dststride,
    const ssize_t itemsize,
    const int decode
);


ssize_t imcd_diff(
    void* src,
    const ssize_t srcsize,
    const ssize_t srcstride,
    void* dst,
    const ssize_t dstsize,
    const ssize_t dststride,
    const ssize_t itemsize,
    const char itemtype,
    const int decode
);


ssize_t imcd_xor(
    void* src,
    const ssize_t srcsize,
    const ssize_t srcstride,
    void* dst,
    const ssize_t dstsize,
    const ssize_t dststride,
    const ssize_t itemsize,
    const int decode
);


ssize_t imcd_floatpred(
    void* src,
    const ssize_t srcsize,
    const ssize_t srcstride,
    void* dst,
    const ssize_t dstsize,
    const ssize_t dststride,
    const ssize_t itemsize,
    const ssize_t samples,
    const char byteorder,
    const int decode
);


ssize_t imcd_bitorder(
    uint8_t* src,
    const ssize_t srcsize,
    const ssize_t srcstride,
    const ssize_t itemsize,
    uint8_t* dst,
    const ssize_t dstsize,
    const ssize_t dststride
);


ssize_t imcd_float24_decode(
    const uint8_t* src,
    const ssize_t srcsize,
    uint8_t* dst,
    const char byteorder
);


ssize_t imcd_float24_encode(
    const uint8_t* src,
    const ssize_t srcsize,
    uint8_t* dst,
    const char byteorder,
    int rounding
);


typedef struct {
    ssize_t len;
    uint8_t* buf;
} imcd_lzw_table_t;


typedef struct imcd_lzw_handle {
    imcd_lzw_table_t* table;
    uint8_t* buffer;
    ssize_t buffersize;
    ssize_t dummy;  /* imcd_lzw_handle_t multiple of imcd_lzw_table_t */
} imcd_lzw_handle_t;


imcd_lzw_handle_t* imcd_lzw_new(
    ssize_t buffersize
);


void imcd_lzw_del(
    imcd_lzw_handle_t* handle
);


ssize_t imcd_lzw_decode_size(
    imcd_lzw_handle_t* handle,
    const uint8_t* src,
    const ssize_t srcsize
);


ssize_t imcd_lzw_decode(
    imcd_lzw_handle_t* handle,
    const uint8_t* src,
    const ssize_t srcsize,
    uint8_t* dst,
    const ssize_t dstsize
);


int imcd_lzw_check(
    const uint8_t* src,
    const ssize_t size
);


typedef struct {
    uint8_t* data;
    ssize_t pos;
    ssize_t size;
    int64_t bitpos;
    int64_t bitsize;
    ssize_t memsize;
    ssize_t addsize;
    int owner;
} imcd_stream_t;


imcd_stream_t* imcd_stream_new(
    uint8_t* data,
    const ssize_t size,
    const ssize_t addsize
);


void imcd_stream_del(
    imcd_stream_t* handle
);


ssize_t imcd_stream_resize(
    imcd_stream_t* handle,
    const ssize_t size,
    const int exact
);


ssize_t imcd_stream_seek(
    imcd_stream_t* handle,
    const ssize_t offset,
    const int whence
);


int64_t imcd_stream_seek_bit(
    imcd_stream_t* handle,
    const int64_t offset,
    const int whence
);


ssize_t imcd_stream_tell(
    imcd_stream_t* handle
);


int64_t imcd_stream_tell_bit(
    imcd_stream_t* handle
);


ssize_t imcd_stream_read(
    imcd_stream_t* handle,
    const uint8_t* out,
    const ssize_t size
);


ssize_t imcd_stream_write(
    imcd_stream_t* handle,
    const uint8_t* data,
    const ssize_t size
);


void imcd_stream_data(
    imcd_stream_t* handle,
    uint8_t** data,
    ssize_t* size
);


#endif /* IMCD_H */
