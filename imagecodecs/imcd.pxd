# imagecodecs/imcd.pxd
# cython: language_level = 3

# Cython declarations for the `imcd 2024.1.1` library.
# https://github.com/cgohlke/imagecodecs

from libc.stdint cimport uint8_t

cdef extern from 'imcd.h' nogil:

    char* IMCD_VERSION

    int IMCD_OK
    int IMCD_ERROR
    int IMCD_MEMORY_ERROR
    int IMCD_RUNTIME_ERROR
    int IMCD_NOTIMPLEMENTED_ERROR
    int IMCD_VALUE_ERROR
    int IMCD_INPUT_CORRUPT
    int IMCD_OUTPUT_TOO_SMALL

    int IMCD_LZW_INVALID
    int IMCD_LZW_NOTIMPLEMENTED
    int IMCD_LZW_BUFFER_TOO_SMALL
    int IMCD_LZW_TABLE_TOO_SMALL
    int IMCD_LZW_CORRUPT

    char IMCD_BOC
    int SSIZE_MAX

    ssize_t imcd_delta(
        void* src,
        const ssize_t srcsize,
        const ssize_t srcstride,
        void* dst,
        const ssize_t dstsize,
        const ssize_t dststride,
        const ssize_t itemsize,
        const bint decode
    )

    ssize_t imcd_xor(
        void* src,
        const ssize_t srcsize,
        const ssize_t srcstride,
        void* dst,
        const ssize_t dstsize,
        const ssize_t dststride,
        const ssize_t itemsize,
        const bint decode
    )

    ssize_t imcd_byteshuffle(
        void* src,
        const ssize_t srcsize,
        const ssize_t srcstride,
        void* dst,
        const ssize_t dstsize,
        const ssize_t dststride,
        const ssize_t itemsize,
        const ssize_t samples,
        const char byteorder,
        const bint diff,
        const bint decode,
    )

    ssize_t imcd_bitorder(
        uint8_t* src,
        const ssize_t srcsize,
        const ssize_t srcstride,
        const ssize_t itemsize,
        uint8_t* dst,
        const ssize_t dstsize,
        const ssize_t dststride
    )

    ssize_t imcd_packbits_decode_size(
        const uint8_t* src,
        const ssize_t srcsize
    )

    ssize_t imcd_packbits_encode_size(
        const ssize_t srcsize
    )

    ssize_t imcd_packbits_decode(
        const uint8_t* src,
        const ssize_t srcsize,
        uint8_t* dst,
        const ssize_t dstsize,
        const ssize_t dststride
    )

    ssize_t imcd_packbits_encode(
        const uint8_t* src,
        const ssize_t srcsize,
        uint8_t* dst,
        const ssize_t dstsize
    )

    ssize_t imcd_ccittrle_decode_size(
        const uint8_t* src,
        const ssize_t srcsize
    )

    ssize_t imcd_ccittrle_encode_size(
        const ssize_t srcsize
    )

    ssize_t imcd_ccittrle_decode(
        const uint8_t* src,
        const ssize_t srcsize,
        uint8_t* dst,
        const ssize_t dstsize
    )

    ssize_t imcd_ccittrle_encode(
        const uint8_t* src,
        const ssize_t srcsize,
        uint8_t* dst,
        const ssize_t dstsize
    )

    ssize_t imcd_packints_decode(
        const uint8_t* src,
        const ssize_t srcsize,
        uint8_t* dst,
        const ssize_t dstsize,
        const int bps
    )

    ssize_t imcd_packints_encode(
        const uint8_t* src,
        const ssize_t srcsize,
        uint8_t* dst,
        const ssize_t dstsize,
        const int bps
    )

    ssize_t imcd_float24_decode(
        const uint8_t* src,
        const ssize_t srcsize,
        uint8_t* dst,
        const char byteorder
    )

    ssize_t imcd_float24_encode(
        const uint8_t* src,
        const ssize_t srcsize,
        uint8_t* dst,
        const char byteorder,
        int rounding
    )

    ssize_t imcd_eer_decode(
        const uint8_t *src,
        const ssize_t srcsize,
        uint8_t *dst,
        const ssize_t height,
        const ssize_t width,
        const int rlebits,
        const int horzbits,
        const int vertbits,
        const bint superres
    )

    void imcd_swapbytes(
        void* src,
        const ssize_t srcsize,
        const ssize_t itemsize
    )

    ctypedef struct imcd_lzw_handle_t:
        pass

    imcd_lzw_handle_t* imcd_lzw_new(
        ssize_t buffersize
    )

    void imcd_lzw_del(
        imcd_lzw_handle_t* handle
    )

    ssize_t imcd_lzw_decode_size(
        imcd_lzw_handle_t* handle,
        const uint8_t* src,
        const ssize_t srcsize
    )

    ssize_t imcd_lzw_decode(
        imcd_lzw_handle_t* handle,
        const uint8_t* src,
        const ssize_t srcsize,
        uint8_t* dst,
        const ssize_t dstsize
    )

    bint imcd_lzw_check(
        const uint8_t* src,
        const ssize_t size
    )

    ssize_t imcd_lzw_encode_size(
        const ssize_t srcsize
    )

    ssize_t imcd_lzw_encode(
        const uint8_t* src,
        const ssize_t srcsize,
        uint8_t* dst,
        const ssize_t dstsize
    )

    ssize_t imcd_memsearch(
        const char *src,
        const ssize_t srclen,
        const char *dst,
        const ssize_t dstlen
    )

    ssize_t imcd_strsearch(
        const char *src,
        const ssize_t srclen,
        const char *dst,
        const ssize_t dstlen
    )

cdef extern from 'fenv.h' nogil:

    int FE_TONEAREST
    int FE_UPWARD
    int FE_DOWNWARD
    int FE_TOWARDZERO

    int fegetround()
    int fesetround(int)
