# imagecodecs/imcd.pxd
# cython: language_level = 3

# Cython declarations for the `imcd 2020.12.22` library.
# https://www.lfd.uci.edu/~gohlke/

from libc.stdint cimport uint8_t

cdef extern from 'imcd.h':

    char* IMCD_VERSION

    int IMCD_OK
    int IMCD_ERROR
    int IMCD_MEMORY_ERROR
    int IMCD_RUNTIME_ERROR
    int IMCD_NOTIMPLEMENTED_ERROR
    int IMCD_VALUE_ERROR
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
        const int decode
    ) nogil

    ssize_t imcd_xor(
        void* src,
        const ssize_t srcsize,
        const ssize_t srcstride,
        void* dst,
        const ssize_t dstsize,
        const ssize_t dststride,
        const ssize_t itemsize,
        const int decode
    ) nogil

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
    ) nogil

    ssize_t imcd_bitorder(
        uint8_t* src,
        const ssize_t srcsize,
        const ssize_t srcstride,
        const ssize_t itemsize,
        uint8_t* dst,
        const ssize_t dstsize,
        const ssize_t dststride
    ) nogil

    ssize_t imcd_packbits_size(
        const uint8_t* src,
        const ssize_t srcsize
    ) nogil

    ssize_t imcd_packbits_decode(
        const uint8_t* src,
        const ssize_t srcsize,
        uint8_t* dst,
        const ssize_t dstsize
    ) nogil

    ssize_t imcd_packbits_encode(
        const uint8_t* src,
        const ssize_t srcsize,
        uint8_t* dst,
        const ssize_t dstsize
    ) nogil

    ssize_t imcd_packints_decode(
        const uint8_t* src,
        const ssize_t srcsize,
        uint8_t* dst,
        const ssize_t dstsize,
        const int bps
    ) nogil

    ssize_t imcd_packints_encode(
        const uint8_t* src,
        const ssize_t srcsize,
        uint8_t* dst,
        const ssize_t dstsize,
        const int bps
    ) nogil

    ssize_t imcd_float24_decode(
        const uint8_t* src,
        const ssize_t srcsize,
        uint8_t* dst,
        const char byteorder
    ) nogil

    ssize_t imcd_float24_encode(
        const uint8_t* src,
        const ssize_t srcsize,
        uint8_t* dst,
        const char byteorder,
        int rounding
    ) nogil

    void imcd_swapbytes(
        void* src,
        const ssize_t srcsize,
        const ssize_t itemsize
    ) nogil

    ctypedef struct imcd_lzw_handle_t:
        pass

    imcd_lzw_handle_t* imcd_lzw_new(
        ssize_t buffersize
    ) nogil

    void imcd_lzw_del(
        imcd_lzw_handle_t* handle
    ) nogil

    ssize_t imcd_lzw_decode_size(
        imcd_lzw_handle_t* handle,
        const uint8_t* src,
        const ssize_t srcsize
    ) nogil

    ssize_t imcd_lzw_decode(
        imcd_lzw_handle_t* handle,
        const uint8_t* src,
        const ssize_t srcsize,
        uint8_t* dst,
        const ssize_t dstsize
    ) nogil

    bint imcd_lzw_check(
        const uint8_t* src,
        const ssize_t size
    ) nogil


cdef extern from 'fenv.h':

    int FE_TONEAREST
    int FE_UPWARD
    int FE_DOWNWARD
    int FE_TOWARDZERO

    int fegetround() nogil
    int fesetround(int) nogil
