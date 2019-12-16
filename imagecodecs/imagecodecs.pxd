# -*- coding: utf-8 -*-
# icd.pxd
# cython: language_level = 3

# Cython declarations for the `imagecodecs 2019.4.20` C library.
# https://www.lfd.uci.edu/~gohlke/

from libc.stdint cimport uint8_t

cdef extern from 'imagecodecs.h':

    char* ICD_VERSION

    int ICD_OK
    int ICD_ERROR
    int ICD_MEMORY_ERROR
    int ICD_RUNTIME_ERROR
    int ICD_NOTIMPLEMENTED_ERROR
    int ICD_VALUE_ERROR
    int ICD_LZW_INVALID
    int ICD_LZW_NOTIMPLEMENTED
    int ICD_LZW_BUFFER_TOO_SMALL
    int ICD_LZW_TABLE_TOO_SMALL

    char ICD_BOC
    int SSIZE_MAX

    ssize_t icd_delta(
        void *src,
        const ssize_t srcsize,
        const ssize_t srcstride,
        void *dst,
        const ssize_t dstsize,
        const ssize_t dststride,
        const ssize_t itemsize,
        const int decode) nogil

    ssize_t icd_xor(
        void *src,
        const ssize_t srcsize,
        const ssize_t srcstride,
        void *dst,
        const ssize_t dstsize,
        const ssize_t dststride,
        const ssize_t itemsize,
        const int decode) nogil

    ssize_t icd_floatpred(
        void *src,
        const ssize_t srcsize,
        const ssize_t srcstride,
        void *dst,
        const ssize_t dstsize,
        const ssize_t dststride,
        const ssize_t itemsize,
        const ssize_t samples,
        const char byteorder,
        const int decode) nogil

    ssize_t icd_bitorder(
        uint8_t *src,
        const ssize_t srcsize,
        const ssize_t srcstride,
        const ssize_t itemsize,
        uint8_t *dst,
        const ssize_t dstsize,
        const ssize_t dststride) nogil

    ssize_t icd_packbits_size(
        const uint8_t *src,
        const ssize_t srcsize) nogil

    ssize_t icd_packbits_decode(
        const uint8_t *src,
        const ssize_t srcsize,
        uint8_t *dst,
        const ssize_t dstsize) nogil

    ssize_t icd_packbits_encode(
        const uint8_t *src,
        const ssize_t srcsize,
        uint8_t *dst,
        const ssize_t dstsize) nogil

    ssize_t icd_packints_decode(
        const uint8_t *src,
        const ssize_t srcsize,
        uint8_t *dst,
        const ssize_t dstsize,
        const int numbits) nogil

    void icd_swapbytes(
        void *src,
        const ssize_t srcsize,
        const ssize_t itemsize) nogil

    ctypedef struct icd_lzw_handle_t:
        pass

    icd_lzw_handle_t* icd_lzw_new(ssize_t buffersize) nogil

    void icd_lzw_del(icd_lzw_handle_t *handle) nogil

    ssize_t icd_lzw_decode_size(
        icd_lzw_handle_t *handle,
        const uint8_t *src,
        const ssize_t srcsize) nogil

    ssize_t icd_lzw_decode(
        icd_lzw_handle_t *handle,
        const uint8_t *src,
        const ssize_t srcsize,
        uint8_t *dst,
        const ssize_t dstsize) nogil
