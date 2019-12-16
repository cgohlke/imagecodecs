# -*- coding: utf-8 -*-
# lz4.pxd
# cython: language_level = 3

# Cython declarations for the `lz4 1.9.2` library.
# https://github.com/lz4/lz4

cdef extern from 'lz4.h':

    int LZ4_VERSION_MAJOR
    int LZ4_VERSION_MINOR
    int LZ4_VERSION_RELEASE

    int LZ4_MAX_INPUT_SIZE

    int LZ4_compressBound(int isize) nogil

    int LZ4_compress_fast(
        const char* src,
        char* dst,
        int srcSize,
        int dstCapacity,
        int acceleration) nogil

    int LZ4_decompress_safe(
        const char* src,
        char* dst,
        int compressedSize,
        int dstCapacity) nogil
