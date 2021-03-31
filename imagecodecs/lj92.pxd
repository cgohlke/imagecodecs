# imagecodecs/lj92.pxd
# cython: language_level = 3

# Cython declarations for the `liblj92-2014` library.
# https://bitbucket.org/baldand/mlrawviewer/src/master/liblj92/

from libc.stdint cimport uint8_t, uint16_t

cdef extern from 'lj92.h':

    enum LJ92_ERRORS:
        LJ92_ERROR_NONE
        LJ92_ERROR_CORRUPT
        LJ92_ERROR_NO_MEMORY
        LJ92_ERROR_BAD_HANDLE
        LJ92_ERROR_TOO_WIDE

    ctypedef struct lj92:
        pass

    int lj92_open(
        lj92* lj,
        uint8_t* data,
        int datalen,
        int* width,
        int* height,
        int* bitdepth,
        int* components
    ) nogil

    void lj92_close(
        lj92 lj
    ) nogil

    int lj92_decode(
        lj92 lj,
        uint16_t* target,
        int writeLength,
        int skipLength,
        uint16_t* linearize,
        int linearizeLength
    ) nogil

    int lj92_encode(
        uint16_t* image,
        int width,
        int height,
        int bitdepth,
        int components,
        int readLength,
        int skipLength,
        uint16_t* delinearize,
        int delinearizeLength,
        uint8_t** encoded,
        int* encodedLength
    ) nogil
