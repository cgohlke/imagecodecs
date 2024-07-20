# imagecodecs/lj92.pxd
# cython: language_level = 3

# Cython declarations for the `liblj92 2023.1.23` library.
# https://github.com/cgohlke/imagecodecs/tree/master/3rdparty/liblj92/

from libc.stdint cimport uint8_t, uint16_t

cdef extern from 'lj92.h' nogil:

    char* LJ92_VERSION

    enum LJ92_ERRORS:
        LJ92_ERROR_NONE
        LJ92_ERROR_CORRUPT
        LJ92_ERROR_NO_MEMORY
        LJ92_ERROR_BAD_HANDLE
        LJ92_ERROR_TOO_WIDE
        LJ92_ERROR_ENCODER

    ctypedef struct lj92:
        pass

    int lj92_open(
        lj92* lj,
        const uint8_t* data,
        int datalen,
        int* width,
        int* height,
        int* bitdepth,
        int* components
    )

    void lj92_close(
        lj92 lj
    )

    int lj92_decode(
        lj92 lj,
        uint16_t* target,
        int writeLength,
        int skipLength,
        const uint16_t* linearize,
        int linearizeLength
    )

    int lj92_encode(
        const uint16_t* image,
        int width,
        int height,
        int bitdepth,
        int components,
        int readLength,
        int skipLength,
        const uint16_t* delinearize,
        int delinearizeLength,
        uint8_t** encoded,
        int* encodedLength
    )
