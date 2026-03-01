# imagecodecs/ccitt.pxd

# Cython declarations for the `ccitt.c` library.

from libc.stdint cimport uint8_t


cdef extern from 'ccitt.h' nogil:

    char* CCITT_VERSION

    int CCITT_OK
    int CCITT_ERROR
    int CCITT_MEMORY_ERROR
    int CCITT_RUNTIME_ERROR
    int CCITT_VALUE_ERROR
    int CCITT_INPUT_CORRUPT
    int CCITT_OUTPUT_TOO_SMALL

    void ccitt_lut_init()

    ssize_t ccitt_rle_decode_size(
        const uint8_t* src,
        const ssize_t srcsize,
        const ssize_t rowlen
    )

    ssize_t ccitt_rle_decode(
        const uint8_t* src,
        const ssize_t srcsize,
        uint8_t* dst,
        const ssize_t dstsize,
        const ssize_t rowlen
    )

    ssize_t ccitt_fax3_decode_size(
        const uint8_t* src,
        const ssize_t srcsize,
        const ssize_t rowlen,
        const int t4options
    )

    ssize_t ccitt_fax3_decode(
        const uint8_t* src,
        const ssize_t srcsize,
        uint8_t* dst,
        const ssize_t dstsize,
        const ssize_t rowlen,
        const int t4options
    )

    ssize_t ccitt_fax4_decode_size(
        const uint8_t* src,
        const ssize_t srcsize,
        const ssize_t rowlen
    )

    ssize_t ccitt_fax4_decode(
        const uint8_t* src,
        const ssize_t srcsize,
        uint8_t* dst,
        const ssize_t dstsize,
        const ssize_t rowlen
    )
