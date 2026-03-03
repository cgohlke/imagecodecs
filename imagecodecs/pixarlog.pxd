# imagecodecs/pixarlog.pxd

# Cython declarations for the `pixarlog.c` library.

from libc.stdint cimport uint8_t


cdef extern from 'pixarlog.h' nogil:

    char* PIXARLOG_VERSION

    int PIXARLOG_FMT_8BIT
    int PIXARLOG_FMT_8BITABGR
    int PIXARLOG_FMT_11BITLOG
    int PIXARLOG_FMT_12BITPICIO
    int PIXARLOG_FMT_16BIT
    int PIXARLOG_FMT_FLOAT

    int PIXARLOG_OK
    int PIXARLOG_ERROR
    int PIXARLOG_MEMORY_ERROR
    int PIXARLOG_VALUE_ERROR
    int PIXARLOG_OUTPUT_TOO_SMALL
    int PIXARLOG_ZLIB_ERROR

    void pixarlog_init()

    ssize_t pixarlog_decode_ "pixarlog_decode" (
        const uint8_t* src,
        const ssize_t srcsize,
        uint8_t* dst,
        const ssize_t dstsize,
        const ssize_t width,
        const ssize_t stride,
        const int datafmt
    )

    ssize_t pixarlog_encode_ "pixarlog_encode" (
        const uint8_t* src,
        const ssize_t srcsize,
        uint8_t* dst,
        const ssize_t dstsize,
        const ssize_t width,
        const ssize_t stride,
        const int datafmt,
        const int level
    )

    ssize_t pixarlog_decode_raw_ "pixarlog_decode_raw" (
        const uint8_t* src,
        const ssize_t srcsize,
        uint8_t* dst,
        const ssize_t dstsize,
        const ssize_t width,
        const ssize_t stride,
        const int datafmt
    )

    ssize_t pixarlog_encode_raw_ "pixarlog_encode_raw" (
        const uint8_t* src,
        const ssize_t srcsize,
        uint8_t* dst,
        const ssize_t dstsize,
        const ssize_t width,
        const ssize_t stride,
        const int datafmt
    )
