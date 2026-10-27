# imagecodecs/bmp.pxd

# Cython declarations for the `bmp.c` library.

from libc.stdint cimport int32_t, uint8_t, uint16_t, uint32_t


cdef extern from 'bmp.h' nogil:

    char* BMP_VERSION

    int BMP_OK
    int BMP_ERROR
    int BMP_VALUE_ERROR
    int BMP_OUTPUT_TOO_SMALL
    int BMP_INVALID_HEADER
    int BMP_UNSUPPORTED_FORMAT
    int BMP_DATA_TRUNCATED
    int BMP_IMAGE_TOO_LARGE
    int BMP_INDEX_ERROR

    int BMP_BI_RGB
    int BMP_BI_RLE8
    int BMP_BI_RLE4
    int BMP_BI_BITFIELDS
    int BMP_BI_JPEG
    int BMP_BI_PNG

    ctypedef struct bmp_info_t:
        ssize_t width
        ssize_t height
        ssize_t samples
        ssize_t rowstride
        int topdown
        int has_alpha
        uint32_t filesize
        uint32_t offbits
        uint32_t header_size
        uint32_t compression
        uint32_t size_image
        uint32_t clr_used
        uint16_t planes
        uint16_t bitcount
        int32_t x_ppm
        int32_t y_ppm
        uint32_t[4] masks
        int[4] shifts
        int[4] bits
        uint8_t[768] palette

    int bmp_check_ "bmp_check" (
        const uint8_t* src,
        const ssize_t srcsize
    )

    int bmp_info_ "bmp_info" (
        const uint8_t* src,
        const ssize_t srcsize,
        const int asrgb,
        bmp_info_t* info
    )

    ssize_t bmp_decode_ "bmp_decode" (
        const uint8_t* src,
        const ssize_t srcsize,
        const bmp_info_t* info,
        uint8_t* dst,
        const ssize_t dstsize
    )

    ssize_t bmp_encode_size(
        const ssize_t width,
        const ssize_t height,
        const ssize_t samples
    )

    ssize_t bmp_encode_ "bmp_encode" (
        const uint8_t* src,
        const ssize_t width,
        const ssize_t height,
        const ssize_t samples,
        const int32_t ppm,
        uint8_t* dst,
        const ssize_t dstsize
    )
