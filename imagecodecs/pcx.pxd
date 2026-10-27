# imagecodecs/pcx.pxd

# Cython declarations for the `pcx.c` library.

from libc.stdint cimport uint8_t, uint16_t


cdef extern from 'pcx.h' nogil:

    char* PCX_VERSION

    int PCX_OK
    int PCX_ERROR
    int PCX_VALUE_ERROR
    int PCX_OUTPUT_TOO_SMALL
    int PCX_INVALID_HEADER
    int PCX_UNSUPPORTED_FORMAT
    int PCX_DATA_TRUNCATED
    int PCX_IMAGE_TOO_LARGE
    int PCX_INDEX_ERROR
    int PCX_SHAPE_MISMATCH

    int PCX_DCX_MAXPAGES

    ctypedef struct pcx_info_t:
        ssize_t width
        ssize_t height
        ssize_t samples
        ssize_t bytes_per_line
        ssize_t frames
        ssize_t index
        ssize_t offset
        ssize_t end
        int dcx
        int isbool
        uint8_t version
        uint8_t encoding
        uint8_t bits_per_pixel
        uint8_t nplanes
        uint16_t hdpi
        uint16_t vdpi
        uint16_t palette_type
        uint8_t[768] palette

    int pcx_check_ "pcx_check" (
        const uint8_t* src,
        const ssize_t srcsize
    )

    int pcx_info_ "pcx_info" (
        const uint8_t* src,
        const ssize_t srcsize,
        const ssize_t index,
        pcx_info_t* info
    )

    ssize_t pcx_decode_ "pcx_decode" (
        const uint8_t* src,
        const ssize_t srcsize,
        const ssize_t index,
        pcx_info_t* info,
        uint8_t* dst,
        const ssize_t dstsize
    )

    ssize_t pcx_encode_size(
        const ssize_t width,
        const ssize_t height,
        const ssize_t samples,
        const ssize_t frames
    )

    ssize_t pcx_encode_ "pcx_encode" (
        const uint8_t* src,
        const ssize_t width,
        const ssize_t height,
        const ssize_t samples,
        const ssize_t frames,
        uint8_t* dst,
        const ssize_t dstsize
    )
