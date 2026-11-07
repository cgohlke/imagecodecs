# imagecodecs/jpeg_tools.pxd

# Cython declarations for the `jpeg_tools.c` library.

from libc.stdint cimport uint8_t


cdef extern from 'jpeg_tools.h' nogil:

    char* JPEG_TOOLS_VERSION

    int JPEG_TOOLS_OK
    int JPEG_TOOLS_ERROR
    int JPEG_TOOLS_VALUE_ERROR
    int JPEG_TOOLS_OUTPUT_TOO_SMALL
    int JPEG_TOOLS_INVALID_HEADER
    int JPEG_TOOLS_UNSUPPORTED_FORMAT
    int JPEG_TOOLS_DATA_TRUNCATED
    int JPEG_TOOLS_IMAGE_TOO_LARGE
    int JPEG_TOOLS_SHAPE_MISMATCH
    int JPEG_TOOLS_TILE_MISSING
    int JPEG_TOOLS_MEMORY_ERROR
    int JPEG_TOOLS_LIBJPEG_ERROR

    int JPEG_TOOLS_MSG_LENGTH

    ctypedef struct jpeg_tools_input_t:
        const uint8_t** data
        const ssize_t* datasize
        ssize_t rows
        ssize_t columns
        const uint8_t* header
        ssize_t headersize
        const uint8_t* tables
        ssize_t tablessize
        int colorspace

    ctypedef struct jpeg_tools_options_t:
        ssize_t crop_top
        ssize_t crop_left
        ssize_t crop_height
        ssize_t crop_width
        int optimize
        int progressive
        int arithmetic
        int restart
        int grayscale
        int copy

    ssize_t jpeg_tools_merge(
        const jpeg_tools_input_t* input,
        uint8_t* dst,
        const ssize_t dstsize
    )

    int jpeg_tools_transcode(
        const jpeg_tools_input_t* input,
        const jpeg_tools_options_t* options,
        uint8_t** dst,
        unsigned long* dstsize,
        char* msg
    )
