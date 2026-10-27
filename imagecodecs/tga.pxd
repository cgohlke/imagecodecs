# imagecodecs/tga.pxd

# Cython declarations for the `tga.c` library.

from libc.stdint cimport uint8_t, uint16_t, uint32_t


cdef extern from 'tga.h' nogil:

    char* TGA_VERSION

    int TGA_OK
    int TGA_ERROR
    int TGA_VALUE_ERROR
    int TGA_OUTPUT_TOO_SMALL
    int TGA_INVALID_HEADER
    int TGA_UNSUPPORTED_FORMAT
    int TGA_DATA_TRUNCATED
    int TGA_IMAGE_TOO_LARGE

    int TGA_TYPE_NONE
    int TGA_TYPE_COLORMAP
    int TGA_TYPE_TRUECOLOR
    int TGA_TYPE_GRAYSCALE
    int TGA_TYPE_RLE_COLORMAP
    int TGA_TYPE_RLE_TRUECOLOR
    int TGA_TYPE_RLE_GRAYSCALE

    ctypedef struct tga_extension_t:
        int version
        int valid
        char[42] author
        char comments[4][82]
        uint16_t month
        uint16_t day
        uint16_t year
        uint16_t hour
        uint16_t minute
        uint16_t second
        char[42] job_name
        uint16_t job_hours
        uint16_t job_minutes
        uint16_t job_seconds
        char[42] software
        uint16_t software_version
        uint32_t key_color
        uint16_t aspect_numerator
        uint16_t aspect_denominator
        uint16_t gamma_numerator
        uint16_t gamma_denominator
        uint32_t color_correction_offset
        uint32_t postage_stamp_offset
        uint32_t scanline_offset
        uint8_t software_letter
        uint8_t attributes_type

    ctypedef struct tga_info_t:
        ssize_t width
        ssize_t height
        ssize_t samples
        ssize_t pixel_size
        ssize_t pixel_offset
        ssize_t colormap_offset
        ssize_t cmap_entry_size
        int alpha_bits
        int topdown
        int righttoleft
        int rle
        int colormapped
        uint8_t id_length
        uint8_t colormap_type
        uint8_t image_type
        uint8_t colormap_depth
        uint8_t pixel_depth
        uint8_t image_descriptor
        uint16_t colormap_origin
        uint16_t colormap_length
        uint16_t x_origin
        uint16_t y_origin
        char[256] image_id
        tga_extension_t extension

    int tga_check_ "tga_check" (
        const uint8_t* src,
        const ssize_t srcsize
    )

    int tga_info_ "tga_info" (
        const uint8_t* src,
        const ssize_t srcsize,
        tga_info_t* info
    )

    ssize_t tga_decode_ "tga_decode" (
        const uint8_t* src,
        const ssize_t srcsize,
        const tga_info_t* info,
        uint8_t* dst,
        const ssize_t dstsize
    )

    ssize_t tga_encode_size(
        const ssize_t width,
        const ssize_t height,
        const ssize_t samples,
        const int rle
    )

    ssize_t tga_encode_ "tga_encode" (
        const uint8_t* src,
        const ssize_t width,
        const ssize_t height,
        const ssize_t samples,
        const int rle,
        uint8_t* dst,
        const ssize_t dstsize
    )
