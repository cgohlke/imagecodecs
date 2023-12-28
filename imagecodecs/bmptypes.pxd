# imagecodecs/bmptypes.pxd
# cython: language_level = 3

# Cython declarations for the BMP file format.

from libc.stdint cimport uint8_t, uint16_t, uint32_t, int32_t

ctypedef enum bmp_colorspace_t:
    LCS_CALIBRATED_RGB = 0
    LCS_sRGB = 1111970419  # b'sRGB'
    LCS_WINDOWS_COLOR_SPACE = 544106839  # b'Win '
    PROFILE_LINKED = 1263421772  # b'LINK'
    PROFILE_EMBEDDED = 1145389645  # b'MBED'

ctypedef enum bmp_compression_t:
    BI_RGB = 0
    BI_RLE8 = 1
    BI_RLE4 = 2
    BI_BITFIELDS = 3
    BI_JPEG = 4
    BI_PNG = 5

# RGBTRIPLE
cdef packed struct bmp_rgbtriple_t:
    uint8_t blue
    uint8_t green
    uint8_t red

# RGBQUAD
cdef packed struct bmp_rgbquad_t:
    uint8_t blue
    uint8_t green
    uint8_t red
    uint8_t reserved

# CIEXYZ  12 bytes
cdef packed struct bmp_ciexyz_t:
    uint32_t x
    uint32_t y
    uint32_t z

# CIEXYZTRIPLE  36 bytes
cdef packed struct bmp_ciexyztriple_t:
    bmp_ciexyz_t red
    bmp_ciexyz_t green
    bmp_ciexyz_t blue

# BITMAPFILEHEADER  14 bytes
cdef packed struct bmp_fileheader_t:
    uint16_t type
    uint32_t size
    uint16_t reserved1
    uint16_t reserved2
    uint32_t offbits

# BITMAPINFOHEADER, BITMAPV4HEADER, BITMAPV5HEADER
cdef packed struct bmp_infoheader_t:
    # v3 40 bytes
    uint32_t size
    int32_t width
    int32_t height
    uint16_t planes
    uint16_t bitcount
    uint32_t compression_type
    uint32_t size_image
    int32_t x_ppm
    int32_t y_ppm
    uint32_t clr_used
    uint32_t clr_important
    # v4 +68 bytes
    uint32_t red_mask
    uint32_t green_mask
    uint32_t blue_mask
    uint32_t alpha_mask
    uint32_t colorspace_type
    bmp_ciexyztriple_t endpoints
    uint32_t gamma_red
    uint32_t gamma_green
    uint32_t gamma_blue
    # v5 +16 bytes
    uint32_t intent
    uint32_t profile_data
    uint32_t profile_size
    uint32_t reserved
