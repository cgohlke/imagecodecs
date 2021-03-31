# imagecodecs/szlib.pxd
# cython: language_level = 3

# Cython declarations for the `szlib 1.0.4` library (part of libaec).
# https://gitlab.dkrz.de/k202009/libaec

cdef extern from 'szlib.h':

    int SZ_ALLOW_K13_OPTION_MASK
    int SZ_CHIP_OPTION_MASK
    int SZ_EC_OPTION_MASK
    int SZ_LSB_OPTION_MASK
    int SZ_MSB_OPTION_MASK
    int SZ_NN_OPTION_MASK
    int SZ_RAW_OPTION_MASK

    int SZ_OK
    int SZ_OUTBUFF_FULL

    int SZ_NO_ENCODER_ERROR
    int SZ_PARAM_ERROR
    int SZ_MEM_ERROR

    int SZ_MAX_PIXELS_PER_BLOCK
    int SZ_MAX_BLOCKS_PER_SCANLINE

    struct SZ_com_t:
        int options_mask
        int bits_per_pixel
        int pixels_per_block
        int pixels_per_scanline

    int SZ_BufftoBuffCompress(
        void* dest,
        size_t* destLen,
        const void* source,
        size_t sourceLen,
        SZ_com_t* param
    ) nogil

    int SZ_BufftoBuffDecompress(
        void* dest,
        size_t* destLen,
        const void* source,
        size_t sourceLen,
        SZ_com_t* param
    ) nogil

    int SZ_encoder_enabled() nogil
