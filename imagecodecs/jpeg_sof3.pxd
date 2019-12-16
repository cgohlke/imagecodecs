# -*- coding: utf-8 -*-
# jpeg_sof3.pxd
# cython: language_level = 3

# Cython declarations for the `jpeg_sof3 2019.1.1` library.

cdef extern from 'jpeg_sof3.h':

    char* JPEG_SOF3_VERSION

    int JPEG_SOF3_OK
    int JPEG_SOF3_INVALID_OUTPUT
    int JPEG_SOF3_INVALID_SIGNATURE
    int JPEG_SOF3_INVALID_HEADER_TAG
    int JPEG_SOF3_SEGMENT_GT_IMAGE
    int JPEG_SOF3_INVALID_ITU_T81
    int JPEG_SOF3_INVALID_BIT_DEPTH
    int JPEG_SOF3_TABLE_CORRUPTED
    int JPEG_SOF3_TABLE_SIZE_CORRUPTED
    int JPEG_SOF3_INVALID_RESTART_SEGMENTS
    int JPEG_SOF3_NO_TABLE

    int jpeg_sof3_decode(
        unsigned char *lRawRA,
        ssize_t lRawSz,
        unsigned char *lImgRA8,
        ssize_t lImgSz,
        int *dimX,
        int *dimY,
        int *bits,
        int *frames) nogil
