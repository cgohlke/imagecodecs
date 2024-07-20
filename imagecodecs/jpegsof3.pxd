# imagecodecs/jpegsof3.pxd
# cython: language_level = 3

# Cython declarations for the `jpegsof3 2023.1.23` library.
# https://github.com/cgohlke/imagecodecs/tree/master/3rdparty/jpegsof3/

cdef extern from 'jpegsof3.h' nogil:

    char* JPEGSOF3_VERSION

    int JPEGSOF3_OK
    int JPEGSOF3_INVALID_OUTPUT
    int JPEGSOF3_INVALID_SIGNATURE
    int JPEGSOF3_INVALID_HEADER_TAG
    int JPEGSOF3_SEGMENT_GT_IMAGE
    int JPEGSOF3_INVALID_ITU_T81
    int JPEGSOF3_INVALID_BIT_DEPTH
    int JPEGSOF3_TABLE_CORRUPTED
    int JPEGSOF3_TABLE_SIZE_CORRUPTED
    int JPEGSOF3_INVALID_RESTART_SEGMENTS
    int JPEGSOF3_NO_TABLE

    int decode_jpegsof3(
        unsigned char* lRawRA,
        ssize_t lRawSz,
        unsigned char* lImgRA8,
        ssize_t lImgSz,
        int* dimX,
        int* dimY,
        int* bits,
        int* frames
    )
