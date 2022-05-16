# imagecodecs/qoi.pxd
# cython: language_level = 3

# Cython declarations for the `qoi 75e7f30` library.
# https://github.com/phoboslab/qoi

cdef extern from 'qoi.h':
    int QOI_SRGB
    int QOI_LINEAR

    ctypedef struct qoi_desc:
        unsigned int width
        unsigned int height
        unsigned char channels
        unsigned char colorspace

    int qoi_write(
        const char* filename,
        const void* data,
        const qoi_desc* desc
    ) nogil

    void* qoi_read(
        const char* filename,
        qoi_desc* desc,
        int channels
    ) nogil

    void* qoi_encode(
        const void* data,
        const qoi_desc* desc,
        int* out_len
    ) nogil

    void* qoi_decode(
        const void* data,
        int size,
        qoi_desc* desc,
        int channels
    ) nogil
