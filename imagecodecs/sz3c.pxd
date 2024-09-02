# imagecodecs/sz3.pxd
# cython: language_level = 3

# Cython declarations for the `sz3 3.1.8` library.
# https://github.com/szcompressor/SZ3

cdef extern from 'SZ3c/sz3c.h' nogil:

    # error bound mode in SZ2
    int ABS
    int REL
    int VR_REL
    int ABS_AND_REL
    int ABS_OR_REL
    int PSNR
    int NORM
    int PW_REL
    int ABS_AND_PW_REL
    int ABS_OR_PW_REL
    int REL_AND_PW_REL
    int REL_OR_PW_REL

    # dataType in SZ2
    int SZ_FLOAT
    int SZ_DOUBLE
    int SZ_UINT8
    int SZ_INT8
    int SZ_UINT16
    int SZ_INT16
    int SZ_UINT32
    int SZ_INT32
    int SZ_UINT64
    int SZ_INT64

    unsigned char* SZ_compress_args(
        int dataType,
        void* data,
        size_t* outSize,
        int errBoundMode,
        double absErrBound,
        double relBoundRatio,
        double pwrBoundRatio,
        size_t r5,
        size_t r4,
        size_t r3,
        size_t r2,
        size_t r1
    )

    void* SZ_decompress(
        int dataType,
        unsigned char* bytes,
        size_t byteLength,
        size_t r5,
        size_t r4,
        size_t r3,
        size_t r2,
        size_t r1
    )
