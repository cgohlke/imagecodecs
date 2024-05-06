# imagecodecs/pcodec.pxd
# cython: language_level = 3

# Cython declarations for the `pcodec 0.2.5` library.
# https://github.com/mwlon/pcodec

cdef extern from 'cpcodec.h':

    unsigned char PCO_TYPE_U32
    unsigned char PCO_TYPE_U64
    unsigned char PCO_TYPE_I32
    unsigned char PCO_TYPE_I64
    unsigned char PCO_TYPE_F32
    unsigned char PCO_TYPE_F64

    ctypedef enum PcoError:
        PcoSuccess
        PcoInvalidType
        PcoCompressionError
        PcoDecompressionError

    ctypedef struct PcoFfiVec:
        const void *ptr
        unsigned int len
        const void *raw_box

    PcoError pco_simpler_compress(
        const void *nums,
        unsigned int len,
        unsigned char dtype,
        unsigned int level,
        PcoFfiVec *dst
    ) nogil

    PcoError pco_simple_decompress(
        const void *compressed,
        unsigned int len,
        unsigned char dtype,
        PcoFfiVec *dst
    ) nogil

    PcoError pco_free_pcovec(
        PcoFfiVec *ffi_vec
    ) nogil
