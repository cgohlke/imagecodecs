# imagecodecs/pcodec.pxd

# Cython declarations for the `pcodec 1.0.1` library.
# https://github.com/mwlon/pcodec

cdef extern from 'cpcodec.h' nogil:

    enum:
        PCO_TYPE_U32
        PCO_TYPE_U64
        PCO_TYPE_I32
        PCO_TYPE_I64
        PCO_TYPE_F32
        PCO_TYPE_F64
        PCO_TYPE_U16
        PCO_TYPE_I16
        PCO_TYPE_F16
        PCO_TYPE_U8
        PCO_TYPE_I8

    ctypedef enum PcoError:
        PcoSuccess
        PcoInvalidType
        PcoCompressionError
        PcoDecompressionError

    ctypedef struct PcoChunkConfig:
        unsigned int compression_level
        size_t max_page_n

    size_t pco_standalone_guarantee_file_size(
        size_t n,
        unsigned char dtype,
    ) nogil

    PcoError pco_standalone_simple_compress_into(
        const void *nums,
        size_t n,
        unsigned char dtype,
        const PcoChunkConfig *config,
        void *dst,
        size_t dst_cap,
        size_t *n_written,
    ) nogil

    PcoError pco_standalone_simple_decompress_into(
        const void *compressed,
        size_t compressed_len,
        unsigned char dtype,
        void *dst,
        size_t dst_cap,
        size_t *n_written,
    ) nogil
