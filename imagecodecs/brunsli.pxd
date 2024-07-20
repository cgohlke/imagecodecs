# imagecodecs/brunsli.pxd
# cython: language_level = 3

# Cython declarations for the `Brunsli 0.1` library.
# https://github.com/google/brunsli

from libc.stdint cimport uint8_t

cdef extern from 'brunsli/decode.h' nogil:

    ctypedef size_t (*DecodeBrunsliSink)(
        void* sink,
        const uint8_t* buf,
        size_t size
    ) nogil

    int DecodeBrunsli(
        size_t size,
        const uint8_t* data,
        void* sink,
        DecodeBrunsliSink out_fun
    )


cdef extern from 'brunsli/encode.h' nogil:

    int EncodeBrunsli(
        size_t size,
        const unsigned char* data,
        void* sink,
        DecodeBrunsliSink out_fun
    )


# ctypedef enum brunsli_status:
#     # defined in brunsli/status.h'
#     BRUNSLI_OK = 0
#     BRUNSLI_NON_REPRESENTABLE
#     BRUNSLI_MEMORY_ERROR
#     BRUNSLI_INVALID_PARAM
#     BRUNSLI_COMPRESSION_ERROR
#     BRUNSLI_INVALID_BRN
#     BRUNSLI_DECOMPRESSION_ERROR
#     BRUNSLI_NOT_ENOUGH_DATA
