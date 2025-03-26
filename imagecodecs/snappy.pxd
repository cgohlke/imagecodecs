# imagecodecs/snappy.pxd
# cython: language_level = 3

# Cython declarations for the `Snappy 1.2.2` library.
# https://github.com/google/snappy

from libc.stdint cimport uint8_t, uint32_t

cdef extern from 'snappy-c.h' nogil:

    ctypedef enum snappy_status:
        SNAPPY_OK
        SNAPPY_INVALID_INPUT
        SNAPPY_BUFFER_TOO_SMALL

    snappy_status snappy_compress(
        const char* input,
        size_t input_length,
        char* compressed,
        size_t* compressed_length
    )

    snappy_status snappy_uncompress(
        const char* compressed,
        size_t compressed_length,
        char* uncompressed,
        size_t* uncompressed_length
    )

    size_t snappy_max_compressed_length(
        size_t source_length
    )

    snappy_status snappy_uncompressed_length(
        const char* compressed,
        size_t compressed_length,
        size_t* result
    )

    snappy_status snappy_validate_compressed_buffer(
        const char* compressed,
        size_t compressed_length
    )
