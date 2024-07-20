# imagecodecs/lzfse.pxd
# cython: language_level = 3

# Cython declarations for the `lzfse 1.0` library.
# https://github.com/lzfse/lzfse

from libc.stdint cimport uint8_t

cdef extern from 'lzfse.h' nogil:

    size_t lzfse_encode_scratch_size()

    size_t lzfse_encode_buffer(
        uint8_t * dst_buffer,
        size_t dst_size,
        const uint8_t *src_buffer,
        size_t src_size,
        void *scratch_buffer
    )

    size_t lzfse_decode_scratch_size()

    size_t lzfse_decode_buffer(
        uint8_t *dst_buffer,
        size_t dst_size,
        const uint8_t *src_buffer,
        size_t src_size,
        void *scratch_buffer
    )
