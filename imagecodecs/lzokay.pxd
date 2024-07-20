# imagecodecs/lzokay.pxd
# cython: language_level = 3

# Cython declarations for the `lzokay` library.
# https://github.com/AxioDL/lzokay

from libc.stdint cimport uint8_t

cdef extern from 'lzokay/lzokay-c.h' nogil:

    ctypedef enum lzokay_EResult:
        EResult_LookbehindOverrun
        EResult_OutputOverrun
        EResult_InputOverrun
        EResult_Error
        EResult_Success
        EResult_InputNotConsumed

    lzokay_EResult lzokay_decompress(
        const uint8_t* src,
        size_t src_size,
        uint8_t* output,
        size_t* output_len
    )
