# imagecodecs/bitshuffle.pxd
# cython: language_level = 3

# Cython declarations for the `Bitshuffle 0.5.1` library.
# https://github.com/kiyo-masui/bitshuffle

from libc.stdint cimport int64_t

cdef extern from 'bitshuffle.h':

    int BSHUF_VERSION_MAJOR
    int BSHUF_VERSION_MINOR
    int BSHUF_VERSION_POINT

    int bshuf_using_NEON() nogil
    int bshuf_using_SSE2() nogil
    int bshuf_using_AVX2() nogil
    int bshuf_using_AVX512() nogil

    size_t bshuf_default_block_size(
        const size_t elem_size
    ) nogil

    size_t bshuf_compress_lz4_bound(
        const size_t size,
        const size_t elem_size,
        size_t block_size
    ) nogil

    int64_t bshuf_compress_lz4(
        const void* inp,
        void* out,
        const size_t size,
        const size_t elem_size,
        size_t block_size
    ) nogil

    int64_t bshuf_decompress_lz4(
        const void* inp,
        void* out,
        const size_t size,
        const size_t elem_size,
        size_t block_size
    ) nogil

    int64_t bshuf_bitshuffle(
        const void* inp,
        void* out,
        const size_t size,
        const size_t elem_size,
        size_t block_size
    ) nogil

    int64_t bshuf_bitunshuffle(
        const void* inp,
        void* out,
        const size_t size,
        const size_t elem_size,
        size_t block_size
    ) nogil

    size_t bshuf_compress_zstd_bound(
        const size_t size,
        const size_t elem_size,
        size_t block_size
    ) nogil

    int64_t bshuf_compress_zstd(
        const void* inp,
        void* out,
        const size_t size,
        const size_t
        elem_size,
        size_t block_size,
        const int comp_lvl
    ) nogil

    int64_t bshuf_decompress_zstd(
        const void* inp,
        void* out,
        const size_t size,
        const size_t elem_size,
        size_t block_size
     ) nogil
