# imagecodecs/libdeflate.pxd
# cython: language_level = 3

# Cython declarations for the `libdeflate 1.7` library.
# https://github.com/ebiggers/libdeflate

from libc.stdint cimport uint32_t

cdef extern from 'libdeflate.h':

    char* LIBDEFLATE_VERSION_STRING
    int LIBDEFLATE_VER_MAJOR
    int LIBDEFLATE_VER_MINOR

    struct libdeflate_compressor:
        pass

    struct libdeflate_decompressor:
        pass

    enum libdeflate_result:
        LIBDEFLATE_SUCCESS
        LIBDEFLATE_BAD_DATA
        LIBDEFLATE_SHORT_OUTPUT
        LIBDEFLATE_INSUFFICIENT_SPACE

    # Compression

    libdeflate_compressor* libdeflate_alloc_compressor(
        int compression_level
    ) nogil

    size_t libdeflate_deflate_compress(
        libdeflate_compressor* compressor,
        const void* in_,
        size_t in_nbytes,
        void* out,
        size_t out_nbytes_avail
    ) nogil

    size_t libdeflate_deflate_compress_bound(
        libdeflate_compressor* compressor,
        size_t in_nbytes
    ) nogil

    size_t libdeflate_zlib_compress(
        libdeflate_compressor* compressor,
        const void* in_,
        size_t in_nbytes,
        void* out,
        size_t out_nbytes_avail
    ) nogil

    size_t libdeflate_zlib_compress_bound(
        libdeflate_compressor* compressor,
        size_t in_nbytes
    ) nogil

    size_t libdeflate_gzip_compress(
        libdeflate_compressor* compressor,
        const void* in_,
        size_t in_nbytes,
        void* out,
        size_t out_nbytes_avail
    ) nogil

    size_t libdeflate_gzip_compress_bound(
        libdeflate_compressor* compressor,
        size_t in_nbytes
    ) nogil

    void libdeflate_free_compressor(
        libdeflate_compressor* compressor
    ) nogil

    # Decompression

    libdeflate_decompressor* libdeflate_alloc_decompressor() nogil

    libdeflate_result libdeflate_deflate_decompress(
        libdeflate_decompressor* decompressor,
        const void* in_,
        size_t in_nbytes,
        void* out,
        size_t out_nbytes_avail,
        size_t* actual_out_nbytes_ret
    ) nogil

    libdeflate_result libdeflate_deflate_decompress_ex(
        libdeflate_decompressor* decompressor,
        const void* in_,
        size_t in_nbytes,
        void* out,
        size_t out_nbytes_avail,
        size_t* actual_in_nbytes_ret,
        size_t* actual_out_nbytes_ret
    ) nogil

    libdeflate_result libdeflate_zlib_decompress(
        libdeflate_decompressor* decompressor,
        const void* in_,
        size_t in_nbytes,
        void* out,
        size_t out_nbytes_avail,
        size_t* actual_out_nbytes_ret
    ) nogil

    libdeflate_result libdeflate_zlib_decompress_ex(
        libdeflate_decompressor* decompressor,
        const void* in_,
        size_t in_nbytes,
        void* out,
        size_t out_nbytes_avail,
        size_t* actual_in_nbytes_ret,
        size_t* actual_out_nbytes_ret
    ) nogil

    libdeflate_result libdeflate_gzip_decompress(
        libdeflate_decompressor* decompressor,
        const void* in_,
        size_t in_nbytes,
        void* out,
        size_t out_nbytes_avail,
        size_t* actual_out_nbytes_ret
    ) nogil

    libdeflate_result libdeflate_gzip_decompress_ex(
        libdeflate_decompressor* decompressor,
        const void* in_,
        size_t in_nbytes,
        void* out,
        size_t out_nbytes_avail,
        size_t* actual_in_nbytes_ret,
        size_t* actual_out_nbytes_ret
    ) nogil

    void libdeflate_free_decompressor(
        libdeflate_decompressor* decompressor
    ) nogil

    # Checksums

    uint32_t libdeflate_adler32(
        uint32_t adler,
        const void* buffer,
        size_t len
    ) nogil

    uint32_t libdeflate_crc32(
        uint32_t crc,
        const void* buffer,
        size_t len
    ) nogil

    # Custom memory allocator

    void libdeflate_set_memory_allocator(
        void* (*malloc_func)(size_t),
        void (*free_func)(void*)
    ) nogil
