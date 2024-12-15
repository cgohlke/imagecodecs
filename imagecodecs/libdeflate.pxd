# imagecodecs/libdeflate.pxd
# cython: language_level = 3

# Cython declarations for the `libdeflate 1.23` library.
# https://github.com/ebiggers/libdeflate

from libc.stdint cimport uint32_t

cdef extern from 'libdeflate.h' nogil:

    char* LIBDEFLATE_VERSION_STRING
    int LIBDEFLATE_VER_MAJOR
    int LIBDEFLATE_VER_MINOR

    struct libdeflate_compressor:
        pass

    struct libdeflate_decompressor:
        pass

    struct libdeflate_options:
        pass

    enum libdeflate_result:
        LIBDEFLATE_SUCCESS
        LIBDEFLATE_BAD_DATA
        LIBDEFLATE_SHORT_OUTPUT
        LIBDEFLATE_INSUFFICIENT_SPACE

    # Compression

    libdeflate_compressor* libdeflate_alloc_compressor(
        int compression_level
    )

    libdeflate_compressor* libdeflate_alloc_compressor_ex(
        int compression_level,
        const libdeflate_options *options
    )

    size_t libdeflate_deflate_compress(
        libdeflate_compressor* compressor,
        const void* in_,
        size_t in_nbytes,
        void* out,
        size_t out_nbytes_avail
    )

    size_t libdeflate_deflate_compress_bound(
        libdeflate_compressor* compressor,
        size_t in_nbytes
    )

    size_t libdeflate_zlib_compress(
        libdeflate_compressor* compressor,
        const void* in_,
        size_t in_nbytes,
        void* out,
        size_t out_nbytes_avail
    )

    size_t libdeflate_zlib_compress_bound(
        libdeflate_compressor* compressor,
        size_t in_nbytes
    )

    size_t libdeflate_gzip_compress(
        libdeflate_compressor* compressor,
        const void* in_,
        size_t in_nbytes,
        void* out,
        size_t out_nbytes_avail
    )

    size_t libdeflate_gzip_compress_bound(
        libdeflate_compressor* compressor,
        size_t in_nbytes
    )

    void libdeflate_free_compressor(
        libdeflate_compressor* compressor
    )

    # Decompression

    libdeflate_decompressor* libdeflate_alloc_decompressor()

    libdeflate_decompressor* libdeflate_alloc_decompressor_ex(
        const libdeflate_options *options
    )

    libdeflate_result libdeflate_deflate_decompress(
        libdeflate_decompressor* decompressor,
        const void* in_,
        size_t in_nbytes,
        void* out,
        size_t out_nbytes_avail,
        size_t* actual_out_nbytes_ret
    )

    libdeflate_result libdeflate_deflate_decompress_ex(
        libdeflate_decompressor* decompressor,
        const void* in_,
        size_t in_nbytes,
        void* out,
        size_t out_nbytes_avail,
        size_t* actual_in_nbytes_ret,
        size_t* actual_out_nbytes_ret
    )

    libdeflate_result libdeflate_zlib_decompress(
        libdeflate_decompressor* decompressor,
        const void* in_,
        size_t in_nbytes,
        void* out,
        size_t out_nbytes_avail,
        size_t* actual_out_nbytes_ret
    )

    libdeflate_result libdeflate_zlib_decompress_ex(
        libdeflate_decompressor* decompressor,
        const void* in_,
        size_t in_nbytes,
        void* out,
        size_t out_nbytes_avail,
        size_t* actual_in_nbytes_ret,
        size_t* actual_out_nbytes_ret
    )

    libdeflate_result libdeflate_gzip_decompress(
        libdeflate_decompressor* decompressor,
        const void* in_,
        size_t in_nbytes,
        void* out,
        size_t out_nbytes_avail,
        size_t* actual_out_nbytes_ret
    )

    libdeflate_result libdeflate_gzip_decompress_ex(
        libdeflate_decompressor* decompressor,
        const void* in_,
        size_t in_nbytes,
        void* out,
        size_t out_nbytes_avail,
        size_t* actual_in_nbytes_ret,
        size_t* actual_out_nbytes_ret
    )

    void libdeflate_free_decompressor(
        libdeflate_decompressor* decompressor
    )

    # Checksums

    uint32_t libdeflate_adler32(
        uint32_t adler,
        const void* buffer,
        size_t len
    )

    uint32_t libdeflate_crc32(
        uint32_t crc,
        const void* buffer,
        size_t len
    )

    # Custom memory allocator

    void libdeflate_set_memory_allocator(
        void* (*malloc_func)(size_t),
        void (*free_func)(void*)
    )

    struct libdeflate_options:
        size_t sizeof_options
        void *(*malloc_func)(size_t) nogil
        void (*free_func)(void *) nogil
