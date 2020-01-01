# blosc.pxd
# cython: language_level = 3

# Cython declarations for the `Blosc 1.17.1` C library.
# https://github.com/Blosc/c-blosc

cdef extern from 'blosc.h':

    char* BLOSC_VERSION_STRING

    int BLOSC_MAX_OVERHEAD
    int BLOSC_NOSHUFFLE
    int BLOSC_SHUFFLE
    int BLOSC_BITSHUFFLE

    int blosc_compress_ctx(
        int clevel,
        int doshuffle,
        size_t typesize,
        size_t nbytes,
        const void* src,
        void* dest,
        size_t destsize,
        const char* compressor,
        size_t blocksize,
        int numinternalthreads) nogil

    int blosc_decompress_ctx(
        const void* src,
        void* dest,
        size_t destsize,
        int numinternalthreads) nogil

    void blosc_cbuffer_sizes(
        const void* cbuffer,
        size_t* nbytes,
        size_t* cbytes,
        size_t* blocksize) nogil

    int blosc_get_blocksize() nogil
