# imagecodecs/blosc.pxd
# cython: language_level = 3

# Cython declarations for the `Blosc 1.21.0` C library.
# https://github.com/Blosc/c-blosc

cdef extern from 'blosc.h':

    char* BLOSC_VERSION_STRING
    int BLOSC_VERSION_MAJOR
    int BLOSC_VERSION_MINOR
    int BLOSC_VERSION_RELEASE

    int BLOSC_MAX_OVERHEAD
    int BLOSC_NOSHUFFLE
    int BLOSC_SHUFFLE
    int BLOSC_BITSHUFFLE

    int BLOSC_BLOSCLZ
    int BLOSC_LZ4
    int BLOSC_LZ4HC
    int BLOSC_SNAPPY
    int BLOSC_ZLIB
    int BLOSC_ZSTD

    char* BLOSC_BLOSCLZ_COMPNAME
    char* BLOSC_LZ4_COMPNAME
    char* BLOSC_LZ4HC_COMPNAME
    char* BLOSC_SNAPPY_COMPNAME
    char* BLOSC_ZLIB_COMPNAME
    char* BLOSC_ZSTD_COMPNAME

    void blosc_init() nogil

    void blosc_destroy() nogil

    int blosc_compress(
        int clevel,
        int doshuffle,
        size_t typesize,
        size_t nbytes,
        const void *src,
        void *dest,
        size_t destsize
    ) nogil

    int blosc_compress_ctx(
        int clevel, int doshuffle,
        size_t typesize,
        size_t nbytes,
        const void* src,
        void* dest,
        size_t destsize,
        const char* compressor,
        size_t blocksize,
        int numinternalthreads
    ) nogil

    int blosc_decompress(
        const void *src,
        void *dest,
        size_t destsize
    ) nogil

    int blosc_decompress_ctx(
        const void *src,
        void *dest,
        size_t destsize,
        int numinternalthreads
    ) nogil

    int blosc_getitem(
        const void *src,
        int start,
        int nitems,
        void *dest
    ) nogil

    int blosc_get_nthreads() nogil

    int blosc_set_nthreads(
        int nthreads
    ) nogil

    const char* blosc_get_compressor() nogil

    int blosc_set_compressor(
        const char* compname
    ) nogil

    int blosc_compcode_to_compname(
        int compcode,
        const char **compname
    ) nogil

    int blosc_compname_to_compcode(
        const char *compname
    ) nogil

    const char* blosc_list_compressors() nogil

    const char* blosc_get_version_string() nogil

    int blosc_get_complib_info(
        const char *compname,
        char **complib,
        char **version
    ) nogil

    int blosc_free_resources() nogil

    void blosc_cbuffer_sizes(
        const void *cbuffer,
        size_t *nbytes,
        size_t *cbytes,
        size_t *blocksize
    ) nogil

    int blosc_cbuffer_validate(
        const void* cbuffer,
        size_t cbytes,
        size_t* nbytes
    ) nogil

    void blosc_cbuffer_metainfo(
        const void *cbuffer,
        size_t *typesize,
        int *flags
    ) nogil

    void blosc_cbuffer_versions(
        const void *cbuffer,
        int *version,
        int *compversion
    ) nogil

    const char *blosc_cbuffer_complib(
        const void *cbuffer
    ) nogil

    int blosc_get_blocksize() nogil

    void blosc_set_blocksize(
        size_t blocksize
    ) nogil

    void blosc_set_splitmode(
        int splitmode
    ) nogil
