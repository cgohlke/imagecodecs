# imagecodecs/blosc.pxd
# cython: language_level = 3

# Cython declarations for the `c-blosc 1.21.6` library.
# https://github.com/Blosc/c-blosc

cdef extern from 'blosc.h' nogil:

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

    void blosc_init()

    void blosc_destroy()

    int blosc_compress(
        int clevel,
        int doshuffle,
        size_t typesize,
        size_t nbytes,
        const void *src,
        void *dest,
        size_t destsize
    )

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
    )

    int blosc_decompress(
        const void *src,
        void *dest,
        size_t destsize
    )

    int blosc_decompress_ctx(
        const void *src,
        void *dest,
        size_t destsize,
        int numinternalthreads
    )

    int blosc_getitem(
        const void *src,
        int start,
        int nitems,
        void *dest
    )

    int blosc_get_nthreads()

    int blosc_set_nthreads(
        int nthreads
    )

    const char* blosc_get_compressor()

    int blosc_set_compressor(
        const char* compname
    )

    int blosc_compcode_to_compname(
        int compcode,
        const char **compname
    )

    int blosc_compname_to_compcode(
        const char *compname
    )

    const char* blosc_list_compressors()

    const char* blosc_get_version_string()

    int blosc_get_complib_info(
        const char *compname,
        char **complib,
        char **version
    )

    int blosc_free_resources()

    void blosc_cbuffer_sizes(
        const void *cbuffer,
        size_t *nbytes,
        size_t *cbytes,
        size_t *blocksize
    )

    int blosc_cbuffer_validate(
        const void* cbuffer,
        size_t cbytes,
        size_t* nbytes
    )

    void blosc_cbuffer_metainfo(
        const void *cbuffer,
        size_t *typesize,
        int *flags
    )

    void blosc_cbuffer_versions(
        const void *cbuffer,
        int *version,
        int *compversion
    )

    const char *blosc_cbuffer_complib(
        const void *cbuffer
    )

    int blosc_get_blocksize()

    void blosc_set_blocksize(
        size_t blocksize
    )

    void blosc_set_splitmode(
        int splitmode
    )
