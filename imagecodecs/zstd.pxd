# zstd.pxd
# cython: language_level = 3

# Cython declarations for the `zstd 1.4.5` library (aka Zstandard).
# https://github.com/facebook/zstd

cdef extern from 'zstd.h':

    int ZSTD_VERSION_MAJOR
    int ZSTD_VERSION_MINOR
    int ZSTD_VERSION_RELEASE
    int ZSTD_VERSION_NUMBER

    int ZSTD_CONTENTSIZE_UNKNOWN
    int ZSTD_CONTENTSIZE_ERROR

    unsigned int ZSTD_isError(size_t code) nogil
    size_t ZSTD_compressBound(size_t srcSize) nogil
    const char* ZSTD_getErrorName(size_t code) nogil

    unsigned ZSTD_versionNumber() nogil
    const char* ZSTD_versionString() nogil

    unsigned long long ZSTD_getFrameContentSize(
        const void* src,
        size_t srcSize) nogil

    size_t ZSTD_decompress(
        void* dst,
        size_t dstCapacity,
        const void* src,
        size_t compressedSize) nogil

    size_t ZSTD_compress(
        void* dst,
        size_t dstCapacity,
        const void* src,
        size_t srcSize,
        int compressionLevel) nogil

# TODO: add missing declarations
