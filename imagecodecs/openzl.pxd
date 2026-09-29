# imagecodecs/openzl.pxd

# Cython declarations for the `OpenZL 0.3.0` library.
# https://github.com/facebook/openzl

from libc.stddef cimport size_t


cdef extern from 'openzl/openzl.h' nogil:

    int ZL_LIBRARY_VERSION_MAJOR
    int ZL_LIBRARY_VERSION_MINOR
    int ZL_LIBRARY_VERSION_PATCH

    ctypedef size_t ZL_Report

    ctypedef enum ZL_ErrorCode:
        ZL_ErrorCode_no_error

    int ZL_isError(
        ZL_Report report
    )

    ZL_ErrorCode ZL_errorCode(
        ZL_Report report
    )

    size_t ZL_validResult(
        ZL_Report report
    )

    const char* ZL_ErrorCode_toString(
        ZL_ErrorCode code
    )

    ZL_Report ZL_getDecompressedSize(
        const void* compressed,
        size_t cSize
    )

    ZL_Report ZL_getCompressedSize(
        const void* compressed,
        size_t testedSize
    )

    ZL_Report ZL_decompress(
        void* dst,
        size_t dstCapacity,
        const void* src,
        size_t srcSize
    )

    # Frame info API (replaces ZL_getFormatVersionFromFrame)
    ctypedef struct ZL_FrameInfo:
        pass

    ZL_FrameInfo* ZL_FrameInfo_create(
        const void* compressed,
        size_t cSize
    )

    void ZL_FrameInfo_free(
        ZL_FrameInfo* fi
    )

    ZL_Report ZL_FrameInfo_getFormatVersion(
        const ZL_FrameInfo* fi
    )

    # TODO: complete these definitions when the API is more stable
