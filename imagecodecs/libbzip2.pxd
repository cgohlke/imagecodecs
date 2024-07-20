# imagecodecs/libbzip2.pxd
# cython: language_level = 3

# Cython declarations for the `libbzip2 1.0.8` library.
# https://sourceware.org/bzip2/

cdef extern from 'bzlib.h' nogil:

    int BZ_RUN
    int BZ_FLUSH
    int BZ_FINISH

    int BZ_OK
    int BZ_RUN_OK
    int BZ_FLUSH_OK
    int BZ_FINISH_OK
    int BZ_STREAM_END
    int BZ_SEQUENCE_ERROR
    int BZ_PARAM_ERROR
    int BZ_MEM_ERROR
    int BZ_DATA_ERROR
    int BZ_DATA_ERROR_MAGIC
    int BZ_IO_ERROR
    int BZ_UNEXPECTED_EOF
    int BZ_OUTBUFF_FULL
    int BZ_CONFIG_ERROR

    ctypedef struct bz_stream:
        char* next_in
        unsigned int avail_in
        unsigned int total_in_lo32
        unsigned int total_in_hi32
        char* next_out
        unsigned int avail_out
        unsigned int total_out_lo32
        unsigned int total_out_hi32
        void* state
        void* (*bzalloc)(void*, int, int) nogil
        void (*bzfree)(void*, void*) nogil
        void* opaque

    int BZ2_bzCompressInit(
        bz_stream* strm,
        int blockSize100k,
        int verbosity,
        int workFactor
    )

    int BZ2_bzCompress(
        bz_stream* strm,
        int action
    )

    int BZ2_bzCompressEnd(
        bz_stream* strm
    )

    int BZ2_bzDecompressInit(
        bz_stream* strm,
        int verbosity,
        int small
    )

    int BZ2_bzDecompress(
        bz_stream* strm
    )

    int BZ2_bzDecompressEnd(
        bz_stream* strm
    )

    const char* BZ2_bzlibVersion()
