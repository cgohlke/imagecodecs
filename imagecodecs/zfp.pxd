# imagecodecs/zfp.pxd
# cython: language_level = 3

# Cython declarations for the `zfp 0.5.5` library.
# https://github.com/LLNL/zfp

cdef extern from 'bitstream.h':

    ctypedef struct bitstream:
        pass

    bitstream* stream_open(
        void* buffer,
        size_t bytes
    ) nogil

    void stream_close(
        bitstream* stream
    ) nogil


cdef extern from 'zfp.h':

    char* ZFP_VERSION_STRING

    int ZFP_HEADER_MAGIC
    int ZFP_HEADER_META
    int ZFP_HEADER_MODE
    int ZFP_HEADER_FULL
    int ZFP_MIN_BITS
    int ZFP_MAX_BITS
    int ZFP_MAX_PREC
    int ZFP_MIN_EXP

    ctypedef unsigned int uint

    ctypedef enum zfp_exec_policy:
        zfp_exec_serial
        zfp_exec_omp
        zfp_exec_cuda

    ctypedef enum zfp_type:
        zfp_type_none
        zfp_type_int32
        zfp_type_int64
        zfp_type_float
        zfp_type_double

    ctypedef enum zfp_mode:
        zfp_mode_null
        zfp_mode_expert
        zfp_mode_fixed_rate
        zfp_mode_fixed_precision
        zfp_mode_fixed_accuracy
        zfp_mode_reversible

    ctypedef struct zfp_exec_params_omp:
        uint threads
        uint chunk_size

    ctypedef union zfp_exec_params:
        zfp_exec_params_omp omp

    ctypedef struct zfp_execution:
        zfp_exec_policy policy
        zfp_exec_params params

    ctypedef struct zfp_stream:
        uint minbits
        uint maxbits
        uint maxprec
        int minexp
        bitstream* stream
        zfp_execution zexec 'exec'

    ctypedef struct zfp_field:
        zfp_type dtype 'type'
        uint nx, ny, nz, nw
        int sx, sy, sz, sw
        void* data

    zfp_stream* zfp_stream_open(
        zfp_stream*
    ) nogil

    void zfp_stream_close(
        zfp_stream*
    ) nogil

    void zfp_stream_rewind(
        zfp_stream*
    ) nogil

    void zfp_stream_set_bit_stream(
        zfp_stream*,
        bitstream*
    ) nogil

    size_t zfp_stream_flush(
        zfp_stream*
    ) nogil

    size_t zfp_write_header(
        zfp_stream*,
        const zfp_field*,
        uint mask
    ) nogil

    size_t zfp_read_header(
        zfp_stream*,
        zfp_field*,
        uint mask
    ) nogil

    size_t zfp_stream_maximum_size(
        const zfp_stream*,
        const zfp_field*
    ) nogil

    size_t zfp_stream_compressed_size(
        const zfp_stream*
    ) nogil

    size_t zfp_compress(
        zfp_stream*,
        const zfp_field*
    ) nogil

    size_t zfp_decompress(
        zfp_stream*,
        zfp_field*
    ) nogil

    int zfp_stream_set_execution(
        zfp_stream*,
        zfp_exec_policy
    ) nogil

    void zfp_stream_set_reversible(
        zfp_stream*
    ) nogil

    uint zfp_stream_set_precision(
        zfp_stream*,
        uint precision
    ) nogil

    double zfp_stream_set_accuracy(
        zfp_stream*,
        double tolerance
    ) nogil

    double zfp_stream_set_rate(
        zfp_stream*,
        double rate,
        zfp_type type,
        uint dims,
        int wra
    ) nogil

    int zfp_stream_set_params(
        zfp_stream*,
        uint minbits,
        uint maxbits,
        uint maxprec,
        int minexp
    ) nogil

    zfp_field* zfp_field_alloc() nogil

    zfp_field* zfp_field_1d(
        void*,
        zfp_type,
        uint
    ) nogil

    zfp_field* zfp_field_2d(
        void*,
        zfp_type,
        uint nx,
        uint
    ) nogil

    zfp_field* zfp_field_3d(
        void*,
        zfp_type,
        uint,
        uint,
        uint
    ) nogil

    zfp_field* zfp_field_4d(
        void*,
        zfp_type,
        uint,
        uint,
        uint,
        uint
    ) nogil

    void zfp_field_free(
        zfp_field*
    ) nogil

    void zfp_field_set_pointer(
        zfp_field*,
        void* pointer
    ) nogil

    void zfp_field_set_size_1d(
        zfp_field*,
        uint
    ) nogil

    void zfp_field_set_size_2d(
        zfp_field*,
        uint,
        uint
    ) nogil

    void zfp_field_set_size_3d(
        zfp_field*,
        uint,
        uint,
        uint
    ) nogil

    void zfp_field_set_size_4d(
        zfp_field*,
        uint,
        uint,
        uint,
        uint
    ) nogil

    void zfp_field_set_stride_1d(
        zfp_field*,
        int
    ) nogil

    void zfp_field_set_stride_2d(
        zfp_field*,
        int,
        int
    ) nogil

    void zfp_field_set_stride_3d(
        zfp_field*,
        int,
        int,
        int
    ) nogil

    void zfp_field_set_stride_4d(
        zfp_field*,
        int,
        int,
        int,
        int
    ) nogil

    zfp_type zfp_field_set_type(
        zfp_field*,
        zfp_type type
    ) nogil
