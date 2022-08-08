# imagecodecs/zfp.pxd
# cython: language_level = 3

# Cython declarations for the `zfp 1.0.0` library.
# https://github.com/LLNL/zfp

from libc.stdint cimport (
    int8_t, int16_t, int32_t, int64_t, uint64_t, uint8_t, uint16_t
)

ctypedef unsigned int uint
ctypedef int8_t int8
ctypedef int16_t int16
ctypedef int32_t int32
ctypedef int64_t int64
ctypedef uint8_t uint8
ctypedef uint16_t uint16
ctypedef uint64_t uint64

cdef extern from 'zfp/version.h':

    char* ZFP_VERSION_STRING

    int ZFP_VERSION_MAJOR
    int ZFP_VERSION_MINOR
    int ZFP_VERSION_PATCH
    int ZFP_VERSION_TWEAK


cdef extern from 'zfp/bitstream.h':

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

    int ZFP_MIN_BITS
    int ZFP_MAX_BITS
    int ZFP_MAX_PREC
    int ZFP_MIN_EXP

    int ZFP_HEADER_NONE
    int ZFP_HEADER_MAGIC
    int ZFP_HEADER_META
    int ZFP_HEADER_MODE
    int ZFP_HEADER_FULL

    int ZFP_DATA_UNUSED
    int ZFP_DATA_PADDING
    int ZFP_DATA_META
    int ZFP_DATA_MISC
    int ZFP_DATA_PAYLOAD
    int ZFP_DATA_INDEX
    int ZFP_DATA_CACHE
    int ZFP_DATA_HEADER
    int ZFP_DATA_ALL

    int ZFP_META_NULL

    int ZFP_MAGIC_BITS
    int ZFP_META_BITS
    int ZFP_MODE_SHORT_BITS
    int ZFP_MODE_LONG_BITS
    int ZFP_HEADER_MAX_BITS
    int ZFP_MODE_SHORT_MAX

    int ZFP_ROUND_FIRST
    int ZFP_ROUND_NEVER
    int ZFP_ROUND_LAST

    enum bool_:
        zfp_false
        zfp_true

    ctypedef int zfp_bool

    ctypedef enum zfp_exec_policy:
        zfp_exec_serial
        zfp_exec_omp
        zfp_exec_cuda

    ctypedef struct zfp_exec_params_omp:
        uint threads
        uint chunk_size

    ctypedef struct zfp_execution:
        zfp_exec_policy policy
        void* params

    ctypedef struct zfp_stream:
        uint minbits
        uint maxbits
        uint maxprec
        int minexp
        bitstream* stream
        zfp_execution zexec 'exec'

    ctypedef enum zfp_mode:
        zfp_mode_null
        zfp_mode_expert
        zfp_mode_fixed_rate
        zfp_mode_fixed_precision
        zfp_mode_fixed_accuracy
        zfp_mode_reversible

    ctypedef struct expert_t:
        uint minbits
        uint maxbits
        uint maxprec
        int minexp

    ctypedef union arg_t:
        double rate
        uint precision
        double tolerance
        expert_t expert

    ctypedef struct zfp_config:
        zfp_mode mode
        arg_t arg

    ctypedef enum zfp_type:
        zfp_type_none
        zfp_type_int32
        zfp_type_int64
        zfp_type_float
        zfp_type_double

    ctypedef struct zfp_field:
        zfp_type dtype 'type'
        size_t nx, ny, nz, nw
        ptrdiff_t sx, sy, sz, sw
        void* data

    const uint zfp_codec_version
    const uint zfp_library_version
    const char* const zfp_version_string

    size_t zfp_type_size(
        zfp_type dtype
    ) nogil

    zfp_stream* zfp_stream_open(
        bitstream* stream
    ) nogil

    void zfp_stream_close(
        zfp_stream* stream
    ) nogil

    bitstream* zfp_stream_bit_stream(
        const zfp_stream* stream
    ) nogil

    zfp_mode zfp_stream_compression_mode(
        const zfp_stream* stream
    ) nogil

    double zfp_stream_rate(
        const zfp_stream* stream,
        uint dims
    ) nogil

    uint zfp_stream_precision(
        const zfp_stream* stream
    ) nogil

    double zfp_stream_accuracy(
        const zfp_stream* stream
    ) nogil

    uint64 zfp_stream_mode(
        const zfp_stream* stream
    ) nogil

    void zfp_stream_params(
        const zfp_stream* stream,
        uint* minbits,
        uint* maxbits,
        uint* maxprec,
        int* minexp
    ) nogil

    size_t zfp_stream_compressed_size(
        const zfp_stream* stream
    ) nogil

    size_t zfp_stream_maximum_size(
        const zfp_stream* stream,
        const zfp_field* field
    ) nogil

    void zfp_stream_rewind(
        zfp_stream* stream
    ) nogil

    void zfp_stream_set_bit_stream(
        zfp_stream* stream,
        bitstream* bs
    ) nogil

    void zfp_stream_set_reversible(
        zfp_stream* stream
    ) nogil

    double zfp_stream_set_rate(
        zfp_stream* stream,
        double rate,
        zfp_type type,
        uint dims,
        zfp_bool align
    ) nogil

    uint zfp_stream_set_precision(
        zfp_stream* stream,
        uint precision
    ) nogil

    double zfp_stream_set_accuracy(
        zfp_stream* stream,
        double tolerance
    ) nogil

    zfp_mode zfp_stream_set_mode(
        zfp_stream* stream,
        uint64 mode
    ) nogil

    zfp_bool zfp_stream_set_params(
        zfp_stream* stream,
        uint minbits,
        uint maxbits,
        uint maxprec,
        int minexp
    ) nogil

    zfp_exec_policy zfp_stream_execution(
        const zfp_stream* stream
    ) nogil

    uint zfp_stream_omp_threads(
        const zfp_stream* stream
    ) nogil

    uint zfp_stream_omp_chunk_size(
        const zfp_stream* stream
    ) nogil

    zfp_bool zfp_stream_set_execution(
        zfp_stream* stream,
        zfp_exec_policy policy
    ) nogil

    zfp_bool zfp_stream_set_omp_threads(
        zfp_stream* stream,
        uint threads
    ) nogil

    zfp_bool zfp_stream_set_omp_chunk_size(
        zfp_stream* stream,
        uint chunk_size
    ) nogil

    zfp_config zfp_config_none() nogil

    zfp_config zfp_config_rate(
        double rate,
        zfp_bool align
    ) nogil

    zfp_config zfp_config_precision(
        uint precision
    ) nogil

    zfp_config zfp_config_accuracy(
        double tolerance
    ) nogil

    zfp_config zfp_config_reversible() nogil

    zfp_config zfp_config_expert(
        uint minbits,
        uint maxbits,
        uint maxprec,
        int minexp
    ) nogil

    zfp_field* zfp_field_alloc() nogil

    zfp_field* zfp_field_1d(
        void* pointer,
        zfp_type type,
        size_t nx
    ) nogil

    zfp_field* zfp_field_2d(
        void* pointer,
        zfp_type type,
        size_t nx,
        size_t ny
    ) nogil

    zfp_field* zfp_field_3d(
        void* pointer,
        zfp_type type,
        size_t nx,
        size_t ny,
        size_t nz
    ) nogil

    zfp_field* zfp_field_4d(
        void* pointer,
        zfp_type type,
        size_t nx,
        size_t ny,
        size_t nz,
        size_t nw
    ) nogil

    void zfp_field_free(
        zfp_field* field
    ) nogil

    void* zfp_field_pointer(
        const zfp_field* field
    ) nogil

    void* zfp_field_begin(
        const zfp_field* field
    ) nogil

    zfp_type zfp_field_type(
        const zfp_field* field
    ) nogil

    uint zfp_field_precision(
        const zfp_field* field
    ) nogil

    uint zfp_field_dimensionality(
        const zfp_field* field
    ) nogil

    size_t zfp_field_size(
        const zfp_field* field,
        size_t* size
    ) nogil

    size_t zfp_field_size_bytes(
        const zfp_field* field
    ) nogil

    size_t zfp_field_blocks(
        const zfp_field* field
    ) nogil

    zfp_bool zfp_field_stride(
        const zfp_field* field,
        ptrdiff_t* stride
    ) nogil

    zfp_bool zfp_field_is_contiguous(
        const zfp_field* field
    ) nogil

    uint64 zfp_field_metadata(
        const zfp_field* field
    ) nogil

    void zfp_field_set_pointer(
        zfp_field* field,
        void* pointer
    ) nogil

    zfp_type zfp_field_set_type(
        zfp_field* field,
        zfp_type type
    ) nogil

    void zfp_field_set_size_1d(
    zfp_field* field,
    size_t nx
    ) nogil

    void zfp_field_set_size_2d(
        zfp_field* field,
        size_t nx,
        size_t ny
    ) nogil

    void zfp_field_set_size_3d(
        zfp_field* field,
        size_t nx,
        size_t ny,
        size_t nz
    ) nogil

    void zfp_field_set_size_4d(
        zfp_field* field,
        size_t nx,
        size_t ny,
        size_t nz,
        size_t nw
    ) nogil

    void zfp_field_set_stride_1d(
        zfp_field* field,
        ptrdiff_t sx
    ) nogil

    void zfp_field_set_stride_2d(
        zfp_field* field,
        ptrdiff_t sx,
        ptrdiff_t sy
    ) nogil

    void zfp_field_set_stride_3d(
        zfp_field* field,
        ptrdiff_t sx,
        ptrdiff_t sy,
        ptrdiff_t sz
    ) nogil

    void zfp_field_set_stride_4d(
        zfp_field* field,
        ptrdiff_t sx,
        ptrdiff_t sy,
        ptrdiff_t sz,
        ptrdiff_t sw
    ) nogil

    zfp_bool zfp_field_set_metadata(
        zfp_field* field,
        uint64 meta
    ) nogil

    size_t zfp_compress(
        zfp_stream* stream,
        const zfp_field* field
    ) nogil

    size_t zfp_decompress(
        zfp_stream* stream,
        zfp_field* field
    ) nogil

    size_t zfp_write_header(
        zfp_stream* stream,
        const zfp_field* field,
        uint mask
    ) nogil

    size_t zfp_read_header(
        zfp_stream* stream,
        zfp_field* field,
        uint mask
    ) nogil

    size_t zfp_stream_flush(
        zfp_stream* stream
    ) nogil

    size_t zfp_stream_align(
        zfp_stream* stream
    ) nogil

    size_t zfp_encode_block_int32_1(
        zfp_stream* stream,
        const int32* block
    ) nogil

    size_t zfp_encode_block_int64_1(
        zfp_stream* stream,
        const int64* block
    ) nogil

    size_t zfp_encode_block_float_1(
        zfp_stream* stream,
        const float* block
    ) nogil

    size_t zfp_encode_block_double_1(
        zfp_stream* stream,
        const double* block
    ) nogil

    size_t zfp_encode_block_strided_int32_1(
        zfp_stream* stream,
        const int32* p,
        ptrdiff_t sx
    ) nogil

    size_t zfp_encode_block_strided_int64_1(
        zfp_stream* stream,
        const int64* p,
        ptrdiff_t sx
    ) nogil

    size_t zfp_encode_block_strided_float_1(
        zfp_stream* stream,
        const float* p,
        ptrdiff_t sx
    ) nogil

    size_t zfp_encode_block_strided_double_1(
        zfp_stream* stream,
        const double* p,
        ptrdiff_t sx
    ) nogil

    size_t zfp_encode_partial_block_strided_int32_1(
        zfp_stream* stream,
        const int32* p,
        size_t nx,
        ptrdiff_t sx
    ) nogil

    size_t zfp_encode_partial_block_strided_int64_1(
        zfp_stream* stream,
        const int64* p,
        size_t nx,
        ptrdiff_t sx
    ) nogil

    size_t zfp_encode_partial_block_strided_float_1(
        zfp_stream* stream,
        const float* p,
        size_t nx,
        ptrdiff_t sx
    ) nogil

    size_t zfp_encode_partial_block_strided_double_1(
        zfp_stream* stream,
        const double* p,
        size_t nx,
        ptrdiff_t sx
    ) nogil

    size_t zfp_encode_block_int32_2(
        zfp_stream* stream,
        const int32* block
    ) nogil

    size_t zfp_encode_block_int64_2(
        zfp_stream* stream,
        const int64* block
    ) nogil

    size_t zfp_encode_block_float_2(
        zfp_stream* stream,
        const float* block
    ) nogil

    size_t zfp_encode_block_double_2(
        zfp_stream* stream,
        const double* block
    ) nogil

    size_t zfp_encode_partial_block_strided_int32_2(
        zfp_stream* stream,
        const int32* p,
        size_t nx,
        size_t ny,
        ptrdiff_t sx,
        ptrdiff_t sy
    ) nogil

    size_t zfp_encode_partial_block_strided_int64_2(
        zfp_stream* stream,
        const int64* p,
        size_t nx,
        size_t ny,
        ptrdiff_t sx,
        ptrdiff_t sy
    ) nogil

    size_t zfp_encode_partial_block_strided_float_2(
        zfp_stream* stream,
        const float* p,
        size_t nx,
        size_t ny,
        ptrdiff_t sx,
        ptrdiff_t sy
    ) nogil

    size_t zfp_encode_partial_block_strided_double_2(
        zfp_stream* stream,
        const double* p,
        size_t nx,
        size_t ny,
        ptrdiff_t sx,
        ptrdiff_t sy
    ) nogil

    size_t zfp_encode_block_strided_int32_2(
        zfp_stream* stream,
        const int32* p,
        ptrdiff_t sx,
        ptrdiff_t sy
    ) nogil

    size_t zfp_encode_block_strided_int64_2(
        zfp_stream* stream,
        const int64* p,
        ptrdiff_t sx,
        ptrdiff_t sy
    ) nogil

    size_t zfp_encode_block_strided_float_2(
        zfp_stream* stream,
        const float* p,
        ptrdiff_t sx,
        ptrdiff_t sy
    ) nogil

    size_t zfp_encode_block_strided_double_2(
        zfp_stream* stream,
        const double* p,
        ptrdiff_t sx,
        ptrdiff_t sy
    ) nogil

    size_t zfp_encode_block_int32_3(
        zfp_stream* stream,
        const int32* block
    ) nogil

    size_t zfp_encode_block_int64_3(
        zfp_stream* stream,
        const int64* block
    ) nogil

    size_t zfp_encode_block_float_3(
        zfp_stream* stream,
        const float* block
    ) nogil

    size_t zfp_encode_block_double_3(
        zfp_stream* stream,
        const double* block
    ) nogil

    size_t zfp_encode_block_strided_int32_3(
        zfp_stream* stream,
        const int32* p,
        ptrdiff_t sx,
        ptrdiff_t sy,
        ptrdiff_t sz
    ) nogil

    size_t zfp_encode_block_strided_int64_3(
        zfp_stream* stream,
        const int64* p,
        ptrdiff_t sx,
        ptrdiff_t sy,
        ptrdiff_t sz
    ) nogil

    size_t zfp_encode_block_strided_float_3(
        zfp_stream* stream,
        const float* p,
        ptrdiff_t sx,
        ptrdiff_t sy,
        ptrdiff_t sz
    ) nogil

    size_t zfp_encode_block_strided_double_3(
        zfp_stream* stream,
        const double* p,
        ptrdiff_t sx,
        ptrdiff_t sy,
        ptrdiff_t sz
    ) nogil

    size_t zfp_encode_partial_block_strided_int32_3(
        zfp_stream* stream,
        const int32* p,
        size_t nx,
        size_t ny,
        size_t nz,
        ptrdiff_t sx,
        ptrdiff_t sy,
        ptrdiff_t sz
    ) nogil

    size_t zfp_encode_partial_block_strided_int64_3(
        zfp_stream* stream,
        const int64* p,
        size_t nx,
        size_t ny,
        size_t nz,
        ptrdiff_t sx,
        ptrdiff_t sy,
        ptrdiff_t sz
    ) nogil

    size_t zfp_encode_partial_block_strided_float_3(
        zfp_stream* stream,
        const float* p,
        size_t nx,
        size_t ny,
        size_t nz,
        ptrdiff_t sx,
        ptrdiff_t sy,
        ptrdiff_t sz
    ) nogil

    size_t zfp_encode_partial_block_strided_double_3(
        zfp_stream* stream,
        const double* p,
        size_t nx,
        size_t ny,
        size_t nz,
        ptrdiff_t sx,
        ptrdiff_t sy,
        ptrdiff_t sz
    ) nogil

    size_t zfp_encode_block_int32_4(
        zfp_stream* stream,
        const int32* block
    ) nogil

    size_t zfp_encode_block_int64_4(
        zfp_stream* stream,
        const int64* block
    ) nogil

    size_t zfp_encode_block_float_4(
        zfp_stream* stream,
        const float* block
    ) nogil

    size_t zfp_encode_block_double_4(
        zfp_stream* stream,
        const double* block
    ) nogil

    size_t zfp_encode_block_strided_int32_4(
        zfp_stream* stream,
        const int32* p,
        ptrdiff_t sx,
        ptrdiff_t sy,
        ptrdiff_t sz,
        ptrdiff_t sw
    ) nogil

    size_t zfp_encode_block_strided_int64_4(
        zfp_stream* stream,
        const int64* p,
        ptrdiff_t sx,
        ptrdiff_t sy,
        ptrdiff_t sz,
        ptrdiff_t sw
    ) nogil

    size_t zfp_encode_block_strided_float_4(
        zfp_stream* stream,
        const float* p,
        ptrdiff_t sx,
        ptrdiff_t sy,
        ptrdiff_t sz,
        ptrdiff_t sw
    ) nogil

    size_t zfp_encode_block_strided_double_4(
        zfp_stream* stream,
        const double* p,
        ptrdiff_t sx,
        ptrdiff_t sy,
        ptrdiff_t sz,
        ptrdiff_t sw
    ) nogil

    size_t zfp_encode_partial_block_strided_int32_4(
        zfp_stream* stream,
        const int32* p,
        size_t nx,
        size_t ny,
        size_t nz,
        size_t nw,
        ptrdiff_t sx,
        ptrdiff_t sy,
        ptrdiff_t sz,
        ptrdiff_t sw
    ) nogil

    size_t zfp_encode_partial_block_strided_int64_4(
        zfp_stream* stream,
        const int64* p,
        size_t nx,
        size_t ny,
        size_t nz,
        size_t nw,
        ptrdiff_t sx,
        ptrdiff_t sy,
        ptrdiff_t sz,
        ptrdiff_t sw
    ) nogil

    size_t zfp_encode_partial_block_strided_float_4(
        zfp_stream* stream,
        const float* p,
        size_t nx,
        size_t ny,
        size_t nz,
        size_t nw,
        ptrdiff_t sx,
        ptrdiff_t sy,
        ptrdiff_t sz,
        ptrdiff_t sw
    ) nogil

    size_t zfp_encode_partial_block_strided_double_4(
        zfp_stream* stream,
        const double* p,
        size_t nx,
        size_t ny,
        size_t nz,
        size_t nw,
        ptrdiff_t sx,
        ptrdiff_t sy,
        ptrdiff_t sz,
        ptrdiff_t sw
    ) nogil

    size_t zfp_decode_block_int32_1(
        zfp_stream* stream,
        int32* block
    ) nogil

    size_t zfp_decode_block_int64_1(
        zfp_stream* stream,
        int64* block
    ) nogil

    size_t zfp_decode_block_float_1(
        zfp_stream* stream,
        float* block
    ) nogil

    size_t zfp_decode_block_double_1(
        zfp_stream* stream,
        double* block
    ) nogil


    size_t zfp_decode_block_strided_int32_1(
        zfp_stream* stream,
        int32* p,
        ptrdiff_t sx
    ) nogil

    size_t zfp_decode_block_strided_int64_1(
        zfp_stream* stream,
        int64* p,
        ptrdiff_t sx
    ) nogil

    size_t zfp_decode_block_strided_float_1(
        zfp_stream* stream,
        float* p,
        ptrdiff_t sx
    ) nogil

    size_t zfp_decode_block_strided_double_1(
        zfp_stream* stream,
        double* p,
        ptrdiff_t sx
    ) nogil

    size_t zfp_decode_partial_block_strided_int32_1(
        zfp_stream* stream,
        int32* p,
        size_t nx,
        ptrdiff_t sx
    ) nogil

    size_t zfp_decode_partial_block_strided_int64_1(
        zfp_stream* stream,
        int64* p,
        size_t nx,
        ptrdiff_t sx
    ) nogil

    size_t zfp_decode_partial_block_strided_float_1(
        zfp_stream* stream,
        float* p,
        size_t nx,
        ptrdiff_t sx
    ) nogil

    size_t zfp_decode_partial_block_strided_double_1(
        zfp_stream* stream,
        double* p,
        size_t nx,
        ptrdiff_t sx
    ) nogil


    size_t zfp_decode_block_int32_2(
        zfp_stream* stream,
        int32* block
    ) nogil

    size_t zfp_decode_block_int64_2(
        zfp_stream* stream,
        int64* block
    ) nogil

    size_t zfp_decode_block_float_2(
        zfp_stream* stream,
        float* block
    ) nogil

    size_t zfp_decode_block_double_2(
        zfp_stream* stream,
        double* block
    ) nogil


    size_t zfp_decode_block_strided_int32_2(
        zfp_stream* stream,
        int32* p,
        ptrdiff_t sx,
        ptrdiff_t sy
    ) nogil

    size_t zfp_decode_block_strided_int64_2(
        zfp_stream* stream,
        int64* p,
        ptrdiff_t sx,
        ptrdiff_t sy
    ) nogil

    size_t zfp_decode_block_strided_float_2(
        zfp_stream* stream,
        float* p,
        ptrdiff_t sx,
        ptrdiff_t sy
    ) nogil

    size_t zfp_decode_block_strided_double_2(
        zfp_stream* stream,
        double* p,
        ptrdiff_t sx,
        ptrdiff_t sy
    ) nogil

    size_t zfp_decode_partial_block_strided_int32_2(
        zfp_stream* stream,
        int32* p,
        size_t nx,
        size_t ny,
        ptrdiff_t sx,
        ptrdiff_t sy
    ) nogil

    size_t zfp_decode_partial_block_strided_int64_2(
        zfp_stream* stream,
        int64* p,
        size_t nx,
        size_t ny,
        ptrdiff_t sx,
        ptrdiff_t sy
    ) nogil

    size_t zfp_decode_partial_block_strided_float_2(
        zfp_stream* stream,
        float* p,
        size_t nx,
        size_t ny,
        ptrdiff_t sx,
        ptrdiff_t sy
    ) nogil

    size_t zfp_decode_partial_block_strided_double_2(
        zfp_stream* stream,
        double* p,
        size_t nx,
        size_t ny,
        ptrdiff_t sx,
        ptrdiff_t sy
    ) nogil


    size_t zfp_decode_block_int32_3(
        zfp_stream* stream,
        int32* block
    ) nogil

    size_t zfp_decode_block_int64_3(
        zfp_stream* stream,
        int64* block
    ) nogil

    size_t zfp_decode_block_float_3(
        zfp_stream* stream,
        float* block
    ) nogil

    size_t zfp_decode_block_double_3(
        zfp_stream* stream,
        double* block
    ) nogil


    size_t zfp_decode_block_strided_int32_3(
        zfp_stream* stream,
        int32* p,
        ptrdiff_t sx,
        ptrdiff_t sy,
        ptrdiff_t sz
    ) nogil

    size_t zfp_decode_block_strided_int64_3(
        zfp_stream* stream,
        int64* p,
        ptrdiff_t sx,
        ptrdiff_t sy,
        ptrdiff_t sz
    ) nogil

    size_t zfp_decode_block_strided_float_3(
        zfp_stream* stream,
        float* p,
        ptrdiff_t sx,
        ptrdiff_t sy,
        ptrdiff_t sz
    ) nogil

    size_t zfp_decode_block_strided_double_3(
        zfp_stream* stream,
        double* p,
        ptrdiff_t sx,
        ptrdiff_t sy,
        ptrdiff_t sz
    ) nogil

    size_t zfp_decode_partial_block_strided_int32_3(
        zfp_stream* stream,
        int32* p,
        size_t nx,
        size_t ny,
        size_t nz,
        ptrdiff_t sx,
        ptrdiff_t sy,
        ptrdiff_t sz
    ) nogil

    size_t zfp_decode_partial_block_strided_int64_3(
        zfp_stream* stream,
        int64* p,
        size_t nx,
        size_t ny,
        size_t nz,
        ptrdiff_t sx,
        ptrdiff_t sy,
        ptrdiff_t sz
    ) nogil

    size_t zfp_decode_partial_block_strided_float_3(
        zfp_stream* stream,
        float* p,
        size_t nx,
        size_t ny,
        size_t nz,
        ptrdiff_t sx,
        ptrdiff_t sy,
        ptrdiff_t sz
    ) nogil

    size_t zfp_decode_partial_block_strided_double_3(
        zfp_stream* stream,
        double* p,
        size_t nx,
        size_t ny,
        size_t nz,
        ptrdiff_t sx,
        ptrdiff_t sy,
        ptrdiff_t sz
    ) nogil


    size_t zfp_decode_block_int32_4(
        zfp_stream* stream,
        int32* block
    ) nogil

    size_t zfp_decode_block_int64_4(
        zfp_stream* stream,
        int64* block
    ) nogil

    size_t zfp_decode_block_float_4(
        zfp_stream* stream,
        float* block
    ) nogil

    size_t zfp_decode_block_double_4(
        zfp_stream* stream,
        double* block
    ) nogil

    size_t zfp_decode_block_strided_int32_4(
        zfp_stream* stream,
        int32* p,
        ptrdiff_t sx,
        ptrdiff_t sy,
        ptrdiff_t sz,
        ptrdiff_t sw
    ) nogil

    size_t zfp_decode_block_strided_int64_4(
        zfp_stream* stream,
        int64* p,
        ptrdiff_t sx,
        ptrdiff_t sy,
        ptrdiff_t sz,
        ptrdiff_t sw
    ) nogil

    size_t zfp_decode_block_strided_float_4(
        zfp_stream* stream,
        float* p,
        ptrdiff_t sx,
        ptrdiff_t sy,
        ptrdiff_t sz,
        ptrdiff_t sw
    ) nogil

    size_t zfp_decode_block_strided_double_4(
        zfp_stream* stream,
        double* p,
        ptrdiff_t sx,
        ptrdiff_t sy,
        ptrdiff_t sz,
        ptrdiff_t sw
    ) nogil

    size_t zfp_decode_partial_block_strided_int32_4(
        zfp_stream* stream,
        int32* p,
        size_t nx,
        size_t ny,
        size_t nz,
        size_t nw,
        ptrdiff_t sx,
        ptrdiff_t sy,
        ptrdiff_t sz,
        ptrdiff_t sw
    ) nogil

    size_t zfp_decode_partial_block_strided_int64_4(
        zfp_stream* stream,
        int64* p,
        size_t nx,
        size_t ny,
        size_t nz,
        size_t nw,
        ptrdiff_t sx,
        ptrdiff_t sy,
        ptrdiff_t sz,
        ptrdiff_t sw
    ) nogil

    size_t zfp_decode_partial_block_strided_float_4(
        zfp_stream* stream,
        float* p,
        size_t nx,
        size_t ny,
        size_t nz,
        size_t nw,
        ptrdiff_t sx,
        ptrdiff_t sy,
        ptrdiff_t sz,
        ptrdiff_t sw
    ) nogil

    size_t zfp_decode_partial_block_strided_double_4(
        zfp_stream* stream,
        double* p,
        size_t nx,
        size_t ny,
        size_t nz,
        size_t nw,
        ptrdiff_t sx,
        ptrdiff_t sy,
        ptrdiff_t sz,
        ptrdiff_t sw
    ) nogil

    void zfp_promote_int8_to_int32(
        int32* oblock,
        const int8* iblock,
        uint dims
    ) nogil

    void zfp_promote_uint8_to_int32(
        int32* oblock,
        const uint8* iblock,
        uint dims
    ) nogil

    void zfp_promote_int16_to_int32(
        int32* oblock,
        const int16* iblock,
        uint dims
    ) nogil

    void zfp_promote_uint16_to_int32(
        int32* oblock,
        const uint16* iblock,
        uint dims
    ) nogil

    void zfp_demote_int32_to_int8(
        int8* oblock,
        const int32* iblock,
        uint dims
    ) nogil

    void zfp_demote_int32_to_uint8(
        uint8* oblock,
        const int32* iblock,
        uint dims
    ) nogil

    void zfp_demote_int32_to_int16(
        int16* oblock,
        const int32* iblock,
        uint dims
    ) nogil

    void zfp_demote_int32_to_uint16(
        uint16* oblock,
        const int32* iblock,
        uint dims
    ) nogil
