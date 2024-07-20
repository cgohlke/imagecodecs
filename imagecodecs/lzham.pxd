# imagecodecs/lzham.pxd
# cython: language_level = 3

# Cython declarations for the `lzham 1.0` library.
# https://github.com/richgel999/lzham_codec

cdef extern from 'lzham.h' nogil:

    ctypedef unsigned char lzham_uint8
    ctypedef signed int lzham_int32
    ctypedef unsigned int lzham_uint32
    ctypedef unsigned int lzham_bool

    lzham_uint32 lzham_get_version()

    int LZHAM_MIN_ALLOC_ALIGNMENT

    ctypedef void*(*lzham_realloc_func)(
        void* p,
        size_t size,
        size_t* pActual_size,
        lzham_bool movable,
        void* pUser_data
    ) nogil

    ctypedef size_t (*lzham_msize_func)(
        void* p,
        void* pUser_data
    ) nogil

    void lzham_set_memory_callbacks(
        lzham_realloc_func pRealloc,
        lzham_msize_func pMSize,
        void* pUser_data
    )

    ctypedef enum lzham_flush_t:
        LZHAM_NO_FLUSH
        LZHAM_SYNC_FLUSH
        LZHAM_FULL_FLUSH
        LZHAM_FINISH
        LZHAM_TABLE_FLUSH

    int LZHAM_MIN_DICT_SIZE_LOG2
    int LZHAM_MAX_DICT_SIZE_LOG2_X86
    int LZHAM_MAX_DICT_SIZE_LOG2_X64

    int LZHAM_MAX_HELPER_THREADS

    ctypedef enum lzham_compress_status_t:
        LZHAM_COMP_STATUS_NOT_FINISHED
        LZHAM_COMP_STATUS_NEEDS_MORE_INPUT
        LZHAM_COMP_STATUS_HAS_MORE_OUTPUT
        LZHAM_COMP_STATUS_FIRST_SUCCESS_OR_FAILURE_CODE
        LZHAM_COMP_STATUS_SUCCESS
        LZHAM_COMP_STATUS_FIRST_FAILURE_CODE
        LZHAM_COMP_STATUS_FAILED
        LZHAM_COMP_STATUS_FAILED_INITIALIZING
        LZHAM_COMP_STATUS_INVALID_PARAMETER
        LZHAM_COMP_STATUS_OUTPUT_BUF_TOO_SMALL
        LZHAM_COMP_STATUS_FORCE_DWORD

    ctypedef enum lzham_compress_level:
        LZHAM_COMP_LEVEL_FASTEST
        LZHAM_COMP_LEVEL_FASTER
        LZHAM_COMP_LEVEL_DEFAULT
        LZHAM_COMP_LEVEL_BETTER
        LZHAM_COMP_LEVEL_UBER
        LZHAM_TOTAL_COMP_LEVELS
        LZHAM_COMP_LEVEL_FORCE_DWORD

    ctypedef void *lzham_compress_state_ptr

    ctypedef enum lzham_compress_flags:
        LZHAM_COMP_FLAG_EXTREME_PARSING
        LZHAM_COMP_FLAG_DETERMINISTIC_PARSING
        LZHAM_COMP_FLAG_TRADEOFF_DECOMPRESSION_RATE_FOR_COMP_RATIO
        LZHAM_COMP_FLAG_WRITE_ZLIB_STREAM

    ctypedef enum lzham_table_update_rate:
        LZHAM_INSANELY_SLOW_TABLE_UPDATE_RATE
        LZHAM_SLOWEST_TABLE_UPDATE_RATE
        LZHAM_DEFAULT_TABLE_UPDATE_RATE
        LZHAM_FASTEST_TABLE_UPDATE_RATE

    ctypedef struct lzham_compress_params:
        lzham_uint32 m_struct_size
        lzham_uint32 m_dict_size_log2
        lzham_compress_level m_level
        lzham_uint32 m_table_update_rate
        lzham_int32 m_max_helper_threads
        lzham_uint32 m_compress_flags
        lzham_uint32 m_num_seed_bytes
        const void *m_pSeed_bytes
        lzham_uint32 m_table_max_update_interval
        lzham_uint32 m_table_update_interval_slow_rate

    lzham_compress_state_ptr lzham_compress_init(
        const lzham_compress_params *pParams
    )

    lzham_compress_state_ptr lzham_compress_reinit(
        lzham_compress_state_ptr pState
    )

    lzham_uint32 lzham_compress_deinit(
        lzham_compress_state_ptr pState
    )

    lzham_compress_status_t lzham_compress(
        lzham_compress_state_ptr pState,
        const lzham_uint8 *pIn_buf,
        size_t *pIn_buf_size,
        lzham_uint8 *pOut_buf,
        size_t *pOut_buf_size,
        lzham_bool no_more_input_bytes_flag
    )

    lzham_compress_status_t lzham_compress2(
        lzham_compress_state_ptr pState,
        const lzham_uint8 *pIn_buf,
        size_t *pIn_buf_size,
        lzham_uint8 *pOut_buf, size_t *pOut_buf_size,
        lzham_flush_t flush_type
    )

    lzham_compress_status_t lzham_compress_memory(
        const lzham_compress_params *pParams,
        lzham_uint8* pDst_buf,
        size_t *pDst_len,
        const lzham_uint8* pSrc_buf,
        size_t src_len,
        lzham_uint32 *pAdler32
    )

    ctypedef enum lzham_decompress_status_t:
        LZHAM_DECOMP_STATUS_NOT_FINISHED
        LZHAM_DECOMP_STATUS_HAS_MORE_OUTPUT
        LZHAM_DECOMP_STATUS_NEEDS_MORE_INPUT
        LZHAM_DECOMP_STATUS_FIRST_SUCCESS_OR_FAILURE_CODE
        LZHAM_DECOMP_STATUS_SUCCESS
        LZHAM_DECOMP_STATUS_FIRST_FAILURE_CODE
        LZHAM_DECOMP_STATUS_FAILED_INITIALIZING
        LZHAM_DECOMP_STATUS_FAILED_DEST_BUF_TOO_SMALL
        LZHAM_DECOMP_STATUS_FAILED_EXPECTED_MORE_RAW_BYTES
        LZHAM_DECOMP_STATUS_FAILED_BAD_CODE
        LZHAM_DECOMP_STATUS_FAILED_ADLER32
        LZHAM_DECOMP_STATUS_FAILED_BAD_RAW_BLOCK
        LZHAM_DECOMP_STATUS_FAILED_BAD_COMP_BLOCK_SYNC_CHECK
        LZHAM_DECOMP_STATUS_FAILED_BAD_ZLIB_HEADER
        LZHAM_DECOMP_STATUS_FAILED_NEED_SEED_BYTES
        LZHAM_DECOMP_STATUS_FAILED_BAD_SEED_BYTES
        LZHAM_DECOMP_STATUS_FAILED_BAD_SYNC_BLOCK
        LZHAM_DECOMP_STATUS_INVALID_PARAMETER

    ctypedef void *lzham_decompress_state_ptr

    ctypedef enum lzham_decompress_flags:
        LZHAM_DECOMP_FLAG_OUTPUT_UNBUFFERED
        LZHAM_DECOMP_FLAG_COMPUTE_ADLER32
        LZHAM_DECOMP_FLAG_READ_ZLIB_STREAM

    ctypedef struct lzham_decompress_params:
        lzham_uint32 m_struct_size
        lzham_uint32 m_dict_size_log2
        lzham_uint32 m_table_update_rate
        lzham_uint32 m_decompress_flags
        lzham_uint32 m_num_seed_bytes
        const void *m_pSeed_bytes
        lzham_uint32 m_table_max_update_interval
        lzham_uint32 m_table_update_interval_slow_rate

    lzham_decompress_state_ptr lzham_decompress_init(
        const lzham_decompress_params *pParams
    )

    lzham_decompress_state_ptr lzham_decompress_reinit(
        lzham_decompress_state_ptr pState,
        const lzham_decompress_params *pParams
    )

    lzham_uint32 lzham_decompress_deinit(
        lzham_decompress_state_ptr pState
    )

    lzham_decompress_status_t lzham_decompress(
        lzham_decompress_state_ptr pState,
        const lzham_uint8 *pIn_buf,
        size_t *pIn_buf_size,
        lzham_uint8 *pOut_buf,
        size_t *pOut_buf_size,
        lzham_bool no_more_input_bytes_flag
    )

    lzham_decompress_status_t lzham_decompress_memory(
        const lzham_decompress_params *pParams,
        lzham_uint8* pDst_buf,
        size_t *pDst_len,
        const lzham_uint8* pSrc_buf,
        size_t src_len,
        lzham_uint32 *pAdler32
    )

    ctypedef unsigned long lzham_z_ulong

    ctypedef void *(*lzham_z_alloc_func)(
        void *opaque,
        size_t items,
        size_t size
    ) nogil

    ctypedef void (*lzham_z_free_func)(
        void *opaque,
        void *address
    ) nogil

    ctypedef void *(*lzham_z_realloc_func)(
        void *opaque,
        void *address,
        size_t items,
        size_t size
    ) nogil

    int LZHAM_Z_ADLER32_INIT

    lzham_z_ulong lzham_z_adler32(
        lzham_z_ulong adler,
        const unsigned char *ptr,
        size_t buf_len
    )

    int LZHAM_Z_CRC32_INIT

    lzham_z_ulong lzham_z_crc32(
        lzham_z_ulong crc,
        const unsigned char *ptr,
        size_t buf_len
    )

    enum:
        LZHAM_Z_DEFAULT_STRATEGY
        LZHAM_Z_FILTERED
        LZHAM_Z_HUFFMAN_ONLY
        LZHAM_Z_RLE
        LZHAM_Z_FIXED

    int LZHAM_Z_DEFLATED
    int LZHAM_Z_LZHAM

    int LZHAM_Z_VERSION
    int LZHAM_Z_VERNUM
    int LZHAM_Z_VER_MAJOR
    int LZHAM_Z_VER_MINOR
    int LZHAM_Z_VER_REVISION
    int LZHAM_Z_VER_SUBREVISION

    enum:
        LZHAM_Z_NO_FLUSH
        LZHAM_Z_PARTIAL_FLUSH
        LZHAM_Z_SYNC_FLUSH
        LZHAM_Z_FULL_FLUSH
        LZHAM_Z_FINISH
        LZHAM_Z_BLOCK
        LZHAM_Z_TABLE_FLUSH

    enum:
        LZHAM_Z_OK
        LZHAM_Z_STREAM_END
        LZHAM_Z_NEED_DICT
        LZHAM_Z_ERRNO
        LZHAM_Z_STREAM_ERROR
        LZHAM_Z_DATA_ERROR
        LZHAM_Z_MEM_ERROR
        LZHAM_Z_BUF_ERROR
        LZHAM_Z_VERSION_ERROR
        LZHAM_Z_PARAM_ERROR

    enum:
        LZHAM_Z_NO_COMPRESSION
        LZHAM_Z_BEST_SPEED
        LZHAM_Z_BEST_COMPRESSION
        LZHAM_Z_UBER_COMPRESSION
        LZHAM_Z_DEFAULT_COMPRESSION

    int LZHAM_Z_DEFAULT_WINDOW_BITS

    int LZHAM_Z_BINARY
    int LZHAM_Z_TEXT
    int LZHAM_Z_ASCII
    int LZHAM_Z_UNKNOWN

    struct lzham_z_internal_state:
        pass

    ctypedef struct lzham_z_stream:
        const unsigned char *next_in
        unsigned int avail_in
        lzham_z_ulong total_in

        unsigned char *next_out
        unsigned int avail_out
        lzham_z_ulong total_out

        char *msg
        lzham_z_internal_state *state

        lzham_z_alloc_func zalloc
        lzham_z_free_func zfree
        void *opaque

        int data_type
        lzham_z_ulong adler
        lzham_z_ulong reserved

    ctypedef lzham_z_stream *lzham_z_streamp

    const char *lzham_z_version()

    int lzham_z_deflateInit(
        lzham_z_streamp pStream,
        int level
    )

    int lzham_z_deflateInit2(
        lzham_z_streamp pStream,
        int level,
        int method,
        int window_bits,
        int mem_level,
        int strategy
    )

    int lzham_z_deflateReset(
        lzham_z_streamp pStream
    )

    int lzham_z_deflate(
        lzham_z_streamp pStream,
        int flush
    )

    int lzham_z_deflateEnd(
        lzham_z_streamp pStream
    )

    lzham_z_ulong lzham_z_deflateBound(
        lzham_z_streamp pStream,
        lzham_z_ulong source_len
    )

    int lzham_z_compress(
        unsigned char *pDest,
        lzham_z_ulong *pDest_len,
        const unsigned char *pSource,
        lzham_z_ulong source_len
    )

    int lzham_z_compress2(
        unsigned char *pDest,
        lzham_z_ulong *pDest_len,
        const unsigned char *pSource,
        lzham_z_ulong source_len,
        int level
    )

    lzham_z_ulong lzham_z_compressBound(
        lzham_z_ulong source_len
    )

    int lzham_z_inflateInit(
        lzham_z_streamp pStream
    )

    int lzham_z_inflateInit2(
        lzham_z_streamp pStream,
        int window_bits
    )

    int lzham_z_inflateReset(
        lzham_z_streamp pStream
    )

    int lzham_z_inflate(
        lzham_z_streamp pStream,
        int flush
    )

    int lzham_z_inflateEnd(
        lzham_z_streamp pStream
    )

    int lzham_z_uncompress(
        unsigned char *pDest,
        lzham_z_ulong *pDest_len,
        const unsigned char *pSource,
        lzham_z_ulong source_len
    )

    const char *lzham_z_error(
        int err
    )
