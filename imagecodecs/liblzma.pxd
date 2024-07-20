# imagecodecs/liblzma.pxd
# cython: language_level = 3

# Cython declarations for the `liblzma 5.6.2` library.
# https://github.com/tukaani-project/xz

from libc.stdint cimport uint8_t, uint32_t, uint64_t

cdef extern from 'lzma.h' nogil:

    # version.h

    int LZMA_VERSION_MAJOR
    int LZMA_VERSION_MINOR
    int LZMA_VERSION_PATCH
    int LZMA_VERSION_STABILITY

    uint32_t LZMA_VERSION

    uint32_t lzma_version_number()

    const char* lzma_version_string()

    # base.h

    ctypedef unsigned char lzma_bool

    ctypedef enum lzma_reserved_enum:
        LZMA_RESERVED_ENUM

    ctypedef enum lzma_ret:
        LZMA_OK
        LZMA_STREAM_END
        LZMA_NO_CHECK
        LZMA_UNSUPPORTED_CHECK
        LZMA_GET_CHECK
        LZMA_MEM_ERROR
        LZMA_MEMLIMIT_ERROR
        LZMA_FORMAT_ERROR
        LZMA_OPTIONS_ERROR
        LZMA_DATA_ERROR
        LZMA_BUF_ERROR
        LZMA_PROG_ERROR
        LZMA_SEEK_NEEDED
        LZMA_RET_INTERNAL1
        LZMA_RET_INTERNAL2
        LZMA_RET_INTERNAL3
        LZMA_RET_INTERNAL4
        LZMA_RET_INTERNAL5
        LZMA_RET_INTERNAL6
        LZMA_RET_INTERNAL7
        LZMA_RET_INTERNAL8

    ctypedef enum lzma_action:
        LZMA_RUN
        LZMA_SYNC_FLUSH
        LZMA_FULL_FLUSH
        LZMA_FULL_BARRIER
        LZMA_FINISH

    ctypedef struct lzma_allocator:
        void* (*alloc)(void* opaque, size_t nmemb, size_t size) nogil
        void (*free)(void* opaque, void* ptr) nogil
        void* opaque

    ctypedef struct lzma_internal:
        pass

    ctypedef struct lzma_stream:
        uint8_t* next_in
        size_t avail_in
        uint64_t total_in
        uint8_t* next_out
        size_t avail_out
        uint64_t total_out
        lzma_allocator* allocator
        lzma_internal* internal
        void* reserved_ptr1
        void* reserved_ptr2
        void* reserved_ptr3
        void* reserved_ptr4
        uint64_t seek_pos
        uint64_t reserved_int2
        size_t reserved_int3
        size_t reserved_int4
        lzma_reserved_enum reserved_enum1
        lzma_reserved_enum reserved_enum2

    # LZMA_STREAM_INIT()

    lzma_ret lzma_code(
        lzma_stream* strm,
        lzma_action action
    )

    void lzma_end(
        lzma_stream* strm
    )

    void lzma_get_progress(
        lzma_stream* strm,
        uint64_t* progress_in,
        uint64_t* progress_out
    )

    uint64_t lzma_memusage(
        const lzma_stream* strm
    )

    uint64_t lzma_memlimit_get(
        const lzma_stream* strm
    )

    lzma_ret lzma_memlimit_set(
        lzma_stream* strm,
        uint64_t memlimit
    )

    # vli.h

    int LZMA_VLI_MAX
    int LZMA_VLI_UNKNOWN
    int LZMA_VLI_BYTES_MAX
    int LZMA_VLI_C(int n)

    ctypedef uint64_t lzma_vli

    int lzma_vli_is_valid(
        int vli
    )

    lzma_ret lzma_vli_encode(
        lzma_vli vli,
        size_t* vli_pos,
        uint8_t* out,
        size_t* out_pos,
        size_t out_size
    )

    lzma_ret lzma_vli_decode(
        lzma_vli* vli,
        size_t* vli_pos,
        const uint8_t* in_,
        size_t* in_pos,
        size_t in_size
    )

    uint32_t lzma_vli_size(
        lzma_vli vli
    )

    # check.h

    ctypedef enum lzma_check_t 'lzma_check':
        LZMA_CHECK_NONE
        LZMA_CHECK_CRC32
        LZMA_CHECK_CRC64
        LZMA_CHECK_SHA256

    lzma_bool lzma_check_is_supported(
        lzma_check_t check
    )

    uint32_t lzma_check_size(
        lzma_check_t check
    )

    int LZMA_CHECK_SIZE_MAX

    uint32_t lzma_crc32(
        const uint8_t* buf,
        size_t size,
        uint32_t crc
    )

    uint64_t lzma_crc64(
        const uint8_t* buf,
        size_t size,
        uint64_t crc
    )

    lzma_check_t lzma_get_check(
        const lzma_stream* strm
    )

    # filter.h

    int LZMA_FILTERS_MAX

    ctypedef struct lzma_filter:
        lzma_vli id_ 'id'
        void* options

    lzma_bool lzma_filter_encoder_is_supported(
        lzma_vli id_
    )

    lzma_bool lzma_filter_decoder_is_supported(
        lzma_vli id_
    )

    lzma_ret lzma_filters_copy(
        const lzma_filter* src,
        lzma_filter* dest,
        const lzma_allocator* allocator
    )

    void lzma_filters_free(
        lzma_filter* filters,
        const lzma_allocator* allocator
    )

    uint64_t lzma_raw_encoder_memusage(
        const lzma_filter* filters
    )

    uint64_t lzma_raw_decoder_memusage(
        const lzma_filter* filters
    )

    lzma_ret lzma_raw_encoder(
        lzma_stream* strm,
        const lzma_filter* filters
    )

    lzma_ret lzma_raw_decoder(
        lzma_stream* strm,
        const lzma_filter* filters
    )

    lzma_ret lzma_filters_update(
        lzma_stream* strm,
        const lzma_filter* filters
    )

    lzma_ret lzma_raw_buffer_encode(
        const lzma_filter* filters,
        const lzma_allocator* allocator,
        const uint8_t* in_,
        size_t in_size,
        uint8_t* out,
        size_t* out_pos,
        size_t out_size
    )

    lzma_ret lzma_raw_buffer_decode(
        const lzma_filter* filters,
        const lzma_allocator* allocator,
        const uint8_t* in_,
        size_t* in_pos,
        size_t in_size,
        uint8_t* out,
        size_t* out_pos,
        size_t out_size
    )

    lzma_ret lzma_properties_size(
        uint32_t* size,
        const lzma_filter* filter_
    )

    lzma_ret lzma_properties_encode(
        const lzma_filter* filter_,
        uint8_t* props
    )

    lzma_ret lzma_properties_decode(
        lzma_filter* filter_,
        const lzma_allocator* allocator,
        const uint8_t* props,
        size_t props_size
    )

    lzma_ret lzma_filter_flags_size(
        uint32_t* size,
        const lzma_filter* filter_
    )

    lzma_ret lzma_filter_flags_encode(
        const lzma_filter* filter_,
        uint8_t* out,
        size_t* out_pos,
        size_t out_size
    )

    lzma_ret lzma_filter_flags_decode(
        lzma_filter* filter_,
        const lzma_allocator* allocator,
        const uint8_t* in_,
        size_t* in_pos,
        size_t in_size
    )

    int LZMA_STR_ALL_FILTERS
    int LZMA_STR_NO_VALIDATION
    int LZMA_STR_ENCODER
    int LZMA_STR_DECODER
    int LZMA_STR_GETOPT_LONG
    int LZMA_STR_NO_SPACES

    const char* lzma_str_to_filters(
        const char* str_,
        int* error_pos,
        lzma_filter* filters,
        uint32_t flags,
        const lzma_allocator* allocator
    )

    lzma_ret lzma_str_from_filters(
        char** str_,
        const lzma_filter* filters,
        uint32_t flags,
        const lzma_allocator* allocator
    )

    lzma_ret lzma_str_list_filters(
        char** str_,
        lzma_vli filter_id,
        uint32_t flags,
        const lzma_allocator* allocator
    )

    # bcj.h

    int LZMA_FILTER_X86
    int LZMA_FILTER_POWERPC
    int LZMA_FILTER_IA64
    int LZMA_FILTER_ARM
    int LZMA_FILTER_ARMTHUMB
    int LZMA_FILTER_SPARC
    int LZMA_FILTER_ARM64

    ctypedef struct lzma_options_bcj:
        uint32_t start_offset

    # delta.h

    int LZMA_FILTER_DELTA
    int LZMA_DELTA_DIST_MIN
    int LZMA_DELTA_DIST_MAX

    ctypedef enum lzma_delta_type:
        LZMA_DELTA_TYPE_BYTE

    ctypedef struct lzma_options_delta:
        lzma_delta_type type_ 'type'
        uint32_t dist
        uint32_t reserved_int1
        uint32_t reserved_int2
        uint32_t reserved_int3
        uint32_t reserved_int4
        void* reserved_ptr1
        void* reserved_ptr2

    # lzma12.h

    int LZMA_FILTER_LZMA1
    int LZMA_FILTER_LZMA1EXT
    int LZMA_FILTER_LZMA2

    ctypedef enum lzma_match_finder:
        LZMA_MF_HC3
        LZMA_MF_HC4
        LZMA_MF_BT2
        LZMA_MF_BT3
        LZMA_MF_BT4

    lzma_bool lzma_mf_is_supported(
        lzma_match_finder match_finder
    )

    ctypedef enum lzma_mode:
        LZMA_MODE_FAST
        LZMA_MODE_NORMAL

    lzma_bool lzma_mode_is_supported(
        lzma_mode mode
    )

    int LZMA_DICT_SIZE_MIN
    int LZMA_DICT_SIZE_DEFAULT
    int LZMA_LCLP_MIN
    int LZMA_LCLP_MAX
    int LZMA_LC_DEFAULT
    int LZMA_LP_DEFAULT
    int LZMA_PB_MIN
    int LZMA_PB_MAX
    int LZMA_PB_DEFAULT
    int LZMA_LZMA1EXT_ALLOW_EOPM

    ctypedef struct lzma_options_lzma:
        uint32_t dict_size
        const uint8_t* preset_dict
        uint32_t preset_dict_size
        uint32_t lc
        uint32_t lp
        uint32_t pb
        lzma_mode mode
        uint32_t nice_len
        lzma_match_finder mf
        uint32_t depth
        uint32_t ext_flags
        uint32_t ext_size_low
        uint32_t ext_size_high
        uint32_t reserved_int4
        uint32_t reserved_int5
        uint32_t reserved_int6
        uint32_t reserved_int7
        uint32_t reserved_int8
        lzma_reserved_enum reserved_enum1
        lzma_reserved_enum reserved_enum2
        lzma_reserved_enum reserved_enum3
        lzma_reserved_enum reserved_enum4
        void* reserved_ptr1
        void* reserved_ptr2

    int lzma_set_ext_size(int, int)

    lzma_bool lzma_lzma_preset(
        lzma_options_lzma* options,
        uint32_t preset
    )

    # container.h

    int LZMA_PRESET_DEFAULT
    int LZMA_PRESET_LEVEL_MASK
    int LZMA_PRESET_EXTREME

    ctypedef struct lzma_mt:
        uint32_t flags
        uint32_t threads
        uint64_t block_size
        uint32_t timeout
        uint32_t preset
        const lzma_filter* filters
        lzma_check_t check
        lzma_reserved_enum reserved_enum1
        lzma_reserved_enum reserved_enum2
        lzma_reserved_enum reserved_enum3
        uint32_t reserved_int1
        uint32_t reserved_int2
        uint32_t reserved_int3
        uint32_t reserved_int4
        uint64_t memlimit_threading
        uint64_t memlimit_stop
        uint64_t reserved_int7
        uint64_t reserved_int8
        void* reserved_ptr1
        void* reserved_ptr2
        void* reserved_ptr3
        void* reserved_ptr4

    uint64_t lzma_easy_encoder_memusage(
        uint32_t preset
    )

    uint64_t lzma_easy_decoder_memusage(
        uint32_t preset
    )

    lzma_ret lzma_easy_encoder(
        lzma_stream* strm,
        uint32_t preset,
        lzma_check_t check
    )

    lzma_ret lzma_easy_buffer_encode(
        uint32_t preset,
        lzma_check_t check,
        const lzma_allocator* allocator,
        const uint8_t* in_,
        size_t in_size,
        uint8_t* out,
        size_t* out_pos,
        size_t out_size
    )

    lzma_ret lzma_stream_encoder(
        lzma_stream* strm,
        const lzma_filter* filters,
        lzma_check_t check
    )

    uint64_t lzma_stream_encoder_mt_memusage(
        const lzma_mt* options
    )

    lzma_ret lzma_stream_encoder_mt(
        lzma_stream* strm,
        const lzma_mt* options
    )

    lzma_ret lzma_alone_encoder(
        lzma_stream* strm,
        const lzma_options_lzma* options
    )

    size_t lzma_stream_buffer_bound(
        size_t uncompressed_size
    )

    lzma_ret lzma_stream_buffer_encode(
        lzma_filter* filters,
        lzma_check_t check,
        const lzma_allocator* allocator,
        const uint8_t* in_,
        size_t in_size,
        uint8_t* out,
        size_t* out_pos,
        size_t out_size
    )

    lzma_ret lzma_microlzma_encoder(
        lzma_stream* strm,
        const lzma_options_lzma* options
    )

    int LZMA_TELL_NO_CHECK
    int LZMA_TELL_UNSUPPORTED_CHECK
    int LZMA_TELL_ANY_CHECK
    int LZMA_IGNORE_CHECK
    int LZMA_CONCATENATED
    int LZMA_FAIL_FAST

    lzma_ret lzma_stream_decoder(
        lzma_stream* strm,
        uint64_t memlimit,
        uint32_t flags
    )

    lzma_ret lzma_stream_decoder_mt(
        lzma_stream* strm,
        const lzma_mt* options
    )

    lzma_ret lzma_auto_decoder(
        lzma_stream* strm,
        uint64_t memlimit,
        uint32_t flags
    )

    lzma_ret lzma_alone_decoder(
        lzma_stream* strm,
        uint64_t memlimit
    )

    lzma_ret lzma_lzip_decoder(
        lzma_stream* strm,
        uint64_t memlimit,
        uint32_t flags
    )

    lzma_ret lzma_stream_buffer_decode(
        uint64_t* memlimit,
        uint32_t flags,
        const lzma_allocator* allocator,
        const uint8_t* in_,
        size_t* in_pos,
        size_t in_size,
        uint8_t* out,
        size_t* out_pos,
        size_t out_size
    )

    lzma_ret lzma_microlzma_decoder(
        lzma_stream* strm,
        uint64_t comp_size,
        uint64_t uncomp_size,
        lzma_bool uncomp_size_is_exact,
        uint32_t dict_size
    )

    # stream_flags.h

    int LZMA_STREAM_HEADER_SIZE
    int LZMA_BACKWARD_SIZE_MIN
    int LZMA_BACKWARD_SIZE_MAX

    ctypedef struct lzma_stream_flags:
        uint32_t version
        lzma_vli backward_size
        lzma_check_t check
        lzma_reserved_enum reserved_enum1
        lzma_reserved_enum reserved_enum2
        lzma_reserved_enum reserved_enum3
        lzma_reserved_enum reserved_enum4
        lzma_bool reserved_bool1
        lzma_bool reserved_bool2
        lzma_bool reserved_bool3
        lzma_bool reserved_bool4
        lzma_bool reserved_bool5
        lzma_bool reserved_bool6
        lzma_bool reserved_bool7
        lzma_bool reserved_bool8
        uint32_t reserved_int1
        uint32_t reserved_int2

    lzma_ret lzma_stream_header_encode(
        const lzma_stream_flags* options,
        uint8_t* out
    )

    lzma_ret lzma_stream_footer_encode(
        const lzma_stream_flags* options,
        uint8_t* out
    )

    lzma_ret lzma_stream_header_decode(
        lzma_stream_flags* options,
        const uint8_t* in_
    )

    lzma_ret lzma_stream_footer_decode(
        lzma_stream_flags* options,
        const uint8_t* in_
    )

    lzma_ret lzma_stream_flags_compare(
        const lzma_stream_flags* a,
        const lzma_stream_flags* b
    )

    # block.h

    int LZMA_BLOCK_HEADER_SIZE_MIN
    int LZMA_BLOCK_HEADER_SIZE_MAX

    ctypedef struct lzma_block:
        uint32_t version
        uint32_t header_size
        lzma_check_t check
        lzma_vli compressed_size
        lzma_vli uncompressed_size
        lzma_filter* filters
        uint8_t[64] raw_check  # LZMA_CHECK_SIZE_MAX
        void* reserved_ptr1
        void* reserved_ptr2
        void* reserved_ptr3
        uint32_t reserved_int1
        uint32_t reserved_int2
        lzma_vli reserved_int3
        lzma_vli reserved_int4
        lzma_vli reserved_int5
        lzma_vli reserved_int6
        lzma_vli reserved_int7
        lzma_vli reserved_int8
        lzma_reserved_enum reserved_enum1
        lzma_reserved_enum reserved_enum2
        lzma_reserved_enum reserved_enum3
        lzma_reserved_enum reserved_enum4
        lzma_bool ignore_check
        lzma_bool reserved_bool2
        lzma_bool reserved_bool3
        lzma_bool reserved_bool4
        lzma_bool reserved_bool5
        lzma_bool reserved_bool6
        lzma_bool reserved_bool7
        lzma_bool reserved_bool8

    int lzma_block_header_size_decode(int)

    lzma_ret lzma_block_header_size(
        lzma_block* block
    )

    lzma_ret lzma_block_header_encode(
        const lzma_block* block,
        uint8_t* out
    )

    lzma_ret lzma_block_header_decode(
        lzma_block* block,
        const lzma_allocator* allocator,
        const uint8_t* in_
    )

    lzma_ret lzma_block_compressed_size(
        lzma_block* block,
        lzma_vli unpadded_size
    )

    lzma_vli lzma_block_unpadded_size(
        const lzma_block* block
    )

    lzma_vli lzma_block_total_size(
        const lzma_block* block
    )

    lzma_ret lzma_block_encoder(
        lzma_stream* strm,
        lzma_block* block
    )

    lzma_ret lzma_block_decoder(
        lzma_stream* strm,
        lzma_block* block
    )

    size_t lzma_block_buffer_bound(
        size_t uncompressed_size
    )

    lzma_ret lzma_block_buffer_encode(
        lzma_block* block,
        const lzma_allocator* allocator,
        const uint8_t* in_,
        size_t in_size,
        uint8_t* out,
        size_t* out_pos,
        size_t out_size
    )

    lzma_ret lzma_block_uncomp_encode(
        lzma_block* block,
        const uint8_t* in_,
        size_t in_size,
        uint8_t* out,
        size_t* out_pos,
        size_t out_size
    )

    lzma_ret lzma_block_buffer_decode(
        lzma_block* block,
        const lzma_allocator* allocator,
        const uint8_t* in_,
        size_t* in_pos,
        size_t in_size,
        uint8_t* out,
        size_t* out_pos,
        size_t out_size
    )

    # index.h

    ctypedef struct lzma_index:
        pass

    struct _stream:
        const lzma_stream_flags* flags
        const void* reserved_ptr1
        const void* reserved_ptr2
        const void* reserved_ptr3
        lzma_vli number
        lzma_vli block_count
        lzma_vli compressed_offset
        lzma_vli uncompressed_offset
        lzma_vli compressed_size
        lzma_vli uncompressed_size
        lzma_vli padding
        lzma_vli reserved_vli1
        lzma_vli reserved_vli2
        lzma_vli reserved_vli3
        lzma_vli reserved_vli4

    struct _block:
        lzma_vli number_in_file
        lzma_vli compressed_file_offset
        lzma_vli uncompressed_file_offset
        lzma_vli number_in_stream
        lzma_vli compressed_stream_offset
        lzma_vli uncompressed_stream_offset
        lzma_vli uncompressed_size
        lzma_vli unpadded_size
        lzma_vli total_size
        lzma_vli reserved_vli1
        lzma_vli reserved_vli2
        lzma_vli reserved_vli3
        lzma_vli reserved_vli4
        const void* reserved_ptr1
        const void* reserved_ptr2
        const void* reserved_ptr3
        const void* reserved_ptr4

    union _internal:
        const void* p
        size_t s
        lzma_vli v

    ctypedef struct lzma_index_iter:
        _stream stream
        _block block
        _internal[6] internal

    ctypedef enum lzma_index_iter_mode:
        LZMA_INDEX_ITER_ANY
        LZMA_INDEX_ITER_STREAM
        LZMA_INDEX_ITER_BLOCK
        LZMA_INDEX_ITER_NONEMPTY_BLOCK

    uint64_t lzma_index_memusage(
        lzma_vli streams,
        lzma_vli blocks
    )

    uint64_t lzma_index_memused(
        const lzma_index* i
    )

    lzma_index* lzma_index_init(
        const lzma_allocator* allocator
    )

    void lzma_index_end(
        lzma_index* i,
        const lzma_allocator* allocator
    )

    lzma_ret lzma_index_append(
        lzma_index* i,
        const lzma_allocator* allocator,
        lzma_vli unpadded_size,
        lzma_vli uncompressed_size
    )

    lzma_ret lzma_index_stream_flags(
        lzma_index* i,
        const lzma_stream_flags* stream_flags
    )

    uint32_t lzma_index_checks(
        const lzma_index* i
    )

    lzma_ret lzma_index_stream_padding(
        lzma_index* i,
        lzma_vli stream_padding
    )

    lzma_vli lzma_index_stream_count(
        const lzma_index* i
    )

    lzma_vli lzma_index_block_count(
        const lzma_index* i
    )

    lzma_vli lzma_index_size(
        const lzma_index* i
    )

    lzma_vli lzma_index_stream_size(
        const lzma_index* i
    )

    lzma_vli lzma_index_total_size(
        const lzma_index* i
    )

    lzma_vli lzma_index_file_size(
        const lzma_index* i
    )

    lzma_vli lzma_index_uncompressed_size(
        const lzma_index* i
    )

    void lzma_index_iter_init(
        lzma_index_iter* iter_,
        const lzma_index* i
    )

    void lzma_index_iter_rewind(
        lzma_index_iter* iter_
    )

    lzma_bool lzma_index_iter_next(
        lzma_index_iter* iter_,
        lzma_index_iter_mode mode
    )

    lzma_bool lzma_index_iter_locate(
        lzma_index_iter* iter_,
        lzma_vli target
    )

    lzma_ret lzma_index_cat(
        lzma_index* dest,
        lzma_index* src,
        const lzma_allocator* allocator
    )

    lzma_index* lzma_index_dup(
        const lzma_index* i,
        const lzma_allocator* allocator
    )

    lzma_ret lzma_index_encoder(
        lzma_stream* strm,
        const lzma_index* i
    )

    lzma_ret lzma_index_decoder(
        lzma_stream* strm,
        lzma_index** i,
        uint64_t memlimit
    )

    lzma_ret lzma_index_buffer_encode(
        const lzma_index* i,
        uint8_t* out,
        size_t* out_pos,
        size_t out_size
    )

    lzma_ret lzma_index_buffer_decode(
        lzma_index** i,
        uint64_t* memlimit,
        const lzma_allocator* allocator,
        const uint8_t* in_,
        size_t* in_pos,
        size_t in_size
    )

    lzma_ret lzma_file_info_decoder(
        lzma_stream* strm,
        lzma_index** dest_index,
        uint64_t memlimit,
        uint64_t file_size
    )

    # index_hash.h

    ctypedef struct lzma_index_hash:
        pass

    lzma_index_hash* lzma_index_hash_init(
        lzma_index_hash *index_hash,
        const lzma_allocator *allocator
    )

    void lzma_index_hash_end(
        lzma_index_hash *index_hash,
        const lzma_allocator *allocator
    )

    lzma_ret lzma_index_hash_append(
        lzma_index_hash *index_hash,
        lzma_vli unpadded_size,
        lzma_vli uncompressed_size
    )

    lzma_ret lzma_index_hash_decode(
        lzma_index_hash *index_hash,
        const uint8_t *in_,
        size_t *in_pos,
        size_t in_size
    )

    lzma_vli lzma_index_hash_size(
        const lzma_index_hash *index_hash
    )

    # hardware.h

    uint64_t lzma_physmem()

    uint32_t lzma_cputhreads()
