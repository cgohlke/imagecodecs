# imagecodecs/openexr.pxd

# Cython declarations for the `OpenEXR 3.4` C library.
# https://github.com/AcademySoftwareFoundation/openexr

from libc.stdint cimport (
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)


cdef extern from 'openexr_errors.h' nogil:

    ctypedef enum exr_error_code_t:
        EXR_ERR_SUCCESS
        EXR_ERR_OUT_OF_MEMORY
        EXR_ERR_MISSING_CONTEXT_ARG
        EXR_ERR_INVALID_ARGUMENT
        EXR_ERR_ARGUMENT_OUT_OF_RANGE
        EXR_ERR_FILE_ACCESS
        EXR_ERR_FILE_BAD_HEADER
        EXR_ERR_NOT_OPEN_READ
        EXR_ERR_NOT_OPEN_WRITE
        EXR_ERR_HEADER_NOT_WRITTEN
        EXR_ERR_READ_IO
        EXR_ERR_WRITE_IO
        EXR_ERR_NAME_TOO_LONG
        EXR_ERR_MISSING_REQ_ATTR
        EXR_ERR_INVALID_ATTR
        EXR_ERR_NO_ATTR_BY_NAME
        EXR_ERR_ATTR_TYPE_MISMATCH
        EXR_ERR_ATTR_SIZE_MISMATCH
        EXR_ERR_SCAN_TILE_MIXEDAPI
        EXR_ERR_TILE_SCAN_MIXEDAPI
        EXR_ERR_MODIFY_SIZE_CHANGE
        EXR_ERR_ALREADY_WROTE_ATTRS
        EXR_ERR_BAD_CHUNK_LEADER
        EXR_ERR_CORRUPT_CHUNK
        EXR_ERR_INCOMPLETE_CHUNK_TABLE
        EXR_ERR_INCORRECT_PART
        EXR_ERR_INCORRECT_CHUNK
        EXR_ERR_USE_SCAN_DEEP_WRITE
        EXR_ERR_USE_TILE_DEEP_WRITE
        EXR_ERR_USE_SCAN_NONDEEP_WRITE
        EXR_ERR_USE_TILE_NONDEEP_WRITE
        EXR_ERR_INVALID_SAMPLE_DATA
        EXR_ERR_FEATURE_NOT_IMPLEMENTED
        EXR_ERR_UNKNOWN

    ctypedef int32_t exr_result_t

    const char* exr_get_default_error_message(exr_result_t code)

    const char* exr_get_error_code_as_string(exr_result_t code)


cdef extern from 'openexr_base.h' nogil:

    ctypedef void* (*exr_memory_allocation_func_t)(size_t bytes)

    ctypedef void (*exr_memory_free_func_t)(void* ptr)

    void exr_get_library_version(
        int* maj,
        int* min,
        int* patch,
        const char** extra
    )


cdef extern from 'openexr_attr.h' nogil:

    ctypedef enum exr_compression_t:
        EXR_COMPRESSION_NONE
        EXR_COMPRESSION_RLE
        EXR_COMPRESSION_ZIPS
        EXR_COMPRESSION_ZIP
        EXR_COMPRESSION_PIZ
        EXR_COMPRESSION_PXR24
        EXR_COMPRESSION_B44
        EXR_COMPRESSION_B44A
        EXR_COMPRESSION_DWAA
        EXR_COMPRESSION_DWAB
        EXR_COMPRESSION_HTJ2K256
        EXR_COMPRESSION_HTJ2K32
        EXR_COMPRESSION_LAST_TYPE

    ctypedef enum exr_lineorder_t:
        EXR_LINEORDER_INCREASING_Y
        EXR_LINEORDER_DECREASING_Y
        EXR_LINEORDER_RANDOM_Y
        EXR_LINEORDER_LAST_TYPE

    ctypedef enum exr_storage_t:
        EXR_STORAGE_SCANLINE
        EXR_STORAGE_TILED
        EXR_STORAGE_DEEP_SCANLINE
        EXR_STORAGE_DEEP_TILED
        EXR_STORAGE_LAST_TYPE
        EXR_STORAGE_UNKNOWN

    ctypedef enum exr_tile_level_mode_t:
        EXR_TILE_ONE_LEVEL
        EXR_TILE_MIPMAP_LEVELS
        EXR_TILE_RIPMAP_LEVELS
        EXR_TILE_LAST_TYPE

    ctypedef enum exr_tile_round_mode_t:
        EXR_TILE_ROUND_DOWN
        EXR_TILE_ROUND_UP
        EXR_TILE_ROUND_LAST_TYPE

    ctypedef enum exr_pixel_type_t:
        EXR_PIXEL_UINT
        EXR_PIXEL_HALF
        EXR_PIXEL_FLOAT
        EXR_PIXEL_LAST_TYPE

    ctypedef enum exr_perceptual_treatment_t:
        EXR_PERCEPTUALLY_LOGARITHMIC
        EXR_PERCEPTUALLY_LINEAR

    ctypedef struct exr_attr_v2i_t:
        int32_t x
        int32_t y

    ctypedef struct exr_attr_v2f_t:
        float x
        float y

    ctypedef struct exr_attr_box2i_t:
        exr_attr_v2i_t min
        exr_attr_v2i_t max

    ctypedef struct exr_attr_string_t:
        int32_t length
        int32_t alloc_size
        const char* str

    ctypedef struct exr_attr_chlist_entry_t:
        exr_attr_string_t name
        exr_pixel_type_t pixel_type
        uint8_t p_linear
        uint8_t[3] reserved
        int32_t x_sampling
        int32_t y_sampling

    ctypedef struct exr_attr_chlist_t:
        int num_channels
        int num_alloced
        const exr_attr_chlist_entry_t* entries


cdef extern from 'openexr_context.h' nogil:

    ctypedef struct _priv_exr_context_t:
        pass

    ctypedef _priv_exr_context_t* exr_context_t
    ctypedef const _priv_exr_context_t* exr_const_context_t

    ctypedef exr_result_t (*exr_stream_error_func_ptr_t)(
        exr_const_context_t ctxt,
        exr_result_t code,
        const char* fmt,
        ...
    )

    ctypedef void (*exr_error_handler_cb_t)(
        exr_const_context_t ctxt,
        exr_result_t code,
        const char* msg
    )

    ctypedef void (*exr_destroy_stream_func_ptr_t)(
        exr_const_context_t ctxt,
        void* userdata,
        int failed
    )

    ctypedef int64_t (*exr_query_size_func_ptr_t)(
        exr_const_context_t ctxt,
        void* userdata
    )

    ctypedef int64_t (*exr_read_func_ptr_t)(
        exr_const_context_t ctxt,
        void* userdata,
        void* buffer,
        uint64_t sz,
        uint64_t offset,
        exr_stream_error_func_ptr_t error_cb
    )

    ctypedef int64_t (*exr_write_func_ptr_t)(
        exr_const_context_t ctxt,
        void* userdata,
        const void* buffer,
        uint64_t sz,
        uint64_t offset,
        exr_stream_error_func_ptr_t error_cb
    )

    ctypedef struct exr_context_initializer_t:
        size_t size
        exr_error_handler_cb_t error_handler_fn
        exr_memory_allocation_func_t alloc_fn
        exr_memory_free_func_t free_fn
        void* user_data
        exr_read_func_ptr_t read_fn
        exr_query_size_func_ptr_t size_fn
        exr_write_func_ptr_t write_fn
        exr_destroy_stream_func_ptr_t destroy_fn
        int max_image_width
        int max_image_height
        int max_tile_width
        int max_tile_height
        int zip_level
        float dwa_quality
        int flags
        uint8_t[4] pad

    ctypedef enum exr_default_write_mode_t:
        EXR_WRITE_FILE_DIRECTLY
        EXR_INTERMEDIATE_TEMP_FILE

    exr_result_t exr_start_read(
        exr_context_t* ctxt,
        const char* filename,
        const exr_context_initializer_t* ctxtdata
    )

    exr_result_t exr_start_write(
        exr_context_t* ctxt,
        const char* filename,
        exr_default_write_mode_t default_mode,
        const exr_context_initializer_t* ctxtdata
    )

    exr_result_t exr_finish(exr_context_t* ctxt)

    exr_result_t exr_get_file_name(
        exr_const_context_t ctxt,
        const char** name
    )

    exr_result_t exr_write_header(exr_context_t ctxt)


cdef extern from 'openexr_part.h' nogil:

    exr_result_t exr_get_count(
        exr_const_context_t ctxt,
        int* count
    )

    exr_result_t exr_get_storage(
        exr_const_context_t ctxt,
        int part_index,
        exr_storage_t* out
    )

    exr_result_t exr_add_part(
        exr_context_t ctxt,
        const char* partname,
        exr_storage_t type,
        int* new_index
    )

    exr_result_t exr_get_chunk_count(
        exr_const_context_t ctxt,
        int part_index,
        int32_t* out
    )

    exr_result_t exr_get_tile_descriptor(
        exr_const_context_t ctxt,
        int part_index,
        uint32_t* xsize,
        uint32_t* ysize,
        exr_tile_level_mode_t* level,
        exr_tile_round_mode_t* round
    )

    exr_result_t exr_get_tile_levels(
        exr_const_context_t ctxt,
        int part_index,
        int32_t* levelsx,
        int32_t* levelsy
    )

    exr_result_t exr_get_tile_sizes(
        exr_const_context_t ctxt,
        int part_index,
        int levelx,
        int levely,
        int32_t* tilew,
        int32_t* tileh
    )

    exr_result_t exr_get_scanlines_per_chunk(
        exr_const_context_t ctxt,
        int part_index,
        int32_t* out
    )

    exr_result_t exr_initialize_required_attr_simple(
        exr_context_t ctxt,
        int part_index,
        int32_t width,
        int32_t height,
        exr_compression_t ctype
    )

    exr_result_t exr_get_channels(
        exr_const_context_t ctxt,
        int part_index,
        const exr_attr_chlist_t** chlist
    )

    exr_result_t exr_add_channel(
        exr_context_t ctxt,
        int part_index,
        const char* name,
        exr_pixel_type_t ptype,
        exr_perceptual_treatment_t percept,
        int32_t xsamp,
        int32_t ysamp
    )

    exr_result_t exr_get_compression(
        exr_const_context_t ctxt,
        int part_index,
        exr_compression_t* compression
    )

    exr_result_t exr_set_compression(
        exr_context_t ctxt,
        int part_index,
        exr_compression_t ctype
    )

    exr_result_t exr_get_data_window(
        exr_const_context_t ctxt,
        int part_index,
        exr_attr_box2i_t* out
    )

    exr_result_t exr_set_data_window(
        exr_context_t ctxt,
        int part_index,
        const exr_attr_box2i_t* dw
    )

    exr_result_t exr_get_display_window(
        exr_const_context_t ctxt,
        int part_index,
        exr_attr_box2i_t* out
    )

    exr_result_t exr_set_display_window(
        exr_context_t ctxt,
        int part_index,
        const exr_attr_box2i_t* dw
    )

    exr_result_t exr_set_lineorder(
        exr_context_t ctxt,
        int part_index,
        exr_lineorder_t lo
    )

    exr_result_t exr_set_zip_compression_level(
        exr_context_t ctxt,
        int part_index,
        int level
    )

    exr_result_t exr_set_dwa_compression_level(
        exr_context_t ctxt,
        int part_index,
        float level
    )

    exr_result_t exr_set_name(
        exr_context_t ctxt,
        int part_index,
        const char* val
    )


cdef extern from 'openexr_coding.h' nogil:

    ctypedef enum exr_transcoding_pipeline_buffer_id_t:
        EXR_TRANSCODE_BUFFER_PACKED
        EXR_TRANSCODE_BUFFER_UNPACKED
        EXR_TRANSCODE_BUFFER_COMPRESSED
        EXR_TRANSCODE_BUFFER_SCRATCH1
        EXR_TRANSCODE_BUFFER_SCRATCH2
        EXR_TRANSCODE_BUFFER_PACKED_SAMPLES
        EXR_TRANSCODE_BUFFER_SAMPLES

    ctypedef struct exr_coding_channel_info_t:
        const char* channel_name
        int32_t height
        int32_t width
        int32_t x_samples
        int32_t y_samples
        uint8_t p_linear
        int8_t bytes_per_element
        uint16_t data_type
        int16_t user_bytes_per_element
        uint16_t user_data_type
        int32_t user_pixel_stride
        int32_t user_line_stride
        uint8_t* decode_to_ptr
        const uint8_t* encode_from_ptr


cdef extern from 'openexr_chunkio.h' nogil:

    ctypedef struct exr_chunk_info_t:
        int32_t idx
        int32_t start_x
        int32_t start_y
        int32_t height
        int32_t width
        uint8_t level_x
        uint8_t level_y
        uint8_t type
        uint8_t compression
        uint64_t data_offset
        uint64_t packed_size
        uint64_t unpacked_size
        uint64_t sample_count_data_offset
        uint64_t sample_count_table_size

    exr_result_t exr_read_scanline_chunk_info(
        exr_const_context_t ctxt,
        int part_index,
        int y,
        exr_chunk_info_t* cinfo
    )

    exr_result_t exr_read_tile_chunk_info(
        exr_const_context_t ctxt,
        int part_index,
        int tilex,
        int tiley,
        int levelx,
        int levely,
        exr_chunk_info_t* cinfo
    )

    exr_result_t exr_write_scanline_chunk_info(
        exr_context_t ctxt,
        int part_index,
        int y,
        exr_chunk_info_t* cinfo
    )


cdef extern from 'openexr_encode.h' nogil:

    ctypedef struct exr_encode_pipeline_t:
        size_t pipe_size
        exr_coding_channel_info_t* channels
        int16_t channel_count
        uint16_t encode_flags
        int part_index
        exr_const_context_t context
        exr_chunk_info_t chunk
        void* encoding_user_data
        void* packed_buffer
        uint64_t packed_bytes
        size_t packed_alloc_size
        int32_t* sample_count_table
        size_t sample_count_alloc_size
        void* packed_sample_count_table
        size_t packed_sample_count_bytes
        size_t packed_sample_count_alloc_size
        void* compressed_buffer
        size_t compressed_bytes
        size_t compressed_alloc_size
        void* scratch_buffer_1
        size_t scratch_alloc_size_1
        void* scratch_buffer_2
        size_t scratch_alloc_size_2

    exr_result_t exr_encoding_initialize(
        exr_const_context_t ctxt,
        int part_index,
        const exr_chunk_info_t* cinfo,
        exr_encode_pipeline_t* encode_pipe
    )

    exr_result_t exr_encoding_choose_default_routines(
        exr_const_context_t ctxt,
        int part_index,
        exr_encode_pipeline_t* encode_pipe
    )

    exr_result_t exr_encoding_update(
        exr_const_context_t ctxt,
        int part_index,
        const exr_chunk_info_t* cinfo,
        exr_encode_pipeline_t* encode_pipe
    )

    exr_result_t exr_encoding_run(
        exr_const_context_t ctxt,
        int part_index,
        exr_encode_pipeline_t* encode_pipe
    )

    exr_result_t exr_encoding_destroy(
        exr_const_context_t ctxt,
        exr_encode_pipeline_t* encode_pipe
    )


cdef extern from 'openexr_decode.h' nogil:

    ctypedef struct exr_decode_pipeline_t:
        size_t pipe_size
        exr_coding_channel_info_t* channels
        int16_t channel_count
        uint16_t decode_flags
        int part_index
        exr_const_context_t context
        exr_chunk_info_t chunk
        int32_t user_line_begin_skip
        int32_t user_line_end_ignore
        uint64_t bytes_decompressed
        void* decoding_user_data
        void* packed_buffer
        size_t packed_alloc_size
        void* unpacked_buffer
        size_t unpacked_alloc_size

    exr_result_t exr_decoding_initialize(
        exr_const_context_t ctxt,
        int part_index,
        const exr_chunk_info_t* cinfo,
        exr_decode_pipeline_t* decode
    )

    exr_result_t exr_decoding_choose_default_routines(
        exr_const_context_t ctxt,
        int part_index,
        exr_decode_pipeline_t* decode
    )

    exr_result_t exr_decoding_update(
        exr_const_context_t ctxt,
        int part_index,
        const exr_chunk_info_t* cinfo,
        exr_decode_pipeline_t* decode
    )

    exr_result_t exr_decoding_run(
        exr_const_context_t ctxt,
        int part_index,
        exr_decode_pipeline_t* decode
    )

    exr_result_t exr_decoding_destroy(
        exr_const_context_t ctxt,
        exr_decode_pipeline_t* decode
    )
