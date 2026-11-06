# imagecodecs/jpegli.pxd

# Cython declarations for the `jpegli 0.12.0` library.
# https://github.com/google/jpegli

# Requires encode.h and decode.h patched to include <stddef.h> and <stdio.h>
# instead of the C++ headers <cstddef> and <cstdio>.

from libc.stdio cimport FILE


# must be included before jpegli headers, which include <jpeglib.h>
cdef extern from 'jpegli/jpeglib.h' nogil:

    # jconfig.h
    int JPEG_LIB_VERSION
    int BITS_IN_JSAMPLE

    # jmorecfg.h
    int MAX_COMPONENTS
    int MAXJSAMPLE
    int CENTERJSAMPLE
    int JPEG_MAX_DIMENSION

    ctypedef unsigned char JSAMPLE
    ctypedef short JCOEF
    ctypedef unsigned char JOCTET
    ctypedef unsigned char UINT8
    ctypedef unsigned short UINT16
    ctypedef short INT16
    ctypedef long INT32
    ctypedef unsigned int JDIMENSION
    ctypedef int boolean

    int FALSE
    int TRUE

    # jpeglib.h
    int DCTSIZE
    int DCTSIZE2
    int NUM_QUANT_TBLS
    int NUM_HUFF_TBLS
    int NUM_ARITH_TBLS
    int MAX_COMPS_IN_SCAN
    int MAX_SAMP_FACTOR

    int C_MAX_BLOCKS_IN_MCU
    int D_MAX_BLOCKS_IN_MCU

    ctypedef JSAMPLE* JSAMPROW
    ctypedef JSAMPROW* JSAMPARRAY
    ctypedef JSAMPARRAY* JSAMPIMAGE

    ctypedef JCOEF[64] JBLOCK  # DCTSIZE2
    ctypedef JBLOCK* JBLOCKROW
    ctypedef JBLOCKROW* JBLOCKARRAY
    ctypedef JBLOCKARRAY* JBLOCKIMAGE

    ctypedef JCOEF* JCOEFPTR

    ctypedef struct JQUANT_TBL:
        UINT16[64] quantval  # DCTSIZE2
        boolean sent_table

    ctypedef struct JHUFF_TBL:
        UINT8[17] bits
        UINT8[256] huffval
        boolean sent_table

    ctypedef struct jpeg_component_info:
        int component_id
        int component_index
        int h_samp_factor
        int v_samp_factor
        int quant_tbl_no
        int dc_tbl_no
        int ac_tbl_no
        JDIMENSION width_in_blocks
        JDIMENSION height_in_blocks
        int DCT_scaled_size
        JDIMENSION downsampled_width
        JDIMENSION downsampled_height
        boolean component_needed
        int MCU_width
        int MCU_height
        int MCU_blocks
        int MCU_sample_width
        int last_col_width
        int last_row_height
        JQUANT_TBL* quant_table
        void* dct_table

    ctypedef struct jpeg_scan_info:
        int comps_in_scan
        int[4] component_index  # MAX_COMPS_IN_SCAN
        int Ss
        int Se
        int Ah
        int Al

    ctypedef jpeg_marker_struct* jpeg_saved_marker_ptr

    struct jpeg_marker_struct:
        jpeg_saved_marker_ptr next
        UINT8 marker
        unsigned int original_length
        unsigned int data_length
        JOCTET* data

    int JCS_EXTENSIONS
    int JCS_ALPHA_EXTENSIONS

    ctypedef enum J_COLOR_SPACE:
        JCS_UNKNOWN
        JCS_GRAYSCALE
        JCS_RGB
        JCS_YCbCr
        JCS_CMYK
        JCS_YCCK
        JCS_EXT_RGB
        JCS_EXT_RGBX
        JCS_EXT_BGR
        JCS_EXT_BGRX
        JCS_EXT_XBGR
        JCS_EXT_XRGB
        JCS_EXT_RGBA
        JCS_EXT_BGRA
        JCS_EXT_ABGR
        JCS_EXT_ARGB
        JCS_RGB565

    ctypedef enum J_DCT_METHOD:
        JDCT_ISLOW
        JDCT_IFAST
        JDCT_FLOAT

    int JDCT_DEFAULT
    int JDCT_FASTEST

    ctypedef enum J_DITHER_MODE:
        JDITHER_NONE
        JDITHER_ORDERED
        JDITHER_FS

    struct jpeg_common_struct:
        # jpeg_common_fields
        jpeg_error_mgr* err
        jpeg_memory_mgr* mem
        jpeg_progress_mgr* progress
        void* client_data
        boolean is_decompressor
        int global_state

    ctypedef jpeg_common_struct* j_common_ptr
    ctypedef jpeg_compress_struct* j_compress_ptr
    ctypedef jpeg_decompress_struct* j_decompress_ptr

    struct jpeg_compress_struct:
        # jpeg_common_fields
        jpeg_error_mgr* err
        jpeg_memory_mgr* mem
        jpeg_progress_mgr* progress
        void* client_data
        boolean is_decompressor
        int global_state

        jpeg_destination_mgr* dest
        JDIMENSION image_width
        JDIMENSION image_height
        int input_components
        J_COLOR_SPACE in_color_space
        double input_gamma
        int data_precision
        int num_components
        J_COLOR_SPACE jpeg_color_space
        jpeg_component_info* comp_info
        (JQUANT_TBL*)[4] quant_tbl_ptrs  # NUM_QUANT_TBLS
        (JHUFF_TBL*)[4] dc_huff_tbl_ptrs  # NUM_HUFF_TBLS
        (JHUFF_TBL*)[4] ac_huff_tbl_ptrs  # NUM_HUFF_TBLS
        UINT8[16] arith_dc_L  # NUM_ARITH_TBLS
        UINT8[16] arith_dc_U  # NUM_ARITH_TBLS
        UINT8[16] arith_ac_K  # NUM_ARITH_TBLS
        int num_scans
        const jpeg_scan_info* scan_info
        boolean raw_data_in
        boolean arith_code
        boolean optimize_coding
        boolean CCIR601_sampling
        int smoothing_factor
        J_DCT_METHOD dct_method
        unsigned int restart_interval
        int restart_in_rows
        boolean write_JFIF_header
        UINT8 JFIF_major_version
        UINT8 JFIF_minor_version
        UINT8 density_unit
        UINT16 X_density
        UINT16 Y_density
        boolean write_Adobe_marker
        JDIMENSION next_scanline
        boolean progressive_mode
        int max_h_samp_factor
        int max_v_samp_factor
        JDIMENSION total_iMCU_rows
        int comps_in_scan
        (jpeg_component_info*)[4] cur_comp_info  # MAX_COMPS_IN_SCAN
        JDIMENSION MCUs_per_row
        JDIMENSION MCU_rows_in_scan
        int blocks_in_MCU
        int[10] MCU_membership  # C_MAX_BLOCKS_IN_MCU
        int Ss
        int Se
        int Ah
        int Al
        jpeg_comp_master* master
        jpeg_scan_info* script_space
        int script_space_size

    struct jpeg_decompress_struct:
        # jpeg_common_fields
        jpeg_error_mgr* err
        jpeg_memory_mgr* mem
        jpeg_progress_mgr* progress
        void* client_data
        boolean is_decompressor
        int global_state

        jpeg_source_mgr* src
        JDIMENSION image_width
        JDIMENSION image_height
        int num_components
        J_COLOR_SPACE jpeg_color_space
        J_COLOR_SPACE out_color_space
        unsigned int scale_num
        unsigned int scale_denom
        double output_gamma
        boolean buffered_image
        boolean raw_data_out
        J_DCT_METHOD dct_method
        boolean do_fancy_upsampling
        boolean do_block_smoothing
        boolean quantize_colors
        J_DITHER_MODE dither_mode
        boolean two_pass_quantize
        int desired_number_of_colors
        boolean enable_1pass_quant
        boolean enable_external_quant
        boolean enable_2pass_quant
        JDIMENSION output_width
        JDIMENSION output_height
        int out_color_components
        int output_components
        int rec_outbuf_height
        int actual_number_of_colors
        JSAMPARRAY colormap
        JDIMENSION output_scanline
        int input_scan_number
        JDIMENSION input_iMCU_row
        int output_scan_number
        JDIMENSION output_iMCU_row
        int (*coef_bits)[64]  # DCTSIZE2
        (JQUANT_TBL*)[4] quant_tbl_ptrs  # NUM_QUANT_TBLS
        (JHUFF_TBL*)[4] dc_huff_tbl_ptrs  # NUM_HUFF_TBLS
        (JHUFF_TBL*)[4] ac_huff_tbl_ptrs  # NUM_HUFF_TBLS
        int data_precision
        jpeg_component_info* comp_info
        boolean progressive_mode
        boolean arith_code
        UINT8[16] arith_dc_L  # NUM_ARITH_TBLS
        UINT8[16] arith_dc_U  # NUM_ARITH_TBLS
        UINT8[16] arith_ac_K  # NUM_ARITH_TBLS
        unsigned int restart_interval
        boolean saw_JFIF_marker
        UINT8 JFIF_major_version
        UINT8 JFIF_minor_version
        UINT8 density_unit
        UINT16 X_density
        UINT16 Y_density
        boolean saw_Adobe_marker
        UINT8 Adobe_transform
        boolean CCIR601_sampling
        jpeg_saved_marker_ptr marker_list
        int max_h_samp_factor
        int max_v_samp_factor
        int min_DCT_scaled_size
        JDIMENSION total_iMCU_rows
        JSAMPLE* sample_range_limit
        int comps_in_scan
        (jpeg_component_info*)[4] cur_comp_info  # MAX_COMPS_IN_SCAN
        JDIMENSION MCUs_per_row
        JDIMENSION MCU_rows_in_scan
        int blocks_in_MCU
        int[10] MCU_membership  # D_MAX_BLOCKS_IN_MCU
        int Ss
        int Se
        int Ah
        int Al
        int unread_marker
        jpeg_decomp_master* master

    int JMSG_STR_PARM_MAX
    int JMSG_LENGTH_MAX

    union _msg_parm:
        int[8] i
        char[80] s  # JMSG_STR_PARM_MAX

    struct jpeg_error_mgr:
        void (*error_exit)(
            j_common_ptr cinfo
        ) nogil

        void (*emit_message)(
            j_common_ptr cinfo,
            int msg_level
        ) nogil

        void (*output_message)(
            j_common_ptr cinfo
        ) nogil

        void (*format_message)(
            j_common_ptr cinfo,
            char* buffer
        ) nogil

        void (*reset_error_mgr)(
            j_common_ptr cinfo
        ) nogil

        int msg_code
        _msg_parm msg_parm
        int trace_level
        long num_warnings
        const char** jpeg_message_table
        int last_jpeg_message
        const char** addon_message_table
        int first_addon_message
        int last_addon_message

    struct jpeg_progress_mgr:
        void (*progress_monitor)(
            j_common_ptr cinfo
        ) nogil

        long pass_counter
        long pass_limit
        int completed_passes
        int total_passes

    struct jpeg_destination_mgr:
        JOCTET* next_output_byte
        size_t free_in_buffer

        void (*init_destination)(
            j_compress_ptr cinfo
        ) nogil

        boolean (*empty_output_buffer)(
            j_compress_ptr cinfo
        ) nogil

        void (*term_destination)(
            j_compress_ptr cinfo
        ) nogil

    struct jpeg_source_mgr:
        const JOCTET* next_input_byte
        size_t bytes_in_buffer

        void (*init_source)(
            j_decompress_ptr cinfo
        ) nogil

        boolean (*fill_input_buffer)(
            j_decompress_ptr cinfo
        ) nogil

        void (*skip_input_data)(
            j_decompress_ptr cinfo,
            long num_bytes
        ) nogil

        boolean (*resync_to_restart)(
            j_decompress_ptr cinfo,
            int desired
        ) nogil

        void (*term_source)(
            j_decompress_ptr cinfo
        ) nogil

    int JPOOL_PERMANENT
    int JPOOL_IMAGE
    int JPOOL_NUMPOOLS

    ctypedef jvirt_sarray_control* jvirt_sarray_ptr
    ctypedef jvirt_barray_control* jvirt_barray_ptr

    struct jpeg_memory_mgr:
        long dummy

    struct jvirt_sarray_control:
        long dummy

    struct jvirt_barray_control:
        long dummy

    struct jpeg_comp_master:
        long dummy

    struct jpeg_decomp_master:
        long dummy

    ctypedef boolean (*jpeg_marker_parser_method)(
            j_decompress_ptr cinfo
    ) nogil

    int JPEG_SUSPENDED
    int JPEG_HEADER_OK
    int JPEG_HEADER_TABLES_ONLY

    int JPEG_REACHED_SOS
    int JPEG_REACHED_EOI
    int JPEG_ROW_COMPLETED
    int JPEG_SCAN_COMPLETED

    int JPEG_RST0
    int JPEG_EOI
    int JPEG_APP0
    int JPEG_COM


cdef extern from 'jpegli/lib/jpegli/types.h' nogil:

    ctypedef enum JpegliDataType:
        JPEGLI_TYPE_FLOAT
        JPEGLI_TYPE_UINT8
        JPEGLI_TYPE_UINT16
        JPEGLI_TYPE_FLOAT16

    ctypedef enum JpegliEndianness:
        JPEGLI_NATIVE_ENDIAN
        JPEGLI_LITTLE_ENDIAN
        JPEGLI_BIG_ENDIAN

    int jpegli_bytes_per_sample(
        JpegliDataType data_type
    )


cdef extern from 'jpegli/lib/jpegli/common.h' nogil:

    jpeg_error_mgr* jpegli_std_error(
        jpeg_error_mgr* err
    )

    void jpegli_abort(
        j_common_ptr cinfo
    )

    void jpegli_destroy(
        j_common_ptr cinfo
    )

    JQUANT_TBL* jpegli_alloc_quant_table(
        j_common_ptr cinfo
    )

    JHUFF_TBL* jpegli_alloc_huff_table(
        j_common_ptr cinfo
    )


cdef extern from 'jpegli/lib/jpegli/encode.h' nogil:

    void jpegli_create_compress(
        j_compress_ptr cinfo
    )

    void jpegli_CreateCompress(
        j_compress_ptr cinfo,
        int version,
        size_t structsize
    )

    void jpegli_stdio_dest(
        j_compress_ptr cinfo,
        FILE* outfile
    )

    void jpegli_mem_dest(
        j_compress_ptr cinfo,
        unsigned char** outbuffer,
        unsigned long* outsize
    )

    void jpegli_set_defaults(
        j_compress_ptr cinfo
    )

    void jpegli_default_colorspace(
        j_compress_ptr cinfo
    )

    void jpegli_set_colorspace(
        j_compress_ptr cinfo,
        J_COLOR_SPACE colorspace
    )

    void jpegli_set_quality(
        j_compress_ptr cinfo,
        int quality,
        boolean force_baseline
    )

    void jpegli_set_linear_quality(
        j_compress_ptr cinfo,
        int scale_factor,
        boolean force_baseline
    )

    int jpegli_quality_scaling(
        int quality
    )

    void jpegli_add_quant_table(
        j_compress_ptr cinfo,
        int which_tbl,
        const unsigned int* basic_table,
        int scale_factor,
        boolean force_baseline
    )

    void jpegli_simple_progression(
        j_compress_ptr cinfo
    )

    void jpegli_suppress_tables(
        j_compress_ptr cinfo,
        boolean suppress
    )

    void jpegli_copy_critical_parameters(
        j_decompress_ptr srcinfo,
        j_compress_ptr dstinfo
    )

    void jpegli_write_m_header(
        j_compress_ptr cinfo,
        int marker,
        unsigned int datalen
    )

    void jpegli_write_m_byte(
        j_compress_ptr cinfo,
        int val
    )

    void jpegli_write_marker(
        j_compress_ptr cinfo,
        int marker,
        const JOCTET* dataptr,
        unsigned int datalen
    )

    void jpegli_write_icc_profile(
        j_compress_ptr cinfo,
        const JOCTET* icc_data_ptr,
        unsigned int icc_data_len
    )

    void jpegli_start_compress(
        j_compress_ptr cinfo,
        boolean write_all_tables
    )

    void jpegli_write_tables(
        j_compress_ptr cinfo
    )

    JDIMENSION jpegli_write_scanlines(
        j_compress_ptr cinfo,
        JSAMPARRAY scanlines,
        JDIMENSION num_lines
    )

    JDIMENSION jpegli_write_raw_data(
        j_compress_ptr cinfo,
        JSAMPIMAGE data,
        JDIMENSION num_lines
    )

    void jpegli_write_coefficients(
        j_compress_ptr cinfo,
        jvirt_barray_ptr* coef_arrays
    )

    void jpegli_finish_compress(
        j_compress_ptr cinfo
    )

    void jpegli_abort_compress(
        j_compress_ptr cinfo
    )

    void jpegli_destroy_compress(
        j_compress_ptr cinfo
    )

    void jpegli_set_distance(
        j_compress_ptr cinfo,
        float distance,
        boolean force_baseline
    )

    float jpegli_quality_to_distance(
        int quality
    )

    void jpegli_set_psnr(
        j_compress_ptr cinfo,
        float psnr,
        float tolerance,
        float min_distance,
        float max_distance
    )

    void jpegli_set_xyb_mode(
        j_compress_ptr cinfo
    )

    void jpegli_set_cicp_transfer_function(
        j_compress_ptr cinfo,
        int code
    )

    void jpegli_set_input_format(
        j_compress_ptr cinfo,
        JpegliDataType data_type,
        JpegliEndianness endianness
    )

    void jpegli_enable_adaptive_quantization(
        j_compress_ptr cinfo,
        boolean value
    )

    void jpegli_set_progressive_level(
        j_compress_ptr cinfo,
        int level
    )

    void jpegli_use_standard_quant_tables(
        j_compress_ptr cinfo
    )


cdef extern from 'jpegli/lib/jpegli/decode.h' nogil:

    void jpegli_create_decompress(
        j_decompress_ptr cinfo
    )

    void jpegli_CreateDecompress(
        j_decompress_ptr cinfo,
        int version,
        size_t structsize
    )

    void jpegli_stdio_src(
        j_decompress_ptr cinfo,
        FILE* infile
    )

    void jpegli_mem_src(
        j_decompress_ptr cinfo,
        const unsigned char* inbuffer,
        unsigned long insize
    )

    int jpegli_read_header(
        j_decompress_ptr cinfo,
        boolean require_image
    )

    boolean jpegli_start_decompress(
        j_decompress_ptr cinfo
    )

    JDIMENSION jpegli_read_scanlines(
        j_decompress_ptr cinfo,
        JSAMPARRAY scanlines,
        JDIMENSION max_lines
    )

    JDIMENSION jpegli_skip_scanlines(
        j_decompress_ptr cinfo,
        JDIMENSION num_lines
    )

    void jpegli_crop_scanline(
        j_decompress_ptr cinfo,
        JDIMENSION* xoffset,
        JDIMENSION* width
    )

    boolean jpegli_finish_decompress(
        j_decompress_ptr cinfo
    )

    JDIMENSION jpegli_read_raw_data(
        j_decompress_ptr cinfo,
        JSAMPIMAGE data,
        JDIMENSION max_lines
    )

    jvirt_barray_ptr* jpegli_read_coefficients(
        j_decompress_ptr cinfo
    )

    boolean jpegli_has_multiple_scans(
        j_decompress_ptr cinfo
    )

    boolean jpegli_start_output(
        j_decompress_ptr cinfo,
        int scan_number
    )

    boolean jpegli_finish_output(
        j_decompress_ptr cinfo
    )

    boolean jpegli_input_complete(
        j_decompress_ptr cinfo
    )

    int jpegli_consume_input(
        j_decompress_ptr cinfo
    )

    void jpegli_calc_output_dimensions(
        j_decompress_ptr cinfo
    )

    void jpegli_save_markers(
        j_decompress_ptr cinfo,
        int marker_code,
        unsigned int length_limit
    )

    void jpegli_set_marker_processor(
        j_decompress_ptr cinfo,
        int marker_code,
        jpeg_marker_parser_method routine
    )

    boolean jpegli_resync_to_restart(
        j_decompress_ptr cinfo,
        int desired
    )

    boolean jpegli_read_icc_profile(
        j_decompress_ptr cinfo,
        JOCTET** icc_data_ptr,
        unsigned int* icc_data_len
    )

    void jpegli_abort_decompress(
        j_decompress_ptr cinfo
    )

    void jpegli_destroy_decompress(
        j_decompress_ptr cinfo
    )

    void jpegli_new_colormap(
        j_decompress_ptr cinfo
    )

    void jpegli_set_output_format(
        j_decompress_ptr cinfo,
        JpegliDataType data_type,
        JpegliEndianness endianness
    )
