# imagecodecs/libjpeg_turbo.pxd
# cython: language_level = 3

# Cython declarations for the `libjpeg-turbo 3.1.0` library.
# https://github.com/libjpeg-turbo/libjpeg-turbo

from libc.stdio cimport FILE

cdef extern from 'jpeglib.h' nogil:

    # jconfig.h
    int JPEG_LIB_VERSION
    int LIBJPEG_TURBO_VERSION
    int LIBJPEG_TURBO_VERSION_NUMBER
    int BITS_IN_JSAMPLE

    # jmorecfg.h
    int MAX_COMPONENTS
    int JPEG_MAX_DIMENSION

    ctypedef unsigned char JSAMPLE
    int MAXJSAMPLE
    int CENTERJSAMPLE

    ctypedef short J12SAMPLE
    int MAXJ12SAMPLE
    int CENTERJ12SAMPLE

    ctypedef unsigned short J16SAMPLE
    int MAXJ16SAMPLE
    int CENTERJ16SAMPLE

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

    int GETJOCTET(int)
    int GETJSAMPLE(int)

    int RGB_RED
    int RGB_GREEN
    int RGB_BLUE
    int RGB_PIXELSIZE

    int JPEG_NUMCS

    int EXT_RGB_RED
    int EXT_RGB_GREEN
    int EXT_RGB_BLUE
    int EXT_RGB_PIXELSIZE

    int EXT_RGBX_RED
    int EXT_RGBX_GREEN
    int EXT_RGBX_BLUE
    int EXT_RGBX_PIXELSIZE

    int EXT_BGR_RED
    int EXT_BGR_GREEN
    int EXT_BGR_BLUE
    int EXT_BGR_PIXELSIZE

    int EXT_BGRX_RED
    int EXT_BGRX_GREEN
    int EXT_BGRX_BLUE
    int EXT_BGRX_PIXELSIZE

    int EXT_XBGR_RED
    int EXT_XBGR_GREEN
    int EXT_XBGR_BLUE
    int EXT_XBGR_PIXELSIZE

    int EXT_XRGB_RED
    int EXT_XRGB_GREEN
    int EXT_XRGB_BLUE
    int EXT_XRGB_PIXELSIZE

    const int[17] rgb_red  # JPEG_NUMCS
    const int[17] rgb_green  # JPEG_NUMCS
    const int[17] rgb_blue  # JPEG_NUMCS
    const int[17] rgb_pixelsize  # JPEG_NUMCS

    int MULTIPLIER
    int FAST_FLOAT

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

    ctypedef J12SAMPLE *J12SAMPROW
    ctypedef J12SAMPROW *J12SAMPARRAY
    ctypedef J12SAMPARRAY *J12SAMPIMAGE

    ctypedef J16SAMPLE *J16SAMPROW
    ctypedef J16SAMPROW *J16SAMPARRAY
    ctypedef J16SAMPARRAY *J16SAMPIMAGE

    ctypedef JCOEF JBLOCK[64]  # DCTSIZE2
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
        int DCT_h_scaled_size
        int DCT_v_scaled_size
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

    struct jpeg_marker_struct:
        pass

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
        unsigned int scale_num
        unsigned int scale_denom
        JDIMENSION jpeg_width
        JDIMENSION jpeg_height
        int data_precision
        int num_components
        J_COLOR_SPACE jpeg_color_space
        jpeg_component_info* comp_info
        (JQUANT_TBL*)[4] quant_tbl_ptrs  # NUM_QUANT_TBLS
        int[4] q_scale_factor  # NUM_QUANT_TBLS
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
        boolean do_fancy_downsampling
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
        int min_DCT_h_scaled_size
        int min_DCT_v_scaled_size
        JDIMENSION total_iMCU_rows
        int comps_in_scan
        (jpeg_component_info*)[4] cur_comp_info  # MAX_COMPS_IN_SCAN
        JDIMENSION MCUs_per_row
        JDIMENSION MCU_rows_in_scan
        int blocks_in_MCU
        int[1] MCU_membership  # C_MAX_BLOCKS_IN_MCU
        int Ss
        int Se
        int Ah
        int Al
        int block_size
        const int* natural_order
        int lim_Se
        jpeg_comp_master* master
        jpeg_c_main_controller* main
        jpeg_c_prep_controller* prep
        jpeg_c_coef_controller* coef
        jpeg_marker_writer* marker
        jpeg_color_converter* cconvert
        jpeg_downsampler* downsample
        jpeg_forward_dct* fdct
        jpeg_entropy_encoder* entropy
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
        boolean is_baseline
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
        int min_DCT_h_scaled_size
        int min_DCT_v_scaled_size
        int min_DCT_scaled_size
        JDIMENSION total_iMCU_rows
        JSAMPLE* sample_range_limit
        int comps_in_scan
        (jpeg_component_info*)[4] cur_comp_info  # MAX_COMPS_IN_SCAN
        JDIMENSION MCUs_per_row
        JDIMENSION MCU_rows_in_scan
        int blocks_in_MCU
        int[1] MCU_membership  # D_MAX_BLOCKS_IN_MCU
        int Ss
        int Se
        int Ah
        int Al
        int block_size
        const int* natural_order
        int lim_Se
        int unread_marker
        jpeg_decomp_master* master
        jpeg_d_main_controller* main
        jpeg_d_coef_controller* coef
        jpeg_d_post_controller* post
        jpeg_input_controller* inputctl
        jpeg_marker_reader* marker
        jpeg_entropy_decoder* entropy
        jpeg_inverse_dct* idct
        jpeg_upsampler* upsample
        jpeg_color_deconverter* cconvert
        jpeg_color_quantizer* cquantize

    ctypedef jpeg_common_struct* j_common_ptr
    ctypedef jpeg_compress_struct* j_compress_ptr
    ctypedef jpeg_decompress_struct* j_decompress_ptr

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
        const char* const* jpeg_message_table
        int last_jpeg_message
        const char* const* addon_message_table
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

    struct jvirt_sarray_control:
        long dummy

    struct jvirt_barray_control:
        long dummy

    ctypedef jvirt_sarray_control* jvirt_sarray_ptr
    ctypedef jvirt_barray_control* jvirt_barray_ptr

    struct jpeg_memory_mgr:
        void* (*alloc_small)(
            j_common_ptr cinfo,
            int pool_id,
            size_t sizeofobject
        ) nogil

        void* (*alloc_large)(
            j_common_ptr cinfo,
            int pool_id,
            size_t sizeofobject
        ) nogil

        JSAMPARRAY (*alloc_sarray)(
            j_common_ptr cinfo,
            int pool_id,
            JDIMENSION samplesperrow,
            JDIMENSION numrows
        ) nogil

        JBLOCKARRAY (*alloc_barray)(
            j_common_ptr cinfo,
            int pool_id,
            JDIMENSION blocksperrow,
            JDIMENSION numrows
        ) nogil

        jvirt_sarray_ptr (*request_virt_sarray)(
            j_common_ptr cinfo,
            int pool_id,
            boolean pre_zero,
            JDIMENSION samplesperrow,
            JDIMENSION numrows,
            JDIMENSION maxaccess
        ) nogil

        jvirt_barray_ptr (*request_virt_barray)(
            j_common_ptr cinfo,
            int pool_id,
            boolean pre_zero,
            JDIMENSION blocksperrow,
            JDIMENSION numrows,
            JDIMENSION maxaccess
        ) nogil

        void (*realize_virt_arrays)(
            j_common_ptr cinfo
        ) nogil

        JSAMPARRAY (*access_virt_sarray)(
            j_common_ptr cinfo,
            jvirt_sarray_ptr ptr,
            JDIMENSION start_row,
            JDIMENSION num_rows,
            boolean writable
        ) nogil

        JBLOCKARRAY (*access_virt_barray)(
            j_common_ptr cinfo,
            jvirt_barray_ptr ptr,
            JDIMENSION start_row,
            JDIMENSION num_rows,
            boolean writable
        ) nogil

        void (*free_pool)(
            j_common_ptr cinfo,
            int pool_id
        ) nogil

        void (*self_destruct)(
            j_common_ptr cinfo
        ) nogil

        long max_memory_to_use
        long max_alloc_chunk

    ctypedef boolean (*jpeg_marker_parser_method)(
            j_decompress_ptr cinfo
    ) nogil

    int JPP(arglist)

    jpeg_error_mgr* jpeg_std_error(
        jpeg_error_mgr* err
    )

    void jpeg_create_compress(
        j_compress_ptr cinfo
    )

    void jpeg_create_decompress(
        j_decompress_ptr cinfo
    )

    void jpeg_CreateCompress(
        j_compress_ptr cinfo,
        int version,
        size_t structsize
    )

    void jpeg_CreateDecompress(
        j_decompress_ptr cinfo,
        int version,
        size_t structsize
    )

    void jpeg_destroy_compress(
        j_compress_ptr cinfo
    )

    void jpeg_destroy_decompress(
        j_decompress_ptr cinfo
    )

    void jpeg_stdio_dest(
        j_compress_ptr cinfo,
        FILE* outfile
    )

    void jpeg_stdio_src(
        j_decompress_ptr cinfo,
        FILE* infile
    )

    void jpeg_mem_dest(
        j_compress_ptr cinfo,
        unsigned char** outbuffer,
        unsigned long* outsize
    )

    void jpeg_mem_src(
        j_decompress_ptr cinfo,
        const unsigned char* inbuffer,
        unsigned long insize
    )

    void jpeg_set_defaults(
        j_compress_ptr cinfo
    )

    void jpeg_set_colorspace(
        j_compress_ptr cinfo,
        J_COLOR_SPACE colorspace
    )

    void jpeg_default_colorspace(
        j_compress_ptr cinfo
    )

    void jpeg_set_quality(
        j_compress_ptr cinfo,
        int quality,
        boolean force_baseline
    )

    void jpeg_set_linear_quality(
        j_compress_ptr cinfo,
        int scale_factor,
        boolean force_baseline
    )

    void jpeg_default_qtables(
        j_compress_ptr cinfo,
        boolean force_baseline
    )

    void jpeg_add_quant_table(
        j_compress_ptr cinfo,
        int which_tbl,
        const unsigned int* basic_table,
        int scale_factor,
        boolean force_baseline
    )

    int jpeg_quality_scaling(
        int quality
    )

    void jpeg_enable_lossless(
        j_compress_ptr cinfo,
        int predictor_selection_value,
        int point_transform
    )

    void jpeg_simple_progression(
        j_compress_ptr cinfo
    )

    void jpeg_suppress_tables(
        j_compress_ptr cinfo,
        boolean suppress
    )

    JQUANT_TBL* jpeg_alloc_quant_table(
        j_common_ptr cinfo
    )

    JHUFF_TBL* jpeg_alloc_huff_table(
        j_common_ptr cinfo
    )

    void jpeg_start_compress(
        j_compress_ptr cinfo,
        boolean write_all_tables
    )

    JDIMENSION jpeg_write_scanlines(
        j_compress_ptr cinfo,
        JSAMPARRAY scanlines,
        JDIMENSION num_lines
    )

    JDIMENSION jpeg12_write_scanlines(
        j_compress_ptr cinfo,
        J12SAMPARRAY scanlines,
        JDIMENSION num_lines
    )

    JDIMENSION jpeg16_write_scanlines(
        j_compress_ptr cinfo,
        J16SAMPARRAY scanlines,
        JDIMENSION num_lines
    )

    void jpeg_finish_compress(
        j_compress_ptr cinfo
    )

    void jpeg_calc_jpeg_dimensions(
        j_compress_ptr cinfo
    )

    JDIMENSION jpeg_write_raw_data(
        j_compress_ptr cinfo,
        JSAMPIMAGE data,
        JDIMENSION num_lines
    )

    JDIMENSION jpeg12_write_raw_data(
        j_compress_ptr cinfo,
        J12SAMPIMAGE data,
        JDIMENSION num_lines
    )

    void jpeg_write_marker(
        j_compress_ptr cinfo,
        int marker,
        const JOCTET* dataptr,
        unsigned int datalen
    )

    void jpeg_write_m_header(
        j_compress_ptr cinfo,
        int marker,
        unsigned int datalen
    )

    void jpeg_write_m_byte(
        j_compress_ptr cinfo,
        int val
    )

    void jpeg_write_tables(
        j_compress_ptr cinfo
    )

    void jpeg_write_icc_profile(
        j_compress_ptr cinfo,
        const JOCTET* icc_data_ptr,
        unsigned int icc_data_len
    )

    int jpeg_read_header(
        j_decompress_ptr cinfo,
        boolean require_image
    )

    int JPEG_SUSPENDED
    int JPEG_HEADER_OK
    int JPEG_HEADER_TABLES_ONLY

    boolean jpeg_start_decompress(
        j_decompress_ptr cinfo
    )

    JDIMENSION jpeg_read_scanlines(
        j_decompress_ptr cinfo,
        JSAMPARRAY scanlines,
        JDIMENSION max_lines
    )

    JDIMENSION jpeg12_read_scanlines(
        j_decompress_ptr cinfo,
        J12SAMPARRAY scanlines,
        JDIMENSION max_lines
    )

    JDIMENSION jpeg16_read_scanlines(
        j_decompress_ptr cinfo,
        J16SAMPARRAY scanlines,
        JDIMENSION max_lines
    )

    JDIMENSION jpeg_skip_scanlines(
        j_decompress_ptr cinfo,
        JDIMENSION num_lines
    )

    JDIMENSION jpeg12_skip_scanlines(
        j_decompress_ptr cinfo,
        JDIMENSION num_lines
    )

    void jpeg_crop_scanline(
        j_decompress_ptr cinfo,
        JDIMENSION* xoffset,
        JDIMENSION* width
    )

    void jpeg12_crop_scanline(
        j_decompress_ptr cinfo,
        JDIMENSION *xoffset,
        JDIMENSION *width
    )

    boolean jpeg_finish_decompress(
        j_decompress_ptr cinfo
    )

    JDIMENSION jpeg_read_raw_data(
        j_decompress_ptr cinfo,
        JSAMPIMAGE data,
        JDIMENSION max_lines
    )

    JDIMENSION jpeg12_read_raw_data(
        j_decompress_ptr cinfo,
        J12SAMPIMAGE data,
        JDIMENSION max_lines
    )

    boolean jpeg_has_multiple_scans(
        j_decompress_ptr cinfo
    )

    boolean jpeg_start_output(
        j_decompress_ptr cinfo,
        int scan_number
    )

    boolean jpeg_finish_output(
        j_decompress_ptr cinfo
    )

    boolean jpeg_input_complete(
        j_decompress_ptr cinfo
    )

    void jpeg_new_colormap(
        j_decompress_ptr cinfo
    )

    int jpeg_consume_input(
        j_decompress_ptr cinfo
    )

    int JPEG_REACHED_SOS
    int JPEG_REACHED_EOI
    int JPEG_ROW_COMPLETED
    int JPEG_SCAN_COMPLETED

    void jpeg_core_output_dimensions(
        j_decompress_ptr cinfo
    )

    void jpeg_calc_output_dimensions(
        j_decompress_ptr cinfo
    )

    void jpeg_save_markers(
        j_decompress_ptr cinfo,
        int marker_code,
        unsigned int length_limit
    )

    void jpeg_set_marker_processor(
        j_decompress_ptr cinfo,
        int marker_code,
        jpeg_marker_parser_method routine
    )

    jvirt_barray_ptr* jpeg_read_coefficients(
        j_decompress_ptr cinfo
    )

    void jpeg_write_coefficients(
        j_compress_ptr cinfo,
        jvirt_barray_ptr* coef_arrays
    )

    void jpeg_copy_critical_parameters(
        j_decompress_ptr srcinfo,
        j_compress_ptr dstinfo
    )

    void jpeg_abort_compress(
        j_compress_ptr cinfo
    )

    void jpeg_abort_decompress(
        j_decompress_ptr cinfo
    )

    void jpeg_abort(
        j_common_ptr cinfo
    )

    void jpeg_destroy(
        j_common_ptr cinfo
    )

    boolean jpeg_resync_to_restart(
        j_decompress_ptr cinfo,
        int desired
    )

    boolean jpeg_read_icc_profile(
        j_decompress_ptr cinfo,
        JOCTET** icc_data_ptr,
        unsigned int* icc_data_len
    )

    int JPEG_RST0
    int JPEG_EOI
    int JPEG_APP0
    int JPEG_COM

    struct jpeg_comp_master:
        long dummy

    struct jpeg_c_main_controller:
        long dummy

    struct jpeg_c_prep_controller:
        long dummy

    struct jpeg_c_coef_controller:
        long dummy

    struct jpeg_marker_writer:
        long dummy

    struct jpeg_color_converter:
        long dummy

    struct jpeg_downsampler:
        long dummy

    struct jpeg_forward_dct:
        long dummy

    struct jpeg_entropy_encoder:
        long dummy

    struct jpeg_decomp_master:
        long dummy

    struct jpeg_d_main_controller:
        long dummy

    struct jpeg_d_coef_controller:
        long dummy

    struct jpeg_d_post_controller:
        long dummy

    struct jpeg_input_controller:
        long dummy

    struct jpeg_marker_reader:
        long dummy

    struct jpeg_entropy_decoder:
        long dummy

    struct jpeg_inverse_dct:
        long dummy

    struct jpeg_upsampler:
        long dummy

    struct jpeg_color_deconverter:
        long dummy

    struct jpeg_color_quantizer:
        long dummy
