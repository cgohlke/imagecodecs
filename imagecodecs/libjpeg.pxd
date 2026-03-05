# imagecodecs/libjpeg.pxd

# Cython declarations for the `libjpeg 8d` library.
# http://libjpeg.sourceforge.net/

from libc.stdio cimport FILE


cdef extern from 'jpeglib.h' nogil:

    int JPEG_LIB_VERSION
    int JPEG_LIB_VERSION_MAJOR
    int JPEG_LIB_VERSION_MINOR

    ctypedef void noreturn_t
    ctypedef int boolean
    ctypedef char JOCTET
    ctypedef unsigned int JDIMENSION
    ctypedef unsigned short JSAMPLE
    ctypedef JSAMPLE* JSAMPROW
    ctypedef JSAMPROW* JSAMPARRAY

    ctypedef enum J_COLOR_SPACE:
        JCS_UNKNOWN
        JCS_GRAYSCALE
        JCS_RGB
        JCS_YCbCr
        JCS_CMYK
        JCS_YCCK

    ctypedef enum J_DITHER_MODE:
        JDITHER_NONE
        JDITHER_ORDERED
        JDITHER_FS

    ctypedef enum J_DCT_METHOD:
        JDCT_ISLOW
        JDCT_IFAST
        JDCT_FLOAT

    struct jpeg_source_mgr:
        pass

    struct jpeg_destination_mgr:
        pass

    int JMSG_LENGTH_MAX

    struct jpeg_error_mgr:
        int msg_code
        const char** jpeg_message_table
        noreturn_t error_exit(jpeg_common_struct*) nogil
        void output_message(jpeg_common_struct*) nogil
        void format_message(jpeg_common_struct* cinfo, char* buffer) nogil

    struct jpeg_common_struct:
        jpeg_error_mgr* err

    struct jpeg_component_info:
        int component_id
        int component_index
        int h_samp_factor
        int v_samp_factor

    struct jpeg_decompress_struct:
        jpeg_error_mgr* err
        void* client_data
        jpeg_source_mgr* src
        JDIMENSION image_width
        JDIMENSION image_height
        JDIMENSION output_width
        JDIMENSION output_height
        JDIMENSION output_scanline
        J_COLOR_SPACE jpeg_color_space
        J_COLOR_SPACE out_color_space
        J_DCT_METHOD dct_method
        J_DITHER_MODE dither_mode
        boolean buffered_image
        boolean raw_data_out
        boolean do_fancy_upsampling
        boolean do_block_smoothing
        boolean quantize_colors
        boolean two_pass_quantize
        unsigned int scale_num
        unsigned int scale_denom
        int num_components
        int out_color_components
        int output_components
        int rec_outbuf_height
        int desired_number_of_colors
        int actual_number_of_colors
        int data_precision
        double output_gamma

    struct jpeg_compress_struct:
        jpeg_error_mgr* err
        void* client_data
        jpeg_destination_mgr* dest
        JDIMENSION image_width
        JDIMENSION image_height
        int input_components
        J_COLOR_SPACE in_color_space
        J_COLOR_SPACE jpeg_color_space
        double input_gamma
        int data_precision
        int num_components
        int smoothing_factor
        boolean optimize_coding
        JDIMENSION next_scanline
        boolean progressive_mode
        jpeg_component_info* comp_info
        # JPEG_LIB_VERSION >= 70
        unsigned int scale_num
        unsigned int scale_denom
        JDIMENSION jpeg_width
        JDIMENSION jpeg_height
        boolean do_fancy_downsampling

    jpeg_error_mgr* jpeg_std_error(
        jpeg_error_mgr*
    )

    void jpeg_create_decompress(
        jpeg_decompress_struct*
    )

    void jpeg_destroy_decompress(
        jpeg_decompress_struct*
    )

    int jpeg_read_header(
        jpeg_decompress_struct*,
        boolean
    )

    boolean jpeg_start_decompress(
        jpeg_decompress_struct*
    )

    boolean jpeg_finish_decompress(
        jpeg_decompress_struct*
    )

    JDIMENSION jpeg_read_scanlines(
        jpeg_decompress_struct*,
        JSAMPARRAY,
        JDIMENSION
    )

    void jpeg_mem_src(
        jpeg_decompress_struct*,
        unsigned char*,
        unsigned long
    )

    void jpeg_mem_dest(
        jpeg_compress_struct*,
        unsigned char**,
        unsigned long*
    )

    void jpeg_create_compress(
        jpeg_compress_struct*
    )

    void jpeg_destroy_compress(
        jpeg_compress_struct*
    )

    void jpeg_set_defaults(
        jpeg_compress_struct*
    )

    void jpeg_set_quality(
        jpeg_compress_struct*,
        int,
        boolean
    )

    void jpeg_start_compress(
        jpeg_compress_struct*,
        boolean
    )

    void jpeg_finish_compress(
        jpeg_compress_struct*
    )

    JDIMENSION jpeg_write_scanlines(
        jpeg_compress_struct*,
        JSAMPARRAY,
        JDIMENSION
    )

    void jpeg_set_colorspace(
        jpeg_compress_struct* cinfo,
        J_COLOR_SPACE colorspace
    )

    ctypedef JSAMPARRAY* JSAMPIMAGE

    ctypedef jpeg_marker_struct* jpeg_saved_marker_ptr

    ctypedef struct jpeg_marker_struct:
        jpeg_marker_struct* next
        unsigned int marker
        unsigned int original_length
        unsigned int data_length
        JOCTET* data

    ctypedef struct jvirt_sarray_control:
        pass

    ctypedef struct jvirt_barray_control:
        pass

    ctypedef jvirt_sarray_control* jvirt_sarray_ptr

    ctypedef jvirt_barray_control* jvirt_barray_ptr

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

    void jpeg_stdio_dest(
        jpeg_compress_struct* cinfo,
        FILE* outfile
    )

    void jpeg_stdio_src(
        jpeg_decompress_struct* cinfo,
        FILE* infile
    )

    void jpeg_simple_progression(
        jpeg_compress_struct* cinfo
    )

    void jpeg_suppress_tables(
        jpeg_compress_struct* cinfo,
        boolean suppress
    )

    void jpeg_write_marker(
        jpeg_compress_struct* cinfo,
        int marker,
        const JOCTET* dataptr,
        unsigned int datalen
    )

    JDIMENSION jpeg_write_raw_data(
        jpeg_compress_struct* cinfo,
        JSAMPIMAGE data,
        JDIMENSION num_lines
    )

    void jpeg_calc_jpeg_dimensions(
        jpeg_compress_struct* cinfo
    )

    JDIMENSION jpeg_read_raw_data(
        jpeg_decompress_struct* cinfo,
        JSAMPIMAGE data,
        JDIMENSION max_lines
    )

    boolean jpeg_has_multiple_scans(
        jpeg_decompress_struct* cinfo
    )

    boolean jpeg_start_output(
        jpeg_decompress_struct* cinfo,
        int scan_number
    )

    boolean jpeg_finish_output(
        jpeg_decompress_struct* cinfo
    )

    boolean jpeg_input_complete(
        jpeg_decompress_struct* cinfo
    )

    void jpeg_new_colormap(
        jpeg_decompress_struct* cinfo
    )

    int jpeg_consume_input(
        jpeg_decompress_struct* cinfo
    )

    void jpeg_calc_output_dimensions(
        jpeg_decompress_struct* cinfo
    )

    void jpeg_save_markers(
        jpeg_decompress_struct* cinfo,
        int marker_code,
        unsigned int length_limit
    )

    jvirt_barray_ptr* jpeg_read_coefficients(
        jpeg_decompress_struct* cinfo
    )

    void jpeg_write_coefficients(
        jpeg_compress_struct* cinfo,
        jvirt_barray_ptr* coef_arrays
    )

    void jpeg_copy_critical_parameters(
        jpeg_decompress_struct* srcinfo,
        jpeg_compress_struct* dstinfo
    )

    void jpeg_abort_compress(
        jpeg_compress_struct* cinfo
    )

    void jpeg_abort_decompress(
        jpeg_decompress_struct* cinfo
    )

    void jpeg_abort(
        jpeg_common_struct* cinfo
    )

    void jpeg_destroy(
        jpeg_common_struct* cinfo
    )
