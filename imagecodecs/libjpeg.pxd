# imagecodecs/libjpeg.pxd
# cython: language_level = 3

# Cython declarations for the `libjpeg 8d` library.
# http://libjpeg.sourceforge.net/

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
        jpeg_compress_struct *cinfo,
        J_COLOR_SPACE colorspace
    )

# TODO: add missing declarations
