# imagecodecs/openjpeg.pxd
# cython: language_level = 3

# Cython declarations for the `OpenJPEG 2.3.1` library.
# https://github.com/uclouvain/openjpeg

from libc.stdint cimport (
    int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t
)

cdef extern from 'openjpeg.h':

    int OPJ_FALSE = 0
    int OPJ_TRUE = 1

    ctypedef int OPJ_BOOL
    ctypedef char OPJ_CHAR
    ctypedef float OPJ_FLOAT32
    ctypedef double OPJ_FLOAT64
    ctypedef unsigned char OPJ_BYTE
    ctypedef int8_t OPJ_INT8
    ctypedef uint8_t OPJ_UINT8
    ctypedef int16_t OPJ_INT16
    ctypedef uint16_t OPJ_UINT16
    ctypedef int32_t OPJ_INT32
    ctypedef uint32_t OPJ_UINT32
    ctypedef int64_t OPJ_INT64
    ctypedef uint64_t OPJ_UINT64
    ctypedef int64_t OPJ_OFF_T
    ctypedef size_t OPJ_SIZE_T

    ctypedef enum OPJ_CODEC_FORMAT:
        OPJ_CODEC_UNKNOWN
        OPJ_CODEC_J2K
        OPJ_CODEC_JPT
        OPJ_CODEC_JP2
        OPJ_CODEC_JPP
        OPJ_CODEC_JPX

    ctypedef enum OPJ_COLOR_SPACE:
        OPJ_CLRSPC_UNKNOWN
        OPJ_CLRSPC_UNSPECIFIED
        OPJ_CLRSPC_SRGB
        OPJ_CLRSPC_GRAY
        OPJ_CLRSPC_SYCC
        OPJ_CLRSPC_EYCC
        OPJ_CLRSPC_CMYK

    ctypedef struct opj_codec_t:
        pass

    ctypedef struct opj_stream_t:
        pass

    ctypedef struct opj_image_cmptparm_t:
        OPJ_UINT32 dx
        OPJ_UINT32 dy
        OPJ_UINT32 w
        OPJ_UINT32 h
        OPJ_UINT32 x0
        OPJ_UINT32 y0
        OPJ_UINT32 prec
        OPJ_UINT32 bpp
        OPJ_UINT32 sgnd

    ctypedef struct opj_cparameters_t:
        OPJ_BOOL tile_size_on
        int cp_tx0
        int cp_ty0
        int cp_tdx
        int cp_tdy
        int cp_disto_alloc
        int cp_fixed_alloc
        int cp_fixed_quality
        int* cp_matrice
        char* cp_comment
        int csty
        # OPJ_PROG_ORDER prog_order
        # opj_poc_t POC[32]
        OPJ_UINT32 numpocs
        int tcp_numlayers
        float tcp_rates[100]
        float tcp_distoratio[100]
        int numresolution
        int cblockw_init
        int cblockh_init
        int mode
        int irreversible

    ctypedef struct opj_dparameters_t:
        OPJ_UINT32 cp_reduce
        OPJ_UINT32 cp_layer
        # char infile[OPJ_PATH_LEN]
        # char outfile[OPJ_PATH_LEN]
        int decod_format
        int cod_format
        OPJ_UINT32 DA_x0
        OPJ_UINT32 DA_x1
        OPJ_UINT32 DA_y0
        OPJ_UINT32 DA_y1
        OPJ_BOOL m_verbose
        OPJ_UINT32 tile_index
        OPJ_UINT32 nb_tile_to_decode
        OPJ_BOOL jpwl_correct
        int jpwl_exp_comps
        int jpwl_max_tiles
        unsigned int flags

    ctypedef struct opj_image_comp_t:
        OPJ_UINT32 dx
        OPJ_UINT32 dy
        OPJ_UINT32 w
        OPJ_UINT32 h
        OPJ_UINT32 x0
        OPJ_UINT32 y0
        OPJ_UINT32 prec
        OPJ_UINT32 bpp
        OPJ_UINT32 sgnd
        OPJ_UINT32 resno_decoded
        OPJ_UINT32 factor
        OPJ_INT32* data
        OPJ_UINT16 alpha

    ctypedef struct opj_image_t:
        OPJ_UINT32 x0
        OPJ_UINT32 y0
        OPJ_UINT32 x1
        OPJ_UINT32 y1
        OPJ_UINT32 numcomps
        OPJ_COLOR_SPACE color_space
        opj_image_comp_t* comps
        OPJ_BYTE* icc_profile_buf
        OPJ_UINT32 icc_profile_len

    ctypedef OPJ_SIZE_T(*opj_stream_read_fn)(
        void*,
        OPJ_SIZE_T,
        void*
    )

    ctypedef OPJ_SIZE_T(*opj_stream_write_fn)(
        void*,
        OPJ_SIZE_T,
        void*
    )

    ctypedef OPJ_OFF_T(*opj_stream_skip_fn)(
        OPJ_OFF_T,
        void*
    )

    ctypedef OPJ_BOOL(*opj_stream_seek_fn)(
        OPJ_OFF_T,
        void*
    )

    ctypedef void(*opj_stream_free_user_data_fn)(
        void*
    )

    ctypedef void(*opj_msg_callback)(
        const char* msg,
        void* client_data
    )

    opj_stream_t* opj_stream_default_create(
        OPJ_BOOL p_is_input
    ) nogil

    opj_codec_t* opj_create_compress(
        OPJ_CODEC_FORMAT format
    ) nogil

    opj_codec_t* opj_create_decompress(
        OPJ_CODEC_FORMAT format
    ) nogil

    void opj_destroy_codec(
        opj_codec_t* p_codec
    ) nogil

    void opj_set_default_encoder_parameters(
        opj_cparameters_t*
    ) nogil

    void opj_set_default_decoder_parameters(
        opj_dparameters_t* params
    ) nogil

    void opj_image_destroy(
        opj_image_t* image
    ) nogil

    void* opj_image_data_alloc(
        OPJ_SIZE_T size
    ) nogil

    void opj_image_data_free(
        void* ptr
    ) nogil

    void opj_stream_destroy(
        opj_stream_t* p_stream
    ) nogil

    void color_sycc_to_rgb(
        opj_image_t* img
    ) nogil

    void color_apply_icc_profile(
        opj_image_t* image
    ) nogil

    void color_cielab_to_rgb(
        opj_image_t* image
    ) nogil

    void color_cmyk_to_rgb(
        opj_image_t* image
    ) nogil

    void color_esycc_to_rgb(
        opj_image_t* image
    ) nogil

    const char* opj_version() nogil

    OPJ_BOOL opj_encode(
        opj_codec_t* p_codec,
        opj_stream_t* p_stream
    ) nogil

    opj_image_t* opj_image_tile_create(
        OPJ_UINT32 numcmpts,
        opj_image_cmptparm_t* cmptparms,
        OPJ_COLOR_SPACE clrspc
    ) nogil

    OPJ_BOOL opj_setup_encoder(
        opj_codec_t* p_codec,
        opj_cparameters_t* parameters,
        opj_image_t* image
    ) nogil

    OPJ_BOOL opj_start_compress(
        opj_codec_t* p_codec,
        opj_image_t* p_image,
        opj_stream_t* p_stream
    ) nogil

    OPJ_BOOL opj_end_compress(
        opj_codec_t* p_codec,
        opj_stream_t* p_stream
    ) nogil

    OPJ_BOOL opj_end_decompress(
        opj_codec_t* p_codec,
        opj_stream_t* p_stream
    ) nogil

    OPJ_BOOL opj_setup_decoder(
        opj_codec_t* p_codec,
        opj_dparameters_t* params
    ) nogil

    OPJ_BOOL opj_codec_set_threads(
        opj_codec_t* p_codec,
        int num_threads
    ) nogil

    OPJ_BOOL opj_read_header(
        opj_stream_t* p_stream,
        opj_codec_t* p_codec,
        opj_image_t** p_image
    ) nogil

    OPJ_BOOL opj_set_decode_area(
        opj_codec_t* p_codec,
        opj_image_t* p_image,
        OPJ_INT32 p_start_x,
        OPJ_INT32 p_start_y,
        OPJ_INT32 p_end_x,
        OPJ_INT32 p_end_y
    ) nogil

    OPJ_BOOL opj_set_info_handler(
        opj_codec_t* p_codec,
        opj_msg_callback p_callback,
        void* p_user_data
    ) nogil

    OPJ_BOOL opj_set_warning_handler(
        opj_codec_t* p_codec,
        opj_msg_callback p_callback,
        void* p_user_data
    ) nogil

    OPJ_BOOL opj_set_error_handler(
        opj_codec_t* p_codec,
        opj_msg_callback p_callback,
        void* p_user_data
    ) nogil

    OPJ_BOOL opj_decode(
        opj_codec_t* p_decompressor,
        opj_stream_t* p_stream,
        opj_image_t* p_image
    ) nogil

    opj_image_t* opj_image_create(
        OPJ_UINT32 numcmpts,
        opj_image_cmptparm_t* cmptparms,
        OPJ_COLOR_SPACE clrspc
    ) nogil

    void opj_stream_set_read_function(
        opj_stream_t* p_stream,
        opj_stream_read_fn p_func
    ) nogil

    void opj_stream_set_write_function(
        opj_stream_t* p_stream,
        opj_stream_write_fn p_func
    ) nogil

    void opj_stream_set_seek_function(
        opj_stream_t* p_stream,
        opj_stream_seek_fn p_func
    ) nogil

    void opj_stream_set_skip_function(
        opj_stream_t* p_stream,
        opj_stream_skip_fn p_func
    ) nogil

    void opj_stream_set_user_data(
        opj_stream_t* p_stream,
        void* p_data,
        opj_stream_free_user_data_fn p_func
    ) nogil

    void opj_stream_set_user_data_length(
        opj_stream_t* p_stream,
        OPJ_UINT64 data_length
    ) nogil

    OPJ_BOOL opj_write_tile(
        opj_codec_t* p_codec,
        OPJ_UINT32 p_tile_index,
        OPJ_BYTE* p_data,
        OPJ_UINT32 p_data_size,
        opj_stream_t* p_stream
    ) nogil


cdef extern from 'color.h':
    # this header is not part of the public OpenJPEG interface

    void color_sycc_to_rgb(
        opj_image_t* img
    ) nogil

    void color_apply_icc_profile(
        opj_image_t* image
    ) nogil

    void color_cielab_to_rgb(
        opj_image_t* image
    ) nogil

    void color_cmyk_to_rgb(
        opj_image_t* image
    ) nogil

    void color_esycc_to_rgb(
        opj_image_t* image
    ) nogil
