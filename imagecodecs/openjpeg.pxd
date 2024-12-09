# imagecodecs/openjpeg.pxd
# cython: language_level = 3

# Cython declarations for the `OpenJPEG 2.5.3` library.
# https://github.com/uclouvain/openjpeg

from libc.stdint cimport (
    int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t
)
from libc.stdio cimport FILE

cdef extern from 'openjpeg.h' nogil:

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
    ctypedef OPJ_INT32 OPJ_UINT32_SEMANTICALLY_BUT_INT32

    int OPJ_PATH_LEN
    int OPJ_J2K_MAXRLVLS
    int OPJ_J2K_MAXBANDS

    int OPJ_J2K_DEFAULT_NB_SEGS
    int OPJ_J2K_STREAM_CHUNK_SIZE
    int OPJ_J2K_DEFAULT_HEADER_SIZE
    int OPJ_J2K_MCC_DEFAULT_NB_RECORDS
    int OPJ_J2K_MCT_DEFAULT_NB_RECORDS

    int JPWL_MAX_NO_TILESPECS
    int JPWL_MAX_NO_PACKSPECS
    int JPWL_MAX_NO_MARKERS
    int JPWL_PRIVATEINDEX_NAMEj
    int JPWL_EXPECTED_COMPONENTS
    int JPWL_MAXIMUM_TILES
    int JPWL_MAXIMUM_HAMMING
    int JPWL_MAXIMUM_EPB_ROOM

    int OPJ_IMG_INFO
    int OPJ_J2K_MH_INFO
    int OPJ_J2K_TH_INFO
    int OPJ_J2K_TCH_INFO
    int OPJ_J2K_MH_IND
    int OPJ_J2K_TH_IND
    int OPJ_J2K_CSTR_IND
    int OPJ_JP2_INFO
    int OPJ_JP2_IND

    int OPJ_PROFILE_NONE
    int OPJ_PROFILE_0
    int OPJ_PROFILE_1
    int OPJ_PROFILE_PART2
    int OPJ_PROFILE_CINEMA_2K
    int OPJ_PROFILE_CINEMA_4K
    int OPJ_PROFILE_CINEMA_S2K
    int OPJ_PROFILE_CINEMA_S4K
    int OPJ_PROFILE_CINEMA_LTS
    int OPJ_PROFILE_BC_SINGLE
    int OPJ_PROFILE_BC_MULTI
    int OPJ_PROFILE_BC_MULTI_R
    int OPJ_PROFILE_IMF_2K
    int OPJ_PROFILE_IMF_4K
    int OPJ_PROFILE_IMF_8K
    int OPJ_PROFILE_IMF_2K_R
    int OPJ_PROFILE_IMF_4K_R
    int OPJ_PROFILE_IMF_8K_R

    int OPJ_EXTENSION_NONE
    int OPJ_EXTENSION_MCT

    int OPJ_IS_CINEMA(v)
    int OPJ_IS_STORAGE(v)
    int OPJ_IS_BROADCAST(v)
    int OPJ_IS_IMF(v)
    int OPJ_IS_PART2(v)

    int OPJ_GET_IMF_PROFILE(v)
    int OPJ_GET_IMF_MAINLEVEL(v)
    int OPJ_GET_IMF_SUBLEVEL(v)

    int OPJ_IMF_MAINLEVEL_MAX

    int OPJ_IMF_MAINLEVEL_1_MSAMPLESEC
    int OPJ_IMF_MAINLEVEL_2_MSAMPLESEC
    int OPJ_IMF_MAINLEVEL_3_MSAMPLESEC
    int OPJ_IMF_MAINLEVEL_4_MSAMPLESEC
    int OPJ_IMF_MAINLEVEL_5_MSAMPLESEC
    int OPJ_IMF_MAINLEVEL_6_MSAMPLESEC
    int OPJ_IMF_MAINLEVEL_7_MSAMPLESEC
    int OPJ_IMF_MAINLEVEL_8_MSAMPLESEC
    int OPJ_IMF_MAINLEVEL_9_MSAMPLESEC
    int OPJ_IMF_MAINLEVEL_10_MSAMPLESEC
    int OPJ_IMF_MAINLEVEL_11_MSAMPLESEC

    int OPJ_IMF_SUBLEVEL_1_MBITSSEC
    int OPJ_IMF_SUBLEVEL_2_MBITSSEC
    int OPJ_IMF_SUBLEVEL_3_MBITSSEC
    int OPJ_IMF_SUBLEVEL_4_MBITSSEC
    int OPJ_IMF_SUBLEVEL_5_MBITSSEC
    int OPJ_IMF_SUBLEVEL_6_MBITSSEC
    int OPJ_IMF_SUBLEVEL_7_MBITSSEC
    int OPJ_IMF_SUBLEVEL_8_MBITSSEC
    int OPJ_IMF_SUBLEVEL_9_MBITSSEC

    int OPJ_CINEMA_24_CS
    int OPJ_CINEMA_48_CS
    int OPJ_CINEMA_24_COMP
    int OPJ_CINEMA_48_COMP

    ctypedef enum OPJ_RSIZ_CAPABILITIES:
        OPJ_STD_RSIZ
        OPJ_CINEMA2K
        OPJ_CINEMA4K
        OPJ_MCT

    ctypedef enum OPJ_CINEMA_MODE:
        OPJ_OFF
        OPJ_CINEMA2K_24
        OPJ_CINEMA2K_48
        OPJ_CINEMA4K_24

    ctypedef enum OPJ_PROG_ORDER:
        OPJ_PROG_UNKNOWN
        OPJ_LRCP
        OPJ_RLCP
        OPJ_RPCL
        OPJ_PCRL
        OPJ_CPRL

    ctypedef enum OPJ_COLOR_SPACE:
        OPJ_CLRSPC_UNKNOWN
        OPJ_CLRSPC_UNSPECIFIED
        OPJ_CLRSPC_SRGB
        OPJ_CLRSPC_GRAY
        OPJ_CLRSPC_SYCC
        OPJ_CLRSPC_EYCC
        OPJ_CLRSPC_CMYK

    ctypedef enum OPJ_CODEC_FORMAT:
        OPJ_CODEC_UNKNOWN
        OPJ_CODEC_J2K
        OPJ_CODEC_JPT
        OPJ_CODEC_JP2
        OPJ_CODEC_JPP
        OPJ_CODEC_JPX

    ctypedef void (*opj_msg_callback)(
        const char* msg,
        void* client_data
    ) nogil

    ctypedef struct opj_poc_t:
        OPJ_UINT32 resno0
        OPJ_UINT32 compno0
        OPJ_UINT32 layno1
        OPJ_UINT32 resno1
        OPJ_UINT32 compno1
        OPJ_UINT32 layno0
        OPJ_UINT32 precno0
        OPJ_UINT32 precno1
        OPJ_PROG_ORDER prg1
        OPJ_PROG_ORDER prg
        OPJ_CHAR[5] progorder
        OPJ_UINT32 tile
        OPJ_UINT32_SEMANTICALLY_BUT_INT32 tx0
        OPJ_UINT32_SEMANTICALLY_BUT_INT32 tx1
        OPJ_UINT32_SEMANTICALLY_BUT_INT32 ty0
        OPJ_UINT32_SEMANTICALLY_BUT_INT32 ty1
        OPJ_UINT32 layS
        OPJ_UINT32 resS
        OPJ_UINT32 compS
        OPJ_UINT32 prcS
        OPJ_UINT32 layE
        OPJ_UINT32 resE
        OPJ_UINT32 compE
        OPJ_UINT32 prcE
        OPJ_UINT32 txS
        OPJ_UINT32 txE
        OPJ_UINT32 tyS
        OPJ_UINT32 tyE
        OPJ_UINT32 dx
        OPJ_UINT32 dy
        OPJ_UINT32 lay_t
        OPJ_UINT32 res_t
        OPJ_UINT32 comp_t
        OPJ_UINT32 prc_t
        OPJ_UINT32 tx0_t
        OPJ_UINT32 ty0_t

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
        OPJ_PROG_ORDER prog_order
        opj_poc_t[32] POC
        OPJ_UINT32 numpocs
        int tcp_numlayers
        float[100] tcp_rates
        float[100] tcp_distoratio
        int numresolution
        int cblockw_init
        int cblockh_init
        int mode
        int irreversible
        int roi_compno
        int roi_shift
        int res_spec
        int[33] prcw_init  # OPJ_J2K_MAXRLVLS
        int[33] prch_init  # OPJ_J2K_MAXRLVLS
        char[4096] infile  # OPJ_PATH_LEN
        char[4096] outfile  # OPJ_PATH_LEN
        int index_on
        char[4096] index  # OPJ_PATH_LEN
        int image_offset_x0
        int image_offset_y0
        int subsampling_dx
        int subsampling_dy
        int decod_format
        int cod_format
        OPJ_BOOL jpwl_epc_on
        int jpwl_hprot_MH
        int[16] jpwl_hprot_TPH_tileno  # JPWL_MAX_NO_TILESPECS
        int[16] jpwl_hprot_TPH  # JPWL_MAX_NO_TILESPECS
        int[16] jpwl_pprot_tileno  # JPWL_MAX_NO_PACKSPECS
        int[16] jpwl_pprot_packno  # JPWL_MAX_NO_PACKSPECS
        int[16] jpwl_pprot  # JPWL_MAX_NO_PACKSPECS
        int jpwl_sens_size
        int jpwl_sens_addr
        int jpwl_sens_range
        int jpwl_sens_MH
        int[16] jpwl_sens_TPH_tileno  # JPWL_MAX_NO_TILESPECS
        int[16] jpwl_sens_TPH  # JPWL_MAX_NO_TILESPECS
        OPJ_CINEMA_MODE cp_cinema
        int max_comp_size
        OPJ_RSIZ_CAPABILITIES cp_rsiz
        char tp_on
        char tp_flag
        char tcp_mct
        OPJ_BOOL jpip_on
        void* mct_data
        int max_cs_size
        OPJ_UINT16 rsiz

    int OPJ_DPARAMETERS_IGNORE_PCLR_CMAP_CDEF_FLAG
    int OPJ_DPARAMETERS_DUMP_FLAG

    ctypedef struct opj_dparameters_t:
        OPJ_UINT32 cp_reduce
        OPJ_UINT32 cp_layer
        char[4096] infile  # OPJ_PATH_LEN
        char[4096] outfile  # OPJ_PATH_LEN
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

    ctypedef void* opj_codec_t

    int OPJ_STREAM_READ
    int OPJ_STREAM_WRITE

    ctypedef OPJ_SIZE_T (*opj_stream_read_fn)(
        void* p_buffer,
        OPJ_SIZE_T p_nb_bytes,
        void* p_user_data
    ) nogil

    ctypedef OPJ_SIZE_T (*opj_stream_write_fn)(
        void* p_buffer,
        OPJ_SIZE_T p_nb_bytes,
        void* p_user_data
    ) nogil

    ctypedef OPJ_OFF_T (*opj_stream_skip_fn)(
        OPJ_OFF_T p_nb_bytes,
        void* p_user_data
    ) nogil

    ctypedef OPJ_BOOL (*opj_stream_seek_fn)(
        OPJ_OFF_T p_nb_bytes,
        void* p_user_data
    ) nogil

    ctypedef void (*opj_stream_free_user_data_fn)(
        void* p_user_data
    ) nogil

    ctypedef void* opj_stream_t

    ctypedef struct opj_image_comp_t:
        OPJ_UINT32 dx
        OPJ_UINT32 dy
        OPJ_UINT32 w
        OPJ_UINT32 h
        OPJ_UINT32 x0
        OPJ_UINT32 y0
        OPJ_UINT32 prec
        OPJ_UINT32 bpp  # obsolete: use prec instead
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

    ctypedef struct opj_packet_info_t:
        OPJ_OFF_T start_pos
        OPJ_OFF_T end_ph_pos
        OPJ_OFF_T end_pos
        double disto

    ctypedef struct opj_marker_info_t:
        unsigned short int c_type 'type'
        OPJ_OFF_T pos
        int len

    ctypedef struct opj_tp_info_t:
        int tp_start_pos
        int tp_end_header
        int tp_end_pos
        int tp_start_pack
        int tp_numpacks

    ctypedef struct opj_tile_info_t:
        double thresh
        int tileno
        int start_pos
        int end_header
        int end_pos
        int[33] pw
        int[33] ph
        int[33] pdx
        int[33] pdy
        opj_packet_info_t* packet
        int numpix
        double distotile
        int marknum
        opj_marker_info_t* marker
        int maxmarknum
        int num_tps
        opj_tp_info_t* tp

    ctypedef struct opj_codestream_info_t:
        double D_max
        int packno
        int index_write
        int image_w
        int image_h
        OPJ_PROG_ORDER prog
        int tile_x
        int tile_y
        int tile_Ox
        int tile_Oy
        int tw
        int th
        int numcomps
        int numlayers
        int* numdecompos
        int marknum
        opj_marker_info_t* marker
        int maxmarknum
        int main_head_start
        int main_head_end
        int codestream_size
        opj_tile_info_t* tile

    ctypedef struct opj_tccp_info_t:
        OPJ_UINT32 compno
        OPJ_UINT32 csty
        OPJ_UINT32 numresolutions
        OPJ_UINT32 cblkw
        OPJ_UINT32 cblkh
        OPJ_UINT32 cblksty
        OPJ_UINT32 qmfbid
        OPJ_UINT32 qntsty
        OPJ_UINT32[97] stepsizes_mant  # OPJ_J2K_MAXBANDS
        OPJ_UINT32[97] stepsizes_expn  # OPJ_J2K_MAXBANDS
        OPJ_UINT32 numgbits
        OPJ_INT32 roishift
        OPJ_UINT32[33] prcw  # OPJ_J2K_MAXRLVLS
        OPJ_UINT32[33] prch  # OPJ_J2K_MAXRLVLS

    ctypedef struct opj_tile_info_v2_t:
        int tileno
        OPJ_UINT32 csty
        OPJ_PROG_ORDER prg
        OPJ_UINT32 numlayers
        OPJ_UINT32 mct
        opj_tccp_info_t* tccp_info

    ctypedef struct opj_codestream_info_v2_t:
        OPJ_UINT32 tx0
        OPJ_UINT32 ty0
        OPJ_UINT32 tdx
        OPJ_UINT32 tdy
        OPJ_UINT32 tw
        OPJ_UINT32 th
        OPJ_UINT32 nbcomps
        opj_tile_info_v2_t m_default_tile_info
        opj_tile_info_v2_t* tile_info

    ctypedef struct opj_tp_index_t:
        OPJ_OFF_T start_pos
        OPJ_OFF_T end_header
        OPJ_OFF_T end_pos

    ctypedef struct opj_tile_index_t:
        OPJ_UINT32 tileno
        OPJ_UINT32 nb_tps
        OPJ_UINT32 current_nb_tps
        OPJ_UINT32 current_tpsno
        opj_tp_index_t* tp_index
        OPJ_UINT32 marknum
        opj_marker_info_t* marker
        OPJ_UINT32 maxmarknum
        OPJ_UINT32 nb_packet
        opj_packet_info_t* packet_index

    ctypedef struct opj_codestream_index_t:
        OPJ_OFF_T main_head_start
        OPJ_OFF_T main_head_end
        OPJ_UINT64 codestream_size
        OPJ_UINT32 marknum
        opj_marker_info_t* marker
        OPJ_UINT32 maxmarknum
        OPJ_UINT32 nb_of_tiles
        opj_tile_index_t* tile_index

    ctypedef struct opj_jp2_metadata_t:
        OPJ_INT32 not_used

    ctypedef struct opj_jp2_index_t:
        OPJ_INT32 not_used

    const char* opj_version()

    opj_image_t* opj_image_create(
        OPJ_UINT32 numcmpts,
        opj_image_cmptparm_t* cmptparms,
        OPJ_COLOR_SPACE clrspc
    )

    void opj_image_destroy(
        opj_image_t* image
    )

    opj_image_t* opj_image_tile_create(
        OPJ_UINT32 numcmpts,
        opj_image_cmptparm_t* cmptparms,
        OPJ_COLOR_SPACE clrspc
    )

    void* opj_image_data_alloc(
        OPJ_SIZE_T size
    )

    void opj_image_data_free(
        void* ptr
    )

    opj_stream_t* opj_stream_default_create(
        OPJ_BOOL p_is_input
    )

    opj_stream_t* opj_stream_create(
        OPJ_SIZE_T p_buffer_size,
        OPJ_BOOL p_is_input
    )

    void opj_stream_destroy(
        opj_stream_t* p_stream
    )

    void opj_stream_set_read_function(
        opj_stream_t* p_stream,
        opj_stream_read_fn p_function
    )

    void opj_stream_set_write_function(
        opj_stream_t* p_stream,
        opj_stream_write_fn p_function
    )

    void opj_stream_set_skip_function(
        opj_stream_t* p_stream,
        opj_stream_skip_fn p_function
    )

    void opj_stream_set_seek_function(
        opj_stream_t* p_stream,
        opj_stream_seek_fn p_function
    )

    void opj_stream_set_user_data(
        opj_stream_t* p_stream,
        void* p_data,
        opj_stream_free_user_data_fn p_function
    )

    void opj_stream_set_user_data_length(
        opj_stream_t* p_stream,
        OPJ_UINT64 data_length
    )

    opj_stream_t* opj_stream_create_default_file_stream(
        const char* fname,
        OPJ_BOOL p_is_read_stream
    )

    opj_stream_t* opj_stream_create_file_stream(
        const char* fname,
        OPJ_SIZE_T p_buffer_size,
        OPJ_BOOL p_is_read_stream
    )

    OPJ_BOOL opj_set_info_handler(
        opj_codec_t* p_codec,
        opj_msg_callback p_callback,
        void* p_user_data
    )

    OPJ_BOOL opj_set_warning_handler(
        opj_codec_t* p_codec,
        opj_msg_callback p_callback,
        void* p_user_data
    )

    OPJ_BOOL opj_set_error_handler(
        opj_codec_t* p_codec,
        opj_msg_callback p_callback,
        void* p_user_data
    )

    opj_codec_t* opj_create_decompress(
        OPJ_CODEC_FORMAT format
    )

    void opj_destroy_codec(
        opj_codec_t* p_codec
    )

    OPJ_BOOL opj_end_decompress(
        opj_codec_t* p_codec,
        opj_stream_t* p_stream
    )

    void opj_set_default_decoder_parameters(
        opj_dparameters_t* parameters
    )

    OPJ_BOOL opj_setup_decoder(
        opj_codec_t* p_codec,
        opj_dparameters_t* parameters
    )

    OPJ_BOOL opj_decoder_set_strict_mode(
        opj_codec_t *p_codec,
        OPJ_BOOL strict
    )

    OPJ_BOOL opj_codec_set_threads(
        opj_codec_t* p_codec,
        int num_threads
    )

    OPJ_BOOL opj_read_header(
        opj_stream_t* p_stream,
        opj_codec_t* p_codec,
        opj_image_t** p_image
    )

    OPJ_BOOL opj_set_decoded_components(
        opj_codec_t* p_codec,
        OPJ_UINT32 numcomps,
        const OPJ_UINT32* comps_indices,
        OPJ_BOOL apply_color_transforms
    )

    OPJ_BOOL opj_set_decode_area(
        opj_codec_t* p_codec,
        opj_image_t* p_image,
        OPJ_INT32 p_start_x,
        OPJ_INT32 p_start_y,
        OPJ_INT32 p_end_x,
        OPJ_INT32 p_end_y
    )

    OPJ_BOOL opj_decode(
        opj_codec_t* p_decompressor,
        opj_stream_t* p_stream,
        opj_image_t* p_image
    )

    OPJ_BOOL opj_get_decoded_tile(
        opj_codec_t* p_codec,
        opj_stream_t* p_stream,
        opj_image_t* p_image,
        OPJ_UINT32 tile_index
    )

    OPJ_BOOL opj_set_decoded_resolution_factor(
        opj_codec_t* p_codec,
        OPJ_UINT32 res_factor
    )

    OPJ_BOOL opj_write_tile(
        opj_codec_t* p_codec,
        OPJ_UINT32 p_tile_index,
        OPJ_BYTE* p_data,
        OPJ_UINT32 p_data_size,
        opj_stream_t* p_stream
    )

    OPJ_BOOL opj_read_tile_header(
        opj_codec_t* p_codec,
        opj_stream_t* p_stream,
        OPJ_UINT32* p_tile_index,
        OPJ_UINT32* p_data_size,
        OPJ_INT32* p_tile_x0,
        OPJ_INT32* p_tile_y0,
        OPJ_INT32* p_tile_x1,
        OPJ_INT32* p_tile_y1,
        OPJ_UINT32* p_nb_comps,
        OPJ_BOOL* p_should_go_on
    )

    OPJ_BOOL opj_decode_tile_data(
        opj_codec_t* p_codec,
        OPJ_UINT32 p_tile_index,
        OPJ_BYTE* p_data,
        OPJ_UINT32 p_data_size,
        opj_stream_t* p_stream
    )

    opj_codec_t* opj_create_compress(
        OPJ_CODEC_FORMAT format
    )

    void opj_set_default_encoder_parameters(
        opj_cparameters_t* parameters
    )

    OPJ_BOOL opj_setup_encoder(
        opj_codec_t* p_codec,
        opj_cparameters_t* parameters,
        opj_image_t* image
    )

    OPJ_BOOL opj_encoder_set_extra_options(
        opj_codec_t* p_codec,
        const char** p_options
    )

    OPJ_BOOL opj_start_compress(
        opj_codec_t* p_codec,
        opj_image_t* p_image,
        opj_stream_t* p_stream
    )

    OPJ_BOOL opj_end_compress(
        opj_codec_t* p_codec,
        opj_stream_t* p_stream
    )

    OPJ_BOOL opj_encode(
        opj_codec_t* p_codec,
        opj_stream_t* p_stream
    )

    void opj_destroy_cstr_info(
        opj_codestream_info_v2_t** cstr_info
    )

    void opj_dump_codec(
        opj_codec_t* p_codec,
        OPJ_INT32 info_flag,
        FILE* output_stream
    )

    opj_codestream_info_v2_t* opj_get_cstr_info(
        opj_codec_t* p_codec
    )

    opj_codestream_index_t* opj_get_cstr_index(
        opj_codec_t* p_codec
    )

    void opj_destroy_cstr_index(
        opj_codestream_index_t** p_cstr_index
    )

    opj_jp2_metadata_t* opj_get_jp2_metadata(
        opj_codec_t* p_codec
    )

    opj_jp2_index_t* opj_get_jp2_index(
        opj_codec_t* p_codec
    )

    OPJ_BOOL opj_set_MCT(
        opj_cparameters_t* parameters,
        OPJ_FLOAT32* pEncodingMatrix,
        OPJ_INT32* p_dc_shift,
        OPJ_UINT32 pNbComp
    )

    OPJ_BOOL opj_has_thread_support()

    int opj_get_num_cpus()


cdef extern from 'color.h' nogil:
    # this header is not part of the public OpenJPEG interface

    void color_sycc_to_rgb(
        opj_image_t* img
    )

    void color_apply_icc_profile(
        opj_image_t* image
    )

    void color_cielab_to_rgb(
        opj_image_t* image
    )

    void color_cmyk_to_rgb(
        opj_image_t* image
    )

    void color_esycc_to_rgb(
        opj_image_t* image
    )
