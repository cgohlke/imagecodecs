# imagecodecs/libultrahdr.pxd
# cython: language_level = 3

# Cython declarations for the `libultrahdr 1.4.0` library.
# https://github.com/google/libultrahdr

cdef extern from 'ultrahdr_api.h' nogil:

    int UHDR_LIB_VER_MAJOR
    int UHDR_LIB_VER_MINOR
    int UHDR_LIB_VER_PATCH
    int UHDR_LIB_VERSION

    char* UHDR_LIB_VERSION_STR

    ctypedef enum uhdr_img_fmt_t:
        UHDR_IMG_FMT_UNSPECIFIED
        UHDR_IMG_FMT_24bppYCbCrP010
        UHDR_IMG_FMT_12bppYCbCr420
        UHDR_IMG_FMT_8bppYCbCr400
        UHDR_IMG_FMT_32bppRGBA8888
        UHDR_IMG_FMT_64bppRGBAHalfFloat
        UHDR_IMG_FMT_32bppRGBA1010102
        UHDR_IMG_FMT_24bppYCbCr444
        UHDR_IMG_FMT_16bppYCbCr422
        UHDR_IMG_FMT_16bppYCbCr440
        UHDR_IMG_FMT_12bppYCbCr411
        UHDR_IMG_FMT_10bppYCbCr410
        UHDR_IMG_FMT_24bppRGB888
        UHDR_IMG_FMT_30bppYCbCr444

    ctypedef enum uhdr_color_gamut_t:
        UHDR_CG_UNSPECIFIED
        UHDR_CG_BT_709
        UHDR_CG_DISPLAY_P3
        UHDR_CG_BT_2100

    ctypedef enum uhdr_color_transfer_t:
        UHDR_CT_UNSPECIFIED
        UHDR_CT_LINEAR
        UHDR_CT_HLG
        UHDR_CT_PQ
        UHDR_CT_SRGB

    ctypedef enum uhdr_color_range_t:
        UHDR_CR_UNSPECIFIED
        UHDR_CR_LIMITED_RANGE
        UHDR_CR_FULL_RANGE

    ctypedef enum uhdr_codec_t:
        UHDR_CODEC_JPG
        UHDR_CODEC_HEIF
        UHDR_CODEC_AVIF

    ctypedef enum uhdr_img_label_t:
        UHDR_HDR_IMG
        UHDR_SDR_IMG
        UHDR_BASE_IMG
        UHDR_GAIN_MAP_IMG

    ctypedef enum uhdr_enc_preset_t:
        UHDR_USAGE_REALTIME
        UHDR_USAGE_BEST_QUALITY

    ctypedef enum uhdr_codec_err_t:
        UHDR_CODEC_OK
        UHDR_CODEC_ERROR
        UHDR_CODEC_UNKNOWN_ERROR
        UHDR_CODEC_INVALID_PARAM
        UHDR_CODEC_MEM_ERROR
        UHDR_CODEC_INVALID_OPERATION
        UHDR_CODEC_UNSUPPORTED_FEATURE
        UHDR_CODEC_LIST_END

    ctypedef enum uhdr_mirror_direction_t:
        UHDR_MIRROR_VERTICAL
        UHDR_MIRROR_HORIZONTAL

    ctypedef struct uhdr_error_info_t:
        uhdr_codec_err_t error_code
        int has_detail
        char[256] detail

    int UHDR_PLANE_PACKED
    int UHDR_PLANE_Y
    int UHDR_PLANE_U
    int UHDR_PLANE_UV
    int UHDR_PLANE_V

    ctypedef struct uhdr_raw_image_t:
        uhdr_img_fmt_t fmt
        uhdr_color_gamut_t cg
        uhdr_color_transfer_t ct
        uhdr_color_range_t range
        unsigned int w
        unsigned int h
        (void*)[3] planes
        unsigned int[3] stride

    ctypedef struct uhdr_compressed_image_t:
        void* data
        size_t data_sz
        size_t capacity
        uhdr_color_gamut_t cg
        uhdr_color_transfer_t ct
        uhdr_color_range_t range

    ctypedef struct uhdr_mem_block_t:
        void* data
        size_t data_sz
        size_t capacity

    ctypedef struct uhdr_gainmap_metadata_t:
        float[3] max_content_boost
        float[3] min_content_boost
        float[3] gamma
        float[3] offset_sdr
        float[3] offset_hdr
        float hdr_capacity_min
        float hdr_capacity_max
        int use_base_cg

    ctypedef struct uhdr_codec_private_t:
        pass

    # Encoder APIs

    uhdr_codec_private_t* uhdr_create_encoder()

    void uhdr_release_encoder(
        uhdr_codec_private_t* enc
    )

    uhdr_error_info_t uhdr_enc_set_raw_image(
        uhdr_codec_private_t* enc,
        uhdr_raw_image_t* img,
        uhdr_img_label_t intent
    )

    uhdr_error_info_t uhdr_enc_set_compressed_image(
        uhdr_codec_private_t* enc,
        uhdr_compressed_image_t* img,
        uhdr_img_label_t intent
    )

    uhdr_error_info_t uhdr_enc_set_gainmap_image(
        uhdr_codec_private_t* enc,
        uhdr_compressed_image_t* img,
        uhdr_gainmap_metadata_t* metadata
    )

    uhdr_error_info_t uhdr_enc_set_quality(
        uhdr_codec_private_t* enc,
        int quality,
        uhdr_img_label_t intent
    )

    uhdr_error_info_t uhdr_enc_set_exif_data(
        uhdr_codec_private_t* enc,
        uhdr_mem_block_t* exif
    )

    uhdr_error_info_t uhdr_enc_set_using_multi_channel_gainmap(
        uhdr_codec_private_t* enc,
        int use_multi_channel_gainmap
    )

    uhdr_error_info_t uhdr_enc_set_gainmap_scale_factor(
        uhdr_codec_private_t* enc,
        int gainmap_scale_factor
    )

    uhdr_error_info_t uhdr_enc_set_gainmap_gamma(
        uhdr_codec_private_t* enc,
        float gamma
    )

    uhdr_error_info_t uhdr_enc_set_min_max_content_boost(
        uhdr_codec_private_t* enc,
        float min_boost,
        float max_boost
    )

    uhdr_error_info_t uhdr_enc_set_target_display_peak_brightness(
        uhdr_codec_private_t* enc,
        float nits
    )

    uhdr_error_info_t uhdr_enc_set_preset(
        uhdr_codec_private_t* enc,
        uhdr_enc_preset_t preset
    )

    uhdr_error_info_t uhdr_enc_set_output_format(
        uhdr_codec_private_t* enc,
        uhdr_codec_t media_type
    )

    uhdr_error_info_t uhdr_encode(
        uhdr_codec_private_t* enc
    )

    uhdr_compressed_image_t* uhdr_get_encoded_stream(
        uhdr_codec_private_t* enc
    )

    void uhdr_reset_encoder(
        uhdr_codec_private_t* enc
    )

    # Decoder APIs

    int is_uhdr_image(
        void* data,
        int size
    )

    uhdr_codec_private_t* uhdr_create_decoder()

    void uhdr_release_decoder(
        uhdr_codec_private_t* dec
    )

    uhdr_error_info_t uhdr_dec_set_image(
        uhdr_codec_private_t* dec,
        uhdr_compressed_image_t* img
    )

    uhdr_error_info_t uhdr_dec_set_out_img_format(
        uhdr_codec_private_t* dec,
        uhdr_img_fmt_t fmt
    )

    uhdr_error_info_t uhdr_dec_set_out_color_transfer(
        uhdr_codec_private_t* dec,
        uhdr_color_transfer_t ct
    )

    uhdr_error_info_t uhdr_dec_set_out_max_display_boost(
        uhdr_codec_private_t* dec,
        float display_boost
    )

    uhdr_error_info_t uhdr_dec_probe(
        uhdr_codec_private_t* dec
    )

    int uhdr_dec_get_image_width(
        uhdr_codec_private_t* dec
    )

    int uhdr_dec_get_image_height(
        uhdr_codec_private_t* dec
    )

    int uhdr_dec_get_gainmap_width(
        uhdr_codec_private_t* dec
    )

    int uhdr_dec_get_gainmap_height(
        uhdr_codec_private_t* dec
    )

    uhdr_mem_block_t* uhdr_dec_get_exif(
        uhdr_codec_private_t* dec
    )

    uhdr_mem_block_t* uhdr_dec_get_icc(
        uhdr_codec_private_t* dec
    )

    uhdr_mem_block_t* uhdr_dec_get_base_image(
        uhdr_codec_private_t* dec
    )

    uhdr_mem_block_t* uhdr_dec_get_gainmap_image(
        uhdr_codec_private_t* dec
    )

    uhdr_gainmap_metadata_t* uhdr_dec_get_gainmap_metadata(
        uhdr_codec_private_t* dec
    )

    uhdr_error_info_t uhdr_decode(
        uhdr_codec_private_t* dec
    )

    uhdr_raw_image_t* uhdr_get_decoded_image(
        uhdr_codec_private_t* dec
    )

    uhdr_raw_image_t* uhdr_get_gainmap_image(
        uhdr_codec_private_t* dec
    )

    void uhdr_reset_decoder(
        uhdr_codec_private_t* dec
    )

    # Common APIs

    uhdr_error_info_t uhdr_enable_gpu_acceleration(
        uhdr_codec_private_t* codec,
        int enable
    )

    uhdr_error_info_t uhdr_add_effect_mirror(
        uhdr_codec_private_t* codec,
        uhdr_mirror_direction_t direction
    )

    uhdr_error_info_t uhdr_add_effect_rotate(
        uhdr_codec_private_t* codec,
        int degrees
    )

    uhdr_error_info_t uhdr_add_effect_crop(
        uhdr_codec_private_t* codec,
        int left,
        int right,
        int top,
        int bottom
    )

    uhdr_error_info_t uhdr_add_effect_resize(
        uhdr_codec_private_t* codec,
        int width,
        int height
    )
