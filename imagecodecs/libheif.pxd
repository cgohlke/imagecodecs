# imagecodecs/libheif.pxd
# cython: language_level = 3

# Cython declarations for the `libheif 1.19.6` library.
# https://github.com/strukturag/libheif

from libc.stdint cimport (
    uint8_t, uint16_t, uint32_t, int32_t, int64_t, uint64_t
)

cdef extern from 'libheif/heif.h' nogil:

    const char* heif_get_version()

    uint32_t heif_get_version_number()

    int heif_get_version_number_major()

    int heif_get_version_number_minor()

    int heif_get_version_number_maintenance()

    struct heif_context:
        pass

    struct heif_image_handle:
        pass

    struct heif_image:
        pass

    enum heif_error_code:
        heif_error_Ok
        heif_error_Input_does_not_exist
        heif_error_Invalid_input
        heif_error_Unsupported_filetype
        heif_error_Unsupported_feature
        heif_error_Usage_error
        heif_error_Memory_allocation_error
        heif_error_Decoder_plugin_error
        heif_error_Encoder_plugin_error
        heif_error_Encoding_error
        heif_error_Color_profile_does_not_exist
        heif_error_Plugin_loading_error
        heif_error_Canceled

    enum heif_suberror_code:
        heif_suberror_Unspecified
        heif_suberror_End_of_data
        heif_suberror_Invalid_box_size
        heif_suberror_No_ftyp_box
        heif_suberror_No_idat_box
        heif_suberror_No_meta_box
        heif_suberror_No_hdlr_box
        heif_suberror_No_hvcC_box
        heif_suberror_No_pitm_box
        heif_suberror_No_ipco_box
        heif_suberror_No_ipma_box
        heif_suberror_No_iloc_box
        heif_suberror_No_iinf_box
        heif_suberror_No_iprp_box
        heif_suberror_No_iref_box
        heif_suberror_No_pict_handler
        heif_suberror_Ipma_box_references_nonexisting_property
        heif_suberror_No_properties_assigned_to_item
        heif_suberror_No_item_data
        heif_suberror_Invalid_grid_data
        heif_suberror_Missing_grid_images
        heif_suberror_Invalid_clean_aperture
        heif_suberror_Invalid_overlay_data
        heif_suberror_Overlay_image_outside_of_canvas
        heif_suberror_Auxiliary_image_type_unspecified
        heif_suberror_No_or_invalid_primary_item
        heif_suberror_No_infe_box
        heif_suberror_Unknown_color_profile_type
        heif_suberror_Wrong_tile_image_chroma_format
        heif_suberror_Invalid_fractional_number
        heif_suberror_Invalid_image_size
        heif_suberror_Invalid_pixi_box
        heif_suberror_No_av1C_box
        heif_suberror_Wrong_tile_image_pixel_depth
        heif_suberror_Unknown_NCLX_color_primaries
        heif_suberror_Unknown_NCLX_transfer_characteristics
        heif_suberror_Unknown_NCLX_matrix_coefficients
        heif_suberror_Invalid_region_data
        heif_suberror_No_ispe_property
        heif_suberror_Camera_intrinsic_matrix_undefined
        heif_suberror_Camera_extrinsic_matrix_undefined
        heif_suberror_Invalid_J2K_codestream
        heif_suberror_No_vvcC_box
        heif_suberror_No_icbr_box
        heif_suberror_No_avcC_box
        heif_suberror_Invalid_mini_box
        heif_suberror_Decompression_invalid_data
        heif_suberror_Security_limit_exceeded
        heif_suberror_Compression_initialisation_error
        heif_suberror_Nonexisting_item_referenced
        heif_suberror_Null_pointer_argument
        heif_suberror_Nonexisting_image_channel_referenced
        heif_suberror_Unsupported_plugin_version
        heif_suberror_Unsupported_writer_version
        heif_suberror_Unsupported_parameter
        heif_suberror_Invalid_parameter_value
        heif_suberror_Item_reference_cycle
        heif_suberror_Invalid_property
        heif_suberror_Unsupported_codec
        heif_suberror_Unsupported_image_type
        heif_suberror_Unsupported_data_version
        heif_suberror_Unsupported_color_conversion
        heif_suberror_Unsupported_item_construction_method
        heif_suberror_Unsupported_header_compression_method
        heif_suberror_Unsupported_generic_compression_method
        heif_suberror_Unsupported_essential_property
        heif_suberror_Unsupported_bit_depth
        heif_suberror_Cannot_write_output_data
        heif_suberror_Encoder_initialization
        heif_suberror_Encoder_encoding
        heif_suberror_Encoder_cleanup
        heif_suberror_Too_many_regions
        heif_suberror_Plugin_loading_error
        heif_suberror_Plugin_is_not_loaded
        heif_suberror_Cannot_read_plugin_directory
        heif_suberror_No_matching_decoder_installed

    struct heif_error:
        heif_error_code code
        heif_suberror_code subcode
        const char* message

    const heif_error heif_error_success

    ctypedef uint32_t heif_item_id
    ctypedef uint32_t heif_property_id

    enum heif_compression_format:
        heif_compression_undefined
        heif_compression_HEVC
        heif_compression_AVC
        heif_compression_JPEG
        heif_compression_AV1
        heif_compression_VVC
        heif_compression_EVC
        heif_compression_JPEG2000
        heif_compression_uncompressed
        heif_compression_mask
        heif_compression_HTJ2K

    enum heif_chroma:
        heif_chroma_undefined
        heif_chroma_monochrome
        heif_chroma_420
        heif_chroma_422
        heif_chroma_444
        heif_chroma_interleaved_RGB
        heif_chroma_interleaved_RGBA
        heif_chroma_interleaved_RRGGBB_BE
        heif_chroma_interleaved_RRGGBBAA_BE
        heif_chroma_interleaved_RRGGBB_LE
        heif_chroma_interleaved_RRGGBBAA_LE

    # int heif_chroma_interleaved_24bit = heif_chroma_interleaved_RGB
    # int heif_chroma_interleaved_32bit = heif_chroma_interleaved_RGBA

    enum heif_colorspace:
        heif_colorspace_undefined
        heif_colorspace_YCbCr
        heif_colorspace_RGB
        heif_colorspace_monochrome
        heif_colorspace_nonvisual

    enum heif_channel:
        heif_channel_Y
        heif_channel_Cb
        heif_channel_Cr
        heif_channel_R
        heif_channel_G
        heif_channel_B
        heif_channel_Alpha
        heif_channel_interleaved
        heif_channel_filter_array
        heif_channel_depth
        heif_channel_disparity

    enum heif_metadata_compression:
        heif_metadata_compression_off
        heif_metadata_compression_auto
        heif_metadata_compression_unknown
        heif_metadata_compression_deflate
        heif_metadata_compression_zlib
        heif_metadata_compression_brotli

    struct heif_init_params:
        int version

    heif_error heif_init(heif_init_params*)

    void heif_deinit()

    enum heif_plugin_type:
        heif_plugin_type_encoder
        heif_plugin_type_decoder

    struct heif_plugin_info:
        int version
        heif_plugin_type type_ 'type'
        const void* plugin
        void* internal_handle

    heif_error heif_load_plugin(
        const char* filename,
        heif_plugin_info **out_plugin
    )

    heif_error heif_load_plugins(
        const char* directory,
        const heif_plugin_info** out_plugins,
        int* out_nPluginsLoaded,
        int output_array_size
    )

    heif_error heif_unload_plugin(
        const heif_plugin_info* plugin
    )

    const char** heif_get_plugin_directories()

    void heif_free_plugin_directories(
        const char**
    )

    enum heif_filetype_result:
        heif_filetype_no
        heif_filetype_yes_supported
        heif_filetype_yes_unsupported
        heif_filetype_maybe

    heif_filetype_result heif_check_filetype(
        const uint8_t* data,
        int len
    )

    heif_error heif_has_compatible_filetype(
        const uint8_t* data,
        int len
    )

    int heif_check_jpeg_filetype(
        const uint8_t* data,
        int len
    )

    enum heif_brand:
        heif_unknown_brand
        heif_heic
        heif_heix
        heif_hevc
        heif_hevx
        heif_heim
        heif_heis
        heif_hevm
        heif_hevs
        heif_mif1
        heif_msf1
        heif_avif
        heif_avis
        heif_vvic
        heif_vvis
        heif_evbi
        heif_evbs
        heif_j2ki
        heif_j2is

    heif_brand heif_main_brand(
        const uint8_t* data,
        int len
    )

    ctypedef uint32_t heif_brand2

    int heif_brand2_mif1
    int heif_brand2_mif2
    int heif_brand2_mif3
    int heif_brand2_msf1
    int heif_brand2_vvic
    int heif_brand2_vvis
    int heif_brand2_evbi
    int heif_brand2_evmi
    int heif_brand2_evbs
    int heif_brand2_evms
    int heif_brand2_jpeg
    int heif_brand2_jpgs
    int heif_brand2_j2ki
    int heif_brand2_j2is
    int heif_brand2_miaf
    int heif_brand2_1pic

    heif_brand2 heif_read_main_brand(
        const uint8_t* data,
        int len
    )

    heif_brand2 heif_read_minor_version_brand(
        const uint8_t* data,
        int len
    )

    heif_brand2 heif_fourcc_to_brand(
        const char* brand_fourcc
    )

    void heif_brand_to_fourcc(
        heif_brand2 brand,
        char* out_fourcc
    )

    int heif_has_compatible_brand(
        const uint8_t* data,
        int len,
        const char* brand_fourcc
    )

    heif_error heif_list_compatible_brands(
        const uint8_t* data,
        int len,
        heif_brand2** out_brands,
        int* out_size
    )

    void heif_free_list_of_compatible_brands(
        heif_brand2* brands_list
    )

    const char* heif_get_file_mime_type(
        const uint8_t* data,
        int len
    )

    heif_context* heif_context_alloc(
    )

    void heif_context_free(
        heif_context*
    )

    struct heif_reading_options:
        pass

    enum heif_reader_grow_status:
        heif_reader_grow_status_size_reached
        heif_reader_grow_status_timeout
        heif_reader_grow_status_size_beyond_eof
        heif_reader_grow_status_error

    struct heif_reader_range_request_result:
        heif_reader_grow_status status
        uint64_t range_end
        int reader_error_code
        const char* reader_error_msg

    struct heif_reader:
        int reader_api_version

        int64_t (* get_position)(
            void* userdata
        ) nogil

        int (* read)(
            void* data,
            size_t size,
            void* userdata
        ) nogil

        int (* seek)(
            int64_t position,
            void* userdata
        ) nogil

        heif_reader_grow_status (* wait_for_file_size)(
            int64_t target_size,
            void* userdata
        ) nogil

        heif_reader_range_request_result (*request_range)(
            uint64_t start_pos,
            uint64_t end_pos,
            void* userdata
        )

        void (*preload_range_hint)(
            uint64_t start_pos,
            uint64_t end_pos,
            void* userdata
        )

        void (*release_file_range)(
            uint64_t start_pos,
            uint64_t end_pos,
            void* userdata
        )

        void (*release_error_msg)(
            const char* msg
        )

    heif_error heif_context_read_from_file(
        heif_context*,
        const char* filename,
        const heif_reading_options*
    )

    heif_error heif_context_read_from_memory(
        heif_context*,
        const void* mem,
        size_t size,
        const heif_reading_options*
    )

    heif_error heif_context_read_from_memory_without_copy(
        heif_context*,
        const void* mem,
        size_t size,
        const heif_reading_options*
    )

    heif_error heif_context_read_from_reader(
        heif_context*,
        const heif_reader* reader,
        void* userdata,
        const heif_reading_options*
    )

    int heif_context_get_number_of_top_level_images(
        heif_context* ctx
    )

    int heif_context_is_top_level_image_ID(
        heif_context* ctx,
        heif_item_id id
    )

    int heif_context_get_list_of_top_level_image_IDs(
        heif_context* ctx,
        heif_item_id* ID_array,
        int count
    )

    heif_error heif_context_get_primary_image_ID(
        heif_context* ctx,
        heif_item_id* id
    )

    heif_error heif_context_get_primary_image_handle(
        heif_context* ctx,
        heif_image_handle**
    )

    heif_error heif_context_get_image_handle(
        heif_context* ctx,
        heif_item_id id_,
        heif_image_handle**
    )

    void heif_context_debug_dump_boxes_to_file(
        heif_context* ctx,
        int fd
    )

    void heif_context_set_maximum_image_size_limit(
        heif_context* ctx,
        int maximum_width
    )

    void heif_context_set_max_decoding_threads(
        heif_context* ctx,
        int max_threads
    )

    struct heif_security_limits:
        uint8_t version
        uint64_t max_image_size_pixels
        uint64_t max_number_of_tiles
        uint32_t max_bayer_pattern_pixels
        uint32_t max_items
        uint32_t max_color_profile_size
        uint64_t max_memory_block_size
        uint32_t max_components
        uint32_t max_iloc_extents_per_item
        uint32_t max_size_entity_group
        uint32_t max_children_per_box

    const heif_security_limits* heif_get_global_security_limits()

    const heif_security_limits* heif_get_disabled_security_limits()

    heif_security_limits* heif_context_get_security_limits(
        const heif_context*
    )

    heif_error heif_context_set_security_limits(
        heif_context*,
        const heif_security_limits*
    )

    void heif_image_handle_release(
        const heif_image_handle*
    )

    int heif_image_handle_is_primary_image(
        const heif_image_handle* handle
    )

    heif_item_id heif_image_handle_get_item_id(
        const heif_image_handle* handle
    )

    int heif_image_handle_get_width(
        const heif_image_handle* handle
    )

    int heif_image_handle_get_height(
        const heif_image_handle* handle
    )

    int heif_image_handle_has_alpha_channel(
        const heif_image_handle*
    )

    int heif_image_handle_is_premultiplied_alpha(
        const heif_image_handle*
    )

    int heif_image_handle_get_luma_bits_per_pixel(
        const heif_image_handle*
    )

    int heif_image_handle_get_chroma_bits_per_pixel(
        const heif_image_handle*
    )

    heif_error heif_image_handle_get_preferred_decoding_colorspace(
        const heif_image_handle* image_handle,
        heif_colorspace* out_colorspace,
        heif_chroma* out_chroma
    )

    heif_context* heif_image_handle_get_context(
        const heif_image_handle* handle
    )

    struct heif_image_tiling:
        int version
        uint32_t num_columns
        uint32_t num_rows
        uint32_t tile_width
        uint32_t tile_height
        uint32_t image_width
        uint32_t image_height
        uint32_t top_offset
        uint32_t left_offset
        uint8_t number_of_extra_dimensions
        uint32_t[8] extra_dimension_size

    heif_error heif_image_handle_get_image_tiling(
        const heif_image_handle* handle,
        int process_image_transformations,
        heif_image_tiling* out_tiling
    )

    heif_error heif_image_handle_get_grid_image_tile_id(
        const heif_image_handle* handle,
        int process_image_transformations,
        uint32_t tile_x,
        uint32_t tile_y,
        heif_item_id* out_tile_item_id
    )

    struct heif_decoding_options:
        pass

    heif_error heif_image_handle_decode_image_tile(
        const heif_image_handle* in_handle,
        heif_image** out_img,
        heif_colorspace colorspace,
        heif_chroma chroma,
        const heif_decoding_options* options,
        uint32_t tile_x,
        uint32_t tile_y
    )

    ctypedef uint32_t heif_entity_group_id

    struct heif_entity_group:
        heif_entity_group_id entity_group_id
        uint32_t entity_group_type
        heif_item_id* entities
        uint32_t num_entities

    heif_entity_group* heif_context_get_entity_groups(
        const heif_context*,
        uint32_t type_filter,
        uint32_t item_filter,
        int* out_num_groups
    )

    void heif_entity_groups_release(
        heif_entity_group*,
        int num_groups
    )

    int heif_image_handle_get_ispe_width(
        const heif_image_handle* handle
    )

    int heif_image_handle_get_ispe_height(
        const heif_image_handle* handle
    )

    int heif_image_handle_has_depth_image(
        const heif_image_handle*
    )

    int heif_image_handle_get_number_of_depth_images(
        const heif_image_handle* handle
    )

    int heif_image_handle_get_list_of_depth_image_IDs(
        const heif_image_handle* handle,
        heif_item_id* ids,
        int count
    )

    heif_error heif_image_handle_get_depth_image_handle(
        const heif_image_handle* handle,
        heif_item_id depth_image_id_,
        heif_image_handle** out_depth_handle
    )

    enum heif_depth_representation_type:
        heif_depth_representation_type_uniform_inverse_Z
        heif_depth_representation_type_uniform_disparity
        heif_depth_representation_type_uniform_Z
        heif_depth_representation_type_nonuniform_disparity

    struct heif_depth_representation_info:
        uint8_t version
        uint8_t has_z_near
        uint8_t has_z_far
        uint8_t has_d_min
        uint8_t has_d_max
        double z_near
        double z_far
        double d_min
        double d_max
        heif_depth_representation_type depth_representation_type
        uint32_t disparity_reference_view
        uint32_t depth_nonlinear_representation_model_size
        uint8_t* depth_nonlinear_representation_model

    void heif_depth_representation_info_free(
        const heif_depth_representation_info* info
    )

    int heif_image_handle_get_depth_image_representation_info(
        const heif_image_handle* handle,
        heif_item_id depth_image_id_,
        const heif_depth_representation_info** out
    )

    int heif_image_handle_get_number_of_thumbnails(
        const heif_image_handle* handle
    )

    int heif_image_handle_get_list_of_thumbnail_IDs(
        const heif_image_handle* handle,
        heif_item_id* ids,
        int count
    )

    heif_error heif_image_handle_get_thumbnail(
        const heif_image_handle* main_image_handle,
        heif_item_id thumbnail_id_,
        heif_image_handle** out_thumbnail_handle
    )

    int LIBHEIF_AUX_IMAGE_FILTER_OMIT_ALPHA
    int LIBHEIF_AUX_IMAGE_FILTER_OMIT_DEPTH

    int heif_image_handle_get_number_of_auxiliary_images(
        const heif_image_handle* handle,
        int aux_filter
    )

    int heif_image_handle_get_list_of_auxiliary_image_IDs(
        const heif_image_handle* handle,
        int aux_filter,
        heif_item_id* ids,
        int count
    )

    heif_error heif_image_handle_get_auxiliary_type(
        const heif_image_handle* handle,
        const char** out_type
    )

    void heif_image_handle_release_auxiliary_type(
        const heif_image_handle* handle,
        const char** out_type
    )

    void heif_image_handle_free_auxiliary_types(
        const heif_image_handle* handle,
        const char** out_type
    )

    heif_error heif_image_handle_get_auxiliary_image_handle(
        const heif_image_handle* main_image_handle,
        heif_item_id auxiliary_id_,
        heif_image_handle** out_auxiliary_handle
    )

    int heif_image_handle_get_number_of_metadata_blocks(
        const heif_image_handle* handle,
        const char* type_filter
    )

    int heif_image_handle_get_list_of_metadata_block_IDs(
        const heif_image_handle* handle,
        const char* type_filter,
        heif_item_id* ids,
        int count
    )

    const char* heif_image_handle_get_metadata_type(
        const heif_image_handle* handle,
        heif_item_id metadata_id
    )

    const char* heif_image_handle_get_metadata_content_type(
        const heif_image_handle* handle,
        heif_item_id metadata_id
    )

    size_t heif_image_handle_get_metadata_size(
        const heif_image_handle* handle,
        heif_item_id metadata_id
    )

    heif_error heif_image_handle_get_metadata(
        const heif_image_handle* handle,
        heif_item_id metadata_id_,
        void* out_data
    )

    const char* heif_image_handle_get_metadata_item_uri_type(
        const heif_image_handle* handle,
        heif_item_id metadata_id
    )

    enum heif_color_profile_type:
        heif_color_profile_type_not_present
        heif_color_profile_type_nclx
        heif_color_profile_type_rICC
        heif_color_profile_type_prof

    heif_color_profile_type heif_image_handle_get_color_profile_type(
        const heif_image_handle* handle
    )

    size_t heif_image_handle_get_raw_color_profile_size(
        const heif_image_handle* handle
    )

    heif_error heif_image_handle_get_raw_color_profile(
        const heif_image_handle* handle,
        void* out_data
    )

    enum heif_color_primaries:
        heif_color_primaries_ITU_R_BT_709_5
        heif_color_primaries_unspecified
        heif_color_primaries_ITU_R_BT_470_6_System_M
        heif_color_primaries_ITU_R_BT_470_6_System_B_G
        heif_color_primaries_ITU_R_BT_601_6
        heif_color_primaries_SMPTE_240M
        heif_color_primaries_generic_film
        heif_color_primaries_ITU_R_BT_2020_2_and_2100_0
        heif_color_primaries_SMPTE_ST_428_1
        heif_color_primaries_SMPTE_RP_431_2
        heif_color_primaries_SMPTE_EG_432_1
        heif_color_primaries_EBU_Tech_3213_E

    enum heif_transfer_characteristics:
        heif_transfer_characteristic_ITU_R_BT_709_5
        heif_transfer_characteristic_unspecified
        heif_transfer_characteristic_ITU_R_BT_470_6_System_M
        heif_transfer_characteristic_ITU_R_BT_470_6_System_B_G
        heif_transfer_characteristic_ITU_R_BT_601_6
        heif_transfer_characteristic_SMPTE_240M
        heif_transfer_characteristic_linear
        heif_transfer_characteristic_logarithmic_100
        heif_transfer_characteristic_logarithmic_100_sqrt10
        heif_transfer_characteristic_IEC_61966_2_4
        heif_transfer_characteristic_ITU_R_BT_1361
        heif_transfer_characteristic_IEC_61966_2_1
        heif_transfer_characteristic_ITU_R_BT_2020_2_10bit
        heif_transfer_characteristic_ITU_R_BT_2020_2_12bit
        heif_transfer_characteristic_ITU_R_BT_2100_0_PQ
        heif_transfer_characteristic_SMPTE_ST_428_1
        heif_transfer_characteristic_ITU_R_BT_2100_0_HLG

    enum heif_matrix_coefficients:
        heif_matrix_coefficients_RGB_GBR
        heif_matrix_coefficients_ITU_R_BT_709_5
        heif_matrix_coefficients_unspecified
        heif_matrix_coefficients_US_FCC_T47
        heif_matrix_coefficients_ITU_R_BT_470_6_System_B_G
        heif_matrix_coefficients_ITU_R_BT_601_6
        heif_matrix_coefficients_SMPTE_240M
        heif_matrix_coefficients_YCgCo
        heif_matrix_coefficients_ITU_R_BT_2020_2_non_constant_luminance
        heif_matrix_coefficients_ITU_R_BT_2020_2_constant_luminance
        heif_matrix_coefficients_SMPTE_ST_2085
        heif_matrix_coefficients_chromaticity_derived_non_constant_luminance
        heif_matrix_coefficients_chromaticity_derived_constant_luminance
        heif_matrix_coefficients_ICtCp

    struct heif_color_profile_nclx:
        uint8_t version
        heif_color_primaries color_primaries
        heif_transfer_characteristics transfer_characteristics
        heif_matrix_coefficients matrix_coefficients
        uint8_t full_range_flag
        float color_primary_red_x
        float color_primary_red_y
        float color_primary_green_x
        float color_primary_green_y
        float color_primary_blue_x
        float color_primary_blue_y
        float color_primary_white_x
        float color_primary_white_y

    heif_error heif_nclx_color_profile_set_color_primaries(
        heif_color_profile_nclx* nclx,
        uint16_t cp
    )

    heif_error heif_nclx_color_profile_set_transfer_characteristics(
        heif_color_profile_nclx* nclx,
        uint16_t transfer_characteristics
    )

    heif_error heif_nclx_color_profile_set_matrix_coefficients(
        heif_color_profile_nclx* nclx,
        uint16_t matrix_coefficients
    )

    heif_error heif_image_handle_get_nclx_color_profile(
        const heif_image_handle* handle,
        heif_color_profile_nclx** out_data
    )

    heif_color_profile_nclx* heif_nclx_color_profile_alloc(
    )

    void heif_nclx_color_profile_free(
        heif_color_profile_nclx* nclx_profile
    )

    heif_color_profile_type heif_image_get_color_profile_type(
        const heif_image* image
    )

    size_t heif_image_get_raw_color_profile_size(
        const heif_image* image
    )

    heif_error heif_image_get_raw_color_profile(
        const heif_image* image,
        void* out_data
    )

    heif_error heif_image_get_nclx_color_profile(
        const heif_image* image,
        heif_color_profile_nclx** out_data
    )

    struct heif_camera_intrinsic_matrix:
        double focal_length_x
        double focal_length_y
        double principal_point_x
        double principal_point_y
        double skew

    int heif_image_handle_has_camera_intrinsic_matrix(
        const heif_image_handle* handle
    )

    heif_error heif_image_handle_get_camera_intrinsic_matrix(
        const heif_image_handle* handle,
        heif_camera_intrinsic_matrix* out_matrix
    )

    struct heif_camera_extrinsic_matrix:
        pass

    int heif_image_handle_has_camera_extrinsic_matrix(
        const heif_image_handle* handle
    )

    heif_error heif_image_handle_get_camera_extrinsic_matrix(
        const heif_image_handle* handle,
        heif_camera_extrinsic_matrix** out_matrix
    )

    void heif_camera_extrinsic_matrix_release(
        heif_camera_extrinsic_matrix*
    )

    heif_error heif_camera_extrinsic_matrix_get_rotation_matrix(
        const heif_camera_extrinsic_matrix*,
        double* out_matrix_row_major
    )

    enum heif_progress_step:
        heif_progress_step_total
        heif_progress_step_load_tile

    enum heif_chroma_downsampling_algorithm:
        heif_chroma_downsampling_nearest_neighbor
        heif_chroma_downsampling_average
        heif_chroma_downsampling_sharp_yuv

    enum heif_chroma_upsampling_algorithm:
        heif_chroma_upsampling_nearest_neighbor
        heif_chroma_upsampling_bilinear

    struct heif_color_conversion_options:
        uint8_t version
        heif_chroma_downsampling_algorithm preferred_chroma_downsampling_algorithm
        heif_chroma_upsampling_algorithm preferred_chroma_upsampling_algorithm
        uint8_t only_use_preferred_chroma_algorithm

    void heif_color_conversion_options_set_defaults(
        heif_color_conversion_options*
    )

    struct heif_decoding_options:
        uint8_t version
        uint8_t ignore_transformations

        void (* start_progress)(
            heif_progress_step step,
            int max_progress,
            void* progress_user_data
        ) nogil

        void (* on_progress)(
            heif_progress_step step,
            int progress,
            void* progress_user_data
        ) nogil

        void (* end_progress)(
            heif_progress_step step,
            void* progress_user_data
        ) nogil

        void* progress_user_data
        uint8_t convert_hdr_to_8bit
        uint8_t strict_decoding
        const char* decoder_id

        heif_color_conversion_options color_conversion_options
        int (* cancel_decoding)(void* progress_user_data)

    heif_decoding_options* heif_decoding_options_alloc(
    )

    void heif_decoding_options_free(
        heif_decoding_options*
    )

    heif_error heif_decode_image(
        const heif_image_handle* in_handle,
        heif_image** out_img,
        heif_colorspace colorspace,
        heif_chroma chroma,
        const heif_decoding_options* options
    )

    heif_colorspace heif_image_get_colorspace(
        const heif_image*
    )

    heif_chroma heif_image_get_chroma_format(
        const heif_image*
    )

    int heif_image_get_width(
        const heif_image* img,
        heif_channel channel
    )

    int heif_image_get_height(
        const heif_image* img,
        heif_channel channel
    )

    int heif_image_get_primary_width(
        const heif_image* img
    )

    int heif_image_get_primary_height(
        const heif_image* img
    )

    heif_error heif_image_crop(
        heif_image* img,
        int left,
        int right,
        int top,
        int bottom
    )

    int heif_image_get_bits_per_pixel(
        const heif_image*,
        heif_channel channel
    )

    int heif_image_get_bits_per_pixel_range(
        const heif_image*,
        heif_channel channel
    )

    int heif_image_has_channel(
        const heif_image*,
        heif_channel channel
    )

    const uint8_t* heif_image_get_plane_readonly(
        const heif_image*,
        heif_channel channel,
        int* out_stride
    )

    uint8_t* heif_image_get_plane(
        heif_image*,
        heif_channel channel,
        int* out_stride
    )

    struct heif_scaling_options:
        pass

    heif_error heif_image_scale_image(
        const heif_image* input,
        heif_image** output,
        int width,
        int height,
        const heif_scaling_options* options
    )

    heif_error heif_image_extend_to_size_fill_with_zero(
        heif_image* image,
        uint32_t width,
        uint32_t height
    )

    heif_error heif_image_set_raw_color_profile(
        heif_image* image,
        const char* profile_type_fourcc_string,
        const void* profile_data,
        const size_t profile_size
    )

    heif_error heif_image_set_nclx_color_profile(
        heif_image* image,
        const heif_color_profile_nclx* color_profile
    )

    int heif_image_get_decoding_warnings(
        heif_image* image,
        int first_warning_idx,
        heif_error* out_warnings,
        int max_output_buffer_entries
    )

    void heif_image_add_decoding_warning(
        heif_image* image,
        heif_error err
    )

    void heif_image_release(
        const heif_image*
    )

    struct heif_content_light_level:
        uint16_t max_content_light_level
        uint16_t max_pic_average_light_level

    int heif_image_has_content_light_level(
        const heif_image*
    )

    void heif_image_get_content_light_level(
        const heif_image*,
        heif_content_light_level* out
    )

    int heif_image_handle_get_content_light_level(
        const heif_image_handle*,
        heif_content_light_level* out
    )

    void heif_image_set_content_light_level(
        const heif_image*,
        const heif_content_light_level* inp
    )

    struct heif_mastering_display_colour_volume:
        uint16_t[3] display_primaries_x
        uint16_t[3] display_primaries_y
        uint16_t white_point_x
        uint16_t white_point_y
        uint32_t max_display_mastering_luminance
        uint32_t min_display_mastering_luminance

    struct heif_decoded_mastering_display_colour_volume:
        float[3] display_primaries_x
        float[3] display_primaries_y
        float white_point_x
        float white_point_y
        double max_display_mastering_luminance
        double min_display_mastering_luminance

    struct heif_ambient_viewing_environment:
        uint32_t ambient_illumination
        uint16_t ambient_light_x
        uint16_t ambient_light_y

    int heif_image_has_mastering_display_colour_volume(
        const heif_image*
    )

    void heif_image_get_mastering_display_colour_volume(
        const heif_image*,
        heif_mastering_display_colour_volume* out
    )

    int heif_image_handle_get_mastering_display_colour_volume(
        const heif_image_handle*,
        heif_mastering_display_colour_volume* out
    )

    void heif_image_set_mastering_display_colour_volume(
        const heif_image*,
        const heif_mastering_display_colour_volume* inp
    )

    heif_error heif_mastering_display_colour_volume_decode(
        const heif_mastering_display_colour_volume* inp,
        heif_decoded_mastering_display_colour_volume* out
    )

    void heif_image_get_pixel_aspect_ratio(
        const heif_image*,
        uint32_t* aspect_h,
        uint32_t* aspect_v
    )

    int heif_image_handle_get_pixel_aspect_ratio(
        const heif_image_handle*,
        uint32_t* aspect_h,
        uint32_t* aspect_v
    )

    void heif_image_set_pixel_aspect_ratio(
        heif_image*,
        uint32_t aspect_h,
        uint32_t aspect_v
    )

    heif_error heif_context_write_to_file(
        heif_context*,
        const char* filename
    )

    void heif_context_add_compatible_brand(
        heif_context* ctx,
        heif_brand2 compatible_brand
    )

    struct heif_writer:
        int writer_api_version
        heif_error (* write)(
            heif_context* ctx,
            const void* data,
            size_t size,
            void* userdata
        ) nogil

    heif_error heif_context_write(
        heif_context*,
        heif_writer* writer,
        void* userdata
    )

    struct heif_encoder:
        pass

    struct heif_encoder_descriptor:
        pass

    struct heif_encoder_parameter:
        pass

    int heif_context_get_encoder_descriptors(
        heif_context*,
        heif_compression_format format_filter,
        const char* name_filter,
        const heif_encoder_descriptor** out_encoders,
        int count
    )

    const char* heif_encoder_descriptor_get_name(
        const heif_encoder_descriptor*
    )

    const char* heif_encoder_descriptor_get_id_name(
        const heif_encoder_descriptor*
    )

    heif_compression_format heif_encoder_descriptor_get_compression_format(
        const heif_encoder_descriptor*
    )

    int heif_encoder_descriptor_supports_lossy_compression(
        const heif_encoder_descriptor*
    )

    int heif_encoder_descriptor_supports_lossless_compression(
        const heif_encoder_descriptor*
    )

    heif_error heif_context_get_encoder(
        heif_context* context,
        const heif_encoder_descriptor*,
        heif_encoder** out_encoder
    )

    int heif_have_decoder_for_format(
        heif_compression_format format
    )

    int heif_have_encoder_for_format(
        heif_compression_format format
    )

    heif_error heif_context_get_encoder_for_format(
        heif_context* context,
        heif_compression_format format,
        heif_encoder**
    )

    void heif_encoder_release(
        heif_encoder*
    )

    const char* heif_encoder_get_name(
        const heif_encoder*
    )

    heif_error heif_encoder_set_lossy_quality(
        heif_encoder*,
        int quality
    )

    heif_error heif_encoder_set_lossless(
        heif_encoder*,
        int enable
    )

    heif_error heif_encoder_set_logging_level(
        heif_encoder*,
        int level
    )

    const heif_encoder_parameter* const* heif_encoder_list_parameters(
        heif_encoder*
    )

    const char* heif_encoder_parameter_get_name(
        const heif_encoder_parameter*
    )

    enum heif_encoder_parameter_type:
        heif_encoder_parameter_type_integer
        heif_encoder_parameter_type_boolean
        heif_encoder_parameter_type_string

    heif_encoder_parameter_type heif_encoder_parameter_get_type(
        const heif_encoder_parameter*
    )

    heif_error heif_encoder_parameter_get_valid_integer_range(
        const heif_encoder_parameter*,
        int* have_minimum_maximum,
        int* minimum,
        int* maximum
    )

    heif_error heif_encoder_parameter_get_valid_integer_values(
        const heif_encoder_parameter*,
        int* have_minimum,
        int* have_maximum,
        int* minimum,
        int* maximum,
        int* num_valid_values,
        const int** out_integer_array
    )

    heif_error heif_encoder_parameter_get_valid_string_values(
        const heif_encoder_parameter*,
        const char** out_stringarray
    )

    heif_error heif_encoder_set_parameter_integer(
        heif_encoder*,
        const char* parameter_name,
        int value
    )

    heif_error heif_encoder_get_parameter_integer(
        heif_encoder*,
        const char* parameter_name,
        int* value
    )

    heif_error heif_encoder_parameter_integer_valid_range(
        heif_encoder*,
        const char* parameter_name,
        int* have_minimum_maximum,
        int* minimum,
        int* maximum
    )

    heif_error heif_encoder_set_parameter_boolean(
        heif_encoder*,
        const char* parameter_name,
        int value
    )

    heif_error heif_encoder_get_parameter_boolean(
        heif_encoder*,
        const char* parameter_name,
        int* value
    )

    heif_error heif_encoder_set_parameter_string(
        heif_encoder*,
        const char* parameter_name,
        const char* value
    )

    heif_error heif_encoder_get_parameter_string(
        heif_encoder*,
        const char* parameter_name,
        char* value,
        int value_size
    )

    heif_error heif_encoder_parameter_string_valid_values(
        heif_encoder*,
        const char* parameter_name,
        const char** out_stringarray
    )

    heif_error heif_encoder_parameter_integer_valid_values(
        heif_encoder*,
        const char* parameter_name,
        int* have_minimum,
        int* have_maximum,
        int* minimum,
        int* maximum,
        int* num_valid_values,
        const int** out_integer_array
    )

    heif_error heif_encoder_set_parameter(
        heif_encoder*,
        const char* parameter_name,
        const char* value
    )

    heif_error heif_encoder_get_parameter(
        heif_encoder*,
        const char* parameter_name,
        char* value_ptr,
        int value_size
    )

    int heif_encoder_has_default(
        heif_encoder*,
        const char* parameter_name
    )

    enum heif_orientation:
        heif_orientation_normal
        heif_orientation_flip_horizontally
        heif_orientation_rotate_180
        heif_orientation_flip_vertically
        heif_orientation_rotate_90_cw_then_flip_horizontally
        heif_orientation_rotate_90_cw
        heif_orientation_rotate_90_cw_then_flip_vertically
        heif_orientation_rotate_270_cw

    struct heif_encoding_options:
        uint8_t version
        uint8_t save_alpha_channel
        uint8_t macOS_compatibility_workaround
        uint8_t save_two_colr_boxes_when_ICC_and_nclx_available
        const heif_color_profile_nclx* output_nclx_profile
        uint8_t macOS_compatibility_workaround_no_nclx_profile
        heif_orientation image_orientation
        heif_color_conversion_options color_conversion_options
        uint8_t prefer_uncC_short_form

    heif_encoding_options* heif_encoding_options_alloc()

    void heif_encoding_options_free(
        heif_encoding_options*
    )

    heif_error heif_context_encode_image(
        heif_context*,
        const heif_image* image,
        heif_encoder* encoder,
        const heif_encoding_options* options,
        heif_image_handle** out_image_handle
    )

    heif_error heif_context_encode_grid(
        heif_context* ctx,
        heif_image** tiles,
        uint16_t rows,
        uint16_t columns,
        heif_encoder* encoder,
        const heif_encoding_options* input_options,
        heif_image_handle** out_image_handle
    )

    heif_error heif_context_add_grid_image(
        heif_context* ctx,
        uint32_t image_width,
        uint32_t image_height,
        uint32_t tile_columns,
        uint32_t tile_rows,
        const heif_encoding_options* encoding_options,
        heif_image_handle** out_grid_image_handle
    )

    heif_error heif_context_add_image_tile(
        heif_context* ctx,
        heif_image_handle* tiled_image,
        uint32_t tile_x, uint32_t tile_y,
        const heif_image* image,
        heif_encoder* encoder
    )

    heif_error heif_context_add_overlay_image(
        heif_context* ctx,
        uint32_t image_width,
        uint32_t image_height,
        uint16_t nImages,
        const heif_item_id* image_ids,
        int32_t* offsets,
        const uint16_t background_rgba[4],
        heif_image_handle** out_iovl_image_handle
    )

    heif_error heif_context_set_primary_image(
        heif_context*,
        heif_image_handle* image_handle
    )

    heif_error heif_context_encode_thumbnail(
        heif_context*,
        const heif_image* image,
        const heif_image_handle* master_image_handle,
        heif_encoder* encoder,
        const heif_encoding_options* options,
        int bbox_size,
        heif_image_handle** out_thumb_image_handle
    )

    heif_error heif_context_assign_thumbnail(
        heif_context*,
        const heif_image_handle* master_image,
        const heif_image_handle* thumbnail_image
    )

    heif_error heif_context_add_exif_metadata(
        heif_context*,
        const heif_image_handle* image_handle,
        const void* data,
        int size
    )

    heif_error heif_context_add_XMP_metadata(
        heif_context*,
        const heif_image_handle* image_handle,
        const void* data,
        int size
    )

    heif_error heif_context_add_XMP_metadata2(
        heif_context*,
        const heif_image_handle* image_handle,
        const void* data,
        int size,
        heif_metadata_compression compression
    )

    heif_error heif_context_add_generic_metadata(
        heif_context* ctx,
        const heif_image_handle* image_handle,
        const void* data,
        int size,
        const char* item_type,
        const char* content_type
    )

    heif_error heif_context_add_generic_uri_metadata(
        heif_context* ctx,
        const heif_image_handle* image_handle,
        const void* data, int size,
        const char* item_uri_type,
        heif_item_id* out_item_id
    )

    heif_error heif_image_create(
        int width,
        int height,
        heif_colorspace colorspace,
        heif_chroma chroma,
        heif_image** out_image
    )

    heif_error heif_image_add_plane(
        heif_image* image,
        heif_channel channel,
        int width,
        int height,
        int bit_depth
    )

    void heif_image_set_premultiplied_alpha(
        heif_image* image,
        int is_premultiplied_alpha
    )

    int heif_image_is_premultiplied_alpha(
        heif_image* image
    )

    struct heif_decoder_plugin:
        pass

    struct heif_encoder_plugin:
        pass

    heif_error heif_register_decoder(
        heif_context* heif,
        const heif_decoder_plugin*
    )

    heif_error heif_register_decoder_plugin(
        const heif_decoder_plugin*
    )

    heif_error heif_register_encoder_plugin(
        const heif_encoder_plugin*
    )

    int heif_encoder_descriptor_supportes_lossy_compression(
        const heif_encoder_descriptor*
    )

    int heif_encoder_descriptor_supportes_lossless_compression(
        const heif_encoder_descriptor*
    )


cdef extern from 'libheif/heif_regions.h' nogil:

    struct heif_region_item:
        pass

    enum heif_region_type:
        heif_region_type_point
        heif_region_type_rectangle
        heif_region_type_ellipse
        heif_region_type_polygon
        heif_region_type_referenced_mask
        heif_region_type_inline_mask
        heif_region_type_polyline

    struct heif_region:
        pass

    int heif_image_handle_get_number_of_region_items(
        const heif_image_handle* image_handle
    )

    int heif_image_handle_get_list_of_region_item_ids(
        const heif_image_handle* image_handle,
        heif_item_id* region_item_ids_array,
        int max_count
    )

    heif_error heif_context_get_region_item(
        const heif_context* context,
        heif_item_id region_item_id_,
        heif_region_item** out
    )

    heif_item_id heif_region_item_get_id(
        heif_region_item*
    )

    void heif_region_item_release(
        heif_region_item*
    )

    void heif_region_item_get_reference_size(
        heif_region_item*,
        uint32_t* width,
        uint32_t* height
    )

    int heif_region_item_get_number_of_regions(
        const heif_region_item* region_item
    )

    int heif_region_item_get_list_of_regions(
        const heif_region_item* region_item,
        heif_region** out_regions_array,
        int max_count
    )

    void heif_region_release(
        const heif_region* region
    )

    void heif_region_release_many(
        const heif_region* const* regions_array,
        int num
    )

    heif_region_type heif_region_get_type(
        const heif_region* region
    )

    heif_error heif_region_get_point(
        const heif_region* region,
        int32_t* x,
        int32_t* y
    )

    heif_error heif_region_get_point_transformed(
        const heif_region* region,
        double* x,
        double* y,
        heif_item_id image_id
    )

    heif_error heif_region_get_rectangle(
        const heif_region* region,
        int32_t* x,
        int32_t* y,
        uint32_t* width,
        uint32_t* height
    )

    heif_error heif_region_get_rectangle_transformed(
        const heif_region* region,
        double* x,
        double* y,
        double* width,
        double* height,
        heif_item_id image_id
    )

    heif_error heif_region_get_ellipse(
        const heif_region* region,
        int32_t* x,
        int32_t* y,
        uint32_t* radius_x,
        uint32_t* radius_y
    )

    heif_error heif_region_get_ellipse_transformed(
        const heif_region* region,
        double* x,
        double* y,
        double* radius_x,
        double* radius_y,
        heif_item_id image_id
    )

    int heif_region_get_polygon_num_points(
        const heif_region* region
    )

    heif_error heif_region_get_polygon_points(
        const heif_region* region,
        int32_t* out_pts_array
    )

    heif_error heif_region_get_polygon_points_transformed(
        const heif_region* region,
        double* out_pts_array,
        heif_item_id image_id
    )

    int heif_region_get_polyline_num_points(
        const heif_region* region
    )

    heif_error heif_region_get_polyline_points(
        const heif_region* region,
        int32_t* out_pts_array
    )

    heif_error heif_region_get_polyline_points_transformed(
        const heif_region* region,
        double* out_pts_array,
        heif_item_id image_id
    )

    heif_error heif_image_handle_add_region_item(
        heif_image_handle* image_handle,
        uint32_t reference_width,
        uint32_t reference_height,
        heif_region_item** out_region_item
    )

    heif_error heif_region_item_add_region_point(
        heif_region_item*,
        int32_t x,
        int32_t y,
        heif_region** out_region
    )

    heif_error heif_region_item_add_region_rectangle(
        heif_region_item*,
        int32_t x,
        int32_t y,
        uint32_t width,
        uint32_t height,
        heif_region** out_region
    )

    heif_error heif_region_item_add_region_ellipse(
        heif_region_item*,
        int32_t x,
        int32_t y,
        uint32_t radius_x,
        uint32_t radius_y,
        heif_region** out_region
    )

    heif_error heif_region_item_add_region_polygon(
        heif_region_item*,
        const int32_t* pts_array,
        int nPoints,
        heif_region** out_region
    )

    heif_error heif_region_item_add_region_polyline(
        heif_region_item*,
        const int32_t* pts_array,
        int nPoints,
        heif_region** out_region
    )


cdef extern from 'libheif/heif_properties.h' nogil:

    enum heif_item_property_type:
        # heif_item_property_unknown
        heif_item_property_type_invalid
        heif_item_property_type_user_description
        heif_item_property_type_transform_mirror
        heif_item_property_type_transform_rotation
        heif_item_property_type_transform_crop
        heif_item_property_type_image_size

    int heif_item_get_properties_of_type(
        const heif_context* context,
        heif_item_id id_,
        heif_item_property_type type_,
        heif_property_id* out_list,
        int count
    )

    int heif_item_get_transformation_properties(
        const heif_context* context,
        heif_item_id id_,
        heif_property_id* out_list,
        int count
    )

    heif_item_property_type heif_item_get_property_type(
        const heif_context* context,
        heif_item_id id_,
        heif_property_id property_id
    )

    struct heif_property_user_description:
        int version
        const char* lang
        const char* name
        const char* description
        const char* tags

    heif_error heif_item_get_property_user_description(
        const heif_context* context,
        heif_item_id itemId,
        heif_property_id propertyId,
        heif_property_user_description** out
    )

    heif_error heif_item_add_property_user_description(
        const heif_context* context,
        heif_item_id itemId,
        const heif_property_user_description* description,
        heif_property_id* out_propertyId
    )

    void heif_property_user_description_release(
        heif_property_user_description*
    )

    enum heif_transform_mirror_direction:
        heif_transform_mirror_direction_vertical
        heif_transform_mirror_direction_horizontal

    heif_transform_mirror_direction heif_item_get_property_transform_mirror(
        const heif_context* context,
        heif_item_id itemId,
        heif_property_id propertyId
    )

    int heif_item_get_property_transform_rotation_ccw(
        const heif_context* context,
        heif_item_id itemId,
        heif_property_id propertyId
    )

    void heif_item_get_property_transform_crop_borders(
        const heif_context* context,
        heif_item_id itemId,
        heif_property_id propertyId,
        int image_width,
        int image_height,
        int* left,
        int* top,
        int* right,
        int* bottom
    )
