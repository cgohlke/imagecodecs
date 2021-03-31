# imagecodecs/charls.pxd
# cython: language_level = 3

# Cython declarations for the `CharLS 2.1.0` library.
# https://github.com/team-charls/charls

from libc.stdint cimport int32_t, uint32_t

cdef extern from 'charls/charls.h':

    int CHARLS_VERSION_MAJOR
    int CHARLS_VERSION_MINOR
    int CHARLS_VERSION_PATCH

    ctypedef enum charls_jpegls_errc:
        CHARLS_JPEGLS_ERRC_SUCCESS
        CHARLS_JPEGLS_ERRC_INVALID_ARGUMENT
        CHARLS_JPEGLS_ERRC_PARAMETER_VALUE_NOT_SUPPORTED
        CHARLS_JPEGLS_ERRC_DESTINATION_BUFFER_TOO_SMALL
        CHARLS_JPEGLS_ERRC_SOURCE_BUFFER_TOO_SMALL
        CHARLS_JPEGLS_ERRC_INVALID_ENCODED_DATA
        CHARLS_JPEGLS_ERRC_TOO_MUCH_ENCODED_DATA
        CHARLS_JPEGLS_ERRC_INVALID_OPERATION
        CHARLS_JPEGLS_ERRC_BIT_DEPTH_FOR_TRANSFORM_NOT_SUPPORTED
        CHARLS_JPEGLS_ERRC_COLOR_TRANSFORM_NOT_SUPPORTED
        CHARLS_JPEGLS_ERRC_ENCODING_NOT_SUPPORTED
        CHARLS_JPEGLS_ERRC_UNKNOWN_JPEG_MARKER_FOUND
        CHARLS_JPEGLS_ERRC_JPEG_MARKER_START_BYTE_NOT_FOUND
        CHARLS_JPEGLS_ERRC_NOT_ENOUGH_MEMORY
        CHARLS_JPEGLS_ERRC_UNEXPECTED_FAILURE
        CHARLS_JPEGLS_ERRC_START_OF_IMAGE_MARKER_NOT_FOUND
        CHARLS_JPEGLS_ERRC_START_OF_FRAME_MARKER_NOT_FOUND
        CHARLS_JPEGLS_ERRC_INVALID_MARKER_SEGMENT_SIZE
        CHARLS_JPEGLS_ERRC_DUPLICATE_START_OF_IMAGE_MARKER
        CHARLS_JPEGLS_ERRC_DUPLICATE_START_OF_FRAME_MARKER
        CHARLS_JPEGLS_ERRC_DUPLICATE_COMPONENT_ID_IN_SOF_SEGMENT
        CHARLS_JPEGLS_ERRC_UNEXPECTED_END_OF_IMAGE_MARKER
        CHARLS_JPEGLS_ERRC_INVALID_JPEGLS_PRESET_PARAMETER_TYPE
        CHARLS_JPEGLS_ERRC_JPEGLS_PRESET_EXTENDED_PARAMETER_TYPE_NOT_SUPPORTED
        CHARLS_JPEGLS_ERRC_MISSING_END_OF_SPIFF_DIRECTORY
        CHARLS_JPEGLS_ERRC_INVALID_ARGUMENT_WIDTH
        CHARLS_JPEGLS_ERRC_INVALID_ARGUMENT_HEIGHT
        CHARLS_JPEGLS_ERRC_INVALID_ARGUMENT_COMPONENT_COUNT
        CHARLS_JPEGLS_ERRC_INVALID_ARGUMENT_BITS_PER_SAMPLE
        CHARLS_JPEGLS_ERRC_INVALID_ARGUMENT_INTERLEAVE_MODE
        CHARLS_JPEGLS_ERRC_INVALID_ARGUMENT_NEAR_LOSSLESS
        CHARLS_JPEGLS_ERRC_INVALID_ARGUMENT_PC_PARAMETERS
        CHARLS_JPEGLS_ERRC_INVALID_ARGUMENT_SPIFF_ENTRY_SIZE
        CHARLS_JPEGLS_ERRC_INVALID_ARGUMENT_COLOR_TRANSFORMATION
        CHARLS_JPEGLS_ERRC_INVALID_PARAMETER_WIDTH
        CHARLS_JPEGLS_ERRC_INVALID_PARAMETER_HEIGHT
        CHARLS_JPEGLS_ERRC_INVALID_PARAMETER_COMPONENT_COUNT
        CHARLS_JPEGLS_ERRC_INVALID_PARAMETER_BITS_PER_SAMPLE
        CHARLS_JPEGLS_ERRC_INVALID_PARAMETER_INTERLEAVE_MODE

    ctypedef enum charls_interleave_mode:
        CHARLS_INTERLEAVE_MODE_NONE
        CHARLS_INTERLEAVE_MODE_LINE
        CHARLS_INTERLEAVE_MODE_SAMPLE

    ctypedef enum charls_color_transformation:
        CHARLS_COLOR_TRANSFORMATION_NONE
        CHARLS_COLOR_TRANSFORMATION_HP1
        CHARLS_COLOR_TRANSFORMATION_HP2
        CHARLS_COLOR_TRANSFORMATION_HP3

    ctypedef enum charls_spiff_profile_id:
        CHARLS_SPIFF_PROFILE_ID_NONE
        CHARLS_SPIFF_PROFILE_ID_CONTINUOUS_TONE_BASE
        CHARLS_SPIFF_PROFILE_ID_CONTINUOUS_TONE_PROGRESSIVE
        CHARLS_SPIFF_PROFILE_ID_BI_LEVEL_FACSIMILE
        CHARLS_SPIFF_PROFILE_ID_CONTINUOUS_TONE_FACSIMILE

    ctypedef enum charls_spiff_color_space:
        CHARLS_SPIFF_COLOR_SPACE_BI_LEVEL_BLACK
        CHARLS_SPIFF_COLOR_SPACE_YCBCR_ITU_BT_709_VIDEO
        CHARLS_SPIFF_COLOR_SPACE_NONE
        CHARLS_SPIFF_COLOR_SPACE_YCBCR_ITU_BT_601_1_RGB
        CHARLS_SPIFF_COLOR_SPACE_YCBCR_ITU_BT_601_1_VIDEO
        CHARLS_SPIFF_COLOR_SPACE_GRAYSCALE
        CHARLS_SPIFF_COLOR_SPACE_PHOTO_YCC
        CHARLS_SPIFF_COLOR_SPACE_RGB
        CHARLS_SPIFF_COLOR_SPACE_CMY
        CHARLS_SPIFF_COLOR_SPACE_CMYK
        CHARLS_SPIFF_COLOR_SPACE_YCCK
        CHARLS_SPIFF_COLOR_SPACE_CIE_LAB
        CHARLS_SPIFF_COLOR_SPACE_BI_LEVEL_WHITE

    ctypedef enum charls_spiff_compression_type:
        CHARLS_SPIFF_COMPRESSION_TYPE_UNCOMPRESSED
        CHARLS_SPIFF_COMPRESSION_TYPE_MODIFIED_HUFFMAN
        CHARLS_SPIFF_COMPRESSION_TYPE_MODIFIED_READ
        CHARLS_SPIFF_COMPRESSION_TYPE_MODIFIED_MODIFIED_READ
        CHARLS_SPIFF_COMPRESSION_TYPE_JBIG
        CHARLS_SPIFF_COMPRESSION_TYPE_JPEG
        CHARLS_SPIFF_COMPRESSION_TYPE_JPEG_LS

    ctypedef enum charls_spiff_resolution_units:
        CHARLS_SPIFF_RESOLUTION_UNITS_ASPECT_RATIO
        CHARLS_SPIFF_RESOLUTION_UNITS_DOTS_PER_INCH
        CHARLS_SPIFF_RESOLUTION_UNITS_DOTS_PER_CENTIMETER

    ctypedef enum charls_spiff_entry_tag:
        CHARLS_SPIFF_ENTRY_TAG_TRANSFER_CHARACTERISTICS
        CHARLS_SPIFF_ENTRY_TAG_COMPONENT_REGISTRATION
        CHARLS_SPIFF_ENTRY_TAG_IMAGE_ORIENTATION
        CHARLS_SPIFF_ENTRY_TAG_THUMBNAIL
        CHARLS_SPIFF_ENTRY_TAG_IMAGE_TITLE
        CHARLS_SPIFF_ENTRY_TAG_IMAGE_DESCRIPTION
        CHARLS_SPIFF_ENTRY_TAG_TIME_STAMP
        CHARLS_SPIFF_ENTRY_TAG_VERSION_IDENTIFIER
        CHARLS_SPIFF_ENTRY_TAG_CREATOR_IDENTIFICATION
        CHARLS_SPIFF_ENTRY_TAG_PROTECTION_INDICATOR
        CHARLS_SPIFF_ENTRY_TAG_COPYRIGHT_INFORMATION
        CHARLS_SPIFF_ENTRY_TAG_CONTACT_INFORMATION
        CHARLS_SPIFF_ENTRY_TAG_TILE_INDEX
        CHARLS_SPIFF_ENTRY_TAG_SCAN_INDEX
        CHARLS_SPIFF_ENTRY_TAG_SET_REFERENCE

    struct charls_jpegls_decoder:
        pass

    struct charls_jpegls_encoder:
        pass

    struct charls_spiff_header:
        charls_spiff_profile_id profile_id
        int32_t component_count
        uint32_t height
        uint32_t width
        charls_spiff_color_space color_space
        int32_t bits_per_sample
        charls_spiff_compression_type compression_type
        charls_spiff_resolution_units resolution_units
        uint32_t vertical_resolution
        uint32_t horizontal_resolution

    struct charls_jpegls_pc_parameters:
        int32_t maximum_sample_value
        int32_t threshold1
        int32_t threshold2
        int32_t threshold3
        int32_t reset_value

    struct charls_frame_info:
        uint32_t width
        uint32_t height
        int32_t bits_per_sample
        int32_t component_count

    const void* charls_get_jpegls_category() nogil

    const char* charls_get_error_message(
        charls_jpegls_errc error_value
    ) nogil

    charls_jpegls_decoder* charls_jpegls_decoder_create() nogil

    void charls_jpegls_decoder_destroy(
        const charls_jpegls_decoder* decoder
    ) nogil

    charls_jpegls_errc charls_jpegls_decoder_set_source_buffer(
        charls_jpegls_decoder* decoder,
        const void* source_buffer,
        size_t source_size_bytes
    ) nogil

    charls_jpegls_errc charls_jpegls_decoder_read_spiff_header(
        charls_jpegls_decoder* decoder,
        charls_spiff_header* spiff_header,
        int32_t* header_found
    ) nogil

    charls_jpegls_errc charls_jpegls_decoder_read_header(
        charls_jpegls_decoder* decoder
    ) nogil

    charls_jpegls_errc charls_jpegls_decoder_get_frame_info(
        const charls_jpegls_decoder* decoder,
        charls_frame_info* frame_info
    ) nogil

    charls_jpegls_errc charls_jpegls_decoder_get_near_lossless(
        const charls_jpegls_decoder* decoder,
        int32_t component,
        int32_t* near_lossless
    ) nogil

    charls_jpegls_errc charls_jpegls_decoder_get_interleave_mode(
        const charls_jpegls_decoder* decoder,
        charls_interleave_mode* interleave_mode
    ) nogil

    charls_jpegls_errc charls_jpegls_decoder_get_preset_coding_parameters(
        const charls_jpegls_decoder* decoder,
        int32_t reserved,
        charls_jpegls_pc_parameters* preset_coding_parameters
    ) nogil

    charls_jpegls_errc charls_jpegls_decoder_get_destination_size(
        const charls_jpegls_decoder* decoder,
        size_t* destination_size_bytes
    ) nogil

    charls_jpegls_errc charls_jpegls_decoder_decode_to_buffer(
        const charls_jpegls_decoder* decoder,
        void* destination_buffer,
        size_t destination_size_bytes,
        uint32_t stride
    ) nogil

    charls_jpegls_encoder* charls_jpegls_encoder_create() nogil

    void charls_jpegls_encoder_destroy(
        const charls_jpegls_encoder* encoder
    ) nogil

    charls_jpegls_errc charls_jpegls_encoder_set_frame_info(
        charls_jpegls_encoder* encoder,
        const charls_frame_info* frame_info
    ) nogil

    charls_jpegls_errc charls_jpegls_encoder_set_near_lossless(
        charls_jpegls_encoder* encoder,
        int32_t near_lossless
    ) nogil

    charls_jpegls_errc charls_jpegls_encoder_set_interleave_mode(
        charls_jpegls_encoder* encoder,
        charls_interleave_mode interleave_mode
    ) nogil

    charls_jpegls_errc charls_jpegls_encoder_set_preset_coding_parameters(
        charls_jpegls_encoder* encoder,
        const charls_jpegls_pc_parameters* preset_coding_parameters
    ) nogil

    charls_jpegls_errc charls_jpegls_encoder_set_color_transformation(
        charls_jpegls_encoder* encoder,
        charls_color_transformation color_transformation
    ) nogil

    charls_jpegls_errc charls_jpegls_encoder_get_estimated_destination_size(
        const charls_jpegls_encoder* encoder,
        size_t* size_in_bytes
    ) nogil

    charls_jpegls_errc charls_jpegls_encoder_set_destination_buffer(
        charls_jpegls_encoder* encoder,
        void* destination_buffer,
        size_t destination_size
    ) nogil

    charls_jpegls_errc charls_jpegls_encoder_write_standard_spiff_header(
        charls_jpegls_encoder* encoder,
        charls_spiff_color_space color_space,
        charls_spiff_resolution_units resolution_units,
        uint32_t vertical_resolution,
        uint32_t horizontal_resolution
    ) nogil

    charls_jpegls_errc charls_jpegls_encoder_write_spiff_header(
        charls_jpegls_encoder* encoder,
        const charls_spiff_header* spiff_header
    ) nogil

    charls_jpegls_errc charls_jpegls_encoder_write_spiff_entry(
        charls_jpegls_encoder* encoder,
        uint32_t entry_tag,
        const void* entry_data,
        size_t entry_data_size
    ) nogil

    charls_jpegls_errc charls_jpegls_encoder_encode_from_buffer(
        charls_jpegls_encoder* encoder,
        const void* source_buffer,
        size_t source_size,
        uint32_t stride
    ) nogil

    charls_jpegls_errc charls_jpegls_encoder_get_bytes_written(
        const charls_jpegls_encoder* encoder,
        size_t* bytes_written
    ) nogil
