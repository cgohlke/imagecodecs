# imagecodecs/libjxl.pxd
# cython: language_level = 3

# Cython declarations for the `libjxl 0.11.1` library.
# https://github.com/libjxl/libjxl

from libc.stdint cimport (
    uint8_t, uint16_t, uint32_t, uint64_t, int32_t, int64_t
)

cdef extern from 'jxl/types.h' nogil:

    ctypedef int JXL_BOOL

    int JXL_TRUE
    int JXL_FALSE
    int TO_JXL_BOOL(int)

    ctypedef enum JxlDataType:
        JXL_TYPE_FLOAT
        JXL_TYPE_UINT8
        JXL_TYPE_UINT16
        JXL_TYPE_FLOAT16

    ctypedef enum JxlEndianness:
        JXL_NATIVE_ENDIAN
        JXL_LITTLE_ENDIAN
        JXL_BIG_ENDIAN

    ctypedef struct JxlPixelFormat:
        uint32_t num_channels
        JxlDataType data_type
        JxlEndianness endianness
        size_t align

    ctypedef enum JxlBitDepthType:
        JXL_BIT_DEPTH_FROM_PIXEL_FORMAT
        JXL_BIT_DEPTH_FROM_CODESTREAM
        JXL_BIT_DEPTH_CUSTOM

    ctypedef struct JxlBitDepth:
        JxlBitDepthType dtype 'type'
        uint32_t bits_per_sample
        uint32_t exponent_bits_per_sample

    ctypedef char[4] JxlBoxType


cdef extern from 'jxl/codestream_header.h' nogil:

    ctypedef enum JxlOrientation:
        JXL_ORIENT_IDENTITY
        JXL_ORIENT_FLIP_HORIZONTAL
        JXL_ORIENT_ROTATE_180
        JXL_ORIENT_FLIP_VERTICAL
        JXL_ORIENT_TRANSPOSE
        JXL_ORIENT_ROTATE_90_CW
        JXL_ORIENT_ANTI_TRANSPOSE
        JXL_ORIENT_ROTATE_90_CCW

    ctypedef enum JxlExtraChannelType:
        JXL_CHANNEL_ALPHA
        JXL_CHANNEL_DEPTH
        JXL_CHANNEL_SPOT_COLOR
        JXL_CHANNEL_SELECTION_MASK
        JXL_CHANNEL_BLACK
        JXL_CHANNEL_CFA
        JXL_CHANNEL_THERMAL
        JXL_CHANNEL_RESERVED0
        JXL_CHANNEL_RESERVED1
        JXL_CHANNEL_RESERVED2
        JXL_CHANNEL_RESERVED3
        JXL_CHANNEL_RESERVED4
        JXL_CHANNEL_RESERVED5
        JXL_CHANNEL_RESERVED6
        JXL_CHANNEL_RESERVED7
        JXL_CHANNEL_UNKNOWN
        JXL_CHANNEL_OPTIONAL

    ctypedef struct JxlPreviewHeader:
        uint32_t xsize
        uint32_t ysize

    ctypedef struct JxlAnimationHeader:
        uint32_t tps_numerator
        uint32_t tps_denominator
        uint32_t num_loops
        JXL_BOOL have_timecodes

    ctypedef struct JxlBasicInfo:
        JXL_BOOL have_container
        uint32_t xsize
        uint32_t ysize
        uint32_t bits_per_sample
        uint32_t exponent_bits_per_sample
        float intensity_target
        float min_nits
        JXL_BOOL relative_to_max_display
        float linear_below
        JXL_BOOL uses_original_profile
        JXL_BOOL have_preview
        JXL_BOOL have_animation
        JxlOrientation orientation
        uint32_t num_color_channels
        uint32_t num_extra_channels
        uint32_t alpha_bits
        uint32_t alpha_exponent_bits
        JXL_BOOL alpha_premultiplied
        JxlPreviewHeader preview
        JxlAnimationHeader animation
        uint32_t intrinsic_xsize
        uint32_t intrinsic_ysize
        # uint8_t[100] padding

    ctypedef struct JxlExtraChannelInfo:
        JxlExtraChannelType type
        uint32_t bits_per_sample
        uint32_t exponent_bits_per_sample
        uint32_t dim_shift
        uint32_t name_length
        JXL_BOOL alpha_premultiplied
        float[4] spot_color
        uint32_t cfa_channel

    ctypedef struct JxlHeaderExtensions:
        uint64_t extensions

    ctypedef enum JxlBlendMode:
        JXL_BLEND_REPLACE
        JXL_BLEND_ADD
        JXL_BLEND_BLEND
        JXL_BLEND_MULADD
        JXL_BLEND_MUL

    ctypedef struct JxlBlendInfo:
        JxlBlendMode blendmode
        uint32_t source
        uint32_t alpha
        JXL_BOOL clamp

    ctypedef struct JxlLayerInfo:
        JXL_BOOL have_crop
        int32_t crop_x0
        int32_t crop_y0
        uint32_t xsize
        uint32_t ysize
        JxlBlendInfo blend_info
        uint32_t save_as_reference

    ctypedef struct JxlFrameHeader:
        uint32_t duration
        uint32_t timecode
        uint32_t name_length
        JXL_BOOL is_last
        JxlLayerInfo layer_info


cdef extern from 'jxl/color_encoding.h' nogil:

    ctypedef enum JxlColorSpace:
        JXL_COLOR_SPACE_RGB
        JXL_COLOR_SPACE_GRAY
        JXL_COLOR_SPACE_XYB
        JXL_COLOR_SPACE_UNKNOWN

    ctypedef enum JxlWhitePoint:
        JXL_WHITE_POINT_D65
        JXL_WHITE_POINT_CUSTOM
        JXL_WHITE_POINT_E
        JXL_WHITE_POINT_DCI

    ctypedef enum JxlPrimaries:
        JXL_PRIMARIES_SRGB
        JXL_PRIMARIES_CUSTOM
        JXL_PRIMARIES_2100
        JXL_PRIMARIES_P3

    ctypedef enum JxlTransferFunction:
        JXL_TRANSFER_FUNCTION_709
        JXL_TRANSFER_FUNCTION_UNKNOWN
        JXL_TRANSFER_FUNCTION_LINEAR
        JXL_TRANSFER_FUNCTION_SRGB
        JXL_TRANSFER_FUNCTION_PQ
        JXL_TRANSFER_FUNCTION_DCI
        JXL_TRANSFER_FUNCTION_HLG
        JXL_TRANSFER_FUNCTION_GAMMA

    ctypedef enum JxlRenderingIntent:
        JXL_RENDERING_INTENT_PERCEPTUAL
        JXL_RENDERING_INTENT_RELATIVE
        JXL_RENDERING_INTENT_SATURATION
        JXL_RENDERING_INTENT_ABSOLUTE

    ctypedef struct JxlColorEncoding:
        JxlColorSpace color_space
        JxlWhitePoint white_point
        double[2] white_point_xy
        JxlPrimaries primaries
        double[2] primaries_red_xy
        double[2] primaries_green_xy
        double[2] primaries_blue_xy
        JxlTransferFunction transfer_function
        double gamma
        JxlRenderingIntent rendering_intent

    ctypedef struct JxlInverseOpsinMatrix:
        float[3][3] opsin_inv_matrix
        float[3] opsin_biases
        float[3] quant_biases

    ctypedef struct JxlInverseOpsinMatrix:
        float[3][3] opsin_inv_matrix
        float[3] opsin_biases
        float[3] quant_biases


cdef extern from 'jxl/memory_manager.h' nogil:

    ctypedef void* (*jpegxl_alloc_func)(
        void* opaque,
        size_t size
    ) nogil

    ctypedef void (*jpegxl_free_func)(
        void* opaque,
        void* address
    ) nogil

    ctypedef struct JxlMemoryManager:
        void* opaque
        jpegxl_alloc_func alloc
        jpegxl_free_func free


cdef extern from 'jxl/parallel_runner.h' nogil:

    ctypedef int JxlParallelRetCode

    int JXL_PARALLEL_RET_RUNNER_ERROR

    ctypedef JxlParallelRetCode (*JxlParallelRunInit)(
        void* jpegxl_opaque,
        size_t num_threads
    ) nogil

    ctypedef void (*JxlParallelRunFunction)(
        void* jpegxl_opaque,
        uint32_t value,
        size_t thread_id
    ) nogil

    ctypedef JxlParallelRetCode (*JxlParallelRunner)(
        void* runner_opaque,
        void* jpegxl_opaque,
        JxlParallelRunInit init,
        JxlParallelRunFunction func,
        uint32_t start_range,
        uint32_t end_range
    ) nogil


cdef extern from 'jxl/thread_parallel_runner.h' nogil:

    JxlParallelRetCode JxlThreadParallelRunner(
        void* runner_opaque,
        void* jpegxl_opaque,
        JxlParallelRunInit init,
        JxlParallelRunFunction func,
        uint32_t start_range,
        uint32_t end_range
    )

    void* JxlThreadParallelRunnerCreate(
        const JxlMemoryManager* memory_manager,
        size_t num_worker_threads
    )

    void JxlThreadParallelRunnerDestroy(
        void* runner_opaque
    )

    size_t JxlThreadParallelRunnerDefaultNumWorkerThreads()


cdef extern from 'jxl/cms_interface.h' nogil:

    ctypedef JXL_BOOL (*jpegxl_cms_set_fields_from_icc_func)(
        void* user_data,
        const uint8_t* icc_data,
        size_t icc_size,
        JxlColorEncoding* c,
        JXL_BOOL* cmyk
    ) nogil

    ctypedef struct icc_t:
        const uint8_t* data
        size_t size

    ctypedef struct JxlColorProfile:
        icc_t icc
        JxlColorEncoding color_encoding
        size_t num_channels

    ctypedef void* (*jpegxl_cms_init_func)(
        void* init_data,
        size_t num_threads,
        size_t pixels_per_thread,
        const JxlColorProfile* input_profile,
        const JxlColorProfile* output_profile,
        float intensity_target
    ) nogil

    ctypedef float* (*jpegxl_cms_get_buffer_func)(
        void* user_data,
        size_t thread
    ) nogil

    ctypedef JXL_BOOL (*jpegxl_cms_run_func)(
        void* user_data, size_t thread,
        const float* input_buffer,
        float* output_buffer,
        size_t num_pixels
    ) nogil

    ctypedef void (*jpegxl_cms_destroy_func)(
        void*
    ) nogil

    ctypedef struct JxlCmsInterface:
        void* set_fields_data
        jpegxl_cms_set_fields_from_icc_func set_fields_from_icc
        void* init_data
        jpegxl_cms_init_func init
        jpegxl_cms_get_buffer_func get_src_buf
        jpegxl_cms_get_buffer_func get_dst_buf
        jpegxl_cms_run_func run
        jpegxl_cms_destroy_func destroy


cdef extern from 'jxl/stats.h' nogil:

    ctypedef struct JxlEncoderStats:
        pass

    JxlEncoderStats* JxlEncoderStatsCreate()

    void JxlEncoderStatsDestroy(
        JxlEncoderStats* stats
    )

    ctypedef enum JxlEncoderStatsKey:
        JXL_ENC_STAT_HEADER_BITS
        JXL_ENC_STAT_TOC_BITS
        JXL_ENC_STAT_DICTIONARY_BITS
        JXL_ENC_STAT_SPLINES_BITS
        JXL_ENC_STAT_NOISE_BITS
        JXL_ENC_STAT_QUANT_BITS
        JXL_ENC_STAT_MODULAR_TREE_BITS
        JXL_ENC_STAT_MODULAR_GLOBAL_BITS
        JXL_ENC_STAT_DC_BITS
        JXL_ENC_STAT_MODULAR_DC_GROUP_BITS
        JXL_ENC_STAT_CONTROL_FIELDS_BITS
        JXL_ENC_STAT_COEF_ORDER_BITS
        JXL_ENC_STAT_AC_HISTOGRAM_BITS
        JXL_ENC_STAT_AC_BITS
        JXL_ENC_STAT_MODULAR_AC_GROUP_BITS
        JXL_ENC_STAT_NUM_SMALL_BLOCKS
        JXL_ENC_STAT_NUM_DCT4X8_BLOCKS
        JXL_ENC_STAT_NUM_AFV_BLOCKS
        JXL_ENC_STAT_NUM_DCT8_BLOCKS
        JXL_ENC_STAT_NUM_DCT8X32_BLOCKS
        JXL_ENC_STAT_NUM_DCT16_BLOCKS
        JXL_ENC_STAT_NUM_DCT16X32_BLOCKS
        JXL_ENC_STAT_NUM_DCT32_BLOCKS
        JXL_ENC_STAT_NUM_DCT32X64_BLOCKS
        JXL_ENC_STAT_NUM_DCT64_BLOCKS
        JXL_ENC_STAT_NUM_BUTTERAUGLI_ITERS
        JXL_ENC_NUM_STATS

    size_t JxlEncoderStatsGet(
        const JxlEncoderStats* stats,
        JxlEncoderStatsKey key
    )

    void JxlEncoderStatsMerge(
        JxlEncoderStats* stats,
        const JxlEncoderStats* other
    )


cdef extern from 'jxl/decode.h' nogil:

    uint32_t JxlDecoderVersion()

    ctypedef enum JxlSignature:
        JXL_SIG_NOT_ENOUGH_BYTES
        JXL_SIG_INVALID
        JXL_SIG_CODESTREAM
        JXL_SIG_CONTAINER

    JxlSignature JxlSignatureCheck(
        const uint8_t* buf,
        size_t len
    )

    ctypedef struct JxlDecoder:
        pass

    JxlDecoder* JxlDecoderCreate(
        const JxlMemoryManager* memory_manager
    )

    void JxlDecoderReset(
        JxlDecoder* dec
    )

    void JxlDecoderDestroy(
        JxlDecoder* dec
    )

    ctypedef enum JxlDecoderStatus:
        JXL_DEC_SUCCESS
        JXL_DEC_ERROR
        JXL_DEC_NEED_MORE_INPUT
        JXL_DEC_NEED_PREVIEW_OUT_BUFFER
        JXL_DEC_NEED_IMAGE_OUT_BUFFER
        JXL_DEC_JPEG_NEED_MORE_OUTPUT
        JXL_DEC_BASIC_INFO
        JXL_DEC_COLOR_ENCODING
        JXL_DEC_PREVIEW_IMAGE
        JXL_DEC_FRAME
        JXL_DEC_FULL_IMAGE
        JXL_DEC_JPEG_RECONSTRUCTION
        JXL_DEC_BOX
        JXL_DEC_FRAME_PROGRESSION
        JXL_DEC_BOX_COMPLETE

    ctypedef enum JxlProgressiveDetail:
        kFrames
        kDC
        kLastPasses
        kPasses
        kDCProgressive
        kDCGroups
        kGroups

    void JxlDecoderRewind(
        JxlDecoder* dec
    )

    void JxlDecoderSkipFrames(
        JxlDecoder* dec,
        size_t amount
    )

    JxlDecoderStatus JxlDecoderSkipCurrentFrame(
        JxlDecoder* dec
    )

    JxlDecoderStatus JxlDecoderSetParallelRunner(
        JxlDecoder* dec,
        JxlParallelRunner parallel_runner,
        void* parallel_runner_opaque
    )

    size_t JxlDecoderSizeHintBasicInfo(
        const JxlDecoder* dec
    )

    JxlDecoderStatus JxlDecoderSubscribeEvents(
        JxlDecoder* dec,
        int events_wanted
    )

    JxlDecoderStatus JxlDecoderSetKeepOrientation(
        JxlDecoder* dec,
        JXL_BOOL keep_orientation
    )

    JxlDecoderStatus JxlDecoderSetUnpremultiplyAlpha(
        JxlDecoder* dec,
        JXL_BOOL unpremul_alpha
    )

    JxlDecoderStatus JxlDecoderSetRenderSpotcolors(
        JxlDecoder* dec,
        JXL_BOOL render_spotcolors
    )

    JxlDecoderStatus JxlDecoderSetCoalescing(
        JxlDecoder* dec,
        JXL_BOOL coalescing
    )

    JxlDecoderStatus JxlDecoderProcessInput(
        JxlDecoder* dec
    )

    JxlDecoderStatus JxlDecoderSetInput(
        JxlDecoder* dec,
        const uint8_t* data,
        size_t size
    )

    size_t JxlDecoderReleaseInput(
        JxlDecoder* dec
    )

    void JxlDecoderCloseInput(
        JxlDecoder* dec
    )

    JxlDecoderStatus JxlDecoderGetBasicInfo(
        const JxlDecoder* dec,
        JxlBasicInfo* info
    )

    JxlDecoderStatus JxlDecoderGetExtraChannelInfo(
        const JxlDecoder* dec,
        size_t index,
        JxlExtraChannelInfo* info
    )

    JxlDecoderStatus JxlDecoderGetExtraChannelName(
        const JxlDecoder* dec,
        size_t index,
        char* name,
        size_t size
    )

    ctypedef enum JxlColorProfileTarget:
        JXL_COLOR_PROFILE_TARGET_ORIGINAL
        JXL_COLOR_PROFILE_TARGET_DATA

    JxlDecoderStatus JxlDecoderGetColorAsEncodedProfile(
        const JxlDecoder* dec,
        JxlColorProfileTarget target,
        JxlColorEncoding* color_encoding
    )

    JxlDecoderStatus JxlDecoderGetICCProfileSize(
        const JxlDecoder* dec,
        JxlColorProfileTarget target,
        size_t* size
    )

    JxlDecoderStatus JxlDecoderGetColorAsICCProfile(
        const JxlDecoder* dec,
        JxlColorProfileTarget target,
        uint8_t* icc_profile,
        size_t size
    )

    JxlDecoderStatus JxlDecoderSetPreferredColorProfile(
        JxlDecoder* dec,
        const JxlColorEncoding* color_encoding
    )

    JxlDecoderStatus JxlDecoderSetDesiredIntensityTarget(
        JxlDecoder* dec,
        float desired_intensity_target
    )

    JxlDecoderStatus JxlDecoderSetOutputColorProfile(
        JxlDecoder* dec,
        const JxlColorEncoding* color_encoding,
        const uint8_t* icc_data,
        size_t icc_size
    )

    JxlDecoderStatus JxlDecoderSetCms(
        JxlDecoder* dec,
        JxlCmsInterface cms
    )

    JxlDecoderStatus JxlDecoderPreviewOutBufferSize(
        const JxlDecoder* dec,
        const JxlPixelFormat* format,
        size_t* size
    )

    JxlDecoderStatus JxlDecoderSetPreviewOutBuffer(
        JxlDecoder* dec,
        const JxlPixelFormat* format,
        void* buffer,
        size_t size
    )

    JxlDecoderStatus JxlDecoderGetFrameHeader(
        const JxlDecoder* dec,
        JxlFrameHeader* header
    )

    JxlDecoderStatus JxlDecoderGetFrameName(
        const JxlDecoder* dec,
        char* name,
        size_t size
    )

    JxlDecoderStatus JxlDecoderGetExtraChannelBlendInfo(
        const JxlDecoder* dec,
        size_t index,
        JxlBlendInfo* blend_info
    )

    JxlDecoderStatus JxlDecoderImageOutBufferSize(
        const JxlDecoder* dec,
        const JxlPixelFormat* format,
        size_t* size
    )

    JxlDecoderStatus JxlDecoderSetImageOutBuffer(
        JxlDecoder* dec,
        const JxlPixelFormat* format,
        void* buffer,
        size_t size
    )

    ctypedef void (*JxlImageOutCallback)(
        void* opaque,
        size_t x,
        size_t y,
        size_t num_pixels,
        const void* pixels
    ) nogil

    ctypedef void* (*JxlImageOutInitCallback)(
        void* init_opaque,
        size_t num_threads,
        size_t num_pixels_per_thread
    ) nogil

    ctypedef void (*JxlImageOutRunCallback)(
        void* run_opaque,
        size_t thread_id,
        size_t x,
        size_t y,
        size_t num_pixels,
        const void* pixels
    ) nogil

    ctypedef void (*JxlImageOutDestroyCallback)(
        void* run_opaque
    ) nogil

    JxlDecoderStatus JxlDecoderSetImageOutCallback(
        JxlDecoder* dec,
        const JxlPixelFormat* format,
        JxlImageOutCallback callback,
        void* opaque
    ) nogil

    JxlDecoderStatus JxlDecoderSetMultithreadedImageOutCallback(
        JxlDecoder* dec,
        const JxlPixelFormat* format,
        JxlImageOutInitCallback init_callback,
        JxlImageOutRunCallback run_callback,
        JxlImageOutDestroyCallback destroy_callback,
        void* init_opaque
    ) nogil

    JxlDecoderStatus JxlDecoderExtraChannelBufferSize(
        const JxlDecoder* dec,
        const JxlPixelFormat* format,
        size_t* size,
        uint32_t index
    )

    JxlDecoderStatus JxlDecoderSetExtraChannelBuffer(
        JxlDecoder* dec,
        const JxlPixelFormat* format,
        void* buffer,
        size_t size,
        uint32_t index
    )

    JxlDecoderStatus JxlDecoderSetJPEGBuffer(
        JxlDecoder* dec,
        uint8_t* data,
        size_t size
    )

    size_t JxlDecoderReleaseJPEGBuffer(
        JxlDecoder* dec
    )

    JxlDecoderStatus JxlDecoderSetBoxBuffer(
        JxlDecoder* dec,
        uint8_t* data,
        size_t size
    )

    size_t JxlDecoderReleaseBoxBuffer(
        JxlDecoder* dec
    )

    JxlDecoderStatus JxlDecoderSetDecompressBoxes(
        JxlDecoder* dec,
        JXL_BOOL decompress
    )

    JxlDecoderStatus JxlDecoderGetBoxType(
        JxlDecoder* dec,
        JxlBoxType type,
        JXL_BOOL decompressed
    )

    JxlDecoderStatus JxlDecoderGetBoxSizeRaw(
        const JxlDecoder* dec,
        uint64_t* size
    )

    JxlDecoderStatus JxlDecoderGetBoxSizeContents(
        const JxlDecoder* dec,
        uint64_t* size
    )

    JxlDecoderStatus JxlDecoderSetProgressiveDetail(
        JxlDecoder* dec,
        JxlProgressiveDetail detail
    )

    JxlDecoderStatus JxlDecoderFlushImage(
        JxlDecoder* dec
    )

    JxlDecoderStatus JxlDecoderSetImageOutBitDepth(
        JxlDecoder* dec,
        const JxlBitDepth* bit_depth
    )


cdef extern from 'jxl/encode.h' nogil:

    uint32_t JxlEncoderVersion()

    ctypedef struct JxlEncoder:
        pass

    ctypedef struct JxlEncoderFrameSettings:
        pass

    ctypedef enum JxlEncoderStatus:
        JXL_ENC_SUCCESS
        JXL_ENC_ERROR
        JXL_ENC_NEED_MORE_OUTPUT

    ctypedef enum JxlEncoderError:
        JXL_ENC_ERR_OK
        JXL_ENC_ERR_GENERIC
        JXL_ENC_ERR_OOM
        JXL_ENC_ERR_JBRD
        JXL_ENC_ERR_BAD_INPUT
        JXL_ENC_ERR_NOT_SUPPORTED
        JXL_ENC_ERR_API_USAGE

    ctypedef enum JxlEncoderFrameSettingId:
        JXL_ENC_FRAME_SETTING_EFFORT
        JXL_ENC_FRAME_SETTING_DECODING_SPEED
        JXL_ENC_FRAME_SETTING_RESAMPLING
        JXL_ENC_FRAME_SETTING_EXTRA_CHANNEL_RESAMPLING
        JXL_ENC_FRAME_SETTING_ALREADY_DOWNSAMPLED
        JXL_ENC_FRAME_SETTING_PHOTON_NOISE
        JXL_ENC_FRAME_SETTING_NOISE
        JXL_ENC_FRAME_SETTING_DOTS
        JXL_ENC_FRAME_SETTING_PATCHES
        JXL_ENC_FRAME_SETTING_EPF
        JXL_ENC_FRAME_SETTING_GABORISH
        JXL_ENC_FRAME_SETTING_MODULAR
        JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE
        JXL_ENC_FRAME_SETTING_GROUP_ORDER
        JXL_ENC_FRAME_SETTING_GROUP_ORDER_CENTER_X
        JXL_ENC_FRAME_SETTING_GROUP_ORDER_CENTER_Y
        JXL_ENC_FRAME_SETTING_RESPONSIVE
        JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC
        JXL_ENC_FRAME_SETTING_QPROGRESSIVE_AC
        JXL_ENC_FRAME_SETTING_PROGRESSIVE_DC
        JXL_ENC_FRAME_SETTING_CHANNEL_COLORS_GLOBAL_PERCENT
        JXL_ENC_FRAME_SETTING_CHANNEL_COLORS_GROUP_PERCENT
        JXL_ENC_FRAME_SETTING_PALETTE_COLORS
        JXL_ENC_FRAME_SETTING_LOSSY_PALETTE
        JXL_ENC_FRAME_SETTING_COLOR_TRANSFORM
        JXL_ENC_FRAME_SETTING_MODULAR_COLOR_SPACE
        JXL_ENC_FRAME_SETTING_MODULAR_GROUP_SIZE
        JXL_ENC_FRAME_SETTING_MODULAR_PREDICTOR
        JXL_ENC_FRAME_SETTING_MODULAR_MA_TREE_LEARNING_PERCENT
        JXL_ENC_FRAME_SETTING_MODULAR_NB_PREV_CHANNELS
        JXL_ENC_FRAME_SETTING_JPEG_RECON_CFL
        JXL_ENC_FRAME_INDEX_BOX
        JXL_ENC_FRAME_SETTING_BROTLI_EFFORT
        JXL_ENC_FRAME_SETTING_JPEG_COMPRESS_BOXES
        JXL_ENC_FRAME_SETTING_BUFFERING
        JXL_ENC_FRAME_SETTING_JPEG_KEEP_EXIF
        JXL_ENC_FRAME_SETTING_JPEG_KEEP_XMP
        JXL_ENC_FRAME_SETTING_JPEG_KEEP_JUMBF
        JXL_ENC_FRAME_SETTING_USE_FULL_IMAGE_HEURISTICS
        JXL_ENC_FRAME_SETTING_DISABLE_PERCEPTUAL_HEURISTICS
        JXL_ENC_FRAME_SETTING_FILL_ENUM

    JxlEncoder* JxlEncoderCreate(
        const JxlMemoryManager* memory_manager
    )

    void JxlEncoderReset(
        JxlEncoder* enc
    )

    void JxlEncoderDestroy(
        JxlEncoder* enc
    )

    void JxlEncoderSetCms(
        JxlEncoder* enc,
        JxlCmsInterface cms
    )

    JxlEncoderStatus JxlEncoderSetParallelRunner(
        JxlEncoder* enc,
        JxlParallelRunner parallel_runner,
        void* parallel_runner_opaque
    )

    JxlEncoderError JxlEncoderGetError(
        JxlEncoder* enc
    )

    JxlEncoderStatus JxlEncoderProcessOutput(
        JxlEncoder* enc,
        uint8_t** next_out,
        size_t* avail_out
    )

    JxlEncoderStatus JxlEncoderSetFrameHeader(
        JxlEncoderFrameSettings* frame_settings,
        const JxlFrameHeader* frame_header
    )

    JxlEncoderStatus JxlEncoderSetExtraChannelBlendInfo(
        JxlEncoderFrameSettings* frame_settings,
        size_t index,
        const JxlBlendInfo* blend_info
    )

    JxlEncoderStatus JxlEncoderSetFrameName(
        JxlEncoderFrameSettings* frame_settings,
        const char* frame_name
    )

    JxlEncoderStatus JxlEncoderSetFrameBitDepth(
        JxlEncoderFrameSettings* frame_settings,
        const JxlBitDepth* bit_depth
    )

    JxlEncoderStatus JxlEncoderAddJPEGFrame(
        const JxlEncoderFrameSettings* frame_settings,
        const uint8_t* buffer,
        size_t size
    )

    JxlEncoderStatus JxlEncoderAddImageFrame(
        const JxlEncoderFrameSettings* frame_settings,
        const JxlPixelFormat* pixel_format,
        const void* buffer,
        size_t size
    )

    struct JxlEncoderOutputProcessor:
        void* opaque

        void* (*get_buffer)(
            void* opaque,
            size_t* size
        ) nogil

        void (*release_buffer)(
            void* opaque,
            size_t written_bytes
        ) nogil

        void (*seek)(
            void* opaque,
            uint64_t position
        ) nogil

        void (*set_finalized_position)(
            void* opaque,
            uint64_t finalized_position
        ) nogil

    JxlEncoderStatus JxlEncoderSetOutputProcessor(
        JxlEncoder* enc,
        JxlEncoderOutputProcessor output_processor
    )

    JxlEncoderStatus JxlEncoderFlushInput(
        JxlEncoder* enc
    )

    struct JxlChunkedFrameInputSource:
        void* opaque

        void (*get_color_channels_pixel_format)(
            void* opaque,
            JxlPixelFormat* pixel_format
        )

        const void* (*get_color_channel_data_at)(
            void* opaque,
            size_t xpos,
            size_t ypos,
            size_t xsize,
            size_t ysize,
            size_t* row_offset
        ) nogil

        void (*get_extra_channel_pixel_format)(
            void* opaque,
            size_t ec_index,
            JxlPixelFormat* pixel_format
        ) nogil

        const void* (*get_extra_channel_data_at)(
            void* opaque,
            size_t ec_index,
            size_t xpos,
            size_t ypos,
            size_t xsize,
            size_t ysize,
            size_t* row_offset
        ) nogil

        void (*release_buffer)(
            void* opaque,
            const void* buf
        ) nogil

    JxlEncoderStatus JxlEncoderAddChunkedFrame(
        const JxlEncoderFrameSettings* frame_settings,
        JXL_BOOL is_last_frame,
        JxlChunkedFrameInputSource chunked_frame_input
    )

    JxlEncoderStatus JxlEncoderSetExtraChannelBuffer(
        const JxlEncoderFrameSettings* frame_settings,
        const JxlPixelFormat* pixel_format,
        const void* buffer,
        size_t size,
        uint32_t index
    )

    JxlEncoderStatus JxlEncoderAddBox(
        JxlEncoder* enc,
        const JxlBoxType type,
        const uint8_t* contents,
        size_t size,
        JXL_BOOL compress_box
    )

    JxlEncoderStatus JxlEncoderUseBoxes(
        JxlEncoder* enc
    )

    void JxlEncoderCloseBoxes(
        JxlEncoder* enc
    )

    void JxlEncoderCloseFrames(
        JxlEncoder* enc
    )

    void JxlEncoderCloseInput(
        JxlEncoder* enc
    )

    JxlEncoderStatus JxlEncoderSetColorEncoding(
        JxlEncoder* enc,
        const JxlColorEncoding* color
    )

    JxlEncoderStatus JxlEncoderSetICCProfile(
        JxlEncoder* enc,
        const uint8_t* icc_profile,
        size_t size
    )

    void JxlEncoderInitBasicInfo(
        JxlBasicInfo* info
    )

    void JxlEncoderInitFrameHeader(
        JxlFrameHeader* frame_header
    )

    void JxlEncoderInitBlendInfo(
        JxlBlendInfo* blend_info
    )

    JxlEncoderStatus JxlEncoderSetBasicInfo(
        JxlEncoder* enc,
        const JxlBasicInfo* info
    )

    JxlEncoderStatus JxlEncoderSetUpsamplingMode(
        JxlEncoder* enc,
        int64_t factor,
        int64_t mode
    )

    void JxlEncoderInitExtraChannelInfo(
        JxlExtraChannelType type,
        JxlExtraChannelInfo* info
    )

    JxlEncoderStatus JxlEncoderSetExtraChannelInfo(
        JxlEncoder* enc,
        size_t index,
        const JxlExtraChannelInfo* info
    )

    JxlEncoderStatus JxlEncoderSetExtraChannelName(
        JxlEncoder* enc,
        size_t index,
        const char* name,
        size_t size
    )

    JxlEncoderStatus JxlEncoderFrameSettingsSetOption(
        JxlEncoderFrameSettings* frame_settings,
        JxlEncoderFrameSettingId option,
        int64_t value
    )

    JxlEncoderStatus JxlEncoderFrameSettingsSetFloatOption(
        JxlEncoderFrameSettings* frame_settings,
        JxlEncoderFrameSettingId option,
        float value
    )

    JxlEncoderStatus JxlEncoderUseContainer(
        JxlEncoder* enc,
        JXL_BOOL use_container
    )

    JxlEncoderStatus JxlEncoderStoreJPEGMetadata(
        JxlEncoder* enc,
        JXL_BOOL store_jpeg_metadata
    )

    JxlEncoderStatus JxlEncoderSetCodestreamLevel(
        JxlEncoder* enc,
        int level
    )

    int JxlEncoderGetRequiredCodestreamLevel(
        const JxlEncoder* enc
    )

    JxlEncoderStatus JxlEncoderSetFrameLossless(
        JxlEncoderFrameSettings* frame_settings,
        JXL_BOOL lossless
    )

    JxlEncoderStatus JxlEncoderSetFrameDistance(
        JxlEncoderFrameSettings* frame_settings,
        float distance
    )

    JxlEncoderStatus JxlEncoderSetExtraChannelDistance(
        JxlEncoderFrameSettings* frame_settings,
        size_t index,
        float distance
    )

    float JxlEncoderDistanceFromQuality(
        float quality
    )

    JxlEncoderFrameSettings* JxlEncoderFrameSettingsCreate(
        JxlEncoder* enc,
        const JxlEncoderFrameSettings* source
    )

    void JxlColorEncodingSetToSRGB(
        JxlColorEncoding* color_encoding,
        JXL_BOOL is_gray
    )

    void JxlColorEncodingSetToLinearSRGB(
        JxlColorEncoding* color_encoding,
        JXL_BOOL is_gray
    )

    void JxlEncoderAllowExpertOptions(
        JxlEncoder* enc
    )

    ctypedef void (*JxlDebugImageCallback)(
        void* opaque,
        const char* label,
        size_t xsize,
        size_t ysize,
        const JxlColorEncoding* color,
        const uint16_t* pixels
    ) nogil

    void JxlEncoderSetDebugImageCallback(
        JxlEncoderFrameSettings* frame_settings,
        JxlDebugImageCallback callback,
        void* opaque
    )

    void JxlEncoderCollectStats(
        JxlEncoderFrameSettings* frame_settings,
        JxlEncoderStats* stats
    )


cdef extern from 'jxl/gain_map.h' nogil:

    ctypedef struct JxlGainMapBundle:
        uint8_t jhgm_version
        uint16_t gain_map_metadata_size
        const uint8_t* gain_map_metadata
        JXL_BOOL has_color_encoding
        JxlColorEncoding color_encoding
        uint32_t alt_icc_size
        const uint8_t* alt_icc
        uint32_t gain_map_size
        const uint8_t* gain_map

    JXL_BOOL JxlGainMapGetBundleSize(
        const JxlGainMapBundle* map_bundle,
        size_t* bundle_size
    )

    JXL_BOOL JxlGainMapWriteBundle(
        const JxlGainMapBundle* map_bundle,
        uint8_t* output_buffer,
        size_t output_buffer_size,
        size_t* bytes_written
    )

    JXL_BOOL JxlGainMapReadBundle(
        JxlGainMapBundle* map_bundle,
        const uint8_t* input_buffer,
        size_t input_buffer_size,
        size_t* bytes_read
    )


cdef extern from 'jxl/compressed_icc.h' nogil:
    JXL_BOOL JxlICCProfileEncode(
        const JxlMemoryManager* memory_manager,
        const uint8_t* icc,
        size_t icc_size,
        uint8_t** compressed_icc,
        size_t* compressed_icc_size
    )

    JXL_BOOL JxlICCProfileDecode(
        const JxlMemoryManager* memory_manager,
        const uint8_t* compressed_icc,
        size_t compressed_icc_size,
        uint8_t** icc,
        size_t* icc_size
    )
