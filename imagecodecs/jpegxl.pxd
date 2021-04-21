# imagecodecs/jpegxl.pxd
# cython: language_level = 3

# Cython declarations for the `jpeg-xl 0.3.7` library.
# https://gitlab.com/wg1/jpeg-xl

from libc.stdint cimport uint8_t, uint32_t, uint64_t

cdef extern from 'jxl/types.h':

    ctypedef int JXL_BOOL

    int JXL_TRUE
    int JXL_FALSE

    ctypedef enum JxlDataType:
        JXL_TYPE_FLOAT
        JXL_TYPE_BOOLEAN
        JXL_TYPE_UINT8
        JXL_TYPE_UINT16
        JXL_TYPE_UINT32

    ctypedef enum JxlEndianness:
        JXL_NATIVE_ENDIAN
        JXL_LITTLE_ENDIAN
        JXL_BIG_ENDIAN

    ctypedef struct JxlPixelFormat:
        uint32_t num_channels
        JxlDataType data_type
        JxlEndianness endianness
        size_t align


cdef extern from 'jxl/codestream_header.h':

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

    ctypedef struct JxlExtraChannelInfo:
        JxlExtraChannelType type
        uint32_t bits_per_sample
        uint32_t exponent_bits_per_sample
        uint32_t dim_shift
        uint32_t name_length
        JXL_BOOL alpha_associated
        float spot_color[4]
        uint32_t cfa_channel

    ctypedef struct JxlHeaderExtensions:
        uint64_t extensions;

    ctypedef struct JxlFrameHeader:
        uint32_t duration
        uint32_t timecode
        uint32_t name_length
        JXL_BOOL is_last


cdef extern from 'jxl/color_encoding.h':

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
        double white_point_xy[2]
        JxlPrimaries primaries
        double primaries_red_xy[2]
        double primaries_green_xy[2]
        double primaries_blue_xy[2]
        JxlTransferFunction transfer_function
        double gamma
        JxlRenderingIntent rendering_intent

    ctypedef struct JxlInverseOpsinMatrix:
        float opsin_inv_matrix[3][3]
        float opsin_biases[3]
        float quant_biases[3]


cdef extern from 'jxl/memory_manager.h':

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


cdef extern from 'jxl/parallel_runner.h':

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


cdef extern from 'jxl/thread_parallel_runner.h':

    JxlParallelRetCode JxlThreadParallelRunner(
        void* runner_opaque,
        void* jpegxl_opaque,
        JxlParallelRunInit init,
        JxlParallelRunFunction func,
        uint32_t start_range,
        uint32_t end_range
    ) nogil

    void* JxlThreadParallelRunnerCreate(
        const JxlMemoryManager* memory_manager,
        size_t num_worker_threads
    ) nogil

    void JxlThreadParallelRunnerDestroy(
        void* runner_opaque
    ) nogil

    size_t JxlThreadParallelRunnerDefaultNumWorkerThreads() nogil


cdef extern from 'jxl/decode.h':

    uint32_t JxlDecoderVersion()

    ctypedef enum JxlSignature:
        JXL_SIG_NOT_ENOUGH_BYTES
        JXL_SIG_INVALID
        JXL_SIG_CODESTREAM
        JXL_SIG_CONTAINER

    JxlSignature JxlSignatureCheck(
        const uint8_t* buf,
        size_t len
    ) nogil

    ctypedef struct JxlDecoder:
        pass

    JxlDecoder* JxlDecoderCreate(
        const JxlMemoryManager* memory_manager
    ) nogil

    void JxlDecoderReset(
        JxlDecoder* dec
    ) nogil

    void JxlDecoderDestroy(
        JxlDecoder* dec
    ) nogil

    ctypedef enum JxlDecoderStatus:
        JXL_DEC_SUCCESS
        JXL_DEC_ERROR
        JXL_DEC_NEED_MORE_INPUT
        JXL_DEC_NEED_PREVIEW_OUT_BUFFER
        JXL_DEC_NEED_DC_OUT_BUFFER
        JXL_DEC_NEED_IMAGE_OUT_BUFFER
        JXL_DEC_JPEG_NEED_MORE_OUTPUT
        JXL_DEC_BASIC_INFO
        JXL_DEC_EXTENSIONS
        JXL_DEC_COLOR_ENCODING
        JXL_DEC_PREVIEW_IMAGE
        JXL_DEC_FRAME
        JXL_DEC_DC_IMAGE
        JXL_DEC_FULL_IMAGE
        JXL_DEC_JPEG_RECONSTRUCTION

    JxlDecoderStatus JxlDecoderDefaultPixelFormat(
        const JxlDecoder* dec,
        JxlPixelFormat* format
    ) nogil

    JxlDecoderStatus JxlDecoderSetParallelRunner(
        JxlDecoder* dec,
        JxlParallelRunner parallel_runner,
        void* parallel_runner_opaque
    ) nogil

    size_t JxlDecoderSizeHintBasicInfo(
        const JxlDecoder* dec
    ) nogil

    JxlDecoderStatus JxlDecoderSubscribeEvents(
        JxlDecoder* dec,
        int events_wanted
    ) nogil

    JxlDecoderStatus JxlDecoderSetKeepOrientation(
        JxlDecoder* dec,
        JXL_BOOL keep_orientation
    ) nogil

    JxlDecoderStatus JxlDecoderProcessInput(
        JxlDecoder* dec
    ) nogil

    JxlDecoderStatus JxlDecoderSetInput(
        JxlDecoder* dec,
        const uint8_t* data,
        size_t size
    ) nogil

    size_t JxlDecoderReleaseInput(
        JxlDecoder* dec
    ) nogil

    JxlDecoderStatus JxlDecoderGetBasicInfo(
        const JxlDecoder* dec,
        JxlBasicInfo* info
    ) nogil

    JxlDecoderStatus JxlDecoderGetExtraChannelInfo(
        const JxlDecoder* dec,
        size_t index,
        JxlExtraChannelInfo* info
    ) nogil

    JxlDecoderStatus JxlDecoderGetExtraChannelName(
        const JxlDecoder* dec,
        size_t index,
        char* name,
        size_t size
    ) nogil

    ctypedef enum JxlColorProfileTarget:
        JXL_COLOR_PROFILE_TARGET_ORIGINAL
        JXL_COLOR_PROFILE_TARGET_DATA

    JxlDecoderStatus JxlDecoderGetColorAsEncodedProfile(
        const JxlDecoder* dec,
        const JxlPixelFormat* format,
        JxlColorProfileTarget target,
        JxlColorEncoding* color_encoding
    ) nogil

    JxlDecoderStatus JxlDecoderGetICCProfileSize(
        const JxlDecoder* dec,
        const JxlPixelFormat* format,
        JxlColorProfileTarget target,
        size_t* size
    ) nogil

    JxlDecoderStatus JxlDecoderGetColorAsICCProfile(
        const JxlDecoder* dec,
        const JxlPixelFormat* format,
        JxlColorProfileTarget target,
        uint8_t* icc_profile,
        size_t size
    ) nogil

    JxlDecoderStatus JxlDecoderPreviewOutBufferSize(
        const JxlDecoder* dec,
        const JxlPixelFormat* format,
        size_t* size
    ) nogil

    JxlDecoderStatus JxlDecoderSetPreviewOutBuffer(
        JxlDecoder* dec,
        const JxlPixelFormat* format,
        void* buffer,
        size_t size
    ) nogil

    JxlDecoderStatus JxlDecoderGetFrameHeader(
        const JxlDecoder* dec,
        JxlFrameHeader* header
    ) nogil

    JxlDecoderStatus JxlDecoderGetFrameName(
        const JxlDecoder* dec,
        char* name, size_t size
    ) nogil

    JxlDecoderStatus JxlDecoderDCOutBufferSize(
        const JxlDecoder* dec,
        const JxlPixelFormat* format,
        size_t* size
    ) nogil

    JxlDecoderStatus JxlDecoderSetDCOutBuffer(
        JxlDecoder* dec,
        const JxlPixelFormat* format,
        void* buffer,
        size_t size
    ) nogil

    JxlDecoderStatus JxlDecoderImageOutBufferSize(
        const JxlDecoder* dec,
        const JxlPixelFormat* format,
        size_t* size
    ) nogil

    JxlDecoderStatus JxlDecoderSetJPEGBuffer(
        JxlDecoder* dec,
        uint8_t* data,
        size_t size
    ) nogil

    size_t JxlDecoderReleaseJPEGBuffer(
        JxlDecoder* dec
    ) nogil

    JxlDecoderStatus JxlDecoderSetImageOutBuffer(
        JxlDecoder* dec,
        const JxlPixelFormat* format,
        void* buffer,
        size_t size
    ) nogil

    JxlDecoderStatus JxlDecoderFlushImage(
        JxlDecoder* dec
    ) nogil


cdef extern from 'jxl/encode.h':

    uint32_t JxlEncoderVersion()

    ctypedef struct JxlEncoder:
        pass

    ctypedef struct JxlEncoderOptions:
        pass

    ctypedef enum JxlEncoderStatus:
        JXL_ENC_SUCCESS
        JXL_ENC_ERROR
        JXL_ENC_NEED_MORE_OUTPUT
        JXL_ENC_NOT_SUPPORTED

    JxlEncoder* JxlEncoderCreate(
        const JxlMemoryManager* memory_manager
    ) nogil

    void JxlEncoderReset(
        JxlEncoder* enc
    ) nogil

    void JxlEncoderDestroy(
        JxlEncoder* enc
    ) nogil

    JxlEncoderStatus JxlEncoderSetParallelRunner(
        JxlEncoder* enc,
        JxlParallelRunner parallel_runner,
        void* parallel_runner_opaque
    ) nogil

    JxlEncoderStatus JxlEncoderProcessOutput(
        JxlEncoder* enc,
        uint8_t** next_out,
        size_t* avail_out
    ) nogil

    JxlEncoderStatus JxlEncoderAddJPEGFrame(
        const JxlEncoderOptions* options,
        const uint8_t* buffer,
        size_t size
    ) nogil

    JxlEncoderStatus JxlEncoderAddImageFrame(
        const JxlEncoderOptions* options,
        const JxlPixelFormat* pixel_format,
        const void* buffer,
        size_t size
    ) nogil

    void JxlEncoderCloseInput(
        JxlEncoder* enc
    ) nogil

    JxlEncoderStatus JxlEncoderSetColorEncoding(
        JxlEncoder* enc,
        const JxlColorEncoding* color
    ) nogil

    JxlEncoderStatus JxlEncoderSetBasicInfo(
        JxlEncoder* enc,
        const JxlBasicInfo* info
    ) nogil

    JxlEncoderStatus JxlEncoderStoreJPEGMetadata(
        JxlEncoder* enc,
        JXL_BOOL store_jpeg_metadata
    ) nogil

    JxlEncoderStatus JxlEncoderUseContainer(
        JxlEncoder* enc,
        JXL_BOOL use_container
    ) nogil

    JxlEncoderStatus JxlEncoderOptionsSetLossless(
        JxlEncoderOptions* options,
        JXL_BOOL lossless
    ) nogil

    JxlEncoderStatus JxlEncoderOptionsSetDecodingSpeed(
        JxlEncoderOptions* options,
        int tier
    ) nogil

    JxlEncoderStatus JxlEncoderOptionsSetEffort(
        JxlEncoderOptions* options,
        int effort
    ) nogil

    JxlEncoderStatus JxlEncoderOptionsSetDistance(
        JxlEncoderOptions* options,
        float distance
    ) nogil

    JxlEncoderOptions* JxlEncoderOptionsCreate(
        JxlEncoder* enc,
        const JxlEncoderOptions* source
    ) nogil

    void JxlColorEncodingSetToSRGB(
        JxlColorEncoding* color_encoding,
        JXL_BOOL is_gray
    ) nogil

    void JxlColorEncodingSetToLinearSRGB(
        JxlColorEncoding* color_encoding,
        JXL_BOOL is_gray
    ) nogil
