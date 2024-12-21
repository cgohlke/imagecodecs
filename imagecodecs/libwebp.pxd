# imagecodecs/libwebp.pxd
# cython: language_level = 3

# Cython declarations for the `libwebp 1.5.0` library.
# https://github.com/webmproject/libwebp

from libc.stdint cimport uint8_t, uint32_t

cdef extern from 'webp/decode.h' nogil:

    int WEBP_DECODER_ABI_VERSION

    ctypedef struct WebPIDecoder:
        pass

    int WebPGetDecoderVersion()

    int WebPGetInfo(
        const uint8_t* data,
        size_t data_size,
        int* width,
        int* height
    )

    uint8_t* WebPDecodeRGBA(
        const uint8_t* data,
        size_t data_size,
        int* width,
        int* height
    )

    uint8_t* WebPDecodeARGB(
        const uint8_t* data,
        size_t data_size,
        int* width,
        int* height
    )

    uint8_t* WebPDecodeBGRA(
        const uint8_t* data,
        size_t data_size,
        int* width,
        int* height
    )

    uint8_t* WebPDecodeRGB(
        const uint8_t* data,
        size_t data_size,
        int* width,
        int* height
    )

    uint8_t* WebPDecodeBGR(
        const uint8_t* data,
        size_t data_size,
        int* width,
        int* height
    )

    uint8_t* WebPDecodeYUV(
        const uint8_t* data,
        size_t data_size,
        int* width,
        int* height,
        uint8_t** u,
        uint8_t** v,
        int* stride,
        int* uv_stride
    )

    uint8_t* WebPDecodeRGBAInto(
        const uint8_t* data,
        size_t data_size,
        uint8_t* output_buffer,
        size_t output_buffer_size,
        int output_stride
    )

    uint8_t* WebPDecodeARGBInto(
        const uint8_t* data,
        size_t data_size,
        uint8_t* output_buffer,
        size_t output_buffer_size,
        int output_stride
    )

    uint8_t* WebPDecodeBGRAInto(
        const uint8_t* data,
        size_t data_size,
        uint8_t* output_buffer,
        size_t output_buffer_size,
        int output_stride
    )

    uint8_t* WebPDecodeRGBInto(
        const uint8_t* data,
        size_t data_size,
        uint8_t* output_buffer,
        size_t output_buffer_size,
        int output_stride
    )

    uint8_t* WebPDecodeBGRInto(
        const uint8_t* data,
        size_t data_size,
        uint8_t* output_buffer,
        size_t output_buffer_size,
        int output_stride
    )

    uint8_t* WebPDecodeYUVInto(
        const uint8_t* data,
        size_t data_size,
        uint8_t* luma,
        size_t luma_size,
        int luma_stride,
        uint8_t* u,
        size_t u_size,
        int u_stride,
        uint8_t* v,
        size_t v_size,
        int v_stride
    )

    ctypedef enum WEBP_CSP_MODE:
        MODE_RGB
        MODE_RGBA
        MODE_BGR
        MODE_BGRA
        MODE_ARGB
        MODE_RGBA_4444
        MODE_RGB_565
        MODE_rgbA
        MODE_bgrA
        MODE_Argb
        MODE_rgbA_4444
        MODE_YUV
        MODE_YUVA
        MODE_LAST

    int WebPIsPremultipliedMode(
        WEBP_CSP_MODE mode
    )

    int WebPIsAlphaMode(
        WEBP_CSP_MODE mode
    )

    int WebPIsRGBMode(
        WEBP_CSP_MODE mode
    )

    struct WebPRGBABuffer:
        uint8_t* rgba
        int stride
        size_t size

    struct WebPYUVABuffer:
        uint8_t* y
        uint8_t* u
        uint8_t* v
        uint8_t* a
        int y_stride
        int u_stride
        int v_stride
        int a_stride
        size_t y_size
        size_t u_size
        size_t v_size
        size_t a_size

    cdef union _WebPDecBufferU:
        WebPRGBABuffer RGBA
        WebPYUVABuffer YUVA

    struct WebPDecBuffer:
        WEBP_CSP_MODE colorspace
        int width, height
        int is_external_memory
        _WebPDecBufferU u
        uint32_t[4] pad
        uint8_t* private_memory

    int WebPInitDecBufferInternal(
        WebPDecBuffer*,
        int
    )

    int WebPInitDecBuffer(
        WebPDecBuffer* buffer
    )

    void WebPFreeDecBuffer(
        WebPDecBuffer* buffer
    )

    ctypedef enum VP8StatusCode:
        VP8_STATUS_OK
        VP8_STATUS_OUT_OF_MEMORY
        VP8_STATUS_INVALID_PARAM
        VP8_STATUS_BITSTREAM_ERROR
        VP8_STATUS_UNSUPPORTED_FEATURE
        VP8_STATUS_SUSPENDED
        VP8_STATUS_USER_ABORT
        VP8_STATUS_NOT_ENOUGH_DATA

    WebPIDecoder* WebPINewDecoder(
        WebPDecBuffer* output_buffer
    )

    WebPIDecoder* WebPINewRGB(
        WEBP_CSP_MODE csp,
        uint8_t* output_buffer,
        size_t output_buffer_size,
        int output_stride
    )

    WebPIDecoder* WebPINewYUVA(
        uint8_t* luma,
        size_t luma_size,
        int luma_stride,
        uint8_t* u,
        size_t u_size,
        int u_stride,
        uint8_t* v,
        size_t v_size,
        int v_stride,
        uint8_t* a,
        size_t a_size,
        int a_stride
    )

    WebPIDecoder* WebPINewYUV(
        uint8_t* luma,
        size_t luma_size,
        int luma_stride,
        uint8_t* u,
        size_t u_size,
        int u_stride,
        uint8_t* v,
        size_t v_size,
        int v_stride
    )

    void WebPIDelete(
        WebPIDecoder* idec
    )

    VP8StatusCode WebPIAppend(
        WebPIDecoder* idec,
        const uint8_t* data, size_t data_size
    )

    VP8StatusCode WebPIUpdate(
        WebPIDecoder* idec,
        const uint8_t* data,
        size_t data_size
    )

    uint8_t* WebPIDecGetRGB(
        const WebPIDecoder* idec,
        int* last_y,
        int* width,
        int* height,
        int* stride
    )

    uint8_t* WebPIDecGetYUVA(
        const WebPIDecoder* idec,
        int* last_y,
        uint8_t** u,
        uint8_t** v,
        uint8_t** a,
        int* width,
        int* height,
        int* stride,
        int* uv_stride,
        int* a_stride
    )

    uint8_t* WebPIDecGetYUV(
        const WebPIDecoder* idec,
        int* last_y,
        uint8_t** u,
        uint8_t** v,
        int* width,
        int* height,
        int* stride,
        int* uv_stride
    )

    const WebPDecBuffer* WebPIDecodedArea(
        const WebPIDecoder* idec,
        int* left,
        int* top,
        int* width,
        int* height
    )

    struct WebPBitstreamFeatures:
        int width
        int height
        int has_alpha
        int has_animation
        int format
        uint32_t[5] pad

    VP8StatusCode WebPGetFeaturesInternal(
        const uint8_t*,
        size_t,
        WebPBitstreamFeatures*,
        int
    )

    VP8StatusCode WebPGetFeatures(
        const uint8_t* data,
        size_t data_size,
        WebPBitstreamFeatures* features
    )

    struct WebPDecoderOptions:
        int bypass_filtering
        int no_fancy_upsampling
        int use_cropping
        int crop_left, crop_top
        int crop_width, crop_height
        int use_scaling
        int scaled_width, scaled_height
        int use_threads
        int dithering_strength
        int flip
        int alpha_dithering_strength
        uint32_t[5] pad

    struct WebPDecoderConfig:
        WebPBitstreamFeatures input
        WebPDecBuffer output
        WebPDecoderOptions options

    int WebPInitDecoderConfigInternal(
        WebPDecoderConfig*,
        int
    )

    int WebPInitDecoderConfig(
        WebPDecoderConfig* config
    )

    WebPIDecoder* WebPIDecode(
        const uint8_t* data,
        size_t data_size,
        WebPDecoderConfig* config
    )

    VP8StatusCode WebPDecode(
        const uint8_t* data,
        size_t data_size,
        WebPDecoderConfig* config
    )


cdef extern from 'webp/encode.h' nogil:

    int WEBP_ENCODER_ABI_VERSION

    int WebPGetEncoderVersion()

    size_t WebPEncodeRGB(
        const uint8_t* rgb,
        int width,
        int height,
        int stride,
        float quality_factor,
        uint8_t** output
    )

    size_t WebPEncodeBGR(
        const uint8_t* bgr,
        int width,
        int height,
        int stride,
        float quality_factor,
        uint8_t** output
    )

    size_t WebPEncodeRGBA(
        const uint8_t* rgba,
        int width,
        int height,
        int stride,
        float quality_factor,
        uint8_t** output
    )

    size_t WebPEncodeBGRA(
        const uint8_t* bgra,
        int width,
        int height,
        int stride,
        float quality_factor,
        uint8_t** output
    )

    size_t WebPEncodeLosslessRGB(
        const uint8_t* rgb,
        int width,
        int height,
        int stride,
        uint8_t** output
    )

    size_t WebPEncodeLosslessBGR(
        const uint8_t* bgr,
        int width,
        int height,
        int stride,
        uint8_t** output
    )

    size_t WebPEncodeLosslessRGBA(
        const uint8_t* rgba,
        int width,
        int height,
        int stride,
        uint8_t** output
    )

    size_t WebPEncodeLosslessBGRA(
        const uint8_t* bgra,
        int width,
        int height,
        int stride,
        uint8_t** output
    )

    ctypedef enum WebPImageHint:
        WEBP_HINT_DEFAULT
        WEBP_HINT_PICTURE
        WEBP_HINT_PHOTO
        WEBP_HINT_GRAPH
        WEBP_HINT_LAST

    struct WebPConfig:
        int lossless
        float quality
        int method
        WebPImageHint image_hint
        int target_size
        float target_PSNR
        int segments
        int sns_strength
        int filter_strength
        int filter_sharpness
        int filter_type
        int autofilter
        int alpha_compression
        int alpha_filtering
        int alpha_quality
        # int pass
        int show_compressed
        int preprocessing
        int partitions
        int partition_limit
        int emulate_jpeg_size
        int thread_level
        int low_memory
        int near_lossless
        int exact
        int use_delta_palette
        int use_sharp_yuv
        int qmin
        int qmax

    ctypedef enum WebPPreset:
        WEBP_PRESET_DEFAULT
        WEBP_PRESET_PICTURE
        WEBP_PRESET_PHOTO
        WEBP_PRESET_DRAWING
        WEBP_PRESET_ICON
        WEBP_PRESET_TEXT

    int WebPConfigInitInternal(
        WebPConfig*,
        WebPPreset,
        float,
        int
    )

    int WebPConfigInit(
        WebPConfig* config
    )

    int WebPConfigPreset(
        WebPConfig* config,
        WebPPreset preset,
        float quality
    )

    int WebPConfigLosslessPreset(
        WebPConfig* config,
        int level
    )

    int WebPValidateConfig(
        const WebPConfig* config
    )

    struct WebPAuxStats:
        int coded_size
        float[5] PSNR
        int[3] block_count
        int[2] header_bytes
        int[3][4] residual_bytes
        int[4] segment_size
        int[4] segment_quant
        int[4] segment_level
        int alpha_data_size
        int layer_data_size
        uint32_t lossless_features
        int histogram_bits
        int transform_bits
        int cache_bits
        int palette_size
        int lossless_size
        int lossless_hdr_size
        int lossless_data_size
        int cross_color_transform_bits
        uint32_t[1] pad

    # ctypedef struct WebPPicture:
    #     pass

    ctypedef int (*WebPWriterFunction)(
        const uint8_t* data,
        size_t data_size,
        const WebPPicture* picture
    ) nogil

    struct WebPMemoryWriter:
        uint8_t* mem
        size_t size
        size_t max_size
        uint32_t[1] pad

    void WebPMemoryWriterInit(
        WebPMemoryWriter* writer
    )

    void WebPMemoryWriterClear(
        WebPMemoryWriter* writer
    )

    int WebPMemoryWrite(
        const uint8_t* data,
        size_t data_size,
        const WebPPicture* picture
    )

    ctypedef int (*WebPProgressHook)(
        int percent,
        const WebPPicture* picture
    ) nogil

    ctypedef enum WebPEncCSP:
        WEBP_YUV420
        WEBP_YUV420A
        WEBP_CSP_UV_MASK
        WEBP_CSP_ALPHA_BIT

    ctypedef enum WebPEncodingError:
        VP8_ENC_OK
        VP8_ENC_ERROR_OUT_OF_MEMORY
        VP8_ENC_ERROR_BITSTREAM_OUT_OF_MEMORY
        VP8_ENC_ERROR_NULL_PARAMETER
        VP8_ENC_ERROR_INVALID_CONFIGURATION
        VP8_ENC_ERROR_BAD_DIMENSION
        VP8_ENC_ERROR_PARTITION0_OVERFLOW
        VP8_ENC_ERROR_PARTITION_OVERFLOW
        VP8_ENC_ERROR_BAD_WRITE
        VP8_ENC_ERROR_FILE_TOO_BIG
        VP8_ENC_ERROR_USER_ABORT
        VP8_ENC_ERROR_LAST

    int WEBP_MAX_DIMENSION

    ctypedef struct WebPPicture:
        int use_argb
        WebPEncCSP colorspace
        int width
        int height
        uint8_t* y
        uint8_t* u
        uint8_t* v
        int y_stride, uv_stride
        uint8_t* a
        int a_stride
        uint32_t[2] pad1
        uint32_t* argb
        int argb_stride
        uint32_t[3] pad2
        WebPWriterFunction writer
        void* custom_ptr
        int extra_info_type
        uint8_t* extra_info
        WebPAuxStats* stats
        WebPEncodingError error_code
        WebPProgressHook progress_hook
        void* user_data
        uint32_t[3] pad3
        uint8_t* pad4
        uint8_t* pad5
        uint32_t[8] pad6
        void* memory_
        void* memory_argb_
        (void*)[2] pad7

    int WebPPictureInitInternal(
        WebPPicture*,
        int
    )

    int WebPPictureInit(
        WebPPicture* picture
    )

    int WebPPictureAlloc(
        WebPPicture* picture
    )

    void WebPPictureFree(
        WebPPicture* picture
    )

    int WebPPictureCopy(
        const WebPPicture* src,
        WebPPicture* dst
    )

    int WebPPlaneDistortion(
        const uint8_t* src,
        size_t src_stride,
        const uint8_t* ref,
        size_t ref_stride,
        int width,
        int height,
        size_t x_step,
        int type,
        float* distortion,
        float* result
    )

    int WebPPictureDistortion(
        const WebPPicture* src,
        const WebPPicture* ref,
        int metric_type,
        float[5] result
    )

    int WebPPictureCrop(
        WebPPicture* picture,
        int left,
        int top,
        int width,
        int height
    )

    int WebPPictureView(
        const WebPPicture* src,
        int left,
        int top,
        int width,
        int height,
        WebPPicture* dst
    )

    int WebPPictureIsView(
        const WebPPicture* picture
    )

    int WebPPictureRescale(
        WebPPicture* pic,
        int width,
        int height
    )

    int WebPPictureImportRGB(
        WebPPicture* picture,
        const uint8_t* rgb,
        int rgb_stride
    )

    int WebPPictureImportRGBA(
        WebPPicture* picture,
        const uint8_t* rgba,
        int rgba_stride
    )

    int WebPPictureImportRGBX(
        WebPPicture* picture,
        const uint8_t* rgbx,
        int rgbx_stride
    )

    int WebPPictureImportBGR(
        WebPPicture* picture,
        const uint8_t* bgr,
        int bgr_stride
    )

    int WebPPictureImportBGRA(
        WebPPicture* picture,
        const uint8_t* bgra,
        int bgra_stride
    )

    int WebPPictureImportBGRX(
        WebPPicture* picture,
        const uint8_t* bgrx,
        int bgrx_stride
    )

    int WebPPictureARGBToYUVA(
        WebPPicture* picture,
        WebPEncCSP
    )

    int WebPPictureARGBToYUVADithered(
        WebPPicture* picture,
        WebPEncCSP colorspace,
        float dithering
    )

    int WebPPictureSharpARGBToYUVA(
        WebPPicture* picture
    )

    int WebPPictureSmartARGBToYUVA(
        WebPPicture* picture
    )

    int WebPPictureYUVAToARGB(
        WebPPicture* picture
    )

    void WebPCleanupTransparentArea(
        WebPPicture* picture
    )

    int WebPPictureHasTransparency(
        const WebPPicture* picture
    )

    void WebPBlendAlpha(
        WebPPicture* pic,
        uint32_t background_rgb
    )

    int WebPEncode(
        const WebPConfig* config,
        WebPPicture* picture
    )

    # cdef extern from 'webp/types.h' nogil:

    void* WebPMalloc(
        size_t size
    )

    void WebPFree(
        void* ptr
    )


cdef extern from 'webp/mux_types.h' nogil:

    ctypedef struct WebPData:
        const uint8_t* bytes
        size_t size

    ctypedef enum WebPFeatureFlags:
        ANIMATION_FLAG
        XMP_FLAG
        EXIF_FLAG
        ALPHA_FLAG
        ICCP_FLAG
        ALL_VALID_FLAGS

    ctypedef enum WebPMuxAnimDispose:
        WEBP_MUX_DISPOSE_NONE
        WEBP_MUX_DISPOSE_BACKGROUND

    ctypedef enum WebPMuxAnimBlend:
        WEBP_MUX_BLEND
        WEBP_MUX_NO_BLEND

    void WebPDataInit(
        WebPData* webp_data
    )

    void WebPDataClear(
        WebPData* webp_data
    )

    int WebPDataCopy(
        const WebPData* src,
        WebPData* dst
    )


cdef extern from 'webp/demux.h' nogil:

    int WEBP_DEMUX_ABI_VERSION

    ctypedef struct WebPDemuxer:
        pass

    int WebPGetDemuxVersion()

    ctypedef enum WebPDemuxState:
        WEBP_DEMUX_PARSE_ERROR
        WEBP_DEMUX_PARSING_HEADER
        WEBP_DEMUX_PARSED_HEADER
        WEBP_DEMUX_DONE

    WebPDemuxer* WebPDemuxInternal(
        const WebPData*,
        int,
        WebPDemuxState*,
        int
    )

    WebPDemuxer* WebPDemux(
        const WebPData* data
    )

    WebPDemuxer* WebPDemuxPartial(
        const WebPData* data,
        WebPDemuxState* state
    )

    void WebPDemuxDelete(
        WebPDemuxer* dmux
    )

    ctypedef enum WebPFormatFeature:
        WEBP_FF_FORMAT_FLAGS
        WEBP_FF_CANVAS_WIDTH
        WEBP_FF_CANVAS_HEIGHT
        WEBP_FF_LOOP_COUNT
        WEBP_FF_BACKGROUND_COLOR
        WEBP_FF_FRAME_COUNT

    uint32_t WebPDemuxGetI(
        const WebPDemuxer* dmux,
        WebPFormatFeature feature
    )

    ctypedef struct WebPIterator:
        int frame_num
        int num_frames
        int x_offset
        int y_offset
        int width
        int height
        int duration
        WebPMuxAnimDispose dispose_method
        int complete
        WebPData fragment
        int has_alpha
        WebPMuxAnimBlend blend_method
        uint32_t[2] pad
        void* private_

    int WebPDemuxGetFrame(
        const WebPDemuxer* dmux,
        int frame_number,
        WebPIterator* iter
    )

    int WebPDemuxNextFrame(
        WebPIterator* iter
    )

    int WebPDemuxPrevFrame(
        WebPIterator* iter
    )

    void WebPDemuxReleaseIterator(
        WebPIterator* iter
    )

    ctypedef struct WebPChunkIterator:
        int chunk_num
        int num_chunks
        WebPData chunk
        uint32_t[6] pad
        void* private_

    int WebPDemuxGetChunk(
        const WebPDemuxer* dmux,
        const char[4] fourcc,
        int chunk_number,
        WebPChunkIterator* iter
    )

    int WebPDemuxNextChunk(
        WebPChunkIterator* iter
    )

    int WebPDemuxPrevChunk(
        WebPChunkIterator* iter
    )

    void WebPDemuxReleaseChunkIterator(
        WebPChunkIterator* iter
    )

    ctypedef struct WebPAnimDecoder:
        pass

    ctypedef struct WebPAnimDecoderOptions:
        WEBP_CSP_MODE color_mode
        int use_threads
        uint32_t[7] padding

    int WebPAnimDecoderOptionsInitInternal(
        WebPAnimDecoderOptions*,
        int
    )

    int WebPAnimDecoderOptionsInit(
        WebPAnimDecoderOptions* dec_options
    )

    WebPAnimDecoder* WebPAnimDecoderNewInternal(
        const WebPData*,
        const WebPAnimDecoderOptions*,
        int
    )

    WebPAnimDecoder* WebPAnimDecoderNew(
        const WebPData* webp_data,
        const WebPAnimDecoderOptions* dec_options
    )

    ctypedef struct WebPAnimInfo:
        uint32_t canvas_width
        uint32_t canvas_height
        uint32_t loop_count
        uint32_t bgcolor
        uint32_t frame_count
        uint32_t[4] pad

    int WebPAnimDecoderGetInfo(
        const WebPAnimDecoder* dec,
        WebPAnimInfo* info
    )

    int WebPAnimDecoderGetNext(
        WebPAnimDecoder* dec,
        uint8_t** buf,
        int* timestamp
    )

    int WebPAnimDecoderHasMoreFrames(
        const WebPAnimDecoder* dec
    )

    void WebPAnimDecoderReset(
        WebPAnimDecoder* dec
    )

    const WebPDemuxer* WebPAnimDecoderGetDemuxer(
        const WebPAnimDecoder* dec
    )

    void WebPAnimDecoderDelete(
        WebPAnimDecoder* dec
    )
