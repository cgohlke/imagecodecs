# imagecodecs/libwebp.pxd
# cython: language_level = 3

# Cython declarations for the `libwebp 1.2.0` library.
# https://github.com/webmproject/libwebp

from libc.stdint cimport uint8_t, uint32_t

cdef extern from 'webp/decode.h':

    int WEBP_DECODER_ABI_VERSION

    ctypedef struct WebPIDecoder:
        pass

    int WebPGetDecoderVersion() nogil

    int WebPGetInfo(
        const uint8_t* data,
        size_t data_size,
        int* width,
        int* height
    ) nogil

    uint8_t* WebPDecodeRGBA(
        const uint8_t* data,
        size_t data_size,
        int* width,
        int* height
    ) nogil

    uint8_t* WebPDecodeARGB(
        const uint8_t* data,
        size_t data_size,
        int* width,
        int* height
    ) nogil

    uint8_t* WebPDecodeBGRA(
        const uint8_t* data,
        size_t data_size,
        int* width,
        int* height
    ) nogil

    uint8_t* WebPDecodeRGB(
        const uint8_t* data,
        size_t data_size,
        int* width,
        int* height
    ) nogil

    uint8_t* WebPDecodeBGR(
        const uint8_t* data,
        size_t data_size,
        int* width,
        int* height
    ) nogil

    uint8_t* WebPDecodeYUV(
        const uint8_t* data,
        size_t data_size,
        int* width,
        int* height,
        uint8_t** u,
        uint8_t** v,
        int* stride,
        int* uv_stride
    ) nogil

    uint8_t* WebPDecodeRGBAInto(
        const uint8_t* data,
        size_t data_size,
        uint8_t* output_buffer,
        size_t output_buffer_size,
        int output_stride
    ) nogil

    uint8_t* WebPDecodeARGBInto(
        const uint8_t* data,
        size_t data_size,
        uint8_t* output_buffer,
        size_t output_buffer_size,
        int output_stride
    ) nogil

    uint8_t* WebPDecodeBGRAInto(
        const uint8_t* data,
        size_t data_size,
        uint8_t* output_buffer,
        size_t output_buffer_size,
        int output_stride
    ) nogil

    uint8_t* WebPDecodeRGBInto(
        const uint8_t* data,
        size_t data_size,
        uint8_t* output_buffer,
        size_t output_buffer_size,
        int output_stride
    ) nogil

    uint8_t* WebPDecodeBGRInto(
        const uint8_t* data,
        size_t data_size,
        uint8_t* output_buffer,
        size_t output_buffer_size,
        int output_stride
    ) nogil

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
    ) nogil

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
    ) nogil

    int WebPIsAlphaMode(
        WEBP_CSP_MODE mode
    ) nogil

    int WebPIsRGBMode(
        WEBP_CSP_MODE mode
    ) nogil

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
        uint32_t pad[4]
        uint8_t* private_memory

    int WebPInitDecBufferInternal(
        WebPDecBuffer*,
        int
    ) nogil

    int WebPInitDecBuffer(
        WebPDecBuffer* buffer
    ) nogil

    void WebPFreeDecBuffer(
        WebPDecBuffer* buffer
    ) nogil

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
    ) nogil

    WebPIDecoder* WebPINewRGB(
        WEBP_CSP_MODE csp,
        uint8_t* output_buffer,
        size_t output_buffer_size,
        int output_stride
    ) nogil

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
    ) nogil

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
    ) nogil

    void WebPIDelete(
        WebPIDecoder* idec
    ) nogil

    VP8StatusCode WebPIAppend(
        WebPIDecoder* idec,
        const uint8_t* data, size_t data_size
    ) nogil

    VP8StatusCode WebPIUpdate(
        WebPIDecoder* idec,
        const uint8_t* data,
        size_t data_size
    ) nogil

    uint8_t* WebPIDecGetRGB(
        const WebPIDecoder* idec,
        int* last_y,
        int* width,
        int* height,
        int* stride
    ) nogil

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
    ) nogil

    uint8_t* WebPIDecGetYUV(
        const WebPIDecoder* idec,
        int* last_y,
        uint8_t** u,
        uint8_t** v,
        int* width,
        int* height,
        int* stride,
        int* uv_stride
    ) nogil

    const WebPDecBuffer* WebPIDecodedArea(
        const WebPIDecoder* idec,
        int* left,
        int* top,
        int* width,
        int* height
    ) nogil

    struct WebPBitstreamFeatures:
        int width
        int height
        int has_alpha
        int has_animation
        int format
        uint32_t pad[5]

    VP8StatusCode WebPGetFeaturesInternal(
        const uint8_t*,
        size_t,
        WebPBitstreamFeatures*,
        int
    ) nogil

    VP8StatusCode WebPGetFeatures(
        const uint8_t* data,
        size_t data_size,
        WebPBitstreamFeatures* features
    ) nogil

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
        uint32_t pad[5]

    struct WebPDecoderConfig:
        WebPBitstreamFeatures input
        WebPDecBuffer output
        WebPDecoderOptions options

    int WebPInitDecoderConfigInternal(
        WebPDecoderConfig*,
        int
    ) nogil

    int WebPInitDecoderConfig(
        WebPDecoderConfig* config
    ) nogil

    WebPIDecoder* WebPIDecode(
        const uint8_t* data,
        size_t data_size,
        WebPDecoderConfig* config
    ) nogil

    VP8StatusCode WebPDecode(
        const uint8_t* data,
        size_t data_size,
        WebPDecoderConfig* config
    ) nogil


cdef extern from 'webp/encode.h':

    int WEBP_ENCODER_ABI_VERSION

    ctypedef struct WebPPicture:
        pass

    int WebPGetEncoderVersion() nogil

    size_t WebPEncodeRGB(
        const uint8_t* rgb,
        int width,
        int height,
        int stride,
        float quality_factor,
        uint8_t** output
    ) nogil

    size_t WebPEncodeBGR(
        const uint8_t* bgr,
        int width,
        int height,
        int stride,
        float quality_factor,
        uint8_t** output
    ) nogil

    size_t WebPEncodeRGBA(
        const uint8_t* rgba,
        int width,
        int height,
        int stride,
        float quality_factor,
        uint8_t** output
    ) nogil

    size_t WebPEncodeBGRA(
        const uint8_t* bgra,
        int width,
        int height,
        int stride,
        float quality_factor,
        uint8_t** output
    ) nogil

    size_t WebPEncodeLosslessRGB(
        const uint8_t* rgb,
        int width,
        int height,
        int stride,
        uint8_t** output
    ) nogil

    size_t WebPEncodeLosslessBGR(
        const uint8_t* bgr,
        int width,
        int height,
        int stride,
        uint8_t** output
    ) nogil

    size_t WebPEncodeLosslessRGBA(
        const uint8_t* rgba,
        int width,
        int height,
        int stride,
        uint8_t** output
    ) nogil

    size_t WebPEncodeLosslessBGRA(
        const uint8_t* bgra,
        int width,
        int height,
        int stride,
        uint8_t** output
    ) nogil

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
    ) nogil

    int WebPConfigInit(
        WebPConfig* config
    ) nogil

    int WebPConfigPreset(
        WebPConfig* config,
        WebPPreset preset,
        float quality
    ) nogil

    int WebPConfigLosslessPreset(
        WebPConfig* config,
        int level
    ) nogil

    int WebPValidateConfig(
        const WebPConfig* config
    ) nogil

    struct WebPAuxStats:
        int coded_size
        float PSNR[5]
        int block_count[3]
        int header_bytes[2]
        int residual_bytes[3][4]
        int segment_size[4]
        int segment_quant[4]
        int segment_level[4]
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
        uint32_t pad[2]

    ctypedef int (*WebPWriterFunction)(
        const uint8_t* data,
        size_t data_size,
        const WebPPicture* picture
    ) nogil

    struct WebPMemoryWriter:
        uint8_t* mem
        size_t size
        size_t max_size
        uint32_t pad[1]

    void WebPMemoryWriterInit(
        WebPMemoryWriter* writer
    ) nogil

    void WebPMemoryWriterClear(
        WebPMemoryWriter* writer
    ) nogil

    int WebPMemoryWrite(
        const uint8_t* data,
        size_t data_size,
        const WebPPicture* picture
    ) nogil

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

    struct WebPPicture:
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
        uint32_t pad1[2]
        uint32_t* argb
        int argb_stride
        uint32_t pad2[3]
        WebPWriterFunction writer
        void* custom_ptr
        int extra_info_type
        uint8_t* extra_info
        WebPAuxStats* stats
        WebPEncodingError error_code
        WebPProgressHook progress_hook
        void* user_data
        uint32_t pad3[3]
        uint8_t* pad4
        uint8_t* pad5
        uint32_t pad6[8]
        void* memory_
        void* memory_argb_
        void* pad7[2]

    int WebPPictureInitInternal(
        WebPPicture*,
        int
    ) nogil

    int WebPPictureInit(
        WebPPicture* picture
    ) nogil

    int WebPPictureAlloc(
        WebPPicture* picture
    ) nogil

    void WebPPictureFree(
        WebPPicture* picture
    ) nogil

    int WebPPictureCopy(
        const WebPPicture* src,
        WebPPicture* dst
    ) nogil

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
    ) nogil

    int WebPPictureDistortion(
        const WebPPicture* src,
        const WebPPicture* ref,
        int metric_type,
        float result[5]
    ) nogil

    int WebPPictureCrop(
        WebPPicture* picture,
        int left,
        int top,
        int width,
        int height
    ) nogil

    int WebPPictureView(
        const WebPPicture* src,
        int left,
        int top,
        int width,
        int height,
        WebPPicture* dst
    ) nogil

    int WebPPictureIsView(
        const WebPPicture* picture
    ) nogil

    int WebPPictureRescale(
        WebPPicture* pic,
        int width,
        int height
    ) nogil

    int WebPPictureImportRGB(
        WebPPicture* picture,
        const uint8_t* rgb,
        int rgb_stride
    ) nogil

    int WebPPictureImportRGBA(
        WebPPicture* picture,
        const uint8_t* rgba,
        int rgba_stride
    ) nogil

    int WebPPictureImportRGBX(
        WebPPicture* picture,
        const uint8_t* rgbx,
        int rgbx_stride
    ) nogil

    int WebPPictureImportBGR(
        WebPPicture* picture,
        const uint8_t* bgr,
        int bgr_stride
    ) nogil

    int WebPPictureImportBGRA(
        WebPPicture* picture,
        const uint8_t* bgra,
        int bgra_stride
    ) nogil

    int WebPPictureImportBGRX(
        WebPPicture* picture,
        const uint8_t* bgrx,
        int bgrx_stride
    ) nogil

    int WebPPictureARGBToYUVA(
        WebPPicture* picture,
        WebPEncCSP
    ) nogil

    int WebPPictureARGBToYUVADithered(
        WebPPicture* picture,
        WebPEncCSP colorspace,
        float dithering
    ) nogil

    int WebPPictureSharpARGBToYUVA(
        WebPPicture* picture
    ) nogil

    int WebPPictureSmartARGBToYUVA(
        WebPPicture* picture
    ) nogil

    int WebPPictureYUVAToARGB(
        WebPPicture* picture
    ) nogil

    void WebPCleanupTransparentArea(
        WebPPicture* picture
    ) nogil

    int WebPPictureHasTransparency(
        const WebPPicture* picture
    ) nogil

    void WebPBlendAlpha(
        WebPPicture* pic,
        uint32_t background_rgb
    ) nogil

    int WebPEncode(
        const WebPConfig* config,
        WebPPicture* picture
    ) nogil

    # cdef extern from 'webp/types.h':

    void* WebPMalloc(
        size_t size
    ) nogil

    void WebPFree(
        void* ptr
    ) nogil
