# imagecodecs/libavif.pxd
# cython: language_level = 3

# Cython declarations for the `libavif 1.0.3` library.
# https://github.com/AOMediaCodec/libavif

from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int32_t

cdef extern from 'avif/avif.h':

    int AVIF_VERSION_MAJOR
    int AVIF_VERSION_MINOR
    int AVIF_VERSION_PATCH
    int AVIF_VERSION

    ctypedef int avifBool

    int AVIF_TRUE
    int AVIF_FALSE

    int AVIF_DEFAULT_IMAGE_SIZE_LIMIT
    int AVIF_DEFAULT_IMAGE_DIMENSION_LIMIT

    int AVIF_DIAGNOSTICS_ERROR_BUFFER_SIZE
    int AVIF_DEFAULT_IMAGE_COUNT_LIMIT

    int AVIF_QUALITY_DEFAULT
    int AVIF_QUALITY_LOSSLESS
    int AVIF_QUALITY_WORST
    int AVIF_QUALITY_BEST

    int AVIF_QUANTIZER_LOSSLESS
    int AVIF_QUANTIZER_BEST_QUALITY
    int AVIF_QUANTIZER_WORST_QUALITY

    int AVIF_PLANE_COUNT_YUV

    int AVIF_SPEED_DEFAULT
    int AVIF_SPEED_SLOWEST
    int AVIF_SPEED_FASTEST

    int AVIF_REPETITION_COUNT_INFINITE
    int AVIF_REPETITION_COUNT_UNKNOWN

    int AVIF_MAX_AV1_LAYER_COUNT

    ctypedef enum avifPlanesFlag:
        AVIF_PLANES_YUV
        AVIF_PLANES_A
        AVIF_PLANES_ALL

    ctypedef uint32_t avifPlanesFlags

    ctypedef enum avifChannelIndex:
        AVIF_CHAN_Y
        AVIF_CHAN_U
        AVIF_CHAN_V
        AVIF_CHAN_A

    # Version

    const char* avifVersion() nogil

    unsigned int avifLibYUVVersion() nogil

    void avifCodecVersions(
        char* outBuffer
    ) nogil

    # Memory management

    void* avifAlloc(
        size_t size
    ) nogil

    void avifFree(
        void* p
    ) nogil

    # avifResult

    ctypedef enum avifResult:
        AVIF_RESULT_OK
        AVIF_RESULT_UNKNOWN_ERROR
        AVIF_RESULT_INVALID_FTYP
        AVIF_RESULT_NO_CONTENT
        AVIF_RESULT_NO_YUV_FORMAT_SELECTED
        AVIF_RESULT_REFORMAT_FAILED
        AVIF_RESULT_UNSUPPORTED_DEPTH
        AVIF_RESULT_ENCODE_COLOR_FAILED
        AVIF_RESULT_ENCODE_ALPHA_FAILED
        AVIF_RESULT_BMFF_PARSE_FAILED
        AVIF_RESULT_MISSING_IMAGE_ITEM
        AVIF_RESULT_DECODE_COLOR_FAILED
        AVIF_RESULT_DECODE_ALPHA_FAILED
        AVIF_RESULT_COLOR_ALPHA_SIZE_MISMATCH
        AVIF_RESULT_ISPE_SIZE_MISMATCH
        AVIF_RESULT_NO_CODEC_AVAILABLE
        AVIF_RESULT_NO_IMAGES_REMAINING
        AVIF_RESULT_INVALID_EXIF_PAYLOAD
        AVIF_RESULT_INVALID_IMAGE_GRID
        AVIF_RESULT_INVALID_CODEC_SPECIFIC_OPTION
        AVIF_RESULT_TRUNCATED_DATA
        AVIF_RESULT_IO_NOT_SET
        AVIF_RESULT_IO_ERROR
        AVIF_RESULT_WAITING_ON_IO
        AVIF_RESULT_INVALID_ARGUMENT
        AVIF_RESULT_NOT_IMPLEMENTED
        AVIF_RESULT_OUT_OF_MEMORY
        AVIF_RESULT_CANNOT_CHANGE_SETTING
        AVIF_RESULT_INCOMPATIBLE_IMAGE
        AVIF_RESULT_NO_AV1_ITEMS_FOUND

    const char* avifResultToString(
        avifResult result
    ) nogil

    # avifROData/avifRWData: Generic raw memory storage

    ctypedef struct avifROData:
        const uint8_t* data
        size_t size

    ctypedef struct avifRWData:
        uint8_t* data
        size_t size

    # int AVIF_DATA_EMPTY { NULL, 0 }

    avifResult avifRWDataRealloc(
        avifRWData* raw,
        size_t newSize
    ) nogil

    avifResult avifRWDataSet(
        avifRWData* raw,
        const uint8_t* data,
        size_t len
    ) nogil

    void avifRWDataFree(
        avifRWData* raw
    ) nogil

    # Metadata

    avifResult avifGetExifTiffHeaderOffset(
        const uint8_t* exif,
        size_t exifSize,
        size_t* offset
    ) nogil

    avifResult avifGetExifOrientationOffset(
        const uint8_t* exif,
        size_t exifSize,
        size_t* offset
    ) nogil

    # avifPixelFormat

    ctypedef enum avifPixelFormat:
        AVIF_PIXEL_FORMAT_NONE
        AVIF_PIXEL_FORMAT_YUV444
        AVIF_PIXEL_FORMAT_YUV422
        AVIF_PIXEL_FORMAT_YUV420
        AVIF_PIXEL_FORMAT_YUV400
        AVIF_PIXEL_FORMAT_COUNT

    const char* avifPixelFormatToString(
        avifPixelFormat format
    ) nogil

    ctypedef struct avifPixelFormatInfo:
        avifBool monochrome
        int chromaShiftX
        int chromaShiftY

    void avifGetPixelFormatInfo(
        avifPixelFormat format,
        avifPixelFormatInfo* info
    ) nogil

    # avifChromaSamplePosition

    ctypedef enum avifChromaSamplePosition:
        AVIF_CHROMA_SAMPLE_POSITION_UNKNOWN
        AVIF_CHROMA_SAMPLE_POSITION_VERTICAL
        AVIF_CHROMA_SAMPLE_POSITION_COLOCATED

    # avifRange

    ctypedef enum avifRange:
        AVIF_RANGE_LIMITED
        AVIF_RANGE_FULL

    # CICP enums - https://www.itu.int/rec/T-REC-H.273-201612-I/en

    ctypedef enum avifColorPrimaries:
        AVIF_COLOR_PRIMARIES_UNKNOWN
        AVIF_COLOR_PRIMARIES_BT709
        AVIF_COLOR_PRIMARIES_IEC61966_2_4
        AVIF_COLOR_PRIMARIES_UNSPECIFIED
        AVIF_COLOR_PRIMARIES_BT470M
        AVIF_COLOR_PRIMARIES_BT470BG
        AVIF_COLOR_PRIMARIES_BT601
        AVIF_COLOR_PRIMARIES_SMPTE240
        AVIF_COLOR_PRIMARIES_GENERIC_FILM
        AVIF_COLOR_PRIMARIES_BT2020
        AVIF_COLOR_PRIMARIES_XYZ
        AVIF_COLOR_PRIMARIES_SMPTE431
        AVIF_COLOR_PRIMARIES_SMPTE432
        AVIF_COLOR_PRIMARIES_EBU3213

    void avifColorPrimariesGetValues(
        avifColorPrimaries acp,
        float outPrimaries[8]
    ) nogil

    avifColorPrimaries avifColorPrimariesFind(
        const float inPrimaries[8],
        const char** outName
    ) nogil

    ctypedef enum avifTransferCharacteristics:
        AVIF_TRANSFER_CHARACTERISTICS_UNKNOWN
        AVIF_TRANSFER_CHARACTERISTICS_BT709
        AVIF_TRANSFER_CHARACTERISTICS_UNSPECIFIED
        AVIF_TRANSFER_CHARACTERISTICS_BT470M
        AVIF_TRANSFER_CHARACTERISTICS_BT470BG
        AVIF_TRANSFER_CHARACTERISTICS_BT601
        AVIF_TRANSFER_CHARACTERISTICS_SMPTE240
        AVIF_TRANSFER_CHARACTERISTICS_LINEAR
        AVIF_TRANSFER_CHARACTERISTICS_LOG100
        AVIF_TRANSFER_CHARACTERISTICS_LOG100_SQRT10
        AVIF_TRANSFER_CHARACTERISTICS_IEC61966
        AVIF_TRANSFER_CHARACTERISTICS_BT1361
        AVIF_TRANSFER_CHARACTERISTICS_SRGB
        AVIF_TRANSFER_CHARACTERISTICS_BT2020_10BIT
        AVIF_TRANSFER_CHARACTERISTICS_BT2020_12BIT
        AVIF_TRANSFER_CHARACTERISTICS_SMPTE2084
        AVIF_TRANSFER_CHARACTERISTICS_SMPTE428
        AVIF_TRANSFER_CHARACTERISTICS_HLG

    avifResult avifTransferCharacteristicsGetGamma(
        avifTransferCharacteristics atc,
        float* gamma
    ) nogil

    avifTransferCharacteristics avifTransferCharacteristicsFindByGamma(
        float gamma
    ) nogil

    ctypedef enum avifMatrixCoefficients:
        AVIF_MATRIX_COEFFICIENTS_IDENTITY
        AVIF_MATRIX_COEFFICIENTS_BT709
        AVIF_MATRIX_COEFFICIENTS_UNSPECIFIED
        AVIF_MATRIX_COEFFICIENTS_FCC
        AVIF_MATRIX_COEFFICIENTS_BT470BG
        AVIF_MATRIX_COEFFICIENTS_BT601
        AVIF_MATRIX_COEFFICIENTS_SMPTE240
        AVIF_MATRIX_COEFFICIENTS_YCGCO
        AVIF_MATRIX_COEFFICIENTS_BT2020_NCL
        AVIF_MATRIX_COEFFICIENTS_BT2020_CL
        AVIF_MATRIX_COEFFICIENTS_SMPTE2085
        AVIF_MATRIX_COEFFICIENTS_CHROMA_DERIVED_NCL
        AVIF_MATRIX_COEFFICIENTS_CHROMA_DERIVED_CL
        AVIF_MATRIX_COEFFICIENTS_ICTCP
        AVIF_MATRIX_COEFFICIENTS_YCGCO_RE
        AVIF_MATRIX_COEFFICIENTS_YCGCO_RO
        AVIF_MATRIX_COEFFICIENTS_LAST

    ctypedef struct avifDiagnostics:
        char error[256]  # [AVIF_DIAGNOSTICS_ERROR_BUFFER_SIZE]

    void avifDiagnosticsClearError(avifDiagnostics* diag) nogil

    # Fraction utility

    ctypedef struct avifFraction:
        int32_t n
        int32_t d

    # Optional transformation structs

    ctypedef enum avifTransformFlag:
        AVIF_TRANSFORM_NONE
        AVIF_TRANSFORM_PASP
        AVIF_TRANSFORM_CLAP
        AVIF_TRANSFORM_IROT
        AVIF_TRANSFORM_IMIR

    ctypedef uint32_t avifTransformFlags

    ctypedef struct avifPixelAspectRatioBox:
        uint32_t hSpacing
        uint32_t vSpacing

    ctypedef struct avifCleanApertureBox:
        uint32_t widthN
        uint32_t widthD
        uint32_t heightN
        uint32_t heightD
        uint32_t horizOffN
        uint32_t horizOffD
        uint32_t vertOffN
        uint32_t vertOffD

    ctypedef struct avifImageRotation:
        uint8_t angle

    ctypedef struct avifImageMirror:
        uint8_t axis

    ctypedef struct avifCropRect:
        uint32_t x
        uint32_t y
        uint32_t width
        uint32_t height

    avifBool avifCropRectConvertCleanApertureBox(
        avifCropRect* cropRect,
        const avifCleanApertureBox* clap,
        uint32_t imageW,
        uint32_t imageH,
        avifPixelFormat yuvFormat,
        avifDiagnostics* diag
    ) nogil

    avifBool avifCleanApertureBoxConvertCropRect(
        avifCleanApertureBox* clap,
        const avifCropRect* cropRect,
        uint32_t imageW,
        uint32_t imageH,
        avifPixelFormat yuvFormat,
        avifDiagnostics* diag
    ) nogil

    ctypedef struct avifContentLightLevelInformationBox:
        uint16_t maxCLL
        uint16_t maxPALL

    # avifImage

    ctypedef struct avifImage:
        uint32_t width
        uint32_t height
        uint32_t depth
        avifPixelFormat yuvFormat
        avifRange yuvRange
        avifChromaSamplePosition yuvChromaSamplePosition
        uint8_t* yuvPlanes[3]  # AVIF_PLANE_COUNT_YUV = 3
        uint32_t yuvRowBytes[3]  # AVIF_PLANE_COUNT_YUV = 3
        avifBool imageOwnsYUVPlanes
        uint8_t* alphaPlane
        uint32_t alphaRowBytes
        avifBool imageOwnsAlphaPlane
        avifBool alphaPremultiplied
        avifRWData icc
        avifColorPrimaries colorPrimaries
        avifTransferCharacteristics transferCharacteristics
        avifMatrixCoefficients matrixCoefficients
        avifContentLightLevelInformationBox clli
        avifTransformFlags transformFlags
        avifPixelAspectRatioBox pasp
        avifCleanApertureBox clap
        avifImageRotation irot
        avifImageMirror imir
        avifRWData exif
        avifRWData xmp

    avifImage* avifImageCreate(
        uint32_t width,
        uint32_t height,
        uint32_t depth,
        avifPixelFormat yuvFormat
    ) nogil

    avifImage* avifImageCreateEmpty() nogil

    avifResult avifImageCopy(
        avifImage* dstImage,
        const avifImage* srcImage,
        avifPlanesFlags planes
    ) nogil

    avifResult avifImageSetViewRect(
        avifImage* dstImage,
        const avifImage* srcImage,
        const avifCropRect* rect
    ) nogil

    void avifImageDestroy(
        avifImage* image
    ) nogil

    avifResult avifImageSetProfileICC(
        avifImage* image,
        const uint8_t* icc,
        size_t iccSize
    ) nogil

    avifResult avifImageSetMetadataExif(
        avifImage* image,
        const uint8_t* exif,
        size_t exifSize
    ) nogil

    avifResult avifImageSetMetadataXMP(
        avifImage* image,
        const uint8_t* xmp,
        size_t xmpSize
    ) nogil

    avifResult avifImageAllocatePlanes(
        avifImage* image,
        avifPlanesFlags planes
    ) nogil

    void avifImageFreePlanes(
        avifImage* image,
        avifPlanesFlags planes
    ) nogil

    void avifImageStealPlanes(
        avifImage* dstImage,
        avifImage* srcImage,
        avifPlanesFlags planes
    ) nogil

    ctypedef enum avifRGBFormat:
        AVIF_RGB_FORMAT_RGB
        AVIF_RGB_FORMAT_RGBA
        AVIF_RGB_FORMAT_ARGB
        AVIF_RGB_FORMAT_BGR
        AVIF_RGB_FORMAT_BGRA
        AVIF_RGB_FORMAT_ABGR
        AVIF_RGB_FORMAT_RGB_565
        AVIF_RGB_FORMAT_COUNT

    uint32_t avifRGBFormatChannelCount(
        avifRGBFormat format
    ) nogil

    avifBool avifRGBFormatHasAlpha(
        avifRGBFormat format
    ) nogil

    ctypedef enum avifChromaUpsampling:
        AVIF_CHROMA_UPSAMPLING_AUTOMATIC
        AVIF_CHROMA_UPSAMPLING_FASTEST
        AVIF_CHROMA_UPSAMPLING_BEST_QUALITY
        AVIF_CHROMA_UPSAMPLING_NEAREST
        AVIF_CHROMA_UPSAMPLING_BILINEAR

    ctypedef enum avifChromaDownsampling:
        AVIF_CHROMA_DOWNSAMPLING_AUTOMATIC
        AVIF_CHROMA_DOWNSAMPLING_FASTEST
        AVIF_CHROMA_DOWNSAMPLING_BEST_QUALITY
        AVIF_CHROMA_DOWNSAMPLING_AVERAGE
        AVIF_CHROMA_DOWNSAMPLING_SHARP_YUV

    ctypedef struct avifRGBImage:
        uint32_t width
        uint32_t height
        uint32_t depth
        avifRGBFormat format
        avifChromaUpsampling chromaUpsampling
        avifChromaDownsampling chromaDownsampling
        avifBool avoidLibYUV
        avifBool ignoreAlpha
        avifBool alphaPremultiplied
        avifBool isFloat
        int maxThreads
        uint8_t* pixels
        uint32_t rowBytes

    void avifRGBImageSetDefaults(
        avifRGBImage* rgb,
        const avifImage* image
    ) nogil

    uint32_t avifRGBImagePixelSize(
        const avifRGBImage* rgb
    ) nogil

    # Convenience functions

    avifResult avifRGBImageAllocatePixels(
        avifRGBImage* rgb
    ) nogil

    void avifRGBImageFreePixels(
        avifRGBImage* rgb
    ) nogil

    # The main conversion functions

    avifResult avifImageRGBToYUV(
        avifImage* image,
        const avifRGBImage* rgb
    ) nogil

    avifResult avifImageYUVToRGB(
        const avifImage* image,
        avifRGBImage* rgb
    ) nogil

    # Premultiply handling functions

    avifResult avifRGBImagePremultiplyAlpha(
        avifRGBImage* rgb
    ) nogil

    avifResult avifRGBImageUnpremultiplyAlpha(
        avifRGBImage* rgb
    ) nogil

    # YUV Utils

    int avifFullToLimitedY(
        uint32_t depth,
        int v
    ) nogil

    int avifFullToLimitedUV(
        uint32_t depth,
        int v
    ) nogil

    int avifLimitedToFullY(
        uint32_t depth,
        int v
    ) nogil

    int avifLimitedToFullUV(
        uint32_t depth,
        int v
    ) nogil

    # Codec selection

    ctypedef enum avifCodecChoice:
        AVIF_CODEC_CHOICE_AUTO
        AVIF_CODEC_CHOICE_AOM
        AVIF_CODEC_CHOICE_DAV1D
        AVIF_CODEC_CHOICE_LIBGAV1
        AVIF_CODEC_CHOICE_RAV1E
        AVIF_CODEC_CHOICE_SVT
        AVIF_CODEC_CHOICE_AVM

    ctypedef enum avifCodecFlag:
        AVIF_CODEC_FLAG_CAN_DECODE
        AVIF_CODEC_FLAG_CAN_ENCODE

    ctypedef uint32_t avifCodecFlags

    const char* avifCodecName(
        avifCodecChoice choice,
        avifCodecFlags requiredFlags
    ) nogil

    avifCodecChoice avifCodecChoiceFromName(
        const char* name
    ) nogil

    # avifIO

    struct avifIO:
        pass

    ctypedef void (*avifIODestroyFunc)(
        avifIO* io
    ) nogil

    ctypedef avifResult (*avifIOReadFunc)(
        avifIO* io,
        uint32_t readFlags,
        uint64_t offset,
        size_t size,
        avifROData* out
    ) nogil

    ctypedef avifResult (*avifIOWriteFunc)(
        avifIO* io,
        uint32_t writeFlags,
        uint64_t offset,
        const uint8_t* data,
        size_t size
    ) nogil

    ctypedef struct avifIO:
        avifIODestroyFunc destroy
        avifIOReadFunc read
        avifIOWriteFunc write
        uint64_t sizeHint
        avifBool persistent
        void* data

    avifIO* avifIOCreateMemoryReader(
        const uint8_t* data,
        size_t size
    ) nogil

    avifIO* avifIOCreateFileReader(
        const char* filename
    ) nogil

    void avifIODestroy(
        avifIO* io
    ) nogil

    # avifDecoder

    ctypedef enum avifStrictFlag:
        AVIF_STRICT_DISABLED
        AVIF_STRICT_PIXI_REQUIRED
        AVIF_STRICT_CLAP_VALID
        AVIF_STRICT_ENABLED

    ctypedef uint32_t avifStrictFlags

    ctypedef struct avifIOStats:
        size_t colorOBUSize
        size_t alphaOBUSize

    struct avifDecoderData:
        pass

    ctypedef enum avifDecoderSource:
        AVIF_DECODER_SOURCE_AUTO
        AVIF_DECODER_SOURCE_PRIMARY_ITEM
        AVIF_DECODER_SOURCE_TRACKS

    ctypedef struct avifImageTiming:
        uint64_t timescale
        double pts
        uint64_t ptsInTimescales
        double duration
        uint64_t durationInTimescales

    ctypedef enum avifProgressiveState:
        AVIF_PROGRESSIVE_STATE_UNAVAILABLE
        AVIF_PROGRESSIVE_STATE_AVAILABLE
        AVIF_PROGRESSIVE_STATE_ACTIVE

    const char * avifProgressiveStateToString(
        avifProgressiveState progressiveState
    ) nogil

    ctypedef struct avifDecoder:
        avifCodecChoice codecChoice
        int maxThreads
        avifDecoderSource requestedSource
        avifBool allowProgressive
        avifBool allowIncremental
        avifBool ignoreExif
        avifBool ignoreXMP
        uint32_t imageSizeLimit
        uint32_t imageDimensionLimit
        uint32_t imageCountLimit
        avifStrictFlags strictFlags

        avifImage* image
        int imageIndex
        int imageCount
        avifProgressiveState progressiveState
        avifImageTiming imageTiming
        uint64_t timescale
        double duration
        uint64_t durationInTimescales
        int repetitionCount
        avifBool alphaPresent
        avifIOStats ioStats
        avifDiagnostics diag
        avifIO* io
        avifDecoderData* data

    avifDecoder* avifDecoderCreate() nogil

    void avifDecoderDestroy(
        avifDecoder* decoder
    ) nogil

    avifResult avifDecoderRead(
        avifDecoder* decoder,
        avifImage* image
    ) nogil

    avifResult avifDecoderReadMemory(
        avifDecoder* decoder,
        avifImage* image,
        const uint8_t* data,
        size_t size
    ) nogil

    avifResult avifDecoderReadFile(
        avifDecoder* decoder,
        avifImage* image,
        const char* filename
    ) nogil

    avifResult avifDecoderSetSource(
        avifDecoder* decoder,
        avifDecoderSource source
    ) nogil

    void avifDecoderSetIO(
        avifDecoder* decoder,
        avifIO* io
    ) nogil

    avifResult avifDecoderSetIOMemory(
        avifDecoder* decoder,
        const uint8_t* data,
        size_t size
    ) nogil

    avifResult avifDecoderSetIOFile(
        avifDecoder* decoder,
        const char* filename
    ) nogil

    avifResult avifDecoderParse(
        avifDecoder* decoder
    ) nogil

    avifResult avifDecoderNextImage(
        avifDecoder* decoder
    ) nogil

    avifResult avifDecoderNthImage(
        avifDecoder* decoder,
        uint32_t frameIndex
    ) nogil

    avifResult avifDecoderReset(
        avifDecoder* decoder
    ) nogil

    avifBool avifDecoderIsKeyframe(
        const avifDecoder* decoder,
        uint32_t frameIndex
    ) nogil

    uint32_t avifDecoderNearestKeyframe(
        const avifDecoder* decoder,
        uint32_t frameIndex
    ) nogil

    avifResult avifDecoderNthImageTiming(
        const avifDecoder* decoder,
        uint32_t frameIndex,
        avifImageTiming* outTiming
    ) nogil

    uint32_t avifDecoderDecodedRowCount(
        const avifDecoder* decoder
    ) nogil

    # avifEncoder

    struct avifEncoderData:
        pass

    struct avifCodecSpecificOptions:
        pass

    ctypedef struct avifScalingMode:
        avifFraction horizontal
        avifFraction vertical

    ctypedef struct avifEncoder:
        avifCodecChoice codecChoice
        int maxThreads
        int speed
        int keyframeInterval
        uint64_t timescale
        int repetitionCount
        uint32_t extraLayerCount
        int quality
        int qualityAlpha
        int minQuantizer
        int maxQuantizer
        int minQuantizerAlpha
        int maxQuantizerAlpha
        int tileRowsLog2
        int tileColsLog2
        avifBool autoTiling
        avifScalingMode scalingMode
        avifIOStats ioStats
        avifDiagnostics diag
        avifEncoderData* data
        avifCodecSpecificOptions* csOptions

    avifEncoder* avifEncoderCreate() nogil

    avifResult avifEncoderWrite(
        avifEncoder* encoder,
        const avifImage* image,
        avifRWData* output
    ) nogil

    void avifEncoderDestroy(
        avifEncoder* encoder
    ) nogil

    ctypedef enum avifAddImageFlag:
        AVIF_ADD_IMAGE_FLAG_NONE
        AVIF_ADD_IMAGE_FLAG_FORCE_KEYFRAME
        AVIF_ADD_IMAGE_FLAG_SINGLE

    ctypedef uint32_t avifAddImageFlags

    avifResult avifEncoderAddImage(
        avifEncoder* encoder,
        const avifImage* image,
        uint64_t durationInTimescales,
        avifAddImageFlags addImageFlags
    ) nogil

    avifResult avifEncoderAddImageGrid(
        avifEncoder* encoder,
        uint32_t gridCols,
        uint32_t gridRows,
        const avifImage* const* cellImages,
        avifAddImageFlags addImageFlags
    ) nogil

    avifResult avifEncoderFinish(
        avifEncoder* encoder,
        avifRWData* output
    ) nogil

    avifResult avifEncoderSetCodecSpecificOption(
        avifEncoder* encoder,
        const char* key,
        const char* value
    ) nogil

    # Helpers

    avifBool avifImageUsesU16(
        const avifImage* image
    ) nogil

    avifBool avifImageIsOpaque(
        const avifImage* image
    ) nogil

    uint8_t* avifImagePlane(
        const avifImage* image,
        int channel
    ) nogil

    uint32_t avifImagePlaneRowBytes(
        const avifImage* image,
        int channel
    ) nogil

    uint32_t avifImagePlaneWidth(
        const avifImage* image,
        int channel
    ) nogil

    uint32_t avifImagePlaneHeight(
        const avifImage* image,
        int channel
    ) nogil

    avifBool avifPeekCompatibleFileType(
        const avifROData* input
    ) nogil
