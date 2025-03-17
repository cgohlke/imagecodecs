# imagecodecs/libavif.pxd
# cython: language_level = 3

# Cython declarations for the `libavif 1.2.1` library.
# https://github.com/AOMediaCodec/libavif

from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int32_t

cdef extern from 'avif/avif.h' nogil:

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

    const char* avifVersion()

    unsigned int avifLibYUVVersion()

    void avifCodecVersions(
        char* outBuffer
    )

    # Memory management

    void* avifAlloc(
        size_t size
    )

    void avifFree(
        void* p
    )

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
        AVIF_RESULT_INTERNAL_ERROR
        AVIF_RESULT_ENCODE_GAIN_MAP_FAILED
        AVIF_RESULT_DECODE_GAIN_MAP_FAILED
        AVIF_RESULT_INVALID_TONE_MAPPED_IMAGE
        AVIF_RESULT_ENCODE_SAMPLE_TRANSFORM_FAILED
        AVIF_RESULT_DECODE_SAMPLE_TRANSFORM_FAILED
        AVIF_RESULT_NO_AV1_ITEMS_FOUND

    ctypedef enum avifHeaderFormat:
        AVIF_HEADER_DEFAULT
        AVIF_HEADER_MINI
        AVIF_HEADER_EXTENDED_PIXI
        # deprecated
        AVIF_HEADER_FULL

    ctypedef int avifHeaderFormatFlags

    const char* avifResultToString(
        avifResult result
    )

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
    )

    avifResult avifRWDataSet(
        avifRWData* raw,
        const uint8_t* data,
        size_t len
    )

    void avifRWDataFree(
        avifRWData* raw
    )

    # Metadata

    avifResult avifGetExifTiffHeaderOffset(
        const uint8_t* exif,
        size_t exifSize,
        size_t* offset
    )

    avifResult avifGetExifOrientationOffset(
        const uint8_t* exif,
        size_t exifSize,
        size_t* offset
    )

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
    )

    ctypedef struct avifPixelFormatInfo:
        avifBool monochrome
        int chromaShiftX
        int chromaShiftY

    void avifGetPixelFormatInfo(
        avifPixelFormat format,
        avifPixelFormatInfo* info
    )

    # avifChromaSamplePosition

    ctypedef enum avifChromaSamplePosition:
        AVIF_CHROMA_SAMPLE_POSITION_UNKNOWN
        AVIF_CHROMA_SAMPLE_POSITION_VERTICAL
        AVIF_CHROMA_SAMPLE_POSITION_COLOCATED
        AVIF_CHROMA_SAMPLE_POSITION_RESERVED

    # avifRange

    ctypedef enum avifRange:
        AVIF_RANGE_LIMITED
        AVIF_RANGE_FULL

    # CICP enums - https://www.itu.int/rec/T-REC-H.273-201612-I/en

    ctypedef enum avifColorPrimaries:
        AVIF_COLOR_PRIMARIES_UNKNOWN
        AVIF_COLOR_PRIMARIES_BT709
        AVIF_COLOR_PRIMARIES_SRGB
        AVIF_COLOR_PRIMARIES_IEC61966_2_4
        AVIF_COLOR_PRIMARIES_UNSPECIFIED
        AVIF_COLOR_PRIMARIES_BT470M
        AVIF_COLOR_PRIMARIES_BT470BG
        AVIF_COLOR_PRIMARIES_BT601
        AVIF_COLOR_PRIMARIES_SMPTE240
        AVIF_COLOR_PRIMARIES_GENERIC_FILM
        AVIF_COLOR_PRIMARIES_BT2020
        AVIF_COLOR_PRIMARIES_BT2100
        AVIF_COLOR_PRIMARIES_XYZ
        AVIF_COLOR_PRIMARIES_SMPTE431
        AVIF_COLOR_PRIMARIES_SMPTE432
        AVIF_COLOR_PRIMARIES_DCI_P3
        AVIF_COLOR_PRIMARIES_EBU3213

    void avifColorPrimariesGetValues(
        avifColorPrimaries acp,
        float[8] outPrimaries
    )

    avifColorPrimaries avifColorPrimariesFind(
        const float[8] inPrimaries,
        const char** outName
    )

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
        AVIF_TRANSFER_CHARACTERISTICS_PQ
        AVIF_TRANSFER_CHARACTERISTICS_SMPTE2084
        AVIF_TRANSFER_CHARACTERISTICS_SMPTE428
        AVIF_TRANSFER_CHARACTERISTICS_HLG

    avifResult avifTransferCharacteristicsGetGamma(
        avifTransferCharacteristics atc,
        float* gamma
    )

    avifTransferCharacteristics avifTransferCharacteristicsFindByGamma(
        float gamma
    )

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
        char[256] error  # [AVIF_DIAGNOSTICS_ERROR_BUFFER_SIZE]

    void avifDiagnosticsClearError(avifDiagnostics* diag)

    # Fraction utility

    ctypedef struct avifFraction:
        int32_t n
        int32_t d

    ctypedef struct avifSignedFraction:
        int32_t n
        uint32_t d

    ctypedef struct avifUnsignedFraction:
        uint32_t n
        uint32_t d

    avifBool avifDoubleToSignedFraction(
        double v,
        avifSignedFraction* fraction
    )

    avifBool avifDoubleToUnsignedFraction(
        double v,
        avifUnsignedFraction* fraction
    )

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

    avifBool avifCropRectFromCleanApertureBox(
        avifCropRect* cropRect,
        const avifCleanApertureBox* clap,
        uint32_t imageW,
        uint32_t imageH,
        avifDiagnostics* diag
    )

    avifBool avifCleanApertureBoxFromCropRect(
        avifCleanApertureBox* clap,
        const avifCropRect* cropRect,
        uint32_t imageW,
        uint32_t imageH,
        avifDiagnostics* diag
    )

    avifBool avifCropRectRequiresUpsampling(
        const avifCropRect* cropRect,
        avifPixelFormat yuvFormat
    )

    avifBool avifCropRectConvertCleanApertureBox(
        avifCropRect* cropRect,
        const avifCleanApertureBox* clap,
        uint32_t imageW,
        uint32_t imageH,
        avifPixelFormat yuvFormat,
        avifDiagnostics* diag
    )

    avifBool avifCleanApertureBoxConvertCropRect(
        avifCleanApertureBox* clap,
        const avifCropRect* cropRect,
        uint32_t imageW,
        uint32_t imageH,
        avifPixelFormat yuvFormat,
        avifDiagnostics* diag
    )

    ctypedef struct avifContentLightLevelInformationBox:
        uint16_t maxCLL
        uint16_t maxPALL

    # avifImage

    ctypedef struct avifGainMap:
        avifImage* image
        avifSignedFraction[3] gainMapMin
        avifSignedFraction[3] gainMapMax
        avifUnsignedFraction[3] gainMapGamma
        avifSignedFraction[3] baseOffset
        avifSignedFraction[3] alternateOffset
        avifUnsignedFraction baseHdrHeadroom
        avifUnsignedFraction alternateHdrHeadroom
        avifRWData altICC
        avifColorPrimaries altColorPrimaries
        avifTransferCharacteristics altTransferCharacteristics
        avifMatrixCoefficients altMatrixCoefficients
        avifRange altYUVRange
        uint32_t altDepth
        uint32_t altPlaneCount
        avifContentLightLevelInformationBox altCLLI

    avifGainMap* avifGainMapCreate()

    void avifGainMapDestroy(
        avifGainMap* gainMap
    )

    ctypedef enum avifSampleTransformRecipe:
        AVIF_SAMPLE_TRANSFORM_NONE
        AVIF_SAMPLE_TRANSFORM_BIT_DEPTH_EXTENSION_8B_8B
        AVIF_SAMPLE_TRANSFORM_BIT_DEPTH_EXTENSION_12B_4B
        AVIF_SAMPLE_TRANSFORM_BIT_DEPTH_EXTENSION_12B_8B_OVERLAP_4B

    ctypedef struct avifImageItemProperty:
        uint8_t[4] boxtype
        uint8_t[16] usertype
        avifRWData boxPayload

    ctypedef struct avifImage:
        uint32_t width
        uint32_t height
        uint32_t depth
        avifPixelFormat yuvFormat
        avifRange yuvRange
        avifChromaSamplePosition yuvChromaSamplePosition
        (uint8_t*)[3] yuvPlanes  # AVIF_PLANE_COUNT_YUV = 3
        uint32_t[3] yuvRowBytes  # AVIF_PLANE_COUNT_YUV = 3
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
        avifImageItemProperty* properties
        size_t numProperties
        avifGainMap* gainMap

    avifImage* avifImageCreate(
        uint32_t width,
        uint32_t height,
        uint32_t depth,
        avifPixelFormat yuvFormat
    )

    avifImage* avifImageCreateEmpty()

    avifResult avifImageCopy(
        avifImage* dstImage,
        const avifImage* srcImage,
        avifPlanesFlags planes
    )

    avifResult avifImageSetViewRect(
        avifImage* dstImage,
        const avifImage* srcImage,
        const avifCropRect* rect
    )

    void avifImageDestroy(
        avifImage* image
    )

    avifResult avifImageSetProfileICC(
        avifImage* image,
        const uint8_t* icc,
        size_t iccSize
    )

    avifResult avifImageSetMetadataExif(
        avifImage* image,
        const uint8_t* exif,
        size_t exifSize
    )

    avifResult avifImageSetMetadataXMP(
        avifImage* image,
        const uint8_t* xmp,
        size_t xmpSize
    )

    avifResult avifImageAllocatePlanes(
        avifImage* image,
        avifPlanesFlags planes
    )

    void avifImageFreePlanes(
        avifImage* image,
        avifPlanesFlags planes
    )

    void avifImageStealPlanes(
        avifImage* dstImage,
        avifImage* srcImage,
        avifPlanesFlags planes
    )

    avifResult avifImageAddOpaqueProperty(
        avifImage* image,
        const uint8_t[4] boxtype,
        const uint8_t* data,
        size_t dataSize
    )

    avifResult avifImageAddUUIDProperty(
        avifImage* image,
        const uint8_t[16] uuid,
        const uint8_t* data,
        size_t dataSize
    )

    avifResult avifImageScale(
        avifImage* image,
        uint32_t dstWidth,
        uint32_t dstHeight,
        avifDiagnostics* diag
    )

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
    )

    avifBool avifRGBFormatHasAlpha(
        avifRGBFormat format
    )

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
    )

    uint32_t avifRGBImagePixelSize(
        const avifRGBImage* rgb
    )

    # Convenience functions

    avifResult avifRGBImageAllocatePixels(
        avifRGBImage* rgb
    )

    void avifRGBImageFreePixels(
        avifRGBImage* rgb
    )

    # The main conversion functions

    avifResult avifImageRGBToYUV(
        avifImage* image,
        const avifRGBImage* rgb
    )

    avifResult avifImageYUVToRGB(
        const avifImage* image,
        avifRGBImage* rgb
    )

    # Premultiply handling functions

    avifResult avifRGBImagePremultiplyAlpha(
        avifRGBImage* rgb
    )

    avifResult avifRGBImageUnpremultiplyAlpha(
        avifRGBImage* rgb
    )

    # YUV Utils

    int avifFullToLimitedY(
        uint32_t depth,
        int v
    )

    int avifFullToLimitedUV(
        uint32_t depth,
        int v
    )

    int avifLimitedToFullY(
        uint32_t depth,
        int v
    )

    int avifLimitedToFullUV(
        uint32_t depth,
        int v
    )

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
    )

    avifCodecChoice avifCodecChoiceFromName(
        const char* name
    )

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
    )

    avifIO* avifIOCreateFileReader(
        const char* filename
    )

    void avifIODestroy(
        avifIO* io
    )

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
    )

    ctypedef enum avifImageContentTypeFlag:
        AVIF_IMAGE_CONTENT_NONE
        AVIF_IMAGE_CONTENT_COLOR_AND_ALPHA
        AVIF_IMAGE_CONTENT_GAIN_MAP
        AVIF_IMAGE_CONTENT_ALL
        AVIF_IMAGE_CONTENT_DECODE_DEFAULT

    ctypedef uint32_t avifImageContentTypeFlags

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
        avifBool imageSequenceTrackPresent
        avifImageContentTypeFlags imageContentToDecode

    avifDecoder* avifDecoderCreate()

    void avifDecoderDestroy(
        avifDecoder* decoder
    )

    avifResult avifDecoderRead(
        avifDecoder* decoder,
        avifImage* image
    )

    avifResult avifDecoderReadMemory(
        avifDecoder* decoder,
        avifImage* image,
        const uint8_t* data,
        size_t size
    )

    avifResult avifDecoderReadFile(
        avifDecoder* decoder,
        avifImage* image,
        const char* filename
    )

    avifResult avifDecoderSetSource(
        avifDecoder* decoder,
        avifDecoderSource source
    )

    void avifDecoderSetIO(
        avifDecoder* decoder,
        avifIO* io
    )

    avifResult avifDecoderSetIOMemory(
        avifDecoder* decoder,
        const uint8_t* data,
        size_t size
    )

    avifResult avifDecoderSetIOFile(
        avifDecoder* decoder,
        const char* filename
    )

    avifResult avifDecoderParse(
        avifDecoder* decoder
    )

    avifResult avifDecoderNextImage(
        avifDecoder* decoder
    )

    avifResult avifDecoderNthImage(
        avifDecoder* decoder,
        uint32_t frameIndex
    )

    avifResult avifDecoderReset(
        avifDecoder* decoder
    )

    avifBool avifDecoderIsKeyframe(
        const avifDecoder* decoder,
        uint32_t frameIndex
    )

    uint32_t avifDecoderNearestKeyframe(
        const avifDecoder* decoder,
        uint32_t frameIndex
    )

    avifResult avifDecoderNthImageTiming(
        const avifDecoder* decoder,
        uint32_t frameIndex,
        avifImageTiming* outTiming
    )

    uint32_t avifDecoderDecodedRowCount(
        const avifDecoder* decoder
    )

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
        avifHeaderFormatFlags headerFormat
        int qualityGainMap
        avifSampleTransformRecipe sampleTransformRecipe

    avifEncoder* avifEncoderCreate()

    avifResult avifEncoderWrite(
        avifEncoder* encoder,
        const avifImage* image,
        avifRWData* output
    )

    void avifEncoderDestroy(
        avifEncoder* encoder
    )

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
    )

    avifResult avifEncoderAddImageGrid(
        avifEncoder* encoder,
        uint32_t gridCols,
        uint32_t gridRows,
        const avifImage* const* cellImages,
        avifAddImageFlags addImageFlags
    )

    avifResult avifEncoderFinish(
        avifEncoder* encoder,
        avifRWData* output
    )

    avifResult avifEncoderSetCodecSpecificOption(
        avifEncoder* encoder,
        const char* key,
        const char* value
    )

    size_t avifEncoderGetGainMapSizeBytes(
        avifEncoder* encoder
    )

    # Helpers

    avifBool avifImageUsesU16(
        const avifImage* image
    )

    avifBool avifImageIsOpaque(
        const avifImage* image
    )

    uint8_t* avifImagePlane(
        const avifImage* image,
        int channel
    )

    uint32_t avifImagePlaneRowBytes(
        const avifImage* image,
        int channel
    )

    uint32_t avifImagePlaneWidth(
        const avifImage* image,
        int channel
    )

    uint32_t avifImagePlaneHeight(
        const avifImage* image,
        int channel
    )

    avifBool avifPeekCompatibleFileType(
        const avifROData* input
    )

    avifResult avifImageApplyGainMap(
        const avifImage* baseImage,
        const avifGainMap* gainMap,
        float hdrHeadroom,
        avifColorPrimaries outputColorPrimaries,
        avifTransferCharacteristics outputTransferCharacteristics,
        avifRGBImage* toneMappedImage,
        avifContentLightLevelInformationBox* clli,
        avifDiagnostics* diag
    )

    avifResult avifRGBImageApplyGainMap(
        const avifRGBImage* baseImage,
        avifColorPrimaries baseColorPrimaries,
        avifTransferCharacteristics baseTransferCharacteristics,
        const avifGainMap* gainMap,
        float hdrHeadroom,
        avifColorPrimaries outputColorPrimaries,
        avifTransferCharacteristics outputTransferCharacteristics,
        avifRGBImage* toneMappedImage,
        avifContentLightLevelInformationBox* clli,
        avifDiagnostics* diag
    )

    avifResult avifRGBImageComputeGainMap(
        const avifRGBImage* baseRgbImage,
        avifColorPrimaries baseColorPrimaries,
        avifTransferCharacteristics baseTransferCharacteristics,
        const avifRGBImage* altRgbImage,
        avifColorPrimaries altColorPrimaries,
        avifTransferCharacteristics altTransferCharacteristics,
        avifGainMap* gainMap,
        avifDiagnostics* diag
    )

    avifResult avifImageComputeGainMap(
        const avifImage* baseImage,
        const avifImage* altImage,
        avifGainMap* gainMap,
        avifDiagnostics* diag
    )
