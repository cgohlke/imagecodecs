# imagecodecs/libavif.pxd
# cython: language_level = 3

# Cython declarations for the `libavif 0.9.0` library.
# https://github.com/AOMediaCodec/libavif

from libc.stdint cimport uint8_t, uint32_t, uint64_t

cdef extern from 'avif/avif.h':

    int AVIF_VERSION_MAJOR
    int AVIF_VERSION_MINOR
    int AVIF_VERSION_PATCH
    int AVIF_VERSION

    ctypedef int avifBool

    int AVIF_TRUE
    int AVIF_FALSE

    int AVIF_QUANTIZER_LOSSLESS
    int AVIF_QUANTIZER_BEST_QUALITY
    int AVIF_QUANTIZER_WORST_QUALITY

    int AVIF_PLANE_COUNT_YUV

    int AVIF_SPEED_DEFAULT
    int AVIF_SPEED_SLOWEST
    int AVIF_SPEED_FASTEST

    enum avifPlanesFlags:
        AVIF_PLANES_YUV
        AVIF_PLANES_A
        AVIF_PLANES_ALL

    enum avifChannelIndex:
        AVIF_CHAN_R
        AVIF_CHAN_G
        AVIF_CHAN_B
        AVIF_CHAN_Y
        AVIF_CHAN_U
        AVIF_CHAN_V

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
        AVIF_RESULT_NO_AV1_ITEMS_FOUND
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

    void avifRWDataRealloc(
        avifRWData* raw,
        size_t newSize
    ) nogil

    void avifRWDataSet(
        avifRWData* raw,
        const uint8_t* data,
        size_t len
    ) nogil

    void avifRWDataFree(
        avifRWData* raw
    ) nogil

    # avifPixelFormat

    ctypedef enum avifPixelFormat:
        AVIF_PIXEL_FORMAT_NONE
        AVIF_PIXEL_FORMAT_YUV444
        AVIF_PIXEL_FORMAT_YUV422
        AVIF_PIXEL_FORMAT_YUV420
        AVIF_PIXEL_FORMAT_YUV400

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

    # Optional transformation structs

    ctypedef enum avifTransformationFlags:
        AVIF_TRANSFORM_NONE
        AVIF_TRANSFORM_PASP
        AVIF_TRANSFORM_CLAP
        AVIF_TRANSFORM_IROT
        AVIF_TRANSFORM_IMIR

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
        avifRange alphaRange
        uint8_t* alphaPlane
        uint32_t alphaRowBytes
        avifBool imageOwnsAlphaPlane
        avifBool alphaPremultiplied
        avifRWData icc
        avifColorPrimaries colorPrimaries
        avifTransferCharacteristics transferCharacteristics
        avifMatrixCoefficients matrixCoefficients
        uint32_t transformFlags
        avifPixelAspectRatioBox pasp
        avifCleanApertureBox clap
        avifImageRotation irot
        avifImageMirror imir
        avifRWData exif
        avifRWData xmp

    avifImage* avifImageCreate(
        int width,
        int height,
        int depth,
        avifPixelFormat yuvFormat
    ) nogil

    avifImage* avifImageCreateEmpty() nogil

    void avifImageCopy(
        avifImage* dstImage,
        const avifImage* srcImage,
        uint32_t planes
    ) nogil

    void avifImageDestroy(
        avifImage* image
    ) nogil

    void avifImageSetProfileICC(
        avifImage* image,
        const uint8_t* icc,
        size_t iccSize
    ) nogil

    void avifImageSetMetadataExif(
        avifImage* image,
        const uint8_t* exif,
        size_t exifSize
    ) nogil

    void avifImageSetMetadataXMP(
        avifImage* image,
        const uint8_t* xmp,
        size_t xmpSize
    ) nogil

    void avifImageAllocatePlanes(
        avifImage* image,
        uint32_t planes
    ) nogil

    void avifImageFreePlanes(
        avifImage* image,
        uint32_t planes
    ) nogil

    void avifImageStealPlanes(
        avifImage* dstImage,
        avifImage* srcImage,
        uint32_t planes
    ) nogil

    ctypedef enum avifRGBFormat:
        AVIF_RGB_FORMAT_RGB
        AVIF_RGB_FORMAT_RGBA
        AVIF_RGB_FORMAT_ARGB
        AVIF_RGB_FORMAT_BGR
        AVIF_RGB_FORMAT_BGRA
        AVIF_RGB_FORMAT_ABGR

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

    ctypedef struct avifRGBImage:
        uint32_t width
        uint32_t height
        uint32_t depth
        avifRGBFormat format
        avifChromaUpsampling chromaUpsampling
        avifBool ignoreAlpha
        avifBool alphaPremultiplied
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

    void avifRGBImageAllocatePixels(
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
    )

    avifResult avifRGBImageUnpremultiplyAlpha(
        avifRGBImage* rgb
    )

    # YUV Utils

    int avifFullToLimitedY(
        int depth,
        int v
    ) nogil

    int avifFullToLimitedUV(
        int depth,
        int v
    ) nogil

    int avifLimitedToFullY(
        int depth,
        int v
    ) nogil

    int avifLimitedToFullUV(
        int depth,
        int v
    ) nogil

    # removed in v0.9
    #
    # ctypedef enum avifReformatMode:
    #     AVIF_REFORMAT_MODE_YUV_COEFFICIENTS
    #     AVIF_REFORMAT_MODE_IDENTITY

    # ctypedef struct avifReformatState:
    #     float kr
    #     float kg
    #     float kb
    #     uint32_t yuvChannelBytes
    #     uint32_t rgbChannelBytes
    #     uint32_t rgbChannelCount
    #     uint32_t rgbPixelBytes
    #     uint32_t rgbOffsetBytesR
    #     uint32_t rgbOffsetBytesG
    #     uint32_t rgbOffsetBytesB
    #     uint32_t rgbOffsetBytesA
    #     uint32_t yuvDepth
    #     uint32_t rgbDepth
    #     avifRange yuvRange
    #     int yuvMaxChannel
    #     int rgbMaxChannel
    #     float yuvMaxChannelF
    #     float rgbMaxChannelF
    #     int uvBias
    #     avifPixelFormatInfo formatInfo
    #     float unormFloatTableY[1 << 12]
    #     float unormFloatTableUV[1 << 12]
    #     avifReformatMode mode

    # avifBool avifPrepareReformatState(
    #     const avifImage* image,
    #     const avifRGBImage* rgb,
    #     avifReformatState* state
    # ) nogil

    # Codec selection

    ctypedef enum avifCodecChoice:
        AVIF_CODEC_CHOICE_AUTO
        AVIF_CODEC_CHOICE_AOM
        AVIF_CODEC_CHOICE_DAV1D
        AVIF_CODEC_CHOICE_LIBGAV1
        AVIF_CODEC_CHOICE_RAV1E
        AVIF_CODEC_CHOICE_SVT

    ctypedef enum avifCodecFlags:
        AVIF_CODEC_FLAG_CAN_DECODE
        AVIF_CODEC_FLAG_CAN_ENCODE

    const char* avifCodecName(
        avifCodecChoice choice,
        uint32_t requiredFlags
    ) nogil

    avifCodecChoice avifCodecChoiceFromName(
        const char* name
    ) nogil

    ctypedef struct avifCodecConfigurationBox:
        uint8_t seqProfile
        uint8_t seqLevelIdx0
        uint8_t seqTier0
        uint8_t highBitdepth
        uint8_t twelveBit
        uint8_t monochrome
        uint8_t chromaSubsamplingX
        uint8_t chromaSubsamplingY
        uint8_t chromaSamplePosition

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

    ctypedef struct avifDecoder:
        avifCodecChoice codecChoice
        avifDecoderSource requestedSource
        avifImage* image
        int imageIndex
        int imageCount
        avifImageTiming imageTiming
        uint64_t timescale
        double duration
        uint64_t durationInTimescales
        avifBool alphaPresent
        avifBool ignoreExif
        avifBool ignoreXMP
        avifIOStats ioStats
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

    # avifEncoder

    struct avifEncoderData:
        pass

    struct avifCodecSpecificOptions:
        pass

    ctypedef struct avifEncoder:
        avifCodecChoice codecChoice
        int maxThreads
        int minQuantizer
        int maxQuantizer
        int minQuantizerAlpha
        int maxQuantizerAlpha
        int tileRowsLog2
        int tileColsLog2
        int speed
        int keyframeInterval
        uint64_t timescale
        avifIOStats ioStats
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

    enum avifAddImageFlags:
        AVIF_ADD_IMAGE_FLAG_NONE
        AVIF_ADD_IMAGE_FLAG_FORCE_KEYFRAME
        AVIF_ADD_IMAGE_FLAG_SINGLE

    avifResult avifEncoderAddImage(
        avifEncoder* encoder,
        const avifImage* image,
        uint64_t durationInTimescales,
        uint32_t addImageFlags
    ) nogil

    avifResult avifEncoderAddImageGrid(
        avifEncoder* encoder,
        uint32_t gridCols,
        uint32_t gridRows,
        const avifImage* const *cellImages,
        uint32_t addImageFlags
    )

    avifResult avifEncoderFinish(
        avifEncoder* encoder,
        avifRWData* output
    ) nogil

    void avifEncoderSetCodecSpecificOption(
        avifEncoder* encoder,
        const char* key,
        const char* value
    ) nogil

    # Helpers

    avifBool avifImageUsesU16(
        const avifImage* image
    ) nogil

    avifBool avifPeekCompatibleFileType(
        const avifROData* input
    ) nogil
