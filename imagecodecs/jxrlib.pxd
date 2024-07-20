# imagecodecs/jxrlib.pxd
# cython: language_level = 3

# Cython declarations for the `jxrlib 1.1` library.
# https://packages.debian.org/sid/libjxr-dev
# https://salsa.debian.org/debian-phototools-team/jxrlib
# https://github.com/4creators/jxrlib
# https://github.com/glencoesoftware/jxrlib

from libc.stdio cimport FILE

cdef extern from 'windowsmediaphoto.h' nogil:

    ctypedef long ERR
    ctypedef int Bool
    ctypedef char Char
    ctypedef double Double
    ctypedef int Int
    ctypedef signed char I8
    ctypedef short I16
    ctypedef int I32
    ctypedef long Long
    ctypedef unsigned char PixelC
    ctypedef int PixelI
    ctypedef unsigned int UInt
    ctypedef unsigned long ULong
    ctypedef unsigned char U8
    ctypedef unsigned short U16
    ctypedef unsigned int U32
    ctypedef void Void

    ctypedef void* CTXSTRCODEC

    ctypedef enum ERR_CODE:
        ICERR_OK
        ICERR_ERROR

    ctypedef enum BITDEPTH:
        BD_SHORT
        BD_LONG
        BD_MAX

    ctypedef enum BITDEPTH_BITS:
        BD_1
        BD_8
        BD_16
        BD_16S
        BD_16F
        BD_32
        BD_32S
        BD_32F
        BD_5
        BD_10
        BD_565
        BDB_MAX
        BD_1alt

    ctypedef enum OVERLAP:
        OL_NONE
        OL_ONE
        OL_TWO
        OL_MAX

    ctypedef enum BITSTREAMFORMAT:
        SPATIAL
        FREQUENCY

    ctypedef enum COLORFORMAT:
        Y_ONLY
        YUV_420
        YUV_422
        YUV_444
        CMYK
        # CMYKDIRECT
        NCOMPONENT
        CF_RGB
        CF_RGBE
        CFT_MAX

    ctypedef enum ORIENTATION:
        O_NONE
        O_FLIPV
        O_FLIPH
        O_FLIPVH
        O_RCW
        O_RCW_FLIPV
        O_RCW_FLIPH
        O_RCW_FLIPVH
        O_MAX

    ctypedef enum SUBBAND:
        SB_ALL
        SB_NO_FLEXBITS
        SB_NO_HIGHPASS
        SB_DC_ONLY
        SB_ISOLATED
        SB_MAX

    # enum:
    #     RAW
    #     BMP
    #     PPM
    #     TIF
    #     HDR
    #     IYUV
    #     YUV422
    #     YUV444

    ctypedef enum WMIDecoderStatus:
        ERROR_FAIL
        SUCCESS_DONE
        PRE_READ_HDR
        PRE_SETUP
        PRE_DECODE
        POST_READ_HDR

    int REENTRANT_MODE

    int MAX_CHANNELS
    int LOG_MAX_TILES
    int MAX_TILES

    int MB_WIDTH_PIXEL
    int MB_HEIGHT_PIXEL

    int BLK_WIDTH_PIXEL
    int BLK_HEIGHT_PIXEL

    int MB_WIDTH_BLK
    int MB_HEIGHT_BLK

    int FRAMEBUFFER_ALIGNMENT

    ERR WMP_errSuccess
    ERR WMP_errFail
    ERR WMP_errNotYetImplemented
    ERR WMP_errAbstractMethod
    ERR WMP_errOutOfMemory
    ERR WMP_errFileIO
    ERR WMP_errBufferOverflow
    ERR WMP_errInvalidParameter
    ERR WMP_errInvalidArgument
    ERR WMP_errUnsupportedFormat
    ERR WMP_errIncorrectCodecVersion
    ERR WMP_errIndexNotFound
    ERR WMP_errOutOfSequence
    ERR WMP_errNotInitialized
    ERR WMP_errMustBeMultipleOf16LinesUntilLastCall
    ERR WMP_errPlanarAlphaBandedEncRequiresTempFile
    ERR WMP_errAlphaModeCannotBeTranscoded
    ERR WMP_errIncorrectCodecSubVersion

    ctypedef ERR (*WMPStream_CloseFunction)(
        WMPStream**
    ) nogil

    ctypedef ERR (*WMPStream_ReadFunction)(
        WMPStream*,
        void* pv,
        size_t c
    ) nogil

    ctypedef ERR (*WMPStream_WriteFunction)(
        WMPStream*,
        const void* pv,
        size_t c
    ) nogil

    ctypedef ERR (*WMPStream_SetPosFunction)(
        WMPStream*,
        size_t offPos
    ) nogil

    ctypedef ERR (*WMPStream_GetPosFunction)(
        WMPStream*,
        size_t* poffPos
    ) nogil

    ctypedef Bool (*WMPStream_EOSFunction)(
        WMPStream*
    ) nogil

    cdef struct WMPStream_buf:
        U8* pbBuf
        size_t cbBuf
        size_t cbCur
        size_t cbBufCount

    cdef union WMPStream_state:
        # FILE* pFile
        WMPStream_buf buf
        # void* pvObj

    cdef struct WMPStream:
        WMPStream_state state
        WMPStream_CloseFunction Close
        WMPStream_ReadFunction Read
        WMPStream_WriteFunction Write
        WMPStream_SetPosFunction SetPos
        WMPStream_GetPosFunction GetPos
        WMPStream_EOSFunction EOS

    ERR CreateWS_File(
        WMPStream** ppWS,
        const char* szFilename,
        const char* szMode
    )

    ERR CloseWS_File(
        WMPStream** ppWS
    )

    ERR CreateWS_Memory(
        WMPStream** ppWS,
        void* pv,
        size_t cb
    )

    ERR CloseWS_Memory(
        WMPStream** ppWS
    )

    ctypedef struct CWMImageInfo:
        size_t cWidth
        size_t cHeight
        COLORFORMAT cfColorFormat
        BITDEPTH_BITS bdBitDepth
        size_t cBitsPerUnit
        size_t cLeadingPadding
        Bool bRGB
        U8 cChromaCenteringX
        U8 cChromaCenteringY
        size_t cROILeftX
        size_t cROIWidth
        size_t cROITopY
        size_t cROIHeight
        Bool   bSkipFlexbits
        size_t cThumbnailWidth
        size_t cThumbnailHeight
        ORIENTATION oOrientation
        U8 cPostProcStrength
        Bool fPaddedUserBuffer

    ctypedef struct CWMIStrCodecParam:
        Bool bVerbose
        U8 uiDefaultQPIndex
        U8 uiDefaultQPIndexYLP
        U8 uiDefaultQPIndexYHP
        U8 uiDefaultQPIndexU
        U8 uiDefaultQPIndexULP
        U8 uiDefaultQPIndexUHP
        U8 uiDefaultQPIndexV
        U8 uiDefaultQPIndexVLP
        U8 uiDefaultQPIndexVHP
        U8 uiDefaultQPIndexAlpha
        COLORFORMAT cfColorFormat
        BITDEPTH bdBitDepth
        OVERLAP olOverlap
        BITSTREAMFORMAT bfBitstreamFormat
        size_t cChannel
        U8 uAlphaMode
        SUBBAND sbSubband
        U8 uiTrimFlexBits
        WMPStream* pWStream
        size_t cbStream
        U32  cNumOfSliceMinus1V
        U32* uiTileX  # [MAX_TILES]
        U32  cNumOfSliceMinus1H
        U32* uiTileY  # [MAX_TILES]
        U8 nLenMantissaOrShift
        I8 nExpBias
        Bool bBlackWhite
        Bool bUseHardTileBoundaries
        Bool bProgressiveMode
        Bool bYUVData
        Bool bUnscaledArith
        Bool fMeasurePerf

    ctypedef struct CWMImageBufferInfo:
        void* pv
        size_t cLine
        size_t cbStride
        # ifdef REENTRANT_MODE
        unsigned int uiFirstMBRow
        unsigned int uiLastMBRow
        size_t cLinesDecoded

    Int ImageStrEncInit(
        CWMImageInfo* pII,
        CWMIStrCodecParam* pSCP,
        CTXSTRCODEC* pctxSC
    )

    Int ImageStrEncEncode(
        CTXSTRCODEC ctxSC,
        const CWMImageBufferInfo* pBI
    )

    Int ImageStrEncTerm(
        CTXSTRCODEC ctxSC
    )

    Int ImageStrDecGetInfo(
        CWMImageInfo* pII,
        CWMIStrCodecParam* pSCP
    )

    Int ImageStrDecInit(
        CWMImageInfo* pII,
        CWMIStrCodecParam* pSCP,
        CTXSTRCODEC* pctxSC
    )

    Int ImageStrDecDecode(
        CTXSTRCODEC ctxSC,
        const CWMImageBufferInfo* pBI,
        # size_t* pcDecodedLines  # ifdef REENTRANT_MODE
    )

    Int ImageStrDecTerm(
        CTXSTRCODEC ctxSC
    )

    Int WMPhotoValidate(
        CWMImageInfo* pII,
        CWMIStrCodecParam *pSCP
    )

    ctypedef struct CWMTranscodingParam:
        size_t cLeftX
        size_t cWidth
        size_t cTopY
        size_t cHeight
        BITSTREAMFORMAT bfBitstreamFormat
        U8 uAlphaMode
        SUBBAND sbSubband
        ORIENTATION oOrientation
        Bool bIgnoreOverlap

    Int WMPhotoTranscode(
        WMPStream* pStreamDec,
        WMPStream* pStreamEnc,
        CWMTranscodingParam* pParam
    )

    ctypedef struct CWMDetilingParam:
        size_t cWidth
        size_t cHeight
        size_t cChannel
        OVERLAP olOverlap
        BITDEPTH_BITS bdBitdepth
        U32 cNumOfSliceMinus1V
        U32* uiTileX  # [MAX_TILES]
        U32 cNumOfSliceMinus1H
        U32* uiTileY  # [MAX_TILES]
        void* pImage
        size_t cbStride

    Int WMPhotoDetile(
        CWMDetilingParam* pParam
    )


cdef extern from 'guiddef.h' nogil:

    ctypedef struct GUID:
        pass

    int IsEqualGUID(
        GUID*,
        GUID*
    )


cdef extern from 'JXRMeta.h' nogil:

    int WMP_tagNull
    int WMP_tagDocumentName
    int WMP_tagImageDescription
    int WMP_tagCameraMake
    int WMP_tagCameraModel
    int WMP_tagPageName
    int WMP_tagPageNumber
    int WMP_tagSoftware
    int WMP_tagDateTime
    int WMP_tagArtist
    int WMP_tagHostComputer
    int WMP_tagXMPMetadata
    int WMP_tagRatingStars
    int WMP_tagRatingValue
    int WMP_tagCopyright
    int WMP_tagEXIFMetadata
    int WMP_tagGPSInfoMetadata
    int WMP_tagIPTCNAAMetadata
    int WMP_tagPhotoshopMetadata
    int WMP_tagInteroperabilityIFD
    int WMP_tagIccProfile
    int WMP_tagCaption
    int WMP_tagPixelFormat
    int WMP_tagTransformation
    int WMP_tagCompression
    int WMP_tagImageType
    int WMP_tagImageWidth
    int WMP_tagImageHeight
    int WMP_tagWidthResolution
    int WMP_tagHeightResolution
    int WMP_tagImageOffset
    int WMP_tagImageByteCount
    int WMP_tagAlphaOffset
    int WMP_tagAlphaByteCount
    int WMP_tagImageDataDiscard
    int WMP_tagAlphaDataDiscard

    int WMP_typBYTE
    int WMP_typASCII
    int WMP_typSHORT
    int WMP_typLONG
    int WMP_typRATIONAL
    int WMP_typSBYTE
    int WMP_typUNDEFINED
    int WMP_typSSHORT
    int WMP_typSLONG
    int WMP_typSRATIONAL
    int WMP_typFLOAT
    int WMP_typDOUBLE

    int WMP_valCompression
    int WMP_valWMPhotoID

    ctypedef enum DPKVARTYPE:
        DPKVT_EMPTY
        DPKVT_UI1
        DPKVT_UI2
        DPKVT_UI4
        DPKVT_LPSTR
        DPKVT_LPWSTR
        DPKVT_BYREF

    cdef union DPKPROPVARIANT_VT:
        U8 bVal
        U16 uiVal
        U32 ulVal
        char* pszVal
        U16* pwszVal
        U8* pbVal

    ctypedef struct DPKPROPVARIANT:
        DPKVARTYPE vt
        DPKPROPVARIANT_VT VT

    ctypedef struct DESCRIPTIVEMETADATA:
        DPKPROPVARIANT pvarImageDescription
        DPKPROPVARIANT pvarCameraMake
        DPKPROPVARIANT pvarCameraModel
        DPKPROPVARIANT pvarSoftware
        DPKPROPVARIANT pvarDateTime
        DPKPROPVARIANT pvarArtist
        DPKPROPVARIANT pvarCopyright
        DPKPROPVARIANT pvarRatingStars
        DPKPROPVARIANT pvarRatingValue
        DPKPROPVARIANT pvarCaption
        DPKPROPVARIANT pvarDocumentName
        DPKPROPVARIANT pvarPageName
        DPKPROPVARIANT pvarPageNumber
        DPKPROPVARIANT pvarHostComputer

    ctypedef struct WmpDE:
        U16 uTag
        U16 uType
        U32 uCount
        U32 uValueOrOffset

    ctypedef struct WmpDEMisc:
        U32 uImageOffset
        U32 uImageByteCount
        U32 uAlphaOffset
        U32 uAlphaByteCount
        U32 uOffPixelFormat
        U32 uOffImageByteCount
        U32 uOffAlphaOffset
        U32 uOffAlphaByteCount
        U32 uColorProfileOffset
        U32 uColorProfileByteCount
        U32 uXMPMetadataOffset
        U32 uXMPMetadataByteCount
        U32 uEXIFMetadataOffset
        U32 uEXIFMetadataByteCount
        U32 uGPSInfoMetadataOffset
        U32 uGPSInfoMetadataByteCount
        U32 uIPTCNAAMetadataOffset
        U32 uIPTCNAAMetadataByteCount
        U32 uPhotoshopMetadataOffset
        U32 uPhotoshopMetadataByteCount
        U32 uDescMetadataOffset
        U32 uDescMetadataByteCount


cdef extern from 'JXRGlue.h' nogil:

    int WMP_SDK_VERSION
    int PK_SDK_VERSION

    ctypedef float Float
    ctypedef U32 PKIID
    ctypedef unsigned long WMP_GRBIT
    ctypedef GUID PKPixelFormatGUID

    GUID GUID_PKPixelFormatDontCare
    # bool
    GUID GUID_PKPixelFormatBlackWhite
    # uint8
    GUID GUID_PKPixelFormat8bppGray
    GUID GUID_PKPixelFormat16bppRGB555
    GUID GUID_PKPixelFormat16bppRGB565
    GUID GUID_PKPixelFormat24bppBGR
    GUID GUID_PKPixelFormat24bppRGB
    GUID GUID_PKPixelFormat32bppRGB
    GUID GUID_PKPixelFormat32bppRGBA
    GUID GUID_PKPixelFormat32bppBGRA
    GUID GUID_PKPixelFormat32bppPRGBA
    GUID GUID_PKPixelFormat32bppRGBE
    GUID GUID_PKPixelFormat32bppCMYK
    GUID GUID_PKPixelFormat40bppCMYKAlpha
    GUID GUID_PKPixelFormat24bpp3Channels
    GUID GUID_PKPixelFormat32bpp4Channels
    GUID GUID_PKPixelFormat40bpp5Channels
    GUID GUID_PKPixelFormat48bpp6Channels
    GUID GUID_PKPixelFormat56bpp7Channels
    GUID GUID_PKPixelFormat64bpp8Channels
    GUID GUID_PKPixelFormat32bpp3ChannelsAlpha
    GUID GUID_PKPixelFormat40bpp4ChannelsAlpha
    GUID GUID_PKPixelFormat48bpp5ChannelsAlpha
    GUID GUID_PKPixelFormat56bpp6ChannelsAlpha
    GUID GUID_PKPixelFormat64bpp7ChannelsAlpha
    GUID GUID_PKPixelFormat72bpp8ChannelsAlpha
    # uint16
    GUID GUID_PKPixelFormat16bppGray
    GUID GUID_PKPixelFormat32bppRGB101010
    GUID GUID_PKPixelFormat48bppRGB
    GUID GUID_PKPixelFormat64bppRGBA
    GUID GUID_PKPixelFormat64bppPRGBA
    GUID GUID_PKPixelFormat64bppCMYK
    GUID GUID_PKPixelFormat80bppCMYKAlpha
    GUID GUID_PKPixelFormat48bpp3Channels
    GUID GUID_PKPixelFormat64bpp4Channels
    GUID GUID_PKPixelFormat80bpp5Channels
    GUID GUID_PKPixelFormat96bpp6Channels
    GUID GUID_PKPixelFormat112bpp7Channels
    GUID GUID_PKPixelFormat128bpp8Channels
    GUID GUID_PKPixelFormat64bpp3ChannelsAlpha
    GUID GUID_PKPixelFormat80bpp4ChannelsAlpha
    GUID GUID_PKPixelFormat96bpp5ChannelsAlpha
    GUID GUID_PKPixelFormat112bpp6ChannelsAlpha
    GUID GUID_PKPixelFormat128bpp7ChannelsAlpha
    GUID GUID_PKPixelFormat144bpp8ChannelsAlpha
    # float16
    GUID GUID_PKPixelFormat16bppGrayHalf
    GUID GUID_PKPixelFormat48bppRGBHalf
    GUID GUID_PKPixelFormat64bppRGBHalf
    GUID GUID_PKPixelFormat64bppRGBAHalf
    # float32
    GUID GUID_PKPixelFormat32bppGrayFloat
    GUID GUID_PKPixelFormat96bppRGBFloat
    GUID GUID_PKPixelFormat128bppRGBFloat
    GUID GUID_PKPixelFormat128bppRGBAFloat
    GUID GUID_PKPixelFormat128bppPRGBAFloat
    # fixed
    GUID GUID_PKPixelFormat48bppRGBFixedPoint
    GUID GUID_PKPixelFormat16bppGrayFixedPoint
    GUID GUID_PKPixelFormat96bppRGBFixedPoint
    GUID GUID_PKPixelFormat64bppRGBAFixedPoint
    GUID GUID_PKPixelFormat64bppRGBFixedPoint
    GUID GUID_PKPixelFormat128bppRGBAFixedPoint
    GUID GUID_PKPixelFormat128bppRGBFixedPoint
    GUID GUID_PKPixelFormat32bppGrayFixedPoint
    # YCrCb from Advanced Profile
    GUID GUID_PKPixelFormat12bppYCC420
    GUID GUID_PKPixelFormat16bppYCC422
    GUID GUID_PKPixelFormat20bppYCC422
    GUID GUID_PKPixelFormat32bppYCC422
    GUID GUID_PKPixelFormat24bppYCC444
    GUID GUID_PKPixelFormat30bppYCC444
    GUID GUID_PKPixelFormat48bppYCC444
    GUID GUID_PKPixelFormat16bpp48bppYCC444FixedPoint
    GUID GUID_PKPixelFormat20bppYCC420Alpha
    GUID GUID_PKPixelFormat24bppYCC422Alpha
    GUID GUID_PKPixelFormat30bppYCC422Alpha
    GUID GUID_PKPixelFormat48bppYCC422Alpha
    GUID GUID_PKPixelFormat32bppYCC444Alpha
    GUID GUID_PKPixelFormat40bppYCC444Alpha
    GUID GUID_PKPixelFormat64bppYCC444Alpha
    GUID GUID_PKPixelFormat64bppYCC444AlphaFixedPoint
    # CMYKDIRECT from Advanced Profile
    GUID GUID_PKPixelFormat32bppCMYKDIRECT
    GUID GUID_PKPixelFormat64bppCMYKDIRECT
    GUID GUID_PKPixelFormat40bppCMYKDIRECTAlpha
    GUID GUID_PKPixelFormat80bppCMYKDIRECTAlpha

    int PK_PI_W0
    int PK_PI_B0
    int PK_PI_RGB
    int PK_PI_RGBPalette
    int PK_PI_TransparencyMask
    int PK_PI_CMYK
    int PK_PI_YCbCr
    int PK_PI_CIELab
    int PK_PI_NCH
    int PK_PI_RGBE

    int PK_pixfmtNul
    int PK_pixfmtHasAlpha
    int PK_pixfmtPreMul
    int PK_pixfmtBGR
    int PK_pixfmtNeedConvert

    int LOOKUP_FORWARD
    int LOOKUP_BACKWARD_TIF

    U32* IFDEntryTypeSizes
    U32 SizeofIFDEntry

    PKIID IID_PKImageScanEncode
    PKIID IID_PKImageFrameEncode
    PKIID IID_PKImageWmpEncode
    PKIID IID_PKImageWmpDecode

    ctypedef struct PKRect:
        I32 X
        I32 Y
        I32 Width
        I32 Height

    ctypedef struct IFDEntry:
        U16 uTag
        U16 uType
        U32 uCount
        U32 uValue

    ctypedef enum PKStreamFlags:
        PKStreamOpenRead
        PKStreamOpenWrite
        PKStreamOpenReadWrite
        PKStreamNoLock
        PKStreamNoSeek
        PKStreamCompress

    # PKPixelInfo

    ctypedef struct PKPixelInfo:
        PKPixelFormatGUID* pGUIDPixFmt
        size_t cChannel
        COLORFORMAT cfColorFormat
        BITDEPTH_BITS bdBitDepth
        U32 cbitUnit
        WMP_GRBIT grBit
        U32 uInterpretation
        U32 uSamplePerPixel
        U32 uBitsPerSample
        U32 uSampleFormat

    ERR GetImageEncodeIID(
        const char* szExt,
        const PKIID** ppIID
    )

    ERR GetImageDecodeIID(
        const char* szExt,
        const PKIID** ppIID
    )

    ERR PixelFormatLookup(
        PKPixelInfo* pPI,
        U8 uLookupType
    )

    PKPixelFormatGUID* GetPixelFormatFromHash(
        const U8 uPFHash
    )

    # PKImageDecode

    ctypedef ERR (*PKImageDecode_InitializeFunction)(
        PKImageDecode*,
        WMPStream* pStream
    ) nogil

    ctypedef ERR (*PKImageDecode_GetPixelFormatFunction)(
        PKImageDecode*,
        PKPixelFormatGUID*
    ) nogil

    ctypedef ERR (*PKImageDecode_GetSizeFunction)(
        PKImageDecode*,
        I32*,
        I32*
    ) nogil

    ctypedef ERR (*PKImageDecode_GetResolutionFunction)(
        PKImageDecode*,
        Float*,
        Float*
    ) nogil

    ctypedef ERR (*PKImageDecode_GetColorContextFunction)(
        PKImageDecode* pID,
        U8* pbColorContext,
        U32* pcbColorContext
    ) nogil

    ctypedef ERR (*PKImageDecode_GetDescriptiveMetadataFunction)(
        PKImageDecode* pIE,
        DESCRIPTIVEMETADATA* pDescMetadata
    ) nogil

    ctypedef ERR (*PKImageDecode_GetRawStreamFunction)(
        PKImageDecode*,
        WMPStream**
    ) nogil

    ctypedef ERR (*PKImageDecode_CopyFunction)(
        PKImageDecode*,
        const PKRect*,
        U8*,
        U32
    ) nogil

    ctypedef ERR (*PKImageDecode_GetFrameCountFunction)(
        PKImageDecode*,
        U32*
    ) nogil

    ctypedef ERR (*PKImageDecode_SelectFrameFunction)(
        PKImageDecode*,
        U32
    ) nogil

    ctypedef ERR (*PKImageDecode_ReleaseFunction)(
        PKImageDecode**
    ) nogil

    ctypedef struct PKImageDecode_WMP:
        WmpDEMisc wmiDEMisc
        CWMImageInfo wmiI
        CWMIStrCodecParam wmiSCP
        CTXSTRCODEC ctxSC
        CWMImageInfo wmiI_Alpha
        CWMIStrCodecParam wmiSCP_Alpha
        CTXSTRCODEC ctxSC_Alpha
        Bool bHasAlpha
        Long nOffImage
        Long nCbImage
        Long nOffAlpha
        Long nCbAlpha
        Bool bIgnoreOverlap
        size_t DecoderCurrMBRow
        size_t DecoderCurrAlphaMBRow
        size_t cMarker
        size_t cLinesDecoded
        size_t cLinesCropped
        Bool fFirstNonZeroDecode
        Bool fOrientationFromContainer
        ORIENTATION oOrientationFromContainer
        DESCRIPTIVEMETADATA sDescMetadata

    ctypedef struct PKImageDecode:
        PKImageDecode_InitializeFunction Initialize
        PKImageDecode_GetPixelFormatFunction GetPixelFormat
        PKImageDecode_GetSizeFunction GetSize
        PKImageDecode_GetResolutionFunction GetResolution
        PKImageDecode_GetColorContextFunction GetColorContext
        PKImageDecode_GetDescriptiveMetadataFunction GetDescriptiveMetadata
        PKImageDecode_GetRawStreamFunction GetRawStream
        PKImageDecode_CopyFunction Copy
        PKImageDecode_GetFrameCountFunction GetFrameCount
        PKImageDecode_SelectFrameFunction SelectFrame
        PKImageDecode_ReleaseFunction Release
        WMPStream* pStream
        Bool fStreamOwner
        size_t offStart
        PKPixelFormatGUID guidPixFormat
        U32 uWidth
        U32 uHeight
        U32 idxCurrentLine
        Float fResX
        Float fResY
        U32 cFrame
        PKImageDecode_WMP WMP

    ERR PKImageDecode_Create_WMP(
        PKImageDecode** ppID
    )

    ERR PKImageDecode_Initialize(
        PKImageDecode* pID,
        WMPStream* pStream
    )

    ERR PKImageDecode_GetPixelFormat(
        PKImageDecode* pID,
        PKPixelFormatGUID* pPF
    )

    ERR PKImageDecode_GetSize(
        PKImageDecode* pID,
        I32* piWidth,
        I32* piHeight
    )

    ERR PKImageDecode_GetResolution(
        PKImageDecode* pID,
        Float* pfrX,
        Float* pfrY
    )

    ERR PKImageDecode_GetColorContext(
        PKImageDecode* pID,
        U8* pbColorContext,
        U32* pcbColorContext
    )

    ERR PKImageDecode_GetDescriptiveMetadata(
        PKImageDecode* pID,
        DESCRIPTIVEMETADATA* pDescMetadata
    )

    ERR PKImageDecode_Copy(
        PKImageDecode* pID,
        const PKRect* pRect,
        U8* pb,
        U32 cbStride
    )

    ERR PKImageDecode_GetFrameCount(
        PKImageDecode* pID,
        U32* puCount
    )

    ERR PKImageDecode_SelectFrame(
        PKImageDecode* pID,
        U32 uFrame
    )

    ERR PKCodecFactory_CreateDecoderFromFile(
        const char* szFilename,
        PKImageDecode** ppDecoder
    )

    ERR PKImageDecode_Create(
        PKImageDecode** ppID
    )

    ERR PKImageDecode_Release(
        PKImageDecode** ppID
    )

    # PKImageEncode

    ctypedef ERR (*PKImageEncode_InitializeFunction)(
        PKImageEncode*,
        WMPStream*,
        void*,
        size_t
    ) nogil

    ctypedef ERR (*PKImageEncode_TerminateFunction)(
        PKImageEncode*
    ) nogil

    ctypedef ERR (*PKImageEncode_SetPixelFormatFunction)(
        PKImageEncode*,
        PKPixelFormatGUID
    ) nogil

    ctypedef ERR (*PKImageEncode_SetSizeFunction)(
        PKImageEncode*,
        I32,
        I32
    ) nogil

    ctypedef ERR (*PKImageEncode_SetResolutionFunction)(
        PKImageEncode*,
        Float,
        Float
    ) nogil

    ctypedef ERR (*PKImageEncode_SetColorContextFunction)(
        PKImageEncode* pIE,
        const U8* pbColorContext,
        U32 cbColorContext
    ) nogil

    ctypedef ERR (*PKImageEncode_SetDescriptiveMetadataFunction)(
        PKImageEncode* pIE,
        const DESCRIPTIVEMETADATA* pDescMetadata
    ) nogil

    ctypedef ERR (*PKImageEncode_WritePixelsFunction)(
        PKImageEncode*,
        U32,
        U8*,
        U32
    ) nogil

    ctypedef ERR (*PKImageEncode_WriteSourceFunction)(
        PKImageEncode*,
        PKFormatConverter*,
        PKRect*
    ) nogil

    ctypedef ERR (*PKImageEncode_WritePixelsBandedBeginFunction)(
        PKImageEncode* pEncoder,
        WMPStream* pPlanarAlphaTempFile
    ) nogil

    ctypedef ERR (*PKImageEncode_WritePixelsBandedFunction)(
        PKImageEncode* pEncoder,
        U32 cLines,
        U8* pbPixels,
        U32 cbStride,
        Bool fLastCall
    ) nogil

    ctypedef ERR (*PKImageEncode_WritePixelsBandedEndFunction)(
        PKImageEncode* pEncoder
    ) nogil

    ctypedef ERR (*PKImageEncode_TranscodeFunction)(
        PKImageEncode*,
        PKImageDecode*,
        CWMTranscodingParam*
    ) nogil

    ctypedef ERR (*PKImageEncode_CreateNewFrameFunction)(
        PKImageEncode*,
        void*,
        size_t
    ) nogil

    ctypedef ERR (*PKImageEncode_ReleaseFunction)(
        PKImageEncode**
    ) nogil

    ctypedef enum BANDEDENCSTATE:
        BANDEDENCSTATE_UNINITIALIZED
        BANDEDENCSTATE_INIT
        BANDEDENCSTATE_ENCODING
        BANDEDENCSTATE_TERMINATED
        BANDEDENCSTATE_NONBANDEDENCODE

    ctypedef struct PKImageEncode_WMP:
        WmpDEMisc wmiDEMisc
        CWMImageInfo wmiI
        CWMIStrCodecParam wmiSCP
        CTXSTRCODEC ctxSC
        CWMImageInfo wmiI_Alpha
        CWMIStrCodecParam wmiSCP_Alpha
        CTXSTRCODEC ctxSC_Alpha
        Bool bHasAlpha
        Long nOffImage
        Long nCbImage
        Long nOffAlpha
        Long nCbAlpha
        ORIENTATION oOrientation
        BANDEDENCSTATE eBandedEncState
        WMPStream* pPATempFile

    ctypedef struct PKImageEncode:
        PKImageEncode_InitializeFunction Initialize
        PKImageEncode_TerminateFunction Terminate
        PKImageEncode_SetPixelFormatFunction SetPixelFormat
        PKImageEncode_SetSizeFunction SetSize
        PKImageEncode_SetResolutionFunction SetResolution
        PKImageEncode_SetColorContextFunction SetColorContext
        PKImageEncode_SetDescriptiveMetadataFunction SetDescriptiveMetadata
        PKImageEncode_WritePixelsFunction WritePixels
        PKImageEncode_WriteSourceFunction WriteSource
        PKImageEncode_WritePixelsBandedBeginFunction WritePixelsBandedBegin
        PKImageEncode_WritePixelsBandedFunction WritePixelsBanded
        PKImageEncode_WritePixelsBandedEndFunction WritePixelsBandedEnd
        PKImageEncode_TranscodeFunction Transcode
        PKImageEncode_CreateNewFrameFunction CreateNewFrame
        PKImageEncode_ReleaseFunction Release
        WMPStream* pStream
        size_t offStart
        PKPixelFormatGUID guidPixFormat
        U32 uWidth
        U32 uHeight
        U32 idxCurrentLine
        Float fResX
        Float fResY
        U32 cFrame
        Bool fHeaderDone
        size_t offPixel
        size_t cbPixel
        U8* pbColorContext
        U32 cbColorContext
        U8* pbEXIFMetadata
        U32 cbEXIFMetadataByteCount
        U8* pbGPSInfoMetadata
        U32 cbGPSInfoMetadataByteCount
        U8* pbIPTCNAAMetadata
        U32 cbIPTCNAAMetadataByteCount
        U8* pbXMPMetadata
        U32 cbXMPMetadataByteCount
        U8* pbPhotoshopMetadata
        U32 cbPhotoshopMetadataByteCount
        DESCRIPTIVEMETADATA sDescMetadata
        Bool bWMP
        PKImageEncode_WMP WMP

    ERR PKImageEncode_Create_WMP(
        PKImageEncode** ppIE
    )

    ERR PKImageEncode_Initialize(
        PKImageEncode* pIE,
        WMPStream* pStream,
        void* pvParam,
        size_t cbParam
    )

    ERR PKImageEncode_Terminate(
        PKImageEncode* pIE
    )

    ERR PKImageEncode_SetPixelFormat(
        PKImageEncode* pIE,
        PKPixelFormatGUID enPixelFormat
    )

    ERR PKImageEncode_SetSize(
        PKImageEncode* pIE,
        I32 iWidth,
        I32 iHeight
    )

    ERR PKImageEncode_SetResolution(
        PKImageEncode* pIE,
        Float rX,
        Float rY
    )

    ERR PKImageEncode_SetColorContext(
        PKImageEncode* pIE,
        const U8* pbColorContext,
        U32 cbColorContext
    )

    ERR PKImageEncode_SetDescriptiveMetadata(
        PKImageEncode* pIE,
        const DESCRIPTIVEMETADATA* pDescMetadata
    )

    ERR PKImageEncode_WritePixels(
        PKImageEncode* pIE,
        U32 cLine,
        U8* pbPixel,
        U32 cbStride
    )

    ERR PKImageEncode_CreateNewFrame(
        PKImageEncode* pIE,
        void* pvParam,
        size_t cbParam
    )

    ERR PKImageEncode_Release(
        PKImageEncode** ppIE
    )

    ERR PKImageEncode_SetXMPMetadata_WMP(
        PKImageEncode* pIE,
        const U8* pbXMPMetadata,
        U32 cbXMPMetadata
    )

    ERR PKImageEncode_SetEXIFMetadata_WMP(
        PKImageEncode* pIE,
        const U8* pbEXIFMetadata,
        U32 cbEXIFMetadata
    )

    ERR PKImageEncode_SetGPSInfoMetadata_WMP(
        PKImageEncode* pIE,
        const U8* pbGPSInfoMetadata,
        U32 cbGPSInfoMetadata
    )

    ERR PKImageEncode_SetIPTCNAAMetadata_WMP(
        PKImageEncode* pIE,
        const U8* pbIPTCNAAMetadata,
        U32 cbIPTCNAAMetadata
    )

    ERR PKImageEncode_SetPhotoshopMetadata_WMP(
        PKImageEncode* pIE,
        const U8* pbPhotoshopMetadata,
        U32 cbPhotoshopMetadata
    )

    ERR PKImageEncode_Create(
        PKImageEncode** ppIE
    )

    void FreeDescMetadata(
        DPKPROPVARIANT* pvar
    )

    # PKStream

    ctypedef ERR (*PKStream_InitializeFromFilenameFunction)(
        const char*,
        ULong
    ) nogil

    ctypedef ERR (*PKStream_ReleaseFunction)() nogil

    ctypedef struct PKStream:
        PKStream_InitializeFromFilenameFunction InitializeFromFilename
        PKStream_ReleaseFunction Release
        FILE* fp

    # PKFactory

    ctypedef ERR (*PKFactory_CreateCodecFunction)(
        const PKIID*,
        void**
    ) nogil

    ctypedef ERR (*PKFactory_CreateFormatConverterFunction)(
        PKFormatConverter**
    ) nogil

    ctypedef ERR (*PKFactory_ReleaseFunction)(
        PKCodecFactory**
    ) nogil

    ctypedef ERR (*PKFactory_CreateDecoderFromFileFunction)(
        const char*,
        PKImageDecode**
    ) nogil

    ctypedef struct PKFactory:
        PKFactory_CreateCodecFunction CreateCodec
        PKFactory_CreateDecoderFromFileFunction CreateDecoderFromFile
        PKFactory_CreateFormatConverterFunction CreateFormatConverter
        PKFactory_ReleaseFunction Release

    ERR PKCreateFactory_CreateStream(
        PKStream** ppStream
    )

    ERR PKCreateFactory_Release(
        PKFactory** ppFactory
    )

    # extern ERR PKCreateFactory(
    #     PKFactory**,
    #     U32
    # ) nogil

    # PKCodecFactory

    ctypedef ERR (*PKCodecFactory_CreateCodecFunction)(
        const PKIID*,
        void**
    ) nogil

    ctypedef ERR (*PKCodecFactory_CreateDecoderFromFileFunction)(
        const char*,
        PKImageDecode**
    ) nogil

    ctypedef ERR (*PKCodecFactory_CreateFormatConverterFunction)(
        PKFormatConverter**
    ) nogil

    ctypedef ERR (*PKCodecFactory_ReleaseFunction)(
        PKCodecFactory**
    ) nogil

    ctypedef struct PKCodecFactory:
        PKCodecFactory_CreateCodecFunction CreateCodec
        PKCodecFactory_CreateDecoderFromFileFunction CreateDecoderFromFile
        PKCodecFactory_CreateFormatConverterFunction CreateFormatConverter
        PKCodecFactory_ReleaseFunction Release

    # extern ERR PKCreateCodecFactory(
    #     PKCodecFactory**,
    #     U32
    # )

    ERR PKCreateCodecFactory_Release(
        PKCodecFactory**
    )

    ERR PKCodecFactory_CreateCodec(
        const PKIID* iid,
        void** ppv
    )

    # PKFormatConverter

    ctypedef ERR (*PKFormatConverter_InitializeFunction)(
        PKFormatConverter*,
        PKImageDecode*,
        char* pExt,
        PKPixelFormatGUID
    ) nogil

    ctypedef ERR (*PKFormatConverter_InitializeConvertFunction)(
        PKFormatConverter* pFC,
        const PKPixelFormatGUID enPFFrom,
        char* pExt,
        PKPixelFormatGUID enPFTTo
    ) nogil

    ctypedef ERR (*PKFormatConverter_GetPixelFormatFunction)(
        PKFormatConverter*,
        PKPixelFormatGUID*
    ) nogil

    ctypedef ERR (*PKFormatConverter_GetSourcePixelFormatFunction)(
        PKFormatConverter*,
        PKPixelFormatGUID*
    ) nogil

    ctypedef ERR (*PKFormatConverter_GetSizeFunction)(
        PKFormatConverter*,
        I32*,
        I32*
    ) nogil

    ctypedef ERR (*PKFormatConverter_GetResolutionFunction)(
        PKFormatConverter*,
        Float*,
        Float*
    ) nogil

    ctypedef ERR (*PKFormatConverter_CopyFunction)(
        PKFormatConverter*,
        const PKRect*,
        U8*,
        U32
    ) nogil

    ctypedef ERR (*PKFormatConverter_ConvertFunction)(
        PKFormatConverter*,
        const PKRect*,
        U8*,
        U32
    ) nogil

    ctypedef ERR (*PKFormatConverter_ReleaseFunction)(
        PKFormatConverter**
    ) nogil

    ctypedef struct PKFormatConverter:
        PKFormatConverter_InitializeFunction Initialize
        PKFormatConverter_InitializeConvertFunction InitializeConvert
        PKFormatConverter_GetPixelFormatFunction GetPixelFormat
        PKFormatConverter_GetSourcePixelFormatFunction GetSourcePixelFormat
        PKFormatConverter_GetSizeFunction GetSize
        PKFormatConverter_GetResolutionFunction GetResolution
        PKFormatConverter_CopyFunction Copy
        PKFormatConverter_ConvertFunction Convert
        PKFormatConverter_ReleaseFunction Release
        PKImageDecode* pDecoder
        PKPixelFormatGUID enPixelFormat

    ERR PKImageEncode_Transcode(
        PKImageEncode* pIE,
        PKFormatConverter* pFC,
        PKRect* pRect
    )

    ERR PKImageEncode_WriteSource(
        PKImageEncode* pIE,
        PKFormatConverter* pFC,
        PKRect* pRect
    )

    ERR PKFormatConverter_Initialize(
        PKFormatConverter* pFC,
        PKImageDecode* pID,
        char* pExt,
        PKPixelFormatGUID enPF
    )

    ERR PKFormatConverter_InitializeConvert(
        PKFormatConverter* pFC,
        const PKPixelFormatGUID enPFFrom,
        char* pExt,
        PKPixelFormatGUID enPFTo
    )

    ERR PKFormatConverter_GetPixelFormat(
        PKFormatConverter* pFC,
        PKPixelFormatGUID* pPF
    )

    ERR PKFormatConverter_GetSourcePixelFormat(
        PKFormatConverter* pFC,
        PKPixelFormatGUID* pPF
    )

    ERR PKFormatConverter_GetSize(
        PKFormatConverter* pFC,
        I32* piWidth,
        I32* piHeight
    )

    ERR PKFormatConverter_GetResolution(
        PKFormatConverter* pFC,
        Float* pfrX,
        Float* pfrY
    )

    ERR PKFormatConverter_Copy(
        PKFormatConverter* pFC,
        const PKRect* pRect,
        U8* pb,
        U32 cbStride
    )

    ERR PKFormatConverter_Convert(
        PKFormatConverter* pFC,
        const PKRect*
        pRect,
        U8* pb,
        U32 cbStride
    )

    ERR PKFormatConverter_EnumConversions(
        const PKPixelFormatGUID* pguidSourcePF,
        const U32 iIndex,
        const PKPixelFormatGUID** ppguidTargetPF
    )

    ERR PKCodecFactory_CreateFormatConverter(
        PKFormatConverter** ppFConverter
    )

    ERR PKFormatConverter_Release(
        PKFormatConverter** ppFC
    )

    # Memory

    ERR PKAlloc(
        void** ppv,
        size_t cb
    )

    ERR PKFree(
        void** ppv
    )

    ERR PKAllocAligned(
        void** ppv,
        size_t cb,
        size_t iAlign
    )

    ERR PKFreeAligned(
        void** ppv
    )
