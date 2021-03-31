# imagecodecs/lcms2.pxd
# cython: language_level = 3

# Cython declarations for the `Little 2.11` library.
# https://github.com/mm2/Little-CMS

from libc.stdint cimport uint8_t
from libc.stddef cimport wchar_t
from libc.stdio cimport FILE
from libc.time cimport tm

cdef extern from 'lcms2.h':

    int LCMS_VERSION

    ctypedef unsigned char cmsUInt8Number
    ctypedef signed char cmsInt8Number

    ctypedef float cmsFloat32Number
    ctypedef double cmsFloat64Number

    ctypedef unsigned short cmsUInt16Number
    ctypedef short cmsInt16Number
    ctypedef unsigned long cmsUInt32Number
    ctypedef long cmsInt32Number
    ctypedef unsigned long long cmsUInt64Number
    ctypedef long long cmsInt64Number

    ctypedef cmsUInt32Number cmsSignature
    ctypedef cmsUInt16Number cmsU8Fixed8Number
    ctypedef cmsInt32Number cmsS15Fixed16Number
    ctypedef cmsUInt32Number cmsU16Fixed16Number

    ctypedef int cmsBool

    int cmsMAX_PATH

    int FALSE
    int TRUE

    double cmsD50X
    double cmsD50Y
    double cmsD50Z

    double cmsPERCEPTUAL_BLACK_X
    double cmsPERCEPTUAL_BLACK_Y
    double cmsPERCEPTUAL_BLACK_Z

    int cmsMagicNumber
    int lcmsSignature

    ctypedef enum cmsTagTypeSignature:
        cmsSigChromaticityType
        cmsSigColorantOrderType
        cmsSigColorantTableType
        cmsSigCrdInfoType
        cmsSigCurveType
        cmsSigDataType
        cmsSigDictType
        cmsSigDateTimeType
        cmsSigDeviceSettingsType
        cmsSigLut16Type
        cmsSigLut8Type
        cmsSigLutAtoBType
        cmsSigLutBtoAType
        cmsSigMeasurementType
        cmsSigMultiLocalizedUnicodeType
        cmsSigMultiProcessElementType
        cmsSigNamedColorType
        cmsSigNamedColor2Type
        cmsSigParametricCurveType
        cmsSigProfileSequenceDescType
        cmsSigProfileSequenceIdType
        cmsSigResponseCurveSet16Type
        cmsSigS15Fixed16ArrayType
        cmsSigScreeningType
        cmsSigSignatureType
        cmsSigTextType
        cmsSigTextDescriptionType
        cmsSigU16Fixed16ArrayType
        cmsSigUcrBgType
        cmsSigUInt16ArrayType
        cmsSigUInt32ArrayType
        cmsSigUInt64ArrayType
        cmsSigUInt8ArrayType
        cmsSigVcgtType
        cmsSigViewingConditionsType
        cmsSigXYZType

    ctypedef enum cmsTagSignature:
        cmsSigAToB0Tag
        cmsSigAToB1Tag
        cmsSigAToB2Tag
        cmsSigBlueColorantTag
        cmsSigBlueMatrixColumnTag
        cmsSigBlueTRCTag
        cmsSigBToA0Tag
        cmsSigBToA1Tag
        cmsSigBToA2Tag
        cmsSigCalibrationDateTimeTag
        cmsSigCharTargetTag
        cmsSigChromaticAdaptationTag
        cmsSigChromaticityTag
        cmsSigColorantOrderTag
        cmsSigColorantTableTag
        cmsSigColorantTableOutTag
        cmsSigColorimetricIntentImageStateTag
        cmsSigCopyrightTag
        cmsSigCrdInfoTag
        cmsSigDataTag
        cmsSigDateTimeTag
        cmsSigDeviceMfgDescTag
        cmsSigDeviceModelDescTag
        cmsSigDeviceSettingsTag
        cmsSigDToB0Tag
        cmsSigDToB1Tag
        cmsSigDToB2Tag
        cmsSigDToB3Tag
        cmsSigBToD0Tag
        cmsSigBToD1Tag
        cmsSigBToD2Tag
        cmsSigBToD3Tag
        cmsSigGamutTag
        cmsSigGrayTRCTag
        cmsSigGreenColorantTag
        cmsSigGreenMatrixColumnTag
        cmsSigGreenTRCTag
        cmsSigLuminanceTag
        cmsSigMeasurementTag
        cmsSigMediaBlackPointTag
        cmsSigMediaWhitePointTag
        cmsSigNamedColorTag
        cmsSigNamedColor2Tag
        cmsSigOutputResponseTag
        cmsSigPerceptualRenderingIntentGamutTag
        cmsSigPreview0Tag
        cmsSigPreview1Tag
        cmsSigPreview2Tag
        cmsSigProfileDescriptionTag
        cmsSigProfileDescriptionMLTag
        cmsSigProfileSequenceDescTag
        cmsSigProfileSequenceIdTag
        cmsSigPs2CRD0Tag
        cmsSigPs2CRD1Tag
        cmsSigPs2CRD2Tag
        cmsSigPs2CRD3Tag
        cmsSigPs2CSATag
        cmsSigPs2RenderingIntentTag
        cmsSigRedColorantTag
        cmsSigRedMatrixColumnTag
        cmsSigRedTRCTag
        cmsSigSaturationRenderingIntentGamutTag
        cmsSigScreeningDescTag
        cmsSigScreeningTag
        cmsSigTechnologyTag
        cmsSigUcrBgTag
        cmsSigViewingCondDescTag
        cmsSigViewingConditionsTag
        cmsSigVcgtTag
        cmsSigMetaTag
        cmsSigArgyllArtsTag

    ctypedef enum cmsTechnologySignature:
        cmsSigDigitalCamera
        cmsSigFilmScanner
        cmsSigReflectiveScanner
        cmsSigInkJetPrinter
        cmsSigThermalWaxPrinter
        cmsSigElectrophotographicPrinter
        cmsSigElectrostaticPrinter
        cmsSigDyeSublimationPrinter
        cmsSigPhotographicPaperPrinter
        cmsSigFilmWriter
        cmsSigVideoMonitor
        cmsSigVideoCamera
        cmsSigProjectionTelevision
        cmsSigCRTDisplay
        cmsSigPMDisplay
        cmsSigAMDisplay
        cmsSigPhotoCD
        cmsSigPhotoImageSetter
        cmsSigGravure
        cmsSigOffsetLithography
        cmsSigSilkscreen
        cmsSigFlexography
        cmsSigMotionPictureFilmScanner
        cmsSigMotionPictureFilmRecorder
        cmsSigDigitalMotionPictureCamera
        cmsSigDigitalCinemaProjector

    ctypedef enum cmsColorSpaceSignature:
        cmsSigXYZData
        cmsSigLabData
        cmsSigLuvData
        cmsSigYCbCrData
        cmsSigYxyData
        cmsSigRgbData
        cmsSigGrayData
        cmsSigHsvData
        cmsSigHlsData
        cmsSigCmykData
        cmsSigCmyData
        cmsSigMCH1Data
        cmsSigMCH2Data
        cmsSigMCH3Data
        cmsSigMCH4Data
        cmsSigMCH5Data
        cmsSigMCH6Data
        cmsSigMCH7Data
        cmsSigMCH8Data
        cmsSigMCH9Data
        cmsSigMCHAData
        cmsSigMCHBData
        cmsSigMCHCData
        cmsSigMCHDData
        cmsSigMCHEData
        cmsSigMCHFData
        cmsSigNamedData
        cmsSig1colorData
        cmsSig2colorData
        cmsSig3colorData
        cmsSig4colorData
        cmsSig5colorData
        cmsSig6colorData
        cmsSig7colorData
        cmsSig8colorData
        cmsSig9colorData
        cmsSig10colorData
        cmsSig11colorData
        cmsSig12colorData
        cmsSig13colorData
        cmsSig14colorData
        cmsSig15colorData
        cmsSigLuvKData

    ctypedef enum cmsProfileClassSignature:
        cmsSigInputClass
        cmsSigDisplayClass
        cmsSigOutputClass
        cmsSigLinkClass
        cmsSigAbstractClass
        cmsSigColorSpaceClass
        cmsSigNamedColorClass

    ctypedef enum cmsPlatformSignature:
        cmsSigMacintosh
        cmsSigMicrosoft
        cmsSigSolaris
        cmsSigSGI
        cmsSigTaligent
        cmsSigUnices

    int cmsSigPerceptualReferenceMediumGamut
    int cmsSigSceneColorimetryEstimates
    int cmsSigSceneAppearanceEstimates
    int cmsSigFocalPlaneColorimetryEstimates
    int cmsSigReflectionHardcopyOriginalColorimetry
    int cmsSigReflectionPrintOutputColorimetry

    ctypedef enum cmsStageSignature:
        cmsSigCurveSetElemType
        cmsSigMatrixElemType
        cmsSigCLutElemType
        cmsSigBAcsElemType
        cmsSigEAcsElemType
        cmsSigXYZ2LabElemType
        cmsSigLab2XYZElemType
        cmsSigNamedColorElemType
        cmsSigLabV2toV4
        cmsSigLabV4toV2
        cmsSigIdentityElemType
        cmsSigLab2FloatPCS
        cmsSigFloatPCS2Lab
        cmsSigXYZ2FloatPCS
        cmsSigFloatPCS2XYZ
        cmsSigClipNegativesElemType

    ctypedef enum cmsCurveSegSignature:
        cmsSigFormulaCurveSeg
        cmsSigSampledCurveSeg
        cmsSigSegmentedCurve

    int cmsSigStatusA
    int cmsSigStatusE
    int cmsSigStatusI
    int cmsSigStatusT
    int cmsSigStatusM
    int cmsSigDN
    int cmsSigDNP
    int cmsSigDNN
    int cmsSigDNNP

    int cmsReflective
    int cmsTransparency
    int cmsGlossy
    int cmsMatte

    ctypedef struct cmsICCData:
        cmsUInt32Number len
        cmsUInt32Number flag
        cmsUInt8Number data[1]

    ctypedef struct cmsDateTimeNumber:
        cmsUInt16Number year
        cmsUInt16Number month
        cmsUInt16Number day
        cmsUInt16Number hours
        cmsUInt16Number minutes
        cmsUInt16Number seconds

    ctypedef struct cmsEncodedXYZNumber:
        cmsS15Fixed16Number X
        cmsS15Fixed16Number Y
        cmsS15Fixed16Number Z

    ctypedef union cmsProfileID:
        cmsUInt8Number ID8[16]
        cmsUInt16Number ID16[8]
        cmsUInt32Number ID32[4]

    ctypedef struct cmsICCHeader:
        cmsUInt32Number size
        cmsSignature cmmId
        cmsUInt32Number version
        cmsProfileClassSignature deviceClass
        cmsColorSpaceSignature colorSpace
        cmsColorSpaceSignature pcs
        cmsDateTimeNumber date
        cmsSignature magic
        cmsPlatformSignature platform
        cmsUInt32Number flags
        cmsSignature manufacturer
        cmsUInt32Number model
        cmsUInt64Number attributes
        cmsUInt32Number renderingIntent
        cmsEncodedXYZNumber illuminant
        cmsSignature creator
        cmsProfileID profileID
        cmsInt8Number reserved[28]

    ctypedef struct cmsTagBase:
        cmsTagTypeSignature sig
        cmsInt8Number reserved[4]

    ctypedef struct cmsTagEntry:
        cmsTagSignature sig
        cmsUInt32Number offset
        cmsUInt32Number size

    ctypedef void* cmsHANDLE
    ctypedef void* cmsHPROFILE
    ctypedef void* cmsHTRANSFORM

    int cmsMAXCHANNELS

    int FLOAT_SH
    int OPTIMIZED_SH
    int COLORSPACE_SH
    int SWAPFIRST_SH
    int FLAVOR_SH
    int PLANAR_SH
    int ENDIAN16_SH
    int DOSWAP_SH
    int EXTRA_SH
    int CHANNELS_SH
    int BYTES_SH

    int T_FLOAT
    int T_OPTIMIZED
    int T_COLORSPACE
    int T_SWAPFIRST
    int T_FLAVOR
    int T_PLANAR
    int T_ENDIAN16
    int T_DOSWAP
    int T_EXTRA
    int T_CHANNELS
    int T_BYTES

    int PT_ANY
    int PT_GRAY
    int PT_RGB
    int PT_CMY
    int PT_CMYK
    int PT_YCbCr
    int PT_YUV
    int PT_XYZ
    int PT_Lab
    int PT_YUVK
    int PT_HSV
    int PT_HLS
    int PT_Yxy

    int PT_MCH1
    int PT_MCH2
    int PT_MCH3
    int PT_MCH4
    int PT_MCH5
    int PT_MCH6
    int PT_MCH7
    int PT_MCH8
    int PT_MCH9
    int PT_MCH10
    int PT_MCH11
    int PT_MCH12
    int PT_MCH13
    int PT_MCH14
    int PT_MCH15

    int PT_LabV2

    int TYPE_GRAY_8
    int TYPE_GRAY_8_REV
    int TYPE_GRAY_16
    int TYPE_GRAY_16_REV
    int TYPE_GRAY_16_SE
    int TYPE_GRAYA_8
    int TYPE_GRAYA_16
    int TYPE_GRAYA_16_SE
    int TYPE_GRAYA_8_PLANAR
    int TYPE_GRAYA_16_PLANAR

    int TYPE_RGB_8
    int TYPE_RGB_8_PLANAR
    int TYPE_BGR_8
    int TYPE_BGR_8_PLANAR
    int TYPE_RGB_16
    int TYPE_RGB_16_PLANAR
    int TYPE_RGB_16_SE
    int TYPE_BGR_16
    int TYPE_BGR_16_PLANAR
    int TYPE_BGR_16_SE

    int TYPE_RGBA_8
    int TYPE_RGBA_8_PLANAR
    int TYPE_RGBA_16
    int TYPE_RGBA_16_PLANAR
    int TYPE_RGBA_16_SE

    int TYPE_ARGB_8
    int TYPE_ARGB_8_PLANAR
    int TYPE_ARGB_16

    int TYPE_ABGR_8
    int TYPE_ABGR_8_PLANAR
    int TYPE_ABGR_16
    int TYPE_ABGR_16_PLANAR
    int TYPE_ABGR_16_SE

    int TYPE_BGRA_8
    int TYPE_BGRA_8_PLANAR
    int TYPE_BGRA_16
    int TYPE_BGRA_16_SE

    int TYPE_CMY_8
    int TYPE_CMY_8_PLANAR
    int TYPE_CMY_16
    int TYPE_CMY_16_PLANAR
    int TYPE_CMY_16_SE

    int TYPE_CMYK_8
    int TYPE_CMYKA_8
    int TYPE_CMYK_8_REV
    int TYPE_YUVK_8
    int TYPE_CMYK_8_PLANAR
    int TYPE_CMYK_16
    int TYPE_CMYK_16_REV
    int TYPE_YUVK_16
    int TYPE_CMYK_16_PLANAR
    int TYPE_CMYK_16_SE

    int TYPE_KYMC_8
    int TYPE_KYMC_16
    int TYPE_KYMC_16_SE

    int TYPE_KCMY_8
    int TYPE_KCMY_8_REV
    int TYPE_KCMY_16
    int TYPE_KCMY_16_REV
    int TYPE_KCMY_16_SE

    int TYPE_CMYK5_8
    int TYPE_CMYK5_16
    int TYPE_CMYK5_16_SE
    int TYPE_KYMC5_8
    int TYPE_KYMC5_16
    int TYPE_KYMC5_16_SE
    int TYPE_CMYK6_8
    int TYPE_CMYK6_8_PLANAR
    int TYPE_CMYK6_16
    int TYPE_CMYK6_16_PLANAR
    int TYPE_CMYK6_16_SE
    int TYPE_CMYK7_8
    int TYPE_CMYK7_16
    int TYPE_CMYK7_16_SE
    int TYPE_KYMC7_8
    int TYPE_KYMC7_16
    int TYPE_KYMC7_16_SE
    int TYPE_CMYK8_8
    int TYPE_CMYK8_16
    int TYPE_CMYK8_16_SE
    int TYPE_KYMC8_8
    int TYPE_KYMC8_16
    int TYPE_KYMC8_16_SE
    int TYPE_CMYK9_8
    int TYPE_CMYK9_16
    int TYPE_CMYK9_16_SE
    int TYPE_KYMC9_8
    int TYPE_KYMC9_16
    int TYPE_KYMC9_16_SE
    int TYPE_CMYK10_8
    int TYPE_CMYK10_16
    int TYPE_CMYK10_16_SE
    int TYPE_KYMC10_8
    int TYPE_KYMC10_16
    int TYPE_KYMC10_16_SE
    int TYPE_CMYK11_8
    int TYPE_CMYK11_16
    int TYPE_CMYK11_16_SE
    int TYPE_KYMC11_8
    int TYPE_KYMC11_16
    int TYPE_KYMC11_16_SE
    int TYPE_CMYK12_8
    int TYPE_CMYK12_16
    int TYPE_CMYK12_16_SE
    int TYPE_KYMC12_8
    int TYPE_KYMC12_16
    int TYPE_KYMC12_16_SE

    int TYPE_XYZ_16
    int TYPE_Lab_8
    int TYPE_LabV2_8

    int TYPE_ALab_8
    int TYPE_ALabV2_8
    int TYPE_Lab_16
    int TYPE_LabV2_16
    int TYPE_Yxy_16

    int TYPE_YCbCr_8
    int TYPE_YCbCr_8_PLANAR
    int TYPE_YCbCr_16
    int TYPE_YCbCr_16_PLANAR
    int TYPE_YCbCr_16_SE

    int TYPE_YUV_8
    int TYPE_YUV_8_PLANAR
    int TYPE_YUV_16
    int TYPE_YUV_16_PLANAR
    int TYPE_YUV_16_SE

    int TYPE_HLS_8
    int TYPE_HLS_8_PLANAR
    int TYPE_HLS_16
    int TYPE_HLS_16_PLANAR
    int TYPE_HLS_16_SE

    int TYPE_HSV_8
    int TYPE_HSV_8_PLANAR
    int TYPE_HSV_16
    int TYPE_HSV_16_PLANAR
    int TYPE_HSV_16_SE

    int TYPE_NAMED_COLOR_INDEX

    int TYPE_XYZ_FLT
    int TYPE_Lab_FLT
    int TYPE_LabA_FLT
    int TYPE_GRAY_FLT
    int TYPE_RGB_FLT

    int TYPE_RGBA_FLT
    int TYPE_ARGB_FLT
    int TYPE_BGR_FLT
    int TYPE_BGRA_FLT
    int TYPE_ABGR_FLT

    int TYPE_CMYK_FLT

    int TYPE_XYZ_DBL
    int TYPE_Lab_DBL
    int TYPE_GRAY_DBL
    int TYPE_RGB_DBL
    int TYPE_BGR_DBL
    int TYPE_CMYK_DBL

    int TYPE_GRAY_HALF_FLT
    int TYPE_RGB_HALF_FLT
    int TYPE_RGBA_HALF_FLT
    int TYPE_CMYK_HALF_FLT

    int TYPE_ARGB_HALF_FLT
    int TYPE_BGR_HALF_FLT
    int TYPE_BGRA_HALF_FLT
    int TYPE_ABGR_HALF_FLT

    ctypedef struct cmsCIEXYZ:
        cmsFloat64Number X
        cmsFloat64Number Y
        cmsFloat64Number Z

    ctypedef struct cmsCIExyY:
        cmsFloat64Number x
        cmsFloat64Number y
        cmsFloat64Number Y

    ctypedef struct cmsCIELab:
        cmsFloat64Number L
        cmsFloat64Number a
        cmsFloat64Number b

    ctypedef struct cmsCIELCh:
        cmsFloat64Number L
        cmsFloat64Number C
        cmsFloat64Number h

    ctypedef struct cmsJCh:
        cmsFloat64Number J
        cmsFloat64Number C
        cmsFloat64Number h

    ctypedef struct cmsCIEXYZTRIPLE:
        cmsCIEXYZ Red
        cmsCIEXYZ Green
        cmsCIEXYZ Blue

    ctypedef struct cmsCIExyYTRIPLE:
        cmsCIExyY Red
        cmsCIExyY Green
        cmsCIExyY Blue

    int cmsILLUMINANT_TYPE_UNKNOWN
    int cmsILLUMINANT_TYPE_D50
    int cmsILLUMINANT_TYPE_D65
    int cmsILLUMINANT_TYPE_D93
    int cmsILLUMINANT_TYPE_F2
    int cmsILLUMINANT_TYPE_D55
    int cmsILLUMINANT_TYPE_A
    int cmsILLUMINANT_TYPE_E
    int cmsILLUMINANT_TYPE_F8

    ctypedef struct cmsICCMeasurementConditions:
        cmsUInt32Number Observer
        cmsCIEXYZ Backing
        cmsUInt32Number Geometry
        cmsFloat64Number Flare
        cmsUInt32Number IlluminantType

    ctypedef struct cmsICCViewingConditions:
        cmsCIEXYZ IlluminantXYZ
        cmsCIEXYZ SurroundXYZ
        cmsUInt32Number IlluminantType

    int cmsGetEncodedCMMversion() nogil

    int cmsstrcasecmp(
        const char* s1,
        const char* s2
    ) nogil

    long int cmsfilelength(
        FILE* f
    ) nogil

    ctypedef struct cmsContext:
        pass

    cmsContext cmsCreateContext(
        void* Plugin,
        void* UserData
    ) nogil

    void cmsDeleteContext(
        cmsContext ContextID
    ) nogil

    cmsContext cmsDupContext(
        cmsContext ContextID,
        void* NewUserData
    ) nogil

    void* cmsGetContextUserData(
        cmsContext ContextID
    ) nogil

    cmsBool cmsPlugin(
        void* Plugin
    ) nogil

    cmsBool cmsPluginTHR(
        cmsContext ContextID,
        void* Plugin
    ) nogil

    void cmsUnregisterPlugins() nogil

    void cmsUnregisterPluginsTHR(
        cmsContext ContextID
    ) nogil

    int cmsERROR_UNDEFINED
    int cmsERROR_FILE
    int cmsERROR_RANGE
    int cmsERROR_INTERNAL
    int cmsERROR_NULL
    int cmsERROR_READ
    int cmsERROR_SEEK
    int cmsERROR_WRITE
    int cmsERROR_UNKNOWN_EXTENSION
    int cmsERROR_COLORSPACE_CHECK
    int cmsERROR_ALREADY_DEFINED
    int cmsERROR_BAD_SIGNATURE
    int cmsERROR_CORRUPTION_DETECTED
    int cmsERROR_NOT_SUITABLE

    ctypedef void (* cmsLogErrorHandlerFunction)(
        cmsContext ContextID,
        cmsUInt32Number ErrorCode,
        const char *Text
    ) nogil

    void cmsSetLogErrorHandler(
        cmsLogErrorHandlerFunction Fn
    ) nogil

    void cmsSetLogErrorHandlerTHR(
        cmsContext ContextID,
        cmsLogErrorHandlerFunction Fn
    ) nogil

    const cmsCIEXYZ* cmsD50_XYZ() nogil

    const cmsCIExyY* cmsD50_xyY() nogil

    void cmsXYZ2xyY(
        cmsCIExyY* Dest,
        const cmsCIEXYZ* Source
    ) nogil

    void cmsxyY2XYZ(
        cmsCIEXYZ* Dest,
        const cmsCIExyY* Source
    ) nogil

    void cmsXYZ2Lab(
        const cmsCIEXYZ* WhitePoint,
        cmsCIELab* Lab,
        const cmsCIEXYZ* xyz
    ) nogil

    void cmsLab2XYZ(
        const cmsCIEXYZ* WhitePoint,
        cmsCIEXYZ* xyz,
        const cmsCIELab* Lab
    ) nogil

    void cmsLab2LCh(
        cmsCIELCh*LCh,
        const cmsCIELab* Lab
    ) nogil

    void cmsLCh2Lab(
        cmsCIELab* Lab,
        const cmsCIELCh* LCh
    ) nogil

    void cmsLabEncoded2Float(
        cmsCIELab* Lab,
        const cmsUInt16Number wLab[3]
    ) nogil

    void cmsLabEncoded2FloatV2(
        cmsCIELab* Lab,
        const cmsUInt16Number wLab[3]
    ) nogil

    void cmsFloat2LabEncoded(
        cmsUInt16Number wLab[3],
        const cmsCIELab* Lab
    ) nogil

    void cmsFloat2LabEncodedV2(
        cmsUInt16Number wLab[3],
        const cmsCIELab* Lab
    ) nogil

    void cmsXYZEncoded2Float(
        cmsCIEXYZ* fxyz,
        const cmsUInt16Number XYZ[3]
    ) nogil

    void cmsFloat2XYZEncoded(
        cmsUInt16Number XYZ[3],
        const cmsCIEXYZ* fXYZ
    ) nogil

    cmsFloat64Number cmsDeltaE(
        const cmsCIELab* Lab1,
        const cmsCIELab* Lab2
    ) nogil

    cmsFloat64Number cmsCIE94DeltaE(
        const cmsCIELab* Lab1,
        const cmsCIELab* Lab2
    ) nogil

    cmsFloat64Number cmsBFDdeltaE(
        const cmsCIELab* Lab1,
        const cmsCIELab* Lab2
    ) nogil

    cmsFloat64Number cmsCMCdeltaE(
        const cmsCIELab* Lab1,
        const cmsCIELab* Lab2,
        cmsFloat64Number l,
        cmsFloat64Number c
    ) nogil

    cmsFloat64Number cmsCIE2000DeltaE(
        const cmsCIELab* Lab1,
        const cmsCIELab* Lab2,
        cmsFloat64Number Kl,
        cmsFloat64Number Kc,
        cmsFloat64Number Kh
    ) nogil

    cmsBool cmsWhitePointFromTemp(
        cmsCIExyY* WhitePoint,
        cmsFloat64Number TempK
    ) nogil

    cmsBool cmsTempFromWhitePoint(
        cmsFloat64Number* TempK,
        const cmsCIExyY* WhitePoint
    ) nogil

    cmsBool cmsAdaptToIlluminant(
        cmsCIEXYZ* Result,
        const cmsCIEXYZ* SourceWhitePt,
        const cmsCIEXYZ* Illuminant,
        const cmsCIEXYZ* Value
    ) nogil

    int AVG_SURROUND
    int DIM_SURROUND
    int DARK_SURROUND
    int CUTSHEET_SURROUND

    int D_CALCULATE

    ctypedef struct cmsViewingConditions:
        cmsCIEXYZ whitePoint
        cmsFloat64Number Yb
        cmsFloat64Number La
        cmsUInt32Number surround
        cmsFloat64Number D_value

    cmsHANDLE cmsCIECAM02Init(
        cmsContext ContextID,
        const cmsViewingConditions* pVC
    ) nogil

    void cmsCIECAM02Done(
        cmsHANDLE hModel
    ) nogil

    void cmsCIECAM02Forward(
        cmsHANDLE hModel,
        const cmsCIEXYZ* pIn,
        cmsJCh* pOut
    ) nogil

    void cmsCIECAM02Reverse(
        cmsHANDLE hModel,
        const cmsJCh* pIn,
        cmsCIEXYZ* pOut
    ) nogil

    ctypedef struct cmsCurveSegment:
        cmsFloat32Number x0
        cmsFloat32Number x1
        cmsInt32Number Type
        cmsFloat64Number Params[10]
        cmsUInt32Number nGridPoints
        cmsFloat32Number* SampledPoints

    ctypedef struct cmsToneCurve:
        pass

    cmsToneCurve* cmsBuildSegmentedToneCurve(
        cmsContext ContextID,
        cmsUInt32Number nSegments,
        const cmsCurveSegment Segments[]
    ) nogil

    cmsToneCurve* cmsBuildParametricToneCurve(
        cmsContext ContextID,
        cmsInt32Number Type,
        const cmsFloat64Number Params[]
    ) nogil

    cmsToneCurve* cmsBuildGamma(
        cmsContext ContextID,
        cmsFloat64Number Gamma
    ) nogil

    cmsToneCurve* cmsBuildTabulatedToneCurve16(
        cmsContext ContextID,
        cmsUInt32Number nEntries,
        const cmsUInt16Number values[]
    ) nogil

    cmsToneCurve* cmsBuildTabulatedToneCurveFloat(
        cmsContext ContextID,
        cmsUInt32Number nEntries,
        const cmsFloat32Number values[]
    ) nogil

    void cmsFreeToneCurve(
        cmsToneCurve* Curve
    ) nogil

    void cmsFreeToneCurveTriple(
        cmsToneCurve* Curve[3]
    ) nogil

    cmsToneCurve* cmsDupToneCurve(
        const cmsToneCurve* Src
    ) nogil

    cmsToneCurve* cmsReverseToneCurve(
        const cmsToneCurve* InGamma
    ) nogil

    cmsToneCurve* cmsReverseToneCurveEx(
        cmsUInt32Number nResultSamples,
        const cmsToneCurve* InGamma
    ) nogil

    cmsToneCurve* cmsJoinToneCurve(
        cmsContext ContextID,
        const cmsToneCurve* X,
        const cmsToneCurve* Y,
        cmsUInt32Number nPoints
    ) nogil

    cmsBool cmsSmoothToneCurve(
        cmsToneCurve* Tab,
        cmsFloat64Number lambda_
    ) nogil

    cmsFloat32Number cmsEvalToneCurveFloat(
        const cmsToneCurve* Curve,
        cmsFloat32Number v
    ) nogil

    cmsUInt16Number cmsEvalToneCurve16(
        const cmsToneCurve* Curve,
        cmsUInt16Number v
    ) nogil

    cmsBool cmsIsToneCurveMultisegment(
        const cmsToneCurve* InGamma
    ) nogil

    cmsBool cmsIsToneCurveLinear(
        const cmsToneCurve* Curve
    ) nogil

    cmsBool cmsIsToneCurveMonotonic(
        const cmsToneCurve* t
    ) nogil

    cmsBool cmsIsToneCurveDescending(
        const cmsToneCurve* t
    ) nogil

    cmsInt32Number cmsGetToneCurveParametricType(
        const cmsToneCurve* t
    ) nogil

    cmsFloat64Number cmsEstimateGamma(
        const cmsToneCurve* t,
        cmsFloat64Number Precision
    ) nogil

    cmsFloat64Number* cmsGetToneCurveParams(
        const cmsToneCurve* t
    ) nogil

    cmsUInt32Number cmsGetToneCurveEstimatedTableEntries(
        const cmsToneCurve* t
    ) nogil

    const cmsUInt16Number* cmsGetToneCurveEstimatedTable(
        const cmsToneCurve* t
    ) nogil

    ctypedef struct cmsPipeline:
        pass

    ctypedef struct cmsStage:
        pass

    cmsPipeline* cmsPipelineAlloc(
        cmsContext ContextID,
        cmsUInt32Number InputChannels,
        cmsUInt32Number OutputChannels
    ) nogil

    void cmsPipelineFree(
        cmsPipeline* lut
    ) nogil

    cmsPipeline* cmsPipelineDup(
        const cmsPipeline* Orig
    ) nogil

    cmsContext cmsGetPipelineContextID(
        const cmsPipeline* lut
    ) nogil

    cmsUInt32Number cmsPipelineInputChannels(
        const cmsPipeline* lut
    ) nogil

    cmsUInt32Number cmsPipelineOutputChannels(
        const cmsPipeline* lut
    ) nogil

    cmsUInt32Number cmsPipelineStageCount(
        const cmsPipeline* lut
    ) nogil

    cmsStage* cmsPipelineGetPtrToFirstStage(
        const cmsPipeline* lut
    ) nogil

    cmsStage* cmsPipelineGetPtrToLastStage(
        const cmsPipeline* lut
    ) nogil

    void cmsPipelineEval16(
        const cmsUInt16Number In[],
        cmsUInt16Number Out[],
        const cmsPipeline* lut
    ) nogil

    void cmsPipelineEvalFloat(
        const cmsFloat32Number In[],
        cmsFloat32Number Out[],
        const cmsPipeline* lut
    ) nogil

    cmsBool cmsPipelineEvalReverseFloat(
        cmsFloat32Number Target[],
        cmsFloat32Number Result[],
        cmsFloat32Number Hint[],
        const cmsPipeline* lut
    ) nogil

    cmsBool cmsPipelineCat(
        cmsPipeline* l1,
        const cmsPipeline* l2
    ) nogil

    cmsBool cmsPipelineSetSaveAs8bitsFlag(
        cmsPipeline* lut,
        cmsBool On
    ) nogil

    ctypedef enum cmsStageLoc:
       cmsAT_BEGIN
       cmsAT_END

    cmsBool cmsPipelineInsertStage(
        cmsPipeline* lut,
        cmsStageLoc loc,
        cmsStage* mpe
    ) nogil

    void cmsPipelineUnlinkStage(
        cmsPipeline* lut,
        cmsStageLoc loc,
        cmsStage** mpe
    ) nogil

    cmsBool cmsPipelineCheckAndRetreiveStages(
        const cmsPipeline* Lut,
        cmsUInt32Number n,
        ...
    ) nogil

    cmsStage* cmsStageAllocIdentity(
        cmsContext ContextID,
        cmsUInt32Number nChannels
    ) nogil

    cmsStage* cmsStageAllocToneCurves(
        cmsContext ContextID,
        cmsUInt32Number nChannels,
        cmsToneCurve* const Curves[]
    ) nogil

    cmsStage* cmsStageAllocMatrix(
        cmsContext ContextID,
        cmsUInt32Number Rows,
        cmsUInt32Number Cols,
        const cmsFloat64Number* Matrix,
        const cmsFloat64Number* Offset
    ) nogil

    cmsStage* cmsStageAllocCLut16bit(
        cmsContext ContextID,
        cmsUInt32Number nGridPoints,
        cmsUInt32Number inputChan,
        cmsUInt32Number outputChan,
        const cmsUInt16Number* Table
    ) nogil

    cmsStage* cmsStageAllocCLutFloat(
        cmsContext ContextID,
        cmsUInt32Number nGridPoints,
        cmsUInt32Number inputChan,
        cmsUInt32Number outputChan,
        const cmsFloat32Number* Table
    ) nogil

    cmsStage* cmsStageAllocCLut16bitGranular(
        cmsContext ContextID,
        const cmsUInt32Number clutPoints[],
        cmsUInt32Number inputChan,
        cmsUInt32Number outputChan,
        const cmsUInt16Number* Table
    ) nogil

    cmsStage* cmsStageAllocCLutFloatGranular(
        cmsContext ContextID,
        const cmsUInt32Number clutPoints[],
        cmsUInt32Number inputChan,
        cmsUInt32Number outputChan,
        const cmsFloat32Number* Table
    ) nogil

    cmsStage* cmsStageDup(
        cmsStage* mpe
    ) nogil

    void cmsStageFree(
        cmsStage* mpe
    ) nogil

    cmsStage* cmsStageNext(
        const cmsStage* mpe
    ) nogil

    cmsUInt32Number cmsStageInputChannels(
        const cmsStage* mpe
    ) nogil

    cmsUInt32Number cmsStageOutputChannels(
        const cmsStage* mpe
    ) nogil

    cmsStageSignature cmsStageType(
        const cmsStage* mpe
    ) nogil

    void* cmsStageData(
        const cmsStage* mpe
    ) nogil

    ctypedef cmsInt32Number(* cmsSAMPLER16)(
        const cmsUInt16Number In[],
        cmsUInt16Number Out[],
        void* Cargo
    ) nogil

    ctypedef cmsInt32Number(* cmsSAMPLERFLOAT)(
        const cmsFloat32Number In[],
        cmsFloat32Number Out[],
        void* Cargo
    ) nogil

    int SAMPLER_INSPECT

    cmsBool cmsStageSampleCLut16bit(
        cmsStage* mpe,
        cmsSAMPLER16 Sampler,
        void* Cargo,
        cmsUInt32Number dwFlags
    ) nogil

    cmsBool cmsStageSampleCLutFloat(
        cmsStage* mpe,
        cmsSAMPLERFLOAT Sampler,
        void* Cargo,
        cmsUInt32Number dwFlags
    ) nogil

    cmsBool cmsSliceSpace16(
        cmsUInt32Number nInputs,
        const cmsUInt32Number clutPoints[],

    cmsSAMPLER16 Sampler,
        void* Cargo
    ) nogil

    cmsBool cmsSliceSpaceFloat(
        cmsUInt32Number nInputs,
        const cmsUInt32Number clutPoints[],
        cmsSAMPLERFLOAT Sampler,
        void* Cargo
    ) nogil

    ctypedef struct cmsMLU:
        pass

    char* cmsNoLanguage
    char* cmsNoCountry

    cmsMLU* cmsMLUalloc(
        cmsContext ContextID,
        cmsUInt32Number nItems
    ) nogil

    void cmsMLUfree(
        cmsMLU* mlu
    ) nogil

    cmsMLU* cmsMLUdup(
        const cmsMLU* mlu
    ) nogil

    cmsBool cmsMLUsetASCII(
        cmsMLU* mlu,
        const char LanguageCode[3],
        const char CountryCode[3],
        const char* ASCIIString
    ) nogil

    cmsBool cmsMLUsetWide(
        cmsMLU* mlu,
        const char LanguageCode[3],
        const char CountryCode[3],
        const wchar_t* WideString
    ) nogil

    cmsUInt32Number cmsMLUgetASCII(
        const cmsMLU* mlu,
        const char LanguageCode[3],
        const char CountryCode[3],
        char* Buffer,
        cmsUInt32Number BufferSize
    ) nogil

    cmsUInt32Number cmsMLUgetWide(
        const cmsMLU* mlu,
        const char LanguageCode[3],
        const char CountryCode[3],
        wchar_t* Buffer,
        cmsUInt32Number BufferSize
    ) nogil

    cmsBool cmsMLUgetTranslation(
        const cmsMLU* mlu,
        const char LanguageCode[3],
        const char CountryCode[3],
        char ObtainedLanguage[3],
        char ObtainedCountry[3]
    ) nogil

    cmsUInt32Number cmsMLUtranslationsCount(
        const cmsMLU* mlu
    ) nogil

    cmsBool cmsMLUtranslationsCodes(
        const cmsMLU* mlu,
        cmsUInt32Number idx,
        char LanguageCode[3],
        char CountryCode[3]
    ) nogil

    ctypedef struct cmsUcrBg:
        cmsToneCurve* Ucr
        cmsToneCurve* Bg
        cmsMLU* Desc

    int cmsPRINTER_DEFAULT_SCREENS
    int cmsFREQUENCE_UNITS_LINES_CM
    int cmsFREQUENCE_UNITS_LINES_INCH

    int cmsSPOT_UNKNOWN
    int cmsSPOT_PRINTER_DEFAULT
    int cmsSPOT_ROUND
    int cmsSPOT_DIAMOND
    int cmsSPOT_ELLIPSE
    int cmsSPOT_LINE
    int cmsSPOT_SQUARE
    int cmsSPOT_CROSS

    ctypedef struct cmsScreeningChannel:
        cmsFloat64Number Frequency
        cmsFloat64Number ScreenAngle
        cmsUInt32Number SpotShape

    ctypedef struct cmsScreening:
        cmsUInt32Number Flag
        cmsUInt32Number nChannels
        cmsScreeningChannel Channels[16]  # [cmsMAXCHANNELS]

    ctypedef struct cmsNAMEDCOLORLIST:
        pass

    cmsNAMEDCOLORLIST* cmsAllocNamedColorList(
        cmsContext ContextID,
        cmsUInt32Number n,
        cmsUInt32Number ColorantCount,
        const char* Prefix,
        const char* Suffix
    ) nogil

    void cmsFreeNamedColorList(
        cmsNAMEDCOLORLIST* v
    ) nogil

    cmsNAMEDCOLORLIST* cmsDupNamedColorList(
        const cmsNAMEDCOLORLIST* v
    ) nogil

    cmsBool cmsAppendNamedColor(
        cmsNAMEDCOLORLIST* v,
        const char* Name,
        cmsUInt16Number PCS[3],
        cmsUInt16Number Colorant[]
    ) nogil

    cmsUInt32Number cmsNamedColorCount(
        const cmsNAMEDCOLORLIST* v
    ) nogil

    cmsInt32Number cmsNamedColorIndex(
        const cmsNAMEDCOLORLIST* v,
        const char* Name
    ) nogil

    cmsBool cmsNamedColorInfo(
        const cmsNAMEDCOLORLIST* NamedColorList,
        cmsUInt32Number nColor,
        char* Name,
        char* Prefix,
        char* Suffix,
        cmsUInt16Number* PCS,
        cmsUInt16Number* Colorant
    ) nogil

    cmsNAMEDCOLORLIST* cmsGetNamedColorList(
        cmsHTRANSFORM xform
    ) nogil

    ctypedef struct cmsPSEQDESC:
        cmsSignature deviceMfg
        cmsSignature deviceModel
        cmsUInt64Number attributes
        cmsTechnologySignature technology
        cmsProfileID ProfileID
        cmsMLU* Manufacturer
        cmsMLU* Model
        cmsMLU* Description

    ctypedef struct cmsSEQ:
        cmsUInt32Number n
        cmsContext ContextID
        cmsPSEQDESC* seq

    cmsSEQ* cmsAllocProfileSequenceDescription(
        cmsContext ContextID,
        cmsUInt32Number n
    ) nogil

    cmsSEQ* cmsDupProfileSequenceDescription(
        const cmsSEQ* pseq
    ) nogil

    void cmsFreeProfileSequenceDescription(
        cmsSEQ* pseq
    ) nogil

    ctypedef struct cmsDICTentry:
        cmsDICTentry* Next
        cmsMLU *DisplayName
        cmsMLU *DisplayValue
        wchar_t* Name
        wchar_t* Value

    cmsHANDLE cmsDictAlloc(
        cmsContext ContextID
    ) nogil

    void cmsDictFree(
        cmsHANDLE hDict
    ) nogil

    cmsHANDLE cmsDictDup(
        cmsHANDLE hDict
    ) nogil

    cmsBool cmsDictAddEntry(
        cmsHANDLE hDict,
        const wchar_t* Name,
        const wchar_t* Value,
        const cmsMLU *DisplayName,
        const cmsMLU *DisplayValue
    ) nogil

    const cmsDICTentry* cmsDictGetEntryList(
        cmsHANDLE hDict
    ) nogil

    const cmsDICTentry* cmsDictNextEntry(
        const cmsDICTentry* e
    ) nogil

    cmsHPROFILE cmsCreateProfilePlaceholder(
        cmsContext ContextID
    ) nogil

    cmsContext cmsGetProfileContextID(
        cmsHPROFILE hProfile
    ) nogil

    cmsInt32Number cmsGetTagCount(
        cmsHPROFILE hProfile
    ) nogil

    cmsTagSignature cmsGetTagSignature(
        cmsHPROFILE hProfile,
        cmsUInt32Number n
    ) nogil

    cmsBool cmsIsTag(
        cmsHPROFILE hProfile,
        cmsTagSignature sig
    ) nogil

    void* cmsReadTag(
        cmsHPROFILE hProfile,
        cmsTagSignature sig
    ) nogil

    cmsBool cmsWriteTag(
        cmsHPROFILE hProfile,
        cmsTagSignature sig,
        const void* data
    ) nogil

    cmsBool cmsLinkTag(
        cmsHPROFILE hProfile,
        cmsTagSignature sig,
        cmsTagSignature dest
    ) nogil

    cmsTagSignature cmsTagLinkedTo(
        cmsHPROFILE hProfile,
        cmsTagSignature sig
    ) nogil

    cmsUInt32Number cmsReadRawTag(
        cmsHPROFILE hProfile,
        cmsTagSignature sig,
        void* Buffer,
        cmsUInt32Number BufferSize
    ) nogil

    cmsBool cmsWriteRawTag(
        cmsHPROFILE hProfile,
        cmsTagSignature sig,
        const void* data,
        cmsUInt32Number Size
    ) nogil

    int cmsEmbeddedProfileFalse
    int cmsEmbeddedProfileTrue
    int cmsUseAnywhere
    int cmsUseWithEmbeddedDataOnly

    cmsUInt32Number cmsGetHeaderFlags(
        cmsHPROFILE hProfile
    ) nogil

    void cmsGetHeaderAttributes(
        cmsHPROFILE hProfile,
        cmsUInt64Number* Flags
    ) nogil

    void cmsGetHeaderProfileID(
        cmsHPROFILE hProfile,
        cmsUInt8Number* ProfileID
    ) nogil

    cmsBool cmsGetHeaderCreationDateTime(
        cmsHPROFILE hProfile,
        tm *Dest
    ) nogil

    cmsUInt32Number cmsGetHeaderRenderingIntent(
        cmsHPROFILE hProfile
    ) nogil

    void cmsSetHeaderFlags(
        cmsHPROFILE hProfile,
        cmsUInt32Number Flags
    ) nogil

    cmsUInt32Number cmsGetHeaderManufacturer(
        cmsHPROFILE hProfile
    ) nogil

    void cmsSetHeaderManufacturer(
        cmsHPROFILE hProfile,
        cmsUInt32Number manufacturer
    ) nogil

    cmsUInt32Number cmsGetHeaderCreator(
        cmsHPROFILE hProfile
    ) nogil

    cmsUInt32Number cmsGetHeaderModel(
        cmsHPROFILE hProfile
    ) nogil

    void cmsSetHeaderModel(
        cmsHPROFILE hProfile,
        cmsUInt32Number model
    ) nogil

    void cmsSetHeaderAttributes(
        cmsHPROFILE hProfile,
        cmsUInt64Number Flags
    ) nogil

    void cmsSetHeaderProfileID(
        cmsHPROFILE hProfile,
        cmsUInt8Number* ProfileID
    ) nogil

    void cmsSetHeaderRenderingIntent(
        cmsHPROFILE hProfile,
        cmsUInt32Number RenderingIntent
    ) nogil

    cmsColorSpaceSignature cmsGetPCS(
        cmsHPROFILE hProfile
    ) nogil

    void cmsSetPCS(
        cmsHPROFILE hProfile,
        cmsColorSpaceSignature pcs
    ) nogil

    cmsColorSpaceSignature cmsGetColorSpace(
        cmsHPROFILE hProfile
    ) nogil

    void cmsSetColorSpace(
        cmsHPROFILE hProfile,
        cmsColorSpaceSignature sig
    ) nogil

    cmsProfileClassSignature cmsGetDeviceClass(
        cmsHPROFILE hProfile
    ) nogil

    void cmsSetDeviceClass(
        cmsHPROFILE hProfile,
        cmsProfileClassSignature sig
    ) nogil

    void cmsSetProfileVersion(
        cmsHPROFILE hProfile,
        cmsFloat64Number Version
    ) nogil

    cmsFloat64Number cmsGetProfileVersion(
        cmsHPROFILE hProfile
    ) nogil

    cmsUInt32Number cmsGetEncodedICCversion(
        cmsHPROFILE hProfile
    ) nogil

    void cmsSetEncodedICCversion(
        cmsHPROFILE hProfile,
        cmsUInt32Number Version
    ) nogil

    int LCMS_USED_AS_INPUT
    int LCMS_USED_AS_OUTPUT
    int LCMS_USED_AS_PROOF

    cmsBool cmsIsIntentSupported(
        cmsHPROFILE hProfile,
        cmsUInt32Number Intent,
        cmsUInt32Number UsedDirection
    ) nogil

    cmsBool cmsIsMatrixShaper(
        cmsHPROFILE hProfile
    ) nogil

    cmsBool cmsIsCLUT(
        cmsHPROFILE hProfile,
        cmsUInt32Number Intent,
        cmsUInt32Number UsedDirection
    ) nogil

    cmsColorSpaceSignature _cmsICCcolorSpace(
        int OurNotation
    ) nogil

    int _cmsLCMScolorSpace(
        cmsColorSpaceSignature ProfileSpace
    ) nogil

    cmsUInt32Number cmsChannelsOf(
        cmsColorSpaceSignature ColorSpace
    ) nogil

    cmsUInt32Number cmsFormatterForColorspaceOfProfile(
        cmsHPROFILE hProfile,
        cmsUInt32Number nBytes,
        cmsBool lIsFloat
    ) nogil

    cmsUInt32Number cmsFormatterForPCSOfProfile(
        cmsHPROFILE hProfile,
        cmsUInt32Number nBytes,
        cmsBool lIsFloat
    ) nogil

    ctypedef enum cmsInfoType:
        cmsInfoDescription
        cmsInfoManufacturer
        cmsInfoModel
        cmsInfoCopyright

    cmsUInt32Number cmsGetProfileInfo(
        cmsHPROFILE hProfile,
        cmsInfoType Info,
        const char LanguageCode[3],
        const char CountryCode[3],
        wchar_t* Buffer,
        cmsUInt32Number BufferSize
    ) nogil

    cmsUInt32Number cmsGetProfileInfoASCII(
        cmsHPROFILE hProfile,
        cmsInfoType Info,
        const char LanguageCode[3],
        const char CountryCode[3],
        char* Buffer,
        cmsUInt32Number BufferSize
    ) nogil

    ctypedef struct cmsIOHANDLER:
        pass

    cmsIOHANDLER* cmsOpenIOhandlerFromFile(
        cmsContext ContextID,
        const char* FileName,
        const char* AccessMode
    ) nogil

    cmsIOHANDLER* cmsOpenIOhandlerFromStream(
        cmsContext ContextID,
        FILE* Stream
    ) nogil

    cmsIOHANDLER* cmsOpenIOhandlerFromMem(
        cmsContext ContextID,
        void*Buffer,
        cmsUInt32Number size,
        const char* AccessMode
    ) nogil

    cmsIOHANDLER* cmsOpenIOhandlerFromNULL(
        cmsContext ContextID
    ) nogil

    cmsIOHANDLER* cmsGetProfileIOhandler(
        cmsHPROFILE hProfile
    ) nogil

    cmsBool cmsCloseIOhandler(
        cmsIOHANDLER* io
    ) nogil

    cmsBool cmsMD5computeID(
        cmsHPROFILE hProfile
    ) nogil

    cmsHPROFILE cmsOpenProfileFromFile(
        const char *ICCProfile,
        const char *sAccess
    ) nogil

    cmsHPROFILE cmsOpenProfileFromFileTHR(
        cmsContext ContextID,
        const char *ICCProfile,
        const char *sAccess
    ) nogil

    cmsHPROFILE cmsOpenProfileFromStream(
        FILE* ICCProfile,
        const char* sAccess
    ) nogil

    cmsHPROFILE cmsOpenProfileFromStreamTHR(
        cmsContext ContextID,
        FILE* ICCProfile,
        const char* sAccess
    ) nogil

    cmsHPROFILE cmsOpenProfileFromMem(
        const void* MemPtr,
        cmsUInt32Number dwSize
    ) nogil

    cmsHPROFILE cmsOpenProfileFromMemTHR(
        cmsContext ContextID,
        const void* MemPtr,
        cmsUInt32Number dwSize
    ) nogil

    cmsHPROFILE cmsOpenProfileFromIOhandlerTHR(
        cmsContext ContextID,
        cmsIOHANDLER* io
    ) nogil

    cmsHPROFILE cmsOpenProfileFromIOhandler2THR(
        cmsContext ContextID,
        cmsIOHANDLER* io,
        cmsBool write
    ) nogil

    cmsBool cmsCloseProfile(
        cmsHPROFILE hProfile
    ) nogil

    cmsBool cmsSaveProfileToFile(
        cmsHPROFILE hProfile,
        const char* FileName
    ) nogil

    cmsBool cmsSaveProfileToStream(
        cmsHPROFILE hProfile,
        FILE* Stream
    ) nogil

    cmsBool cmsSaveProfileToMem(
        cmsHPROFILE hProfile,
        void*MemPtr,
        cmsUInt32Number* BytesNeeded
    ) nogil

    cmsUInt32Number cmsSaveProfileToIOhandler(
        cmsHPROFILE hProfile,
        cmsIOHANDLER* io
    ) nogil

    cmsHPROFILE cmsCreateRGBProfileTHR(
        cmsContext ContextID,
        const cmsCIExyY* WhitePoint,
        const cmsCIExyYTRIPLE* Primaries,
        cmsToneCurve* const TransferFunction[3]
    ) nogil

    cmsHPROFILE cmsCreateRGBProfile(
        const cmsCIExyY* WhitePoint,
        const cmsCIExyYTRIPLE* Primaries,
        cmsToneCurve* const TransferFunction[3]
    ) nogil

    cmsHPROFILE cmsCreateGrayProfileTHR(
        cmsContext ContextID,
        const cmsCIExyY* WhitePoint,
        const cmsToneCurve* TransferFunction
    ) nogil

    cmsHPROFILE cmsCreateGrayProfile(
        const cmsCIExyY* WhitePoint,

    const cmsToneCurve* TransferFunction
    ) nogil

    cmsHPROFILE cmsCreateLinearizationDeviceLinkTHR(
        cmsContext ContextID,
        cmsColorSpaceSignature ColorSpace,
        cmsToneCurve* TransferFunctions[]
    ) nogil

    cmsHPROFILE cmsCreateLinearizationDeviceLink(
        cmsColorSpaceSignature ColorSpace,
        cmsToneCurve* TransferFunctions[]
    ) nogil

    cmsHPROFILE cmsCreateInkLimitingDeviceLinkTHR(
        cmsContext ContextID,
        cmsColorSpaceSignature ColorSpace,
        cmsFloat64Number Limit
    ) nogil

    cmsHPROFILE cmsCreateInkLimitingDeviceLink(
        cmsColorSpaceSignature ColorSpace,
        cmsFloat64Number Limit
    ) nogil

    cmsHPROFILE cmsCreateLab2ProfileTHR(
        cmsContext ContextID,
        const cmsCIExyY* WhitePoint
    ) nogil

    cmsHPROFILE cmsCreateLab2Profile(
        const cmsCIExyY* WhitePoint
    ) nogil

    cmsHPROFILE cmsCreateLab4ProfileTHR(
        cmsContext ContextID,
        const cmsCIExyY* WhitePoint
    ) nogil

    cmsHPROFILE cmsCreateLab4Profile(
        const cmsCIExyY* WhitePoint
    ) nogil

    cmsHPROFILE cmsCreateXYZProfileTHR(
        cmsContext ContextID
    ) nogil

    cmsHPROFILE cmsCreateXYZProfile(
    ) nogil

    cmsHPROFILE cmsCreate_sRGBProfileTHR(
        cmsContext ContextID
    ) nogil

    cmsHPROFILE cmsCreate_sRGBProfile(
    ) nogil

    cmsHPROFILE cmsCreateBCHSWabstractProfileTHR(
        cmsContext ContextID,
        cmsUInt32Number nLUTPoints,
        cmsFloat64Number Bright,
        cmsFloat64Number Contrast,
        cmsFloat64Number Hue,
        cmsFloat64Number Saturation,
        cmsUInt32Number TempSrc,
        cmsUInt32Number TempDest
    ) nogil

    cmsHPROFILE cmsCreateBCHSWabstractProfile(
        cmsUInt32Number nLUTPoints,
        cmsFloat64Number Bright,
        cmsFloat64Number Contrast,
        cmsFloat64Number Hue,
        cmsFloat64Number Saturation,
        cmsUInt32Number TempSrc,
        cmsUInt32Number TempDest
    ) nogil

    cmsHPROFILE cmsCreateNULLProfileTHR(
        cmsContext ContextID
    ) nogil

    cmsHPROFILE cmsCreateNULLProfile(
    ) nogil

    cmsHPROFILE cmsTransform2DeviceLink(
        cmsHTRANSFORM hTransform,
        cmsFloat64Number Version,
        cmsUInt32Number dwFlags
    ) nogil

    int INTENT_PERCEPTUAL
    int INTENT_RELATIVE_COLORIMETRIC
    int INTENT_SATURATION
    int INTENT_ABSOLUTE_COLORIMETRIC

    int INTENT_PRESERVE_K_ONLY_PERCEPTUAL
    int INTENT_PRESERVE_K_ONLY_RELATIVE_COLORIMETRIC
    int INTENT_PRESERVE_K_ONLY_SATURATION
    int INTENT_PRESERVE_K_PLANE_PERCEPTUAL
    int INTENT_PRESERVE_K_PLANE_RELATIVE_COLORIMETRIC
    int INTENT_PRESERVE_K_PLANE_SATURATION

    cmsUInt32Number cmsGetSupportedIntents(
        cmsUInt32Number nMax,
        cmsUInt32Number* Codes,
        char** Descriptions
    ) nogil

    cmsUInt32Number cmsGetSupportedIntentsTHR(
        cmsContext ContextID,
        cmsUInt32Number nMax,
        cmsUInt32Number* Codes,
        char** Descriptions
    ) nogil

    int cmsFLAGS_NOCACHE
    int cmsFLAGS_NOOPTIMIZE
    int cmsFLAGS_NULLTRANSFORM
    int cmsFLAGS_GAMUTCHECK
    int cmsFLAGS_SOFTPROOFING
    int cmsFLAGS_BLACKPOINTCOMPENSATION
    int cmsFLAGS_NOWHITEONWHITEFIXUP
    int cmsFLAGS_HIGHRESPRECALC
    int cmsFLAGS_LOWRESPRECALC
    int cmsFLAGS_8BITS_DEVICELINK
    int cmsFLAGS_GUESSDEVICECLASS
    int cmsFLAGS_KEEP_SEQUENCE
    int cmsFLAGS_FORCE_CLUT
    int cmsFLAGS_CLUT_POST_LINEARIZATION
    int cmsFLAGS_CLUT_PRE_LINEARIZATION
    int cmsFLAGS_NONEGATIVES
    int cmsFLAGS_COPY_ALPHA
    int cmsFLAGS_GRIDPOINTS
    int cmsFLAGS_NODEFAULTRESOURCEDEF

    cmsHTRANSFORM cmsCreateTransformTHR(
        cmsContext ContextID,
        cmsHPROFILE Input,
        cmsUInt32Number InputFormat,
        cmsHPROFILE Output,
        cmsUInt32Number OutputFormat,
        cmsUInt32Number Intent,
        cmsUInt32Number dwFlags
    ) nogil

    cmsHTRANSFORM cmsCreateTransform(
        cmsHPROFILE Input,
        cmsUInt32Number InputFormat,
        cmsHPROFILE Output,
        cmsUInt32Number OutputFormat,
        cmsUInt32Number Intent,
        cmsUInt32Number dwFlags
    ) nogil

    cmsHTRANSFORM cmsCreateProofingTransformTHR(
        cmsContext ContextID,
        cmsHPROFILE Input,
        cmsUInt32Number InputFormat,
        cmsHPROFILE Output,
        cmsUInt32Number OutputFormat,
        cmsHPROFILE Proofing,
        cmsUInt32Number Intent,
        cmsUInt32Number ProofingIntent,
        cmsUInt32Number dwFlags
    ) nogil

    cmsHTRANSFORM cmsCreateProofingTransform(
        cmsHPROFILE Input,
        cmsUInt32Number InputFormat,
        cmsHPROFILE Output,
        cmsUInt32Number OutputFormat,
        cmsHPROFILE Proofing,
        cmsUInt32Number Intent,
        cmsUInt32Number ProofingIntent,
        cmsUInt32Number dwFlags
    ) nogil

    cmsHTRANSFORM cmsCreateMultiprofileTransformTHR(
        cmsContext ContextID,
        cmsHPROFILE hProfiles[],
        cmsUInt32Number nProfiles,
        cmsUInt32Number InputFormat,
        cmsUInt32Number OutputFormat,
        cmsUInt32Number Intent,
        cmsUInt32Number dwFlags
    ) nogil

    cmsHTRANSFORM cmsCreateMultiprofileTransform(
        cmsHPROFILE hProfiles[],
        cmsUInt32Number nProfiles,
        cmsUInt32Number InputFormat,
        cmsUInt32Number OutputFormat,
        cmsUInt32Number Intent,
        cmsUInt32Number dwFlags
    ) nogil

    cmsHTRANSFORM cmsCreateExtendedTransform(
        cmsContext ContextID,
        cmsUInt32Number nProfiles,
        cmsHPROFILE hProfiles[],
        cmsBool BPC[],
        cmsUInt32Number Intents[],
        cmsFloat64Number AdaptationStates[],
        cmsHPROFILE hGamutProfile,
        cmsUInt32Number nGamutPCSposition,
        cmsUInt32Number InputFormat,
        cmsUInt32Number OutputFormat,
        cmsUInt32Number dwFlags
    ) nogil

    void cmsDeleteTransform(
        cmsHTRANSFORM hTransform
    ) nogil

    void cmsDoTransform(
        cmsHTRANSFORM Transform,
        const void* InputBuffer,
        void* OutputBuffer,
        cmsUInt32Number Size
    ) nogil

    void cmsDoTransformStride(
        cmsHTRANSFORM Transform,
        const void* InputBuffer,
        void* OutputBuffer,
        cmsUInt32Number Size,
        cmsUInt32Number Stride
    ) nogil

    void cmsDoTransformLineStride(
        cmsHTRANSFORM Transform,
        const void* InputBuffer,
        void* OutputBuffer,
        cmsUInt32Number PixelsPerLine,
        cmsUInt32Number LineCount,
        cmsUInt32Number BytesPerLineIn,
        cmsUInt32Number BytesPerLineOut,
        cmsUInt32Number BytesPerPlaneIn,
        cmsUInt32Number BytesPerPlaneOut
    ) nogil

    void cmsSetAlarmCodes(
        const cmsUInt16Number NewAlarm[]
    ) nogil

    void cmsGetAlarmCodes(
        cmsUInt16Number NewAlarm[]
    ) nogil

    void cmsSetAlarmCodesTHR(
        cmsContext ContextID,

    const cmsUInt16Number AlarmCodes[]
    ) nogil

    void cmsGetAlarmCodesTHR(
        cmsContext ContextID,
        cmsUInt16Number AlarmCodes[]
    ) nogil

    cmsFloat64Number cmsSetAdaptationState(
        cmsFloat64Number d
    ) nogil

    cmsFloat64Number cmsSetAdaptationStateTHR(
        cmsContext ContextID,
        cmsFloat64Number d
    ) nogil

    cmsContext cmsGetTransformContextID(
        cmsHTRANSFORM hTransform
    ) nogil

    cmsUInt32Number cmsGetTransformInputFormat(
        cmsHTRANSFORM hTransform
    ) nogil

    cmsUInt32Number cmsGetTransformOutputFormat(
        cmsHTRANSFORM hTransform
    ) nogil

    cmsBool cmsChangeBuffersFormat(
        cmsHTRANSFORM hTransform,
        cmsUInt32Number InputFormat,
        cmsUInt32Number OutputFormat
    ) nogil

    ctypedef enum cmsPSResourceType:
       cmsPS_RESOURCE_CSA
       cmsPS_RESOURCE_CRD

    cmsUInt32Number cmsGetPostScriptColorResource(
        cmsContext ContextID,
        cmsPSResourceType Type,
        cmsHPROFILE hProfile,
        cmsUInt32Number Intent,
        cmsUInt32Number dwFlags,
        cmsIOHANDLER* io
    ) nogil

    cmsUInt32Number cmsGetPostScriptCSA(
        cmsContext ContextID,
        cmsHPROFILE hProfile,
        cmsUInt32Number Intent,
        cmsUInt32Number dwFlags,
        void* Buffer,
        cmsUInt32Number dwBufferLen
    ) nogil

    cmsUInt32Number cmsGetPostScriptCRD(
        cmsContext ContextID,
        cmsHPROFILE hProfile,
        cmsUInt32Number Intent,
        cmsUInt32Number dwFlags,
        void* Buffer,
        cmsUInt32Number dwBufferLen
    ) nogil

    cmsHANDLE cmsIT8Alloc(
        cmsContext ContextID
    ) nogil

    void cmsIT8Free(
        cmsHANDLE hIT8
    ) nogil

    cmsUInt32Number cmsIT8TableCount(
        cmsHANDLE hIT8
    ) nogil

    cmsInt32Number cmsIT8SetTable(
        cmsHANDLE hIT8,
        cmsUInt32Number nTable
    ) nogil

    cmsHANDLE cmsIT8LoadFromFile(
        cmsContext ContextID,
        const char* cFileName
    ) nogil

    cmsHANDLE cmsIT8LoadFromMem(
        cmsContext ContextID,
        const void*Ptr,
        cmsUInt32Number len
    ) nogil

    cmsBool cmsIT8SaveToFile(
        cmsHANDLE hIT8,
        const char* cFileName
    ) nogil

    cmsBool cmsIT8SaveToMem(
        cmsHANDLE hIT8,
        void*MemPtr,
        cmsUInt32Number* BytesNeeded
    ) nogil

    const char* cmsIT8GetSheetType(
        cmsHANDLE hIT8
    ) nogil

    cmsBool cmsIT8SetSheetType(
        cmsHANDLE hIT8,
        const char* Type
    ) nogil

    cmsBool cmsIT8SetComment(
        cmsHANDLE hIT8,
        const char* cComment
    ) nogil

    cmsBool cmsIT8SetPropertyStr(
        cmsHANDLE hIT8,
        const char* cProp,
        const char *Str
    ) nogil

    cmsBool cmsIT8SetPropertyDbl(
        cmsHANDLE hIT8,
        const char* cProp,
        cmsFloat64Number Val
    ) nogil

    cmsBool cmsIT8SetPropertyHex(
        cmsHANDLE hIT8,
        const char* cProp,
        cmsUInt32Number Val
    ) nogil

    cmsBool cmsIT8SetPropertyMulti(
        cmsHANDLE hIT8,
        const char* Key,
        const char* SubKey,
        const char *Buffer
    ) nogil

    cmsBool cmsIT8SetPropertyUncooked(
        cmsHANDLE hIT8,
        const char* Key,
        const char* Buffer
    ) nogil

    const char* cmsIT8GetProperty(
        cmsHANDLE hIT8,
        const char* cProp
    ) nogil

    cmsFloat64Number cmsIT8GetPropertyDbl(
        cmsHANDLE hIT8,
        const char* cProp
    ) nogil

    const char* cmsIT8GetPropertyMulti(
        cmsHANDLE hIT8,
        const char* Key,
        const char *SubKey
    ) nogil

    cmsUInt32Number cmsIT8EnumProperties(
        cmsHANDLE hIT8,
        char ***PropertyNames
    ) nogil

    cmsUInt32Number cmsIT8EnumPropertyMulti(
        cmsHANDLE hIT8,
        const char* cProp,
        const char ***SubpropertyNames
    ) nogil

    const char* cmsIT8GetDataRowCol(
        cmsHANDLE hIT8,
        int row,
        int col
    ) nogil

    cmsFloat64Number cmsIT8GetDataRowColDbl(
        cmsHANDLE hIT8,
        int row,
        int col
    ) nogil

    cmsBool cmsIT8SetDataRowCol(
        cmsHANDLE hIT8,
        int row,
        int col,
        const char* Val
    ) nogil

    cmsBool cmsIT8SetDataRowColDbl(
        cmsHANDLE hIT8,
        int row,
        int col,
        cmsFloat64Number Val
    ) nogil

    const char* cmsIT8GetData(
        cmsHANDLE hIT8,
        const char* cPatch,
        const char* cSample
    ) nogil

    cmsFloat64Number cmsIT8GetDataDbl(
        cmsHANDLE hIT8,
        const char* cPatch,
        const char* cSample
    ) nogil

    cmsBool cmsIT8SetData(
        cmsHANDLE hIT8,
        const char* cPatch,
        const char* cSample,
        const char *Val
    ) nogil

    cmsBool cmsIT8SetDataDbl(
        cmsHANDLE hIT8,
        const char* cPatch,
        const char* cSample,
        cmsFloat64Number Val
    ) nogil

    int cmsIT8FindDataFormat(
        cmsHANDLE hIT8,
        const char* cSample
    ) nogil

    cmsBool cmsIT8SetDataFormat(
        cmsHANDLE hIT8,
        int n,
        const char *Sample
    ) nogil

    int cmsIT8EnumDataFormat(
        cmsHANDLE hIT8,
        char ***SampleNames
    ) nogil

    const char* cmsIT8GetPatchName(
        cmsHANDLE hIT8,
        int nPatch,
        char* buffer
    ) nogil

    int cmsIT8GetPatchByName(
        cmsHANDLE hIT8,
        const char *cPatch
    ) nogil

    int cmsIT8SetTableByLabel(
        cmsHANDLE hIT8,
        const char* cSet,
        const char* cField,
        const char* ExpectedType
    ) nogil

    cmsBool cmsIT8SetIndexColumn(
        cmsHANDLE hIT8,
        const char* cSample
    ) nogil

    void cmsIT8DefineDblFormat(
        cmsHANDLE hIT8,
        const char* Formatter
    ) nogil

    cmsHANDLE cmsGBDAlloc(
        cmsContext ContextID
    ) nogil

    void cmsGBDFree(
        cmsHANDLE hGBD
    ) nogil

    cmsBool cmsGDBAddPoint(
        cmsHANDLE hGBD,
        const cmsCIELab* Lab
    ) nogil

    cmsBool cmsGDBCompute(
        cmsHANDLE hGDB,
        cmsUInt32Number dwFlags
    ) nogil

    cmsBool cmsGDBCheckPoint(
        cmsHANDLE hGBD,
        const cmsCIELab* Lab
    ) nogil

    cmsBool cmsDetectBlackPoint(
        cmsCIEXYZ* BlackPoint,
        cmsHPROFILE hProfile,
        cmsUInt32Number Intent,
        cmsUInt32Number dwFlags
    ) nogil

    cmsBool cmsDetectDestinationBlackPoint(
        cmsCIEXYZ* BlackPoint,
        cmsHPROFILE hProfile,
        cmsUInt32Number Intent,
        cmsUInt32Number dwFlags
    ) nogil

    cmsFloat64Number cmsDetectTAC(
        cmsHPROFILE hProfile
    ) nogil

    cmsBool cmsDesaturateLab(
        cmsCIELab* Lab,
        double amax,
        double amin,
        double bmax,
        double bmin
    ) nogil
