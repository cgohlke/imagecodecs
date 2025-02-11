# imagecodecs/lcms2.pxd
# cython: language_level = 3

# Cython declarations for the `Little 2.17.0` library.
# https://github.com/mm2/Little-CMS

from libc.stddef cimport wchar_t
from libc.stdio cimport FILE
from libc.time cimport tm

cdef extern from 'lcms2.h' nogil:

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
        cmsSigcicpType
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
        cmsSigMHC2Type

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
        cmsSigcicpTag
        cmsSigArgyllArtsTag
        cmsSigMHC2Tag

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
        cmsUInt8Number[1] data

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
        cmsUInt8Number[16] ID8
        cmsUInt16Number[8] ID16
        cmsUInt32Number[4] ID32

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
        cmsInt8Number[28] reserved

    ctypedef struct cmsTagBase:
        cmsTagTypeSignature sig
        cmsInt8Number[4] reserved

    ctypedef struct cmsTagEntry:
        cmsTagSignature sig
        cmsUInt32Number offset
        cmsUInt32Number size

    ctypedef void* cmsHANDLE
    ctypedef void* cmsHPROFILE
    ctypedef void* cmsHTRANSFORM

    int cmsMAXCHANNELS

    int PREMUL_SH(int)
    int FLOAT_SH(int)
    int OPTIMIZED_SH(int)
    int COLORSPACE_SH(int)
    int SWAPFIRST_SH(int)
    int FLAVOR_SH(int)
    int PLANAR_SH(int)
    int ENDIAN16_SH(int)
    int DOSWAP_SH(int)
    int EXTRA_SH(int)
    int CHANNELS_SH(int)
    int BYTES_SH(int)

    int T_PREMUL(int)
    int T_FLOAT(int)
    int T_OPTIMIZED(int)
    int T_COLORSPACE(int)
    int T_SWAPFIRST(int)
    int T_FLAVOR(int)
    int T_PLANAR(int)
    int T_ENDIAN16(int)
    int T_DOSWAP(int)
    int T_EXTRA(int)
    int T_CHANNELS(int)
    int T_BYTES(int)

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
    int TYPE_GRAYA_8_PREMUL
    int TYPE_GRAYA_16
    int TYPE_GRAYA_16_PREMUL
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
    int TYPE_RGBA_8_PREMUL
    int TYPE_RGBA_8_PLANAR
    int TYPE_RGBA_16
    int TYPE_RGBA_16_PREMUL
    int TYPE_RGBA_16_PLANAR
    int TYPE_RGBA_16_SE

    int TYPE_ARGB_8
    int TYPE_ARGB_8_PLANAR
    int TYPE_ARGB_16

    int TYPE_ABGR_8
    int TYPE_ARGB_8_PREMUL
    int TYPE_ABGR_8_PLANAR
    int TYPE_ABGR_16
    int TYPE_ARGB_16_PREMUL
    int TYPE_ABGR_16_PLANAR
    int TYPE_ABGR_16_SE

    int TYPE_BGRA_8
    int TYPE_BGRA_8_PREMUL
    int TYPE_BGRA_8_PLANAR
    int TYPE_BGRA_16
    int TYPE_BGRA_16_PREMUL
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
    int TYPE_GRAYA_FLT
    int TYPE_GRAYA_FLT_PREMUL
    int TYPE_RGB_FLT

    int TYPE_RGBA_FLT
    int TYPE_RGBA_FLT_PREMUL
    int TYPE_ARGB_FLT
    int TYPE_ARGB_FLT_PREMUL
    int TYPE_BGR_FLT
    int TYPE_BGRA_FLT
    int TYPE_BGRA_FLT_PREMUL
    int TYPE_ABGR_FLT
    int TYPE_ABGR_FLT_PREMUL

    int TYPE_CMYK_FLT

    int TYPE_XYZ_DBL
    int TYPE_Lab_DBL
    int TYPE_GRAY_DBL
    int TYPE_RGB_DBL
    int TYPE_BGR_DBL
    int TYPE_CMYK_DBL
    int TYPE_OKLAB_DBL

    int TYPE_GRAY_HALF_FLT
    int TYPE_RGB_HALF_FLT
    int TYPE_CMYK_HALF_FLT

    int TYPE_RGBA_HALF_FLT
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

    ctypedef struct cmsVideoSignalType:
        cmsUInt8Number ColourPrimaries
        cmsUInt8Number TransferCharacteristics
        cmsUInt8Number MatrixCoefficients
        cmsUInt8Number VideoFullRangeFlag

    ctypedef struct cmsMHC2Type:
        cmsUInt32Number CurveEntries
        cmsFloat64Number* RedCurve
        cmsFloat64Number* GreenCurve
        cmsFloat64Number* BlueCurve
        cmsFloat64Number MinLuminance
        cmsFloat64Number PeakLuminance
        cmsFloat64Number[3][4] XYZ2XYZmatrix

    int cmsGetEncodedCMMversion()

    int cmsstrcasecmp(
        const char* s1,
        const char* s2
    )

    long int cmsfilelength(
        FILE* f
    )

    ctypedef struct cmsContext:
        pass

    cmsContext cmsCreateContext(
        void* Plugin,
        void* UserData
    )

    void cmsDeleteContext(
        cmsContext ContextID
    )

    cmsContext cmsDupContext(
        cmsContext ContextID,
        void* NewUserData
    )

    void* cmsGetContextUserData(
        cmsContext ContextID
    )

    cmsBool cmsPlugin(
        void* Plugin
    )

    cmsBool cmsPluginTHR(
        cmsContext ContextID,
        void* Plugin
    )

    void cmsUnregisterPlugins()

    void cmsUnregisterPluginsTHR(
        cmsContext ContextID
    )

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
    )

    void cmsSetLogErrorHandlerTHR(
        cmsContext ContextID,
        cmsLogErrorHandlerFunction Fn
    )

    const cmsCIEXYZ* cmsD50_XYZ()

    const cmsCIExyY* cmsD50_xyY()

    void cmsXYZ2xyY(
        cmsCIExyY* Dest,
        const cmsCIEXYZ* Source
    )

    void cmsxyY2XYZ(
        cmsCIEXYZ* Dest,
        const cmsCIExyY* Source
    )

    void cmsXYZ2Lab(
        const cmsCIEXYZ* WhitePoint,
        cmsCIELab* Lab,
        const cmsCIEXYZ* xyz
    )

    void cmsLab2XYZ(
        const cmsCIEXYZ* WhitePoint,
        cmsCIEXYZ* xyz,
        const cmsCIELab* Lab
    )

    void cmsLab2LCh(
        cmsCIELCh*LCh,
        const cmsCIELab* Lab
    )

    void cmsLCh2Lab(
        cmsCIELab* Lab,
        const cmsCIELCh* LCh
    )

    void cmsLabEncoded2Float(
        cmsCIELab* Lab,
        const cmsUInt16Number[3] wLab
    )

    void cmsLabEncoded2FloatV2(
        cmsCIELab* Lab,
        const cmsUInt16Number[3] wLab
    )

    void cmsFloat2LabEncoded(
        cmsUInt16Number[3] wLab,
        const cmsCIELab* Lab
    )

    void cmsFloat2LabEncodedV2(
        cmsUInt16Number[3] wLab,
        const cmsCIELab* Lab
    )

    void cmsXYZEncoded2Float(
        cmsCIEXYZ* fxyz,
        const cmsUInt16Number[3] XYZ
    )

    void cmsFloat2XYZEncoded(
        cmsUInt16Number[3] XYZ,
        const cmsCIEXYZ* fXYZ
    )

    cmsFloat64Number cmsDeltaE(
        const cmsCIELab* Lab1,
        const cmsCIELab* Lab2
    )

    cmsFloat64Number cmsCIE94DeltaE(
        const cmsCIELab* Lab1,
        const cmsCIELab* Lab2
    )

    cmsFloat64Number cmsBFDdeltaE(
        const cmsCIELab* Lab1,
        const cmsCIELab* Lab2
    )

    cmsFloat64Number cmsCMCdeltaE(
        const cmsCIELab* Lab1,
        const cmsCIELab* Lab2,
        cmsFloat64Number l,
        cmsFloat64Number c
    )

    cmsFloat64Number cmsCIE2000DeltaE(
        const cmsCIELab* Lab1,
        const cmsCIELab* Lab2,
        cmsFloat64Number Kl,
        cmsFloat64Number Kc,
        cmsFloat64Number Kh
    )

    cmsBool cmsWhitePointFromTemp(
        cmsCIExyY* WhitePoint,
        cmsFloat64Number TempK
    )

    cmsBool cmsTempFromWhitePoint(
        cmsFloat64Number* TempK,
        const cmsCIExyY* WhitePoint
    )

    cmsBool cmsAdaptToIlluminant(
        cmsCIEXYZ* Result,
        const cmsCIEXYZ* SourceWhitePt,
        const cmsCIEXYZ* Illuminant,
        const cmsCIEXYZ* Value
    )

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
    )

    void cmsCIECAM02Done(
        cmsHANDLE hModel
    )

    void cmsCIECAM02Forward(
        cmsHANDLE hModel,
        const cmsCIEXYZ* pIn,
        cmsJCh* pOut
    )

    void cmsCIECAM02Reverse(
        cmsHANDLE hModel,
        const cmsJCh* pIn,
        cmsCIEXYZ* pOut
    )

    ctypedef struct cmsCurveSegment:
        cmsFloat32Number x0
        cmsFloat32Number x1
        cmsInt32Number Type
        cmsFloat64Number[10] Params
        cmsUInt32Number nGridPoints
        cmsFloat32Number* SampledPoints

    ctypedef struct cmsToneCurve:
        pass

    cmsToneCurve* cmsBuildSegmentedToneCurve(
        cmsContext ContextID,
        cmsUInt32Number nSegments,
        const cmsCurveSegment[] Segments
    )

    cmsToneCurve* cmsBuildParametricToneCurve(
        cmsContext ContextID,
        cmsInt32Number Type,
        const cmsFloat64Number[] Params
    )

    cmsToneCurve* cmsBuildGamma(
        cmsContext ContextID,
        cmsFloat64Number Gamma
    )

    cmsToneCurve* cmsBuildTabulatedToneCurve16(
        cmsContext ContextID,
        cmsUInt32Number nEntries,
        const cmsUInt16Number[] values
    )

    cmsToneCurve* cmsBuildTabulatedToneCurveFloat(
        cmsContext ContextID,
        cmsUInt32Number nEntries,
        const cmsFloat32Number[] values
    )

    void cmsFreeToneCurve(
        cmsToneCurve* Curve
    )

    void cmsFreeToneCurveTriple(
        (cmsToneCurve*)[3] Curve
    )

    cmsToneCurve* cmsDupToneCurve(
        const cmsToneCurve* Src
    )

    cmsToneCurve* cmsReverseToneCurve(
        const cmsToneCurve* InGamma
    )

    cmsToneCurve* cmsReverseToneCurveEx(
        cmsUInt32Number nResultSamples,
        const cmsToneCurve* InGamma
    )

    cmsToneCurve* cmsJoinToneCurve(
        cmsContext ContextID,
        const cmsToneCurve* X,
        const cmsToneCurve* Y,
        cmsUInt32Number nPoints
    )

    cmsBool cmsSmoothToneCurve(
        cmsToneCurve* Tab,
        cmsFloat64Number lambda_
    )

    cmsFloat32Number cmsEvalToneCurveFloat(
        const cmsToneCurve* Curve,
        cmsFloat32Number v
    )

    cmsUInt16Number cmsEvalToneCurve16(
        const cmsToneCurve* Curve,
        cmsUInt16Number v
    )

    cmsBool cmsIsToneCurveMultisegment(
        const cmsToneCurve* InGamma
    )

    cmsBool cmsIsToneCurveLinear(
        const cmsToneCurve* Curve
    )

    cmsBool cmsIsToneCurveMonotonic(
        const cmsToneCurve* t
    )

    cmsBool cmsIsToneCurveDescending(
        const cmsToneCurve* t
    )

    cmsInt32Number cmsGetToneCurveParametricType(
        const cmsToneCurve* t
    )

    cmsFloat64Number cmsEstimateGamma(
        const cmsToneCurve* t,
        cmsFloat64Number Precision
    )

    const cmsCurveSegment* cmsGetToneCurveSegment(
        cmsInt32Number n,
        const cmsToneCurve* t
    )

    cmsUInt32Number cmsGetToneCurveEstimatedTableEntries(
        const cmsToneCurve* t
    )

    const cmsUInt16Number* cmsGetToneCurveEstimatedTable(
        const cmsToneCurve* t
    )

    ctypedef struct cmsPipeline:
        pass

    ctypedef struct cmsStage:
        pass

    cmsPipeline* cmsPipelineAlloc(
        cmsContext ContextID,
        cmsUInt32Number InputChannels,
        cmsUInt32Number OutputChannels
    )

    void cmsPipelineFree(
        cmsPipeline* lut
    )

    cmsPipeline* cmsPipelineDup(
        const cmsPipeline* Orig
    )

    cmsContext cmsGetPipelineContextID(
        const cmsPipeline* lut
    )

    cmsUInt32Number cmsPipelineInputChannels(
        const cmsPipeline* lut
    )

    cmsUInt32Number cmsPipelineOutputChannels(
        const cmsPipeline* lut
    )

    cmsUInt32Number cmsPipelineStageCount(
        const cmsPipeline* lut
    )

    cmsStage* cmsPipelineGetPtrToFirstStage(
        const cmsPipeline* lut
    )

    cmsStage* cmsPipelineGetPtrToLastStage(
        const cmsPipeline* lut
    )

    void cmsPipelineEval16(
        const cmsUInt16Number[] In,
        cmsUInt16Number[] Out,
        const cmsPipeline* lut
    )

    void cmsPipelineEvalFloat(
        const cmsFloat32Number[] In,
        cmsFloat32Number[] Out,
        const cmsPipeline* lut
    )

    cmsBool cmsPipelineEvalReverseFloat(
        cmsFloat32Number[] Target,
        cmsFloat32Number[] Result,
        cmsFloat32Number[] Hint,
        const cmsPipeline* lut
    )

    cmsBool cmsPipelineCat(
        cmsPipeline* l1,
        const cmsPipeline* l2
    )

    cmsBool cmsPipelineSetSaveAs8bitsFlag(
        cmsPipeline* lut,
        cmsBool On
    )

    ctypedef enum cmsStageLoc:
        cmsAT_BEGIN
        cmsAT_END

    cmsBool cmsPipelineInsertStage(
        cmsPipeline* lut,
        cmsStageLoc loc,
        cmsStage* mpe
    )

    void cmsPipelineUnlinkStage(
        cmsPipeline* lut,
        cmsStageLoc loc,
        cmsStage** mpe
    )

    cmsBool cmsPipelineCheckAndRetreiveStages(
        const cmsPipeline* Lut,
        cmsUInt32Number n,
        ...
    )

    cmsStage* cmsStageAllocIdentity(
        cmsContext ContextID,
        cmsUInt32Number nChannels
    )

    cmsStage* cmsStageAllocToneCurves(
        cmsContext ContextID,
        cmsUInt32Number nChannels,
        cmsToneCurve* const Curves[]
    )

    cmsStage* cmsStageAllocMatrix(
        cmsContext ContextID,
        cmsUInt32Number Rows,
        cmsUInt32Number Cols,
        const cmsFloat64Number* Matrix,
        const cmsFloat64Number* Offset
    )

    cmsStage* cmsStageAllocCLut16bit(
        cmsContext ContextID,
        cmsUInt32Number nGridPoints,
        cmsUInt32Number inputChan,
        cmsUInt32Number outputChan,
        const cmsUInt16Number* Table
    )

    cmsStage* cmsStageAllocCLutFloat(
        cmsContext ContextID,
        cmsUInt32Number nGridPoints,
        cmsUInt32Number inputChan,
        cmsUInt32Number outputChan,
        const cmsFloat32Number* Table
    )

    cmsStage* cmsStageAllocCLut16bitGranular(
        cmsContext ContextID,
        const cmsUInt32Number[] clutPoints,
        cmsUInt32Number inputChan,
        cmsUInt32Number outputChan,
        const cmsUInt16Number* Table
    )

    cmsStage* cmsStageAllocCLutFloatGranular(
        cmsContext ContextID,
        const cmsUInt32Number[] clutPoints,
        cmsUInt32Number inputChan,
        cmsUInt32Number outputChan,
        const cmsFloat32Number* Table
    )

    cmsStage* cmsStageDup(
        cmsStage* mpe
    )

    void cmsStageFree(
        cmsStage* mpe
    )

    cmsStage* cmsStageNext(
        const cmsStage* mpe
    )

    cmsUInt32Number cmsStageInputChannels(
        const cmsStage* mpe
    )

    cmsUInt32Number cmsStageOutputChannels(
        const cmsStage* mpe
    )

    cmsStageSignature cmsStageType(
        const cmsStage* mpe
    )

    void* cmsStageData(
        const cmsStage* mpe
    )

    cmsContext cmsGetStageContextID(
        const cmsStage* mpe
    )

    ctypedef cmsInt32Number(* cmsSAMPLER16)(
        const cmsUInt16Number[] In,
        cmsUInt16Number[] Out,
        void* Cargo
    ) nogil

    ctypedef cmsInt32Number(* cmsSAMPLERFLOAT)(
        const cmsFloat32Number[] In,
        cmsFloat32Number[] Out,
        void* Cargo
    ) nogil

    int SAMPLER_INSPECT

    cmsBool cmsStageSampleCLut16bit(
        cmsStage* mpe,
        cmsSAMPLER16 Sampler,
        void* Cargo,
        cmsUInt32Number dwFlags
    )

    cmsBool cmsStageSampleCLutFloat(
        cmsStage* mpe,
        cmsSAMPLERFLOAT Sampler,
        void* Cargo,
        cmsUInt32Number dwFlags
    )

    cmsBool cmsSliceSpace16(
        cmsUInt32Number nInputs,
        const cmsUInt32Number[] clutPoints,
        cmsSAMPLER16 Sampler,
        void* Cargo
    )

    cmsBool cmsSliceSpaceFloat(
        cmsUInt32Number nInputs,
        const cmsUInt32Number[] clutPoints,
        cmsSAMPLERFLOAT Sampler,
        void* Cargo
    )

    ctypedef struct cmsMLU:
        pass

    char* cmsNoLanguage
    char* cmsNoCountry
    char* cmsV2Unicode

    cmsMLU* cmsMLUalloc(
        cmsContext ContextID,
        cmsUInt32Number nItems
    )

    void cmsMLUfree(
        cmsMLU* mlu
    )

    cmsMLU* cmsMLUdup(
        const cmsMLU* mlu
    )

    cmsBool cmsMLUsetASCII(
        cmsMLU* mlu,
        const char[3] LanguageCode,
        const char[3] CountryCode,
        const char* ASCIIString
    )

    cmsBool cmsMLUsetWide(
        cmsMLU* mlu,
        const char[3] LanguageCode,
        const char[3] CountryCode,
        const wchar_t* WideString
    )

    cmsBool cmsMLUsetUTF8(
        cmsMLU* mlu,
        const char[3] LanguageCode,
        const char[3] CountryCode,
        const char* UTF8String
    )

    cmsUInt32Number cmsMLUgetASCII(
        const cmsMLU* mlu,
        const char[3] LanguageCode,
        const char[3] CountryCode,
        char* Buffer,
        cmsUInt32Number BufferSize
    )

    cmsUInt32Number cmsMLUgetWide(
        const cmsMLU* mlu,
        const char[3] LanguageCode,
        const char[3] CountryCode,
        wchar_t* Buffer,
        cmsUInt32Number BufferSize
    )

    cmsUInt32Number cmsMLUgetUTF8(
        const cmsMLU* mlu,
        const char[3] LanguageCode,
        const char[3] CountryCode,
        char* Buffer,
        cmsUInt32Number BufferSize
    )

    cmsBool cmsMLUgetTranslation(
        const cmsMLU* mlu,
        const char[3] LanguageCode,
        const char[3] CountryCode,
        char[3] ObtainedLanguage,
        char[3] ObtainedCountry
    )

    cmsUInt32Number cmsMLUtranslationsCount(
        const cmsMLU* mlu
    )

    cmsBool cmsMLUtranslationsCodes(
        const cmsMLU* mlu,
        cmsUInt32Number idx,
        char[3] LanguageCode,
        char[3] CountryCode
    )

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
        cmsScreeningChannel[16] Channels  # [cmsMAXCHANNELS]

    ctypedef struct cmsNAMEDCOLORLIST:
        pass

    cmsNAMEDCOLORLIST* cmsAllocNamedColorList(
        cmsContext ContextID,
        cmsUInt32Number n,
        cmsUInt32Number ColorantCount,
        const char* Prefix,
        const char* Suffix
    )

    void cmsFreeNamedColorList(
        cmsNAMEDCOLORLIST* v
    )

    cmsNAMEDCOLORLIST* cmsDupNamedColorList(
        const cmsNAMEDCOLORLIST* v
    )

    cmsBool cmsAppendNamedColor(
        cmsNAMEDCOLORLIST* v,
        const char* Name,
        cmsUInt16Number[3] PCS,
        cmsUInt16Number[] Colorant
    )

    cmsUInt32Number cmsNamedColorCount(
        const cmsNAMEDCOLORLIST* v
    )

    cmsInt32Number cmsNamedColorIndex(
        const cmsNAMEDCOLORLIST* v,
        const char* Name
    )

    cmsBool cmsNamedColorInfo(
        const cmsNAMEDCOLORLIST* NamedColorList,
        cmsUInt32Number nColor,
        char* Name,
        char* Prefix,
        char* Suffix,
        cmsUInt16Number* PCS,
        cmsUInt16Number* Colorant
    )

    cmsNAMEDCOLORLIST* cmsGetNamedColorList(
        cmsHTRANSFORM xform
    )

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
    )

    cmsSEQ* cmsDupProfileSequenceDescription(
        const cmsSEQ* pseq
    )

    void cmsFreeProfileSequenceDescription(
        cmsSEQ* pseq
    )

    ctypedef struct cmsDICTentry:
        cmsDICTentry* Next
        cmsMLU *DisplayName
        cmsMLU *DisplayValue
        wchar_t* Name
        wchar_t* Value

    cmsHANDLE cmsDictAlloc(
        cmsContext ContextID
    )

    void cmsDictFree(
        cmsHANDLE hDict
    )

    cmsHANDLE cmsDictDup(
        cmsHANDLE hDict
    )

    cmsBool cmsDictAddEntry(
        cmsHANDLE hDict,
        const wchar_t* Name,
        const wchar_t* Value,
        const cmsMLU *DisplayName,
        const cmsMLU *DisplayValue
    )

    const cmsDICTentry* cmsDictGetEntryList(
        cmsHANDLE hDict
    )

    const cmsDICTentry* cmsDictNextEntry(
        const cmsDICTentry* e
    )

    cmsHPROFILE cmsCreateProfilePlaceholder(
        cmsContext ContextID
    )

    cmsContext cmsGetProfileContextID(
        cmsHPROFILE hProfile
    )

    cmsInt32Number cmsGetTagCount(
        cmsHPROFILE hProfile
    )

    cmsTagSignature cmsGetTagSignature(
        cmsHPROFILE hProfile,
        cmsUInt32Number n
    )

    cmsBool cmsIsTag(
        cmsHPROFILE hProfile,
        cmsTagSignature sig
    )

    void* cmsReadTag(
        cmsHPROFILE hProfile,
        cmsTagSignature sig
    )

    cmsBool cmsWriteTag(
        cmsHPROFILE hProfile,
        cmsTagSignature sig,
        const void* data
    )

    cmsBool cmsLinkTag(
        cmsHPROFILE hProfile,
        cmsTagSignature sig,
        cmsTagSignature dest
    )

    cmsTagSignature cmsTagLinkedTo(
        cmsHPROFILE hProfile,
        cmsTagSignature sig
    )

    cmsUInt32Number cmsReadRawTag(
        cmsHPROFILE hProfile,
        cmsTagSignature sig,
        void* Buffer,
        cmsUInt32Number BufferSize
    )

    cmsBool cmsWriteRawTag(
        cmsHPROFILE hProfile,
        cmsTagSignature sig,
        const void* data,
        cmsUInt32Number Size
    )

    int cmsEmbeddedProfileFalse
    int cmsEmbeddedProfileTrue
    int cmsUseAnywhere
    int cmsUseWithEmbeddedDataOnly

    cmsUInt32Number cmsGetHeaderFlags(
        cmsHPROFILE hProfile
    )

    void cmsGetHeaderAttributes(
        cmsHPROFILE hProfile,
        cmsUInt64Number* Flags
    )

    void cmsGetHeaderProfileID(
        cmsHPROFILE hProfile,
        cmsUInt8Number* ProfileID
    )

    cmsBool cmsGetHeaderCreationDateTime(
        cmsHPROFILE hProfile,
        tm *Dest
    )

    cmsUInt32Number cmsGetHeaderRenderingIntent(
        cmsHPROFILE hProfile
    )

    void cmsSetHeaderFlags(
        cmsHPROFILE hProfile,
        cmsUInt32Number Flags
    )

    cmsUInt32Number cmsGetHeaderManufacturer(
        cmsHPROFILE hProfile
    )

    void cmsSetHeaderManufacturer(
        cmsHPROFILE hProfile,
        cmsUInt32Number manufacturer
    )

    cmsUInt32Number cmsGetHeaderCreator(
        cmsHPROFILE hProfile
    )

    cmsUInt32Number cmsGetHeaderModel(
        cmsHPROFILE hProfile
    )

    void cmsSetHeaderModel(
        cmsHPROFILE hProfile,
        cmsUInt32Number model
    )

    void cmsSetHeaderAttributes(
        cmsHPROFILE hProfile,
        cmsUInt64Number Flags
    )

    void cmsSetHeaderProfileID(
        cmsHPROFILE hProfile,
        cmsUInt8Number* ProfileID
    )

    void cmsSetHeaderRenderingIntent(
        cmsHPROFILE hProfile,
        cmsUInt32Number RenderingIntent
    )

    cmsColorSpaceSignature cmsGetPCS(
        cmsHPROFILE hProfile
    )

    void cmsSetPCS(
        cmsHPROFILE hProfile,
        cmsColorSpaceSignature pcs
    )

    cmsColorSpaceSignature cmsGetColorSpace(
        cmsHPROFILE hProfile
    )

    void cmsSetColorSpace(
        cmsHPROFILE hProfile,
        cmsColorSpaceSignature sig
    )

    cmsProfileClassSignature cmsGetDeviceClass(
        cmsHPROFILE hProfile
    )

    void cmsSetDeviceClass(
        cmsHPROFILE hProfile,
        cmsProfileClassSignature sig
    )

    void cmsSetProfileVersion(
        cmsHPROFILE hProfile,
        cmsFloat64Number Version
    )

    cmsFloat64Number cmsGetProfileVersion(
        cmsHPROFILE hProfile
    )

    cmsUInt32Number cmsGetEncodedICCversion(
        cmsHPROFILE hProfile
    )

    void cmsSetEncodedICCversion(
        cmsHPROFILE hProfile,
        cmsUInt32Number Version
    )

    int LCMS_USED_AS_INPUT
    int LCMS_USED_AS_OUTPUT
    int LCMS_USED_AS_PROOF

    cmsBool cmsIsIntentSupported(
        cmsHPROFILE hProfile,
        cmsUInt32Number Intent,
        cmsUInt32Number UsedDirection
    )

    cmsBool cmsIsMatrixShaper(
        cmsHPROFILE hProfile
    )

    cmsBool cmsIsCLUT(
        cmsHPROFILE hProfile,
        cmsUInt32Number Intent,
        cmsUInt32Number UsedDirection
    )

    cmsColorSpaceSignature _cmsICCcolorSpace(
        int OurNotation
    )

    int _cmsLCMScolorSpace(
        cmsColorSpaceSignature ProfileSpace
    )

    cmsUInt32Number cmsChannelsOf(
        cmsColorSpaceSignature ColorSpace
    )

    cmsInt32Number cmsChannelsOfColorSpace(
        cmsColorSpaceSignature ColorSpace
    )

    cmsUInt32Number cmsFormatterForColorspaceOfProfile(
        cmsHPROFILE hProfile,
        cmsUInt32Number nBytes,
        cmsBool lIsFloat
    )

    cmsUInt32Number cmsFormatterForPCSOfProfile(
        cmsHPROFILE hProfile,
        cmsUInt32Number nBytes,
        cmsBool lIsFloat
    )

    ctypedef enum cmsInfoType:
        cmsInfoDescription
        cmsInfoManufacturer
        cmsInfoModel
        cmsInfoCopyright

    cmsUInt32Number cmsGetProfileInfo(
        cmsHPROFILE hProfile,
        cmsInfoType Info,
        const char[3] LanguageCode,
        const char[3] CountryCode,
        wchar_t* Buffer,
        cmsUInt32Number BufferSize
    )

    cmsUInt32Number cmsGetProfileInfoASCII(
        cmsHPROFILE hProfile,
        cmsInfoType Info,
        const char[3] LanguageCode,
        const char[3] CountryCode,
        char* Buffer,
        cmsUInt32Number BufferSize
    )

    cmsUInt32Number cmsGetProfileInfoUTF8(
        cmsHPROFILE hProfile,
        cmsInfoType Info,
        const char[3] LanguageCode,
        const char[3] CountryCode,
        char* Buffer,
        cmsUInt32Number BufferSize
    )

    ctypedef struct cmsIOHANDLER:
        pass

    cmsIOHANDLER* cmsOpenIOhandlerFromFile(
        cmsContext ContextID,
        const char* FileName,
        const char* AccessMode
    )

    cmsIOHANDLER* cmsOpenIOhandlerFromStream(
        cmsContext ContextID,
        FILE* Stream
    )

    cmsIOHANDLER* cmsOpenIOhandlerFromMem(
        cmsContext ContextID,
        void*Buffer,
        cmsUInt32Number size,
        const char* AccessMode
    )

    cmsIOHANDLER* cmsOpenIOhandlerFromNULL(
        cmsContext ContextID
    )

    cmsIOHANDLER* cmsGetProfileIOhandler(
        cmsHPROFILE hProfile
    )

    cmsBool cmsCloseIOhandler(
        cmsIOHANDLER* io
    )

    cmsBool cmsMD5computeID(
        cmsHPROFILE hProfile
    )

    cmsHPROFILE cmsOpenProfileFromFile(
        const char *ICCProfile,
        const char *sAccess
    )

    cmsHPROFILE cmsOpenProfileFromFileTHR(
        cmsContext ContextID,
        const char *ICCProfile,
        const char *sAccess
    )

    cmsHPROFILE cmsOpenProfileFromStream(
        FILE* ICCProfile,
        const char* sAccess
    )

    cmsHPROFILE cmsOpenProfileFromStreamTHR(
        cmsContext ContextID,
        FILE* ICCProfile,
        const char* sAccess
    )

    cmsHPROFILE cmsOpenProfileFromMem(
        const void* MemPtr,
        cmsUInt32Number dwSize
    )

    cmsHPROFILE cmsOpenProfileFromMemTHR(
        cmsContext ContextID,
        const void* MemPtr,
        cmsUInt32Number dwSize
    )

    cmsHPROFILE cmsOpenProfileFromIOhandlerTHR(
        cmsContext ContextID,
        cmsIOHANDLER* io
    )

    cmsHPROFILE cmsOpenProfileFromIOhandler2THR(
        cmsContext ContextID,
        cmsIOHANDLER* io,
        cmsBool write
    )

    cmsBool cmsCloseProfile(
        cmsHPROFILE hProfile
    )

    cmsBool cmsSaveProfileToFile(
        cmsHPROFILE hProfile,
        const char* FileName
    )

    cmsBool cmsSaveProfileToStream(
        cmsHPROFILE hProfile,
        FILE* Stream
    )

    cmsBool cmsSaveProfileToMem(
        cmsHPROFILE hProfile,
        void*MemPtr,
        cmsUInt32Number* BytesNeeded
    )

    cmsUInt32Number cmsSaveProfileToIOhandler(
        cmsHPROFILE hProfile,
        cmsIOHANDLER* io
    )

    cmsHPROFILE cmsCreateRGBProfileTHR(
        cmsContext ContextID,
        const cmsCIExyY* WhitePoint,
        const cmsCIExyYTRIPLE* Primaries,
        (cmsToneCurve* const)[3] TransferFunction
    )

    cmsHPROFILE cmsCreateRGBProfile(
        const cmsCIExyY* WhitePoint,
        const cmsCIExyYTRIPLE* Primaries,
        (cmsToneCurve* const)[3] TransferFunction
    )

    cmsHPROFILE cmsCreateGrayProfileTHR(
        cmsContext ContextID,
        const cmsCIExyY* WhitePoint,
        const cmsToneCurve* TransferFunction
    )

    cmsHPROFILE cmsCreateGrayProfile(
        const cmsCIExyY* WhitePoint,
        const cmsToneCurve* TransferFunction
    )

    cmsHPROFILE cmsCreateLinearizationDeviceLinkTHR(
        cmsContext ContextID,
        cmsColorSpaceSignature ColorSpace,
        (cmsToneCurve*)[] TransferFunctions
    )

    cmsHPROFILE cmsCreateLinearizationDeviceLink(
        cmsColorSpaceSignature ColorSpace,
        (cmsToneCurve*)[] TransferFunctions
    )

    cmsHPROFILE cmsCreateInkLimitingDeviceLinkTHR(
        cmsContext ContextID,
        cmsColorSpaceSignature ColorSpace,
        cmsFloat64Number Limit
    )

    cmsHPROFILE cmsCreateInkLimitingDeviceLink(
        cmsColorSpaceSignature ColorSpace,
        cmsFloat64Number Limit
    )

    cmsHPROFILE cmsCreateDeviceLinkFromCubeFile(
        const char* cFileName
    )

    cmsHPROFILE cmsCreateDeviceLinkFromCubeFileTHR(
        cmsContext ContextID,
        const char* cFileName
    )

    cmsHPROFILE cmsCreateLab2ProfileTHR(
        cmsContext ContextID,
        const cmsCIExyY* WhitePoint
    )

    cmsHPROFILE cmsCreateLab2Profile(
        const cmsCIExyY* WhitePoint
    )

    cmsHPROFILE cmsCreateLab4ProfileTHR(
        cmsContext ContextID,
        const cmsCIExyY* WhitePoint
    )

    cmsHPROFILE cmsCreateLab4Profile(
        const cmsCIExyY* WhitePoint
    )

    cmsHPROFILE cmsCreateXYZProfileTHR(
        cmsContext ContextID
    )

    cmsHPROFILE cmsCreateXYZProfile(
    )

    cmsHPROFILE cmsCreate_sRGBProfileTHR(
        cmsContext ContextID
    )

    cmsHPROFILE cmsCreate_sRGBProfile(
    )

    cmsHPROFILE cmsCreate_OkLabProfile(
        cmsContext ctx
    )

    cmsHPROFILE cmsCreateBCHSWabstractProfileTHR(
        cmsContext ContextID,
        cmsUInt32Number nLUTPoints,
        cmsFloat64Number Bright,
        cmsFloat64Number Contrast,
        cmsFloat64Number Hue,
        cmsFloat64Number Saturation,
        cmsUInt32Number TempSrc,
        cmsUInt32Number TempDest
    )

    cmsHPROFILE cmsCreateBCHSWabstractProfile(
        cmsUInt32Number nLUTPoints,
        cmsFloat64Number Bright,
        cmsFloat64Number Contrast,
        cmsFloat64Number Hue,
        cmsFloat64Number Saturation,
        cmsUInt32Number TempSrc,
        cmsUInt32Number TempDest
    )

    cmsHPROFILE cmsCreateNULLProfileTHR(
        cmsContext ContextID
    )

    cmsHPROFILE cmsCreateNULLProfile(
    )

    cmsHPROFILE cmsTransform2DeviceLink(
        cmsHTRANSFORM hTransform,
        cmsFloat64Number Version,
        cmsUInt32Number dwFlags
    )

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
    )

    cmsUInt32Number cmsGetSupportedIntentsTHR(
        cmsContext ContextID,
        cmsUInt32Number nMax,
        cmsUInt32Number* Codes,
        char** Descriptions
    )

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
    )

    cmsHTRANSFORM cmsCreateTransform(
        cmsHPROFILE Input,
        cmsUInt32Number InputFormat,
        cmsHPROFILE Output,
        cmsUInt32Number OutputFormat,
        cmsUInt32Number Intent,
        cmsUInt32Number dwFlags
    )

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
    )

    cmsHTRANSFORM cmsCreateProofingTransform(
        cmsHPROFILE Input,
        cmsUInt32Number InputFormat,
        cmsHPROFILE Output,
        cmsUInt32Number OutputFormat,
        cmsHPROFILE Proofing,
        cmsUInt32Number Intent,
        cmsUInt32Number ProofingIntent,
        cmsUInt32Number dwFlags
    )

    cmsHTRANSFORM cmsCreateMultiprofileTransformTHR(
        cmsContext ContextID,
        cmsHPROFILE[] hProfiles,
        cmsUInt32Number nProfiles,
        cmsUInt32Number InputFormat,
        cmsUInt32Number OutputFormat,
        cmsUInt32Number Intent,
        cmsUInt32Number dwFlags
    )

    cmsHTRANSFORM cmsCreateMultiprofileTransform(
        cmsHPROFILE[] hProfiles,
        cmsUInt32Number nProfiles,
        cmsUInt32Number InputFormat,
        cmsUInt32Number OutputFormat,
        cmsUInt32Number Intent,
        cmsUInt32Number dwFlags
    )

    cmsHTRANSFORM cmsCreateExtendedTransform(
        cmsContext ContextID,
        cmsUInt32Number nProfiles,
        cmsHPROFILE[] hProfiles,
        cmsBool[] BPC,
        cmsUInt32Number[] Intents,
        cmsFloat64Number[] AdaptationStates,
        cmsHPROFILE hGamutProfile,
        cmsUInt32Number nGamutPCSposition,
        cmsUInt32Number InputFormat,
        cmsUInt32Number OutputFormat,
        cmsUInt32Number dwFlags
    )

    void cmsDeleteTransform(
        cmsHTRANSFORM hTransform
    )

    void cmsDoTransform(
        cmsHTRANSFORM Transform,
        const void* InputBuffer,
        void* OutputBuffer,
        cmsUInt32Number Size
    )

    void cmsDoTransformStride(
        cmsHTRANSFORM Transform,
        const void* InputBuffer,
        void* OutputBuffer,
        cmsUInt32Number Size,
        cmsUInt32Number Stride
    )

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
    )

    void cmsSetAlarmCodes(
        const cmsUInt16Number[] NewAlarm
    )

    void cmsGetAlarmCodes(
        cmsUInt16Number[] NewAlarm
    )

    void cmsSetAlarmCodesTHR(
        cmsContext ContextID,
        const cmsUInt16Number[] AlarmCodes
    )

    void cmsGetAlarmCodesTHR(
        cmsContext ContextID,
        cmsUInt16Number[] AlarmCodes
    )

    cmsFloat64Number cmsSetAdaptationState(
        cmsFloat64Number d
    )

    cmsFloat64Number cmsSetAdaptationStateTHR(
        cmsContext ContextID,
        cmsFloat64Number d
    )

    cmsContext cmsGetTransformContextID(
        cmsHTRANSFORM hTransform
    )

    cmsUInt32Number cmsGetTransformInputFormat(
        cmsHTRANSFORM hTransform
    )

    cmsUInt32Number cmsGetTransformOutputFormat(
        cmsHTRANSFORM hTransform
    )

    cmsBool cmsChangeBuffersFormat(
        cmsHTRANSFORM hTransform,
        cmsUInt32Number InputFormat,
        cmsUInt32Number OutputFormat
    )

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
    )

    cmsUInt32Number cmsGetPostScriptCSA(
        cmsContext ContextID,
        cmsHPROFILE hProfile,
        cmsUInt32Number Intent,
        cmsUInt32Number dwFlags,
        void* Buffer,
        cmsUInt32Number dwBufferLen
    )

    cmsUInt32Number cmsGetPostScriptCRD(
        cmsContext ContextID,
        cmsHPROFILE hProfile,
        cmsUInt32Number Intent,
        cmsUInt32Number dwFlags,
        void* Buffer,
        cmsUInt32Number dwBufferLen
    )

    cmsHANDLE cmsIT8Alloc(
        cmsContext ContextID
    )

    void cmsIT8Free(
        cmsHANDLE hIT8
    )

    cmsUInt32Number cmsIT8TableCount(
        cmsHANDLE hIT8
    )

    cmsInt32Number cmsIT8SetTable(
        cmsHANDLE hIT8,
        cmsUInt32Number nTable
    )

    cmsHANDLE cmsIT8LoadFromFile(
        cmsContext ContextID,
        const char* cFileName
    )

    cmsHANDLE cmsIT8LoadFromMem(
        cmsContext ContextID,
        const void*Ptr,
        cmsUInt32Number len
    )

    cmsBool cmsIT8SaveToFile(
        cmsHANDLE hIT8,
        const char* cFileName
    )

    cmsBool cmsIT8SaveToMem(
        cmsHANDLE hIT8,
        void*MemPtr,
        cmsUInt32Number* BytesNeeded
    )

    const char* cmsIT8GetSheetType(
        cmsHANDLE hIT8
    )

    cmsBool cmsIT8SetSheetType(
        cmsHANDLE hIT8,
        const char* Type
    )

    cmsBool cmsIT8SetComment(
        cmsHANDLE hIT8,
        const char* cComment
    )

    cmsBool cmsIT8SetPropertyStr(
        cmsHANDLE hIT8,
        const char* cProp,
        const char *Str
    )

    cmsBool cmsIT8SetPropertyDbl(
        cmsHANDLE hIT8,
        const char* cProp,
        cmsFloat64Number Val
    )

    cmsBool cmsIT8SetPropertyHex(
        cmsHANDLE hIT8,
        const char* cProp,
        cmsUInt32Number Val
    )

    cmsBool cmsIT8SetPropertyMulti(
        cmsHANDLE hIT8,
        const char* Key,
        const char* SubKey,
        const char *Buffer
    )

    cmsBool cmsIT8SetPropertyUncooked(
        cmsHANDLE hIT8,
        const char* Key,
        const char* Buffer
    )

    const char* cmsIT8GetProperty(
        cmsHANDLE hIT8,
        const char* cProp
    )

    cmsFloat64Number cmsIT8GetPropertyDbl(
        cmsHANDLE hIT8,
        const char* cProp
    )

    const char* cmsIT8GetPropertyMulti(
        cmsHANDLE hIT8,
        const char* Key,
        const char *SubKey
    )

    cmsUInt32Number cmsIT8EnumProperties(
        cmsHANDLE hIT8,
        char ***PropertyNames
    )

    cmsUInt32Number cmsIT8EnumPropertyMulti(
        cmsHANDLE hIT8,
        const char* cProp,
        const char ***SubpropertyNames
    )

    const char* cmsIT8GetDataRowCol(
        cmsHANDLE hIT8,
        int row,
        int col
    )

    cmsFloat64Number cmsIT8GetDataRowColDbl(
        cmsHANDLE hIT8,
        int row,
        int col
    )

    cmsBool cmsIT8SetDataRowCol(
        cmsHANDLE hIT8,
        int row,
        int col,
        const char* Val
    )

    cmsBool cmsIT8SetDataRowColDbl(
        cmsHANDLE hIT8,
        int row,
        int col,
        cmsFloat64Number Val
    )

    const char* cmsIT8GetData(
        cmsHANDLE hIT8,
        const char* cPatch,
        const char* cSample
    )

    cmsFloat64Number cmsIT8GetDataDbl(
        cmsHANDLE hIT8,
        const char* cPatch,
        const char* cSample
    )

    cmsBool cmsIT8SetData(
        cmsHANDLE hIT8,
        const char* cPatch,
        const char* cSample,
        const char *Val
    )

    cmsBool cmsIT8SetDataDbl(
        cmsHANDLE hIT8,
        const char* cPatch,
        const char* cSample,
        cmsFloat64Number Val
    )

    int cmsIT8FindDataFormat(
        cmsHANDLE hIT8,
        const char* cSample
    )

    cmsBool cmsIT8SetDataFormat(
        cmsHANDLE hIT8,
        int n,
        const char *Sample
    )

    int cmsIT8EnumDataFormat(
        cmsHANDLE hIT8,
        char ***SampleNames
    )

    const char* cmsIT8GetPatchName(
        cmsHANDLE hIT8,
        int nPatch,
        char* buffer
    )

    int cmsIT8GetPatchByName(
        cmsHANDLE hIT8,
        const char *cPatch
    )

    int cmsIT8SetTableByLabel(
        cmsHANDLE hIT8,
        const char* cSet,
        const char* cField,
        const char* ExpectedType
    )

    cmsBool cmsIT8SetIndexColumn(
        cmsHANDLE hIT8,
        const char* cSample
    )

    void cmsIT8DefineDblFormat(
        cmsHANDLE hIT8,
        const char* Formatter
    )

    cmsHANDLE cmsGBDAlloc(
        cmsContext ContextID
    )

    void cmsGBDFree(
        cmsHANDLE hGBD
    )

    cmsBool cmsGDBAddPoint(
        cmsHANDLE hGBD,
        const cmsCIELab* Lab
    )

    cmsBool cmsGDBCompute(
        cmsHANDLE hGDB,
        cmsUInt32Number dwFlags
    )

    cmsBool cmsGDBCheckPoint(
        cmsHANDLE hGBD,
        const cmsCIELab* Lab
    )

    cmsBool cmsDetectBlackPoint(
        cmsCIEXYZ* BlackPoint,
        cmsHPROFILE hProfile,
        cmsUInt32Number Intent,
        cmsUInt32Number dwFlags
    )

    cmsBool cmsDetectDestinationBlackPoint(
        cmsCIEXYZ* BlackPoint,
        cmsHPROFILE hProfile,
        cmsUInt32Number Intent,
        cmsUInt32Number dwFlags
    )

    cmsFloat64Number cmsDetectTAC(
        cmsHPROFILE hProfile
    )

    cmsFloat64Number cmsDetectRGBProfileGamma(
        cmsHPROFILE hProfile,
        cmsFloat64Number threshold
    )

    cmsBool cmsDesaturateLab(
        cmsCIELab* Lab,
        double amax,
        double amin,
        double bmax,
        double bmin
    )
