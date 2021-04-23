# imagecodecs/libtiff.pxd
# cython: language_level = 3

# Cython declarations for the `libtiff 4.3.0` library.
# https://gitlab.com/libtiff/libtiff

from libc.stdio cimport FILE
from libc.stdint cimport (
    uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t
)

cdef extern from '<stdarg.h>':
    ctypedef struct va_list:
        pass


cdef extern from 'tiffio.h':

    char* TIFFLIB_VERSION_STR
    int TIFFLIB_VERSION

    int TIFF_VERSION_CLASSIC
    int TIFF_VERSION_BIG

    int TIFF_BIGENDIAN
    int TIFF_LITTLEENDIAN
    int MDI_LITTLEENDIAN
    int MDI_BIGENDIAN

    int TIFFPRINT_NONE
    int TIFFPRINT_STRIPS
    int TIFFPRINT_CURVES
    int TIFFPRINT_COLORMAP
    int TIFFPRINT_JPEGQTABLES
    int TIFFPRINT_JPEGACTABLES
    int TIFFPRINT_JPEGDCTABLES
    int CIELABTORGB_TABLE_RANGE
    int D65_X0
    int D65_Y0
    int D65_Z0
    int D50_X0
    int D50_Y0
    int D50_Z0
    int TIFF_ANY
    int TIFF_VARIABLE
    int TIFF_SPP
    int TIFF_VARIABLE2
    int FIELD_CUSTOM
    int U_NEU
    int V_NEU
    int UVSCALE

    # deprecated
    # ctypedef signed char int8
    # ctypedef unsigned char uint8
    # ctypedef signed short int16
    # ctypedef unsigned short uint16
    # ctypedef signed int int32
    # ctypedef unsigned int uint32
    # ctypedef signed long long int64
    # ctypedef unsigned long long uint64
    # ctypedef int uint16_vap

    ctypedef int16_t TIFF_INT16_T
    ctypedef int32_t TIFF_INT32_T
    ctypedef int64_t TIFF_INT64_T
    ctypedef int8_t TIFF_INT8_T
    ctypedef uint16_t TIFF_UINT16_T
    ctypedef uint32_t TIFF_UINT32_T
    ctypedef uint64_t TIFF_UINT64_T
    ctypedef uint8_t TIFF_UINT8_T
    ctypedef int64_t TIFF_SSIZE_T

    ctypedef int64_t tmsize_t
    ctypedef uint64_t toff_t
    ctypedef uint32_t ttag_t
    ctypedef uint16_t tdir_t
    ctypedef uint16_t tsample_t
    ctypedef uint32_t tstrile_t
    ctypedef tstrile_t tstrip_t
    ctypedef tstrile_t ttile_t
    ctypedef tmsize_t tsize_t
    ctypedef void* tdata_t
    ctypedef void* thandle_t
    ctypedef unsigned char TIFFRGBValue

    ctypedef struct TIFFHeaderCommon:
        uint16_t tiff_magic
        uint16_t tiff_version

    ctypedef struct TIFFHeaderClassic:
        uint16_t tiff_magic
        uint16_t tiff_version
        uint32_t tiff_diroff

    ctypedef struct TIFFHeaderBig:
        uint16_t tiff_magic
        uint16_t tiff_version
        uint16_t tiff_offsetsize
        uint16_t tiff_unused
        uint64_t tiff_diroff

    ctypedef enum TIFFDataType:
        TIFF_NOTYPE
        TIFF_BYTE
        TIFF_ASCII
        TIFF_SHORT
        TIFF_LONG
        TIFF_RATIONAL
        TIFF_SBYTE
        TIFF_UNDEFINED
        TIFF_SSHORT
        TIFF_SLONG
        TIFF_SRATIONAL
        TIFF_FLOAT
        TIFF_DOUBLE
        TIFF_IFD
        TIFF_LONG8
        TIFF_SLONG8
        TIFF_IFD8

    ctypedef struct TIFFDisplay:
        float d_mat[3][3]
        float d_YCR
        float d_YCG
        float d_YCB
        uint32_t d_Vrwr
        uint32_t d_Vrwg
        uint32_t d_Vrwb
        float d_Y0R
        float d_Y0G
        float d_Y0B
        float d_gammaR
        float d_gammaG
        float d_gammaB

    ctypedef struct TIFFYCbCrToRGB:
        TIFFRGBValue* clamptab
        int* Cr_r_tab
        int* Cb_b_tab
        int32_t* Cr_g_tab
        int32_t* Cb_g_tab
        int32_t* Y_tab

    ctypedef struct TIFFCIELabToRGB:
        int range
        float rstep
        float gstep
        float bstep
        float X0
        float Y0
        float Z0
        TIFFDisplay display
        float* Yr2r
        float* Yg2g
        float* Yb2b

    ctypedef struct TIFF:
        pass

    ctypedef struct TIFFRGBAImage:
        pass

    ctypedef void (*tileContigRoutine)(
        TIFFRGBAImage*,
        uint32_t*,
        uint32_t,
        uint32_t,
        uint32_t,
        uint32_t,
        int32_t,
        int32_t,
        unsigned char*
    ) nogil

    ctypedef void (*tileSeparateRoutine)(
        TIFFRGBAImage*,
        uint32_t*,
        uint32_t,
        uint32_t,
        uint32_t,
        uint32_t,
        int32_t,
        int32_t,
        unsigned char*,
        unsigned char*,
        unsigned char*,
        unsigned char*
    ) nogil

    ctypedef int (*TIFFRGBAImage_get_func)(
        TIFFRGBAImage*,
        uint32_t*,
        uint32_t,
        uint32_t
    ) nogil

    ctypedef void (*TIFFRGBAImage_put_any_func)(
        TIFFRGBAImage*
    ) nogil

    cdef union TIFFRGBAImage_put_union:
        TIFFRGBAImage_put_any_func any
        tileContigRoutine contig
        tileSeparateRoutine separate

    cdef struct TIFFRGBAImage:
        TIFF* tif
        int stoponerr
        int isContig
        int alpha
        uint32_t width
        uint32_t height
        uint16_t bitspersample
        uint16_t samplesperpixel
        uint16_t orientation
        uint16_t req_orientation
        uint16_t photometric
        uint16_t* redcmap
        uint16_t* greencmap
        uint16_t* bluecmap
        TIFFRGBAImage_get_func get
        TIFFRGBAImage_put_union put
        TIFFRGBValue* Map
        uint32_t** BWmap
        uint32_t** PALmap
        TIFFYCbCrToRGB* ycbcr
        TIFFCIELabToRGB* cielab
        uint8_t* UaToAa
        uint8_t* Bitdepth16To8
        int row_offset
        int col_offset

    ctypedef int (*TIFFInitMethod)(
        TIFF*,
        int
    ) nogil

    ctypedef struct TIFFCodec:
        char* name
        uint16_t scheme
        TIFFInitMethod init

    ctypedef void (*TIFFErrorHandler)(
        const char*,
        const char*,
        va_list
    ) nogil

    ctypedef void (*TIFFErrorHandlerExt)(
        thandle_t,
        char*,
        char*,
        va_list
    ) nogil

    ctypedef tmsize_t (*TIFFReadWriteProc)(
        thandle_t,
        void*,
        tmsize_t
    ) nogil

    ctypedef toff_t (*TIFFSeekProc)(
        thandle_t,
        toff_t,
        int
    ) nogil

    ctypedef int (*TIFFCloseProc)(
        thandle_t
    ) nogil

    ctypedef toff_t (*TIFFSizeProc)(
        thandle_t
    ) nogil

    ctypedef int (*TIFFMapFileProc)(
        thandle_t,
        void** base,
        toff_t* size
    ) nogil

    ctypedef void (*TIFFUnmapFileProc)(
        thandle_t,
        void* base,
        toff_t size
    ) nogil

    ctypedef void (*TIFFExtendProc)(
        TIFF*
    ) nogil

    int TIFFGetR(int) nogil
    int TIFFGetG(int) nogil
    int TIFFGetB(int) nogil
    int TIFFGetA(int) nogil

    char* TIFFGetVersion() nogil

    TIFFCodec* TIFFFindCODEC(
        uint16_t
    ) nogil

    TIFFCodec* TIFFRegisterCODEC(
        uint16_t,
        char*,
        TIFFInitMethod
    ) nogil

    void TIFFUnRegisterCODEC(
        TIFFCodec*
    ) nogil

    int TIFFIsCODECConfigured(
        uint16_t
    ) nogil

    TIFFCodec* TIFFGetConfiguredCODECs(
    ) nogil

    void* _TIFFmalloc(
        tmsize_t s
    ) nogil

    void* _TIFFcalloc(
        tmsize_t nmemb,
        tmsize_t siz
    ) nogil

    void* _TIFFrealloc(
        void* p,
        tmsize_t s
    ) nogil

    void _TIFFmemset(
        void* p,
        int v,
        tmsize_t c
    ) nogil

    void _TIFFmemcpy(
        void* d,
        void* s,
        tmsize_t c
    ) nogil

    int _TIFFmemcmp(
        void* p1,
        void* p2,
        tmsize_t c
    ) nogil

    void _TIFFfree(
        void* p
    ) nogil

    int TIFFGetTagListCount(
        TIFF*
    ) nogil

    uint32_t TIFFGetTagListEntry(
        TIFF*,
        int tag_index
    ) nogil

    ctypedef struct TIFFField:
        pass

    ctypedef struct TIFFFieldArray:
        pass

    TIFFField* TIFFFindField(
        TIFF*,
        uint32_t,
        TIFFDataType
    ) nogil

    TIFFField* TIFFFieldWithTag(
        TIFF*,
        uint32_t
    ) nogil

    TIFFField* TIFFFieldWithName(
        TIFF*,
        char*
    ) nogil

    uint32_t TIFFFieldTag(
        TIFFField*
    ) nogil

    char* TIFFFieldName(
        TIFFField*
    ) nogil

    TIFFDataType TIFFFieldDataType(
        TIFFField*
    ) nogil

    int TIFFFieldPassCount(
        TIFFField*
    ) nogil

    int TIFFFieldReadCount(
        TIFFField*
    ) nogil

    int TIFFFieldWriteCount(
        TIFFField*
    ) nogil

    ctypedef int (*TIFFVSetMethod)(
        TIFF*,
        uint32_t,
        va_list
    ) nogil

    ctypedef int (*TIFFVGetMethod)(
        TIFF*,
        uint32_t,
        va_list
    ) nogil

    ctypedef void (*TIFFPrintMethod)(
        TIFF*,
        FILE*,
        long
    ) nogil

    ctypedef struct TIFFTagMethods:
        TIFFVSetMethod vsetfield
        TIFFVGetMethod vgetfield
        TIFFPrintMethod printdir

    TIFFTagMethods* TIFFAccessTagMethods(
        TIFF*
    ) nogil

    void* TIFFGetClientInfo(
        TIFF*,
        char*
    ) nogil

    void TIFFSetClientInfo(
        TIFF*,
        void*,
        char*
    ) nogil

    void TIFFCleanup(
        TIFF* tif
    ) nogil

    void TIFFClose(
        TIFF* tif
    ) nogil

    int TIFFFlush(
        TIFF* tif
    ) nogil

    int TIFFFlushData(
        TIFF* tif
    ) nogil

    int TIFFGetField(
        TIFF* tif,
        uint32_t tag,
        ...
    ) nogil

    int TIFFVGetField(
        TIFF* tif,
        uint32_t tag,
        va_list ap
    ) nogil

    int TIFFGetFieldDefaulted(
        TIFF* tif,
        uint32_t tag,
        ...
    ) nogil

    int TIFFVGetFieldDefaulted(
        TIFF* tif,
        uint32_t tag,
        va_list ap
    ) nogil

    int TIFFReadDirectory(
        TIFF* tif
    ) nogil

    int TIFFReadCustomDirectory(
        TIFF* tif,
        toff_t diroff,
        TIFFFieldArray* infoarray
    ) nogil

    int TIFFReadEXIFDirectory(
        TIFF* tif,
        toff_t diroff
    ) nogil

    uint64_t TIFFScanlineSize64(
        TIFF* tif
    ) nogil

    tmsize_t TIFFScanlineSize(
        TIFF* tif
    ) nogil

    uint64_t TIFFRasterScanlineSize64(
        TIFF* tif
    ) nogil

    tmsize_t TIFFRasterScanlineSize(
        TIFF* tif
    ) nogil

    uint64_t TIFFStripSize64(
        TIFF* tif
    ) nogil

    tmsize_t TIFFStripSize(
        TIFF* tif
    ) nogil

    uint64_t TIFFRawStripSize64(
        TIFF* tif,
        uint32_t strip
    ) nogil

    tmsize_t TIFFRawStripSize(
        TIFF* tif,
        uint32_t strip
    ) nogil

    uint64_t TIFFVStripSize64(
        TIFF* tif,
        uint32_t nrows
    ) nogil

    tmsize_t TIFFVStripSize(
        TIFF* tif,
        uint32_t nrows
    ) nogil

    uint64_t TIFFTileRowSize64(
        TIFF* tif
    ) nogil

    tmsize_t TIFFTileRowSize(
        TIFF* tif
    ) nogil

    uint64_t TIFFTileSize64(
        TIFF* tif
    ) nogil

    tmsize_t TIFFTileSize(
        TIFF* tif
    ) nogil

    uint64_t TIFFVTileSize64(
        TIFF* tif,
        uint32_t nrows
    ) nogil

    tmsize_t TIFFVTileSize(
        TIFF* tif,
        uint32_t nrows
    ) nogil

    uint32_t TIFFDefaultStripSize(
        TIFF* tif,
        uint32_t request
    ) nogil

    void TIFFDefaultTileSize(
        TIFF*,
        uint32_t*,
        uint32_t*
    ) nogil

    int TIFFFileno(
        TIFF*
    ) nogil

    int TIFFSetFileno(
        TIFF*,
        int
    ) nogil

    thandle_t TIFFClientdata(
        TIFF*
    ) nogil

    thandle_t TIFFSetClientdata(
        TIFF*,
        thandle_t
    ) nogil

    int TIFFGetMode(
        TIFF*
    ) nogil

    int TIFFSetMode(
        TIFF*,
        int
    ) nogil

    int TIFFIsTiled(
        TIFF*
    ) nogil

    int TIFFIsByteSwapped(
        TIFF*
    ) nogil

    int TIFFIsUpSampled(
        TIFF*
    ) nogil

    int TIFFIsMSB2LSB(
        TIFF*
    ) nogil

    int TIFFIsBigEndian(
        TIFF*
    ) nogil

    TIFFReadWriteProc TIFFGetReadProc(
        TIFF*
    ) nogil

    TIFFReadWriteProc TIFFGetWriteProc(
        TIFF*
    ) nogil

    TIFFSeekProc TIFFGetSeekProc(
        TIFF*
    ) nogil

    TIFFCloseProc TIFFGetCloseProc(
        TIFF*
    ) nogil

    TIFFSizeProc TIFFGetSizeProc(
        TIFF*
    ) nogil

    TIFFMapFileProc TIFFGetMapFileProc(
        TIFF*
    ) nogil

    TIFFUnmapFileProc TIFFGetUnmapFileProc(
        TIFF*
    ) nogil

    uint32_t TIFFCurrentRow(
        TIFF*
    ) nogil

    uint16_t TIFFCurrentDirectory(
        TIFF*
    ) nogil

    uint16_t TIFFNumberOfDirectories(
        TIFF*
    ) nogil

    uint64_t TIFFCurrentDirOffset(
        TIFF*
    ) nogil

    uint32_t TIFFCurrentStrip(
        TIFF*
    ) nogil

    uint32_t TIFFCurrentTile(
        TIFF* tif
    ) nogil

    int TIFFReadBufferSetup(
        TIFF* tif,
        void* bp,
        tmsize_t size
    ) nogil

    int TIFFWriteBufferSetup(
        TIFF* tif,
        void* bp,
        tmsize_t size
    ) nogil

    int TIFFSetupStrips(
        TIFF*
    ) nogil

    int TIFFWriteCheck(
        TIFF*,
        int,
        char*
    ) nogil

    void TIFFFreeDirectory(
        TIFF*
    ) nogil

    int TIFFCreateDirectory(
        TIFF*
    ) nogil

    int TIFFCreateCustomDirectory(
        TIFF*,
        TIFFFieldArray*
    ) nogil

    int TIFFCreateEXIFDirectory(
        TIFF*
    ) nogil

    int TIFFLastDirectory(
        TIFF*
    ) nogil

    int TIFFSetDirectory(
        TIFF*,
        uint16_t
    ) nogil

    int TIFFSetSubDirectory(
        TIFF*,
        uint64_t
    ) nogil

    int TIFFUnlinkDirectory(
        TIFF*,
        uint16_t
    ) nogil

    int TIFFSetField(
        TIFF*,
        uint32_t,
        ...
    ) nogil

    int TIFFVSetField(
        TIFF*,
        uint32_t,
        va_list
    ) nogil

    int TIFFUnsetField(
        TIFF*,
        uint32_t
    ) nogil

    int TIFFWriteDirectory(
        TIFF*
    ) nogil

    int TIFFWriteCustomDirectory(
        TIFF*,
        uint64_t*
    ) nogil

    int TIFFCheckpointDirectory(
        TIFF*
    ) nogil

    int TIFFRewriteDirectory(
        TIFF*
    ) nogil

    int TIFFDeferStrileArrayWriting(
        TIFF*
    ) nogil

    int TIFFForceStrileArrayWriting(
        TIFF*
    ) nogil

    void TIFFPrintDirectory(
        TIFF*,
        FILE*,
        long
    ) nogil

    int TIFFReadScanline(
        TIFF* tif,
        void* buf,
        uint32_t row,
        uint16_t sample
    ) nogil

    int TIFFWriteScanline(
        TIFF* tif,
        void* buf,
        uint32_t row,
        uint16_t sample
    ) nogil

    int TIFFReadRGBAImage(
        TIFF*,
        uint32_t,
        uint32_t,
        uint32_t*,
        int
    ) nogil

    int TIFFReadRGBAImageOriented(
        TIFF*,
        uint32_t,
        uint32_t,
        uint32_t*,
        int,
        int
    ) nogil

    int TIFFReadRGBAStrip(
        TIFF*,
        uint32_t,
        uint32_t*
    ) nogil

    int TIFFReadRGBATile(
        TIFF*,
        uint32_t,
        uint32_t,
        uint32_t*
    ) nogil

    int TIFFReadRGBAStripExt(
        TIFF*,
        uint32_t,
        uint32_t*,
        int stop_on_error
    ) nogil

    int TIFFReadRGBATileExt(
        TIFF*,
        uint32_t,
        uint32_t,
        uint32_t*,
        int stop_on_error
    ) nogil

    int TIFFRGBAImageOK(
        TIFF*,
        char [1024]
    ) nogil

    int TIFFRGBAImageBegin(
        TIFFRGBAImage*,
        TIFF*,
        int,
        char [1024]
    ) nogil

    int TIFFRGBAImageGet(
        TIFFRGBAImage*,
        uint32_t*,
        uint32_t,
        uint32_t
    ) nogil

    void TIFFRGBAImageEnd(
        TIFFRGBAImage*
    ) nogil

    TIFF* TIFFOpen(
        char*,
        char*
    ) nogil

    TIFF* TIFFFdOpen(
        int,
        char*,
        char*
    ) nogil

    TIFF* TIFFClientOpen(
        char*,
        char*,
        thandle_t,
        TIFFReadWriteProc,
        TIFFReadWriteProc,
        TIFFSeekProc,
        TIFFCloseProc,
        TIFFSizeProc,
        TIFFMapFileProc,
        TIFFUnmapFileProc
    ) nogil

    char* TIFFFileName(
        TIFF*
    ) nogil

    char* TIFFSetFileName(
        TIFF*,
        char*
    ) nogil

    void TIFFError(
        char*,
        char*
    ) nogil

    void TIFFErrorExt(
        thandle_t,
        char*,
        char*
    ) nogil

    void TIFFWarning(
        char*,
        char*
    ) nogil

    void TIFFWarningExt(
        thandle_t,
        char*,
        char*
    ) nogil

    TIFFErrorHandler TIFFSetErrorHandler(
        TIFFErrorHandler
    ) nogil

    TIFFErrorHandlerExt TIFFSetErrorHandlerExt(
        TIFFErrorHandlerExt
    ) nogil

    TIFFErrorHandler TIFFSetWarningHandler(
        TIFFErrorHandler
    ) nogil

    TIFFErrorHandlerExt TIFFSetWarningHandlerExt(
        TIFFErrorHandlerExt
    ) nogil

    TIFFExtendProc TIFFSetTagExtender(
        TIFFExtendProc
    ) nogil

    uint32_t TIFFComputeTile(
        TIFF* tif,
        uint32_t x,
        uint32_t y,
        uint32_t z,
        uint16_t s
    ) nogil

    int TIFFCheckTile(
        TIFF* tif,
        uint32_t x,
        uint32_t y,
        uint32_t z,
        uint16_t s
    ) nogil

    uint32_t TIFFNumberOfTiles(
        TIFF*
    ) nogil

    tmsize_t TIFFReadTile(
        TIFF* tif,
        void* buf,
        uint32_t x,
        uint32_t y,
        uint32_t z,
        uint16_t s
    ) nogil

    tmsize_t TIFFWriteTile(
        TIFF* tif,
        void* buf,
        uint32_t x,
        uint32_t y,
        uint32_t z,
        uint16_t s
    ) nogil

    uint32_t TIFFComputeStrip(
        TIFF*,
        uint32_t,
        uint16_t
    ) nogil

    uint32_t TIFFNumberOfStrips(
        TIFF*
    ) nogil

    tmsize_t TIFFReadEncodedStrip(
        TIFF* tif,
        uint32_t strip,
        void* buf,
        tmsize_t size
    ) nogil

    tmsize_t TIFFReadRawStrip(
        TIFF* tif,
        uint32_t strip,
        void* buf,
        tmsize_t size
    ) nogil

    tmsize_t TIFFReadEncodedTile(
        TIFF* tif,
        uint32_t tile,
        void* buf,
        tmsize_t size
    ) nogil

    tmsize_t TIFFReadRawTile(
        TIFF* tif,
        uint32_t tile,
        void* buf,
        tmsize_t size
    ) nogil

    int TIFFReadFromUserBuffer(
        TIFF* tif,
        uint32_t strile,
        void* inbuf,
        tmsize_t insize,
        void* outbuf,
        tmsize_t outsize
    ) nogil

    tmsize_t TIFFWriteEncodedStrip(
        TIFF* tif,
        uint32_t strip,
        void* data,
        tmsize_t cc
    ) nogil

    tmsize_t TIFFWriteRawStrip(
        TIFF* tif,
        uint32_t strip,
        void* data,
        tmsize_t cc
    ) nogil

    tmsize_t TIFFWriteEncodedTile(
        TIFF* tif,
        uint32_t tile,
        void* data,
        tmsize_t cc
    ) nogil

    tmsize_t TIFFWriteRawTile(
        TIFF* tif,
        uint32_t tile,
        void* data,
        tmsize_t cc
    ) nogil

    int TIFFDataWidth(
        TIFFDataType
    ) nogil

    void TIFFSetWriteOffset(
        TIFF* tif,
        toff_t off
    ) nogil

    void TIFFSwabShort(
        uint16_t*
    ) nogil

    void TIFFSwabLong(
        uint32_t*
    ) nogil

    void TIFFSwabLong8(
        uint64_t*
    ) nogil

    void TIFFSwabFloat(
        float*
    ) nogil

    void TIFFSwabDouble(
        double*
    ) nogil

    void TIFFSwabArrayOfShort(
        uint16_t* wp,
        tmsize_t n
    ) nogil

    void TIFFSwabArrayOfTriples(
        uint8_t* tp,
        tmsize_t n
    ) nogil

    void TIFFSwabArrayOfLong(
        uint32_t* lp,
        tmsize_t n
    ) nogil

    void TIFFSwabArrayOfLong8(
        uint64_t* lp,
        tmsize_t n
    ) nogil

    void TIFFSwabArrayOfFloat(
        float* fp,
        tmsize_t n
    ) nogil

    void TIFFSwabArrayOfDouble(
        double* dp,
        tmsize_t n
    ) nogil

    void TIFFReverseBits(
        uint8_t* cp,
        tmsize_t n
    ) nogil

    unsigned char* TIFFGetBitRevTable(
        int
    ) nogil

    uint64_t TIFFGetStrileOffset(
        TIFF* tif,
        uint32_t strile
    ) nogil

    uint64_t TIFFGetStrileByteCount(
        TIFF* tif,
        uint32_t strile
    ) nogil

    uint64_t TIFFGetStrileOffsetWithErr(
        TIFF* tif,
        uint32_t strile,
        int* pbErr
    ) nogil

    uint64_t TIFFGetStrileByteCountWithErr(
        TIFF* tif,
        uint32_t strile,
        int* pbErr
    ) nogil

    double LogL16toY(
        int
    ) nogil

    double LogL10toY(
        int
    ) nogil

    void XYZtoRGB24(
        float*,
        uint8_t*
    ) nogil

    int uv_decode(
        double*,
        double*,
        int
    ) nogil

    void LogLuv24toXYZ(
        uint32_t,
        float*
    ) nogil

    void LogLuv32toXYZ(
        uint32_t,
        float*
    ) nogil

    int LogL16fromY(
        double,
        int
    ) nogil

    int LogL10fromY(
        double,
        int
    ) nogil

    int uv_encode(
        double,
        double,
        int
    ) nogil

    uint32_t LogLuv24fromXYZ(
        float*,
        int
    ) nogil

    uint32_t LogLuv32fromXYZ(
        float*,
        int
    ) nogil

    int TIFFCIELabToRGBInit(
        TIFFCIELabToRGB*,
        TIFFDisplay*,
        float*
    ) nogil

    void TIFFCIELabToXYZ(
        TIFFCIELabToRGB*,
        uint32_t,
        int32_t,
        int32_t,
        float*,
        float*,
        float*
    ) nogil

    void TIFFXYZToRGB(
        TIFFCIELabToRGB*,
        float,
        float,
        float,
        uint32_t*,
        uint32_t*,
        uint32_t*
    ) nogil

    int TIFFYCbCrToRGBInit(
        TIFFYCbCrToRGB*,
        float*,
        float*
    ) nogil

    void TIFFYCbCrtoRGB(
        TIFFYCbCrToRGB*,
        uint32_t,
        int32_t,
        int32_t,
        uint32_t*,
        uint32_t*,
        uint32_t*
    ) nogil

    # ctypedef struct TIFFFieldInfo:
    #     ttag_t field_tag
    #     short field_readcount
    #     short field_writecount
    #     TIFFDataType field_type
    #     unsigned short field_bit
    #     unsigned char field_oktochange
    #     unsigned char field_passcount
    #     char* field_name

    # int TIFFMergeFieldInfo(
    #     TIFF*,
    #     TIFFFieldInfo [],
    #     uint32_t
    # ) nogil

    int TIFFTAG_SUBFILETYPE
    int     FILETYPE_REDUCEDIMAGE
    int     FILETYPE_PAGE
    int     FILETYPE_MASK
    int TIFFTAG_OSUBFILETYPE
    int     OFILETYPE_IMAGE
    int     OFILETYPE_REDUCEDIMAGE
    int     OFILETYPE_PAGE
    int TIFFTAG_IMAGEWIDTH
    int TIFFTAG_IMAGELENGTH
    int TIFFTAG_BITSPERSAMPLE
    int TIFFTAG_COMPRESSION
    int     COMPRESSION_NONE
    int     COMPRESSION_CCITTRLE
    int     COMPRESSION_CCITTFAX3
    int     COMPRESSION_CCITT_T4
    int     COMPRESSION_CCITTFAX4
    int     COMPRESSION_CCITT_T6
    int     COMPRESSION_LZW
    int     COMPRESSION_OJPEG
    int     COMPRESSION_JPEG
    int     COMPRESSION_T85
    int     COMPRESSION_T43
    int     COMPRESSION_NEXT
    int     COMPRESSION_CCITTRLEW
    int     COMPRESSION_PACKBITS
    int     COMPRESSION_THUNDERSCAN
    int     COMPRESSION_IT8CTPAD
    int     COMPRESSION_IT8LW
    int     COMPRESSION_IT8MP
    int     COMPRESSION_IT8BL
    int     COMPRESSION_PIXARFILM
    int     COMPRESSION_PIXARLOG
    int     COMPRESSION_DEFLATE
    int     COMPRESSION_ADOBE_DEFLATE
    int     COMPRESSION_DCS
    int     COMPRESSION_JBIG
    int     COMPRESSION_SGILOG
    int     COMPRESSION_SGILOG24
    int     COMPRESSION_JP2000
    int     COMPRESSION_LERC
    int     COMPRESSION_LZMA
    int     COMPRESSION_ZSTD
    int     COMPRESSION_WEBP
    int     COMPRESSION_JXL
    int TIFFTAG_PHOTOMETRIC
    int     PHOTOMETRIC_MINISWHITE
    int     PHOTOMETRIC_MINISBLACK
    int     PHOTOMETRIC_RGB
    int     PHOTOMETRIC_PALETTE
    int     PHOTOMETRIC_MASK
    int     PHOTOMETRIC_SEPARATED
    int     PHOTOMETRIC_YCBCR
    int     PHOTOMETRIC_CIELAB
    int     PHOTOMETRIC_ICCLAB
    int     PHOTOMETRIC_ITULAB
    int     PHOTOMETRIC_CFA
    int     PHOTOMETRIC_LOGL
    int     PHOTOMETRIC_LOGLUV
    int TIFFTAG_THRESHHOLDING
    int     THRESHHOLD_BILEVEL
    int     THRESHHOLD_HALFTONE
    int     THRESHHOLD_ERRORDIFFUSE
    int TIFFTAG_CELLWIDTH
    int TIFFTAG_CELLLENGTH
    int TIFFTAG_FILLORDER
    int     FILLORDER_MSB2LSB
    int     FILLORDER_LSB2MSB
    int TIFFTAG_DOCUMENTNAME
    int TIFFTAG_IMAGEDESCRIPTION
    int TIFFTAG_MAKE
    int TIFFTAG_MODEL
    int TIFFTAG_STRIPOFFSETS
    int TIFFTAG_ORIENTATION
    int     ORIENTATION_TOPLEFT
    int     ORIENTATION_TOPRIGHT
    int     ORIENTATION_BOTRIGHT
    int     ORIENTATION_BOTLEFT
    int     ORIENTATION_LEFTTOP
    int     ORIENTATION_RIGHTTOP
    int     ORIENTATION_RIGHTBOT
    int     ORIENTATION_LEFTBOT
    int TIFFTAG_SAMPLESPERPIXEL
    int TIFFTAG_ROWSPERSTRIP
    int TIFFTAG_STRIPBYTECOUNTS
    int TIFFTAG_MINSAMPLEVALUE
    int TIFFTAG_MAXSAMPLEVALUE
    int TIFFTAG_XRESOLUTION
    int TIFFTAG_YRESOLUTION
    int TIFFTAG_PLANARCONFIG
    int     PLANARCONFIG_CONTIG
    int     PLANARCONFIG_SEPARATE
    int TIFFTAG_PAGENAME
    int TIFFTAG_XPOSITION
    int TIFFTAG_YPOSITION
    int TIFFTAG_FREEOFFSETS
    int TIFFTAG_FREEBYTECOUNTS
    int TIFFTAG_GRAYRESPONSEUNIT
    int     GRAYRESPONSEUNIT_10S
    int     GRAYRESPONSEUNIT_100S
    int     GRAYRESPONSEUNIT_1000S
    int     GRAYRESPONSEUNIT_10000S
    int     GRAYRESPONSEUNIT_100000S
    int TIFFTAG_GRAYRESPONSECURVE
    int TIFFTAG_GROUP3OPTIONS
    int TIFFTAG_T4OPTIONS
    int     GROUP3OPT_2DENCODING
    int     GROUP3OPT_UNCOMPRESSED
    int     GROUP3OPT_FILLBITS
    int TIFFTAG_GROUP4OPTIONS
    int TIFFTAG_T6OPTIONS
    int     GROUP4OPT_UNCOMPRESSED
    int TIFFTAG_RESOLUTIONUNIT
    int     RESUNIT_NONE
    int     RESUNIT_INCH
    int     RESUNIT_CENTIMETER
    int TIFFTAG_PAGENUMBER
    int TIFFTAG_COLORRESPONSEUNIT
    int     COLORRESPONSEUNIT_10S
    int     COLORRESPONSEUNIT_100S
    int     COLORRESPONSEUNIT_1000S
    int     COLORRESPONSEUNIT_10000S
    int     COLORRESPONSEUNIT_100000S
    int TIFFTAG_TRANSFERFUNCTION
    int TIFFTAG_SOFTWARE
    int TIFFTAG_DATETIME
    int TIFFTAG_ARTIST
    int TIFFTAG_HOSTCOMPUTER
    int TIFFTAG_PREDICTOR
    int     PREDICTOR_NONE
    int     PREDICTOR_HORIZONTAL
    int     PREDICTOR_FLOATINGPOINT
    int TIFFTAG_WHITEPOINT
    int TIFFTAG_PRIMARYCHROMATICITIES
    int TIFFTAG_COLORMAP
    int TIFFTAG_HALFTONEHINTS
    int TIFFTAG_TILEWIDTH
    int TIFFTAG_TILELENGTH
    int TIFFTAG_TILEOFFSETS
    int TIFFTAG_TILEBYTECOUNTS
    int TIFFTAG_BADFAXLINES
    int TIFFTAG_CLEANFAXDATA
    int     CLEANFAXDATA_CLEAN
    int     CLEANFAXDATA_REGENERATED
    int     CLEANFAXDATA_UNCLEAN
    int TIFFTAG_CONSECUTIVEBADFAXLINES
    int TIFFTAG_SUBIFD
    int TIFFTAG_INKSET
    int     INKSET_CMYK
    int     INKSET_MULTIINK
    int TIFFTAG_INKNAMES
    int TIFFTAG_NUMBEROFINKS
    int TIFFTAG_DOTRANGE
    int TIFFTAG_TARGETPRINTER
    int TIFFTAG_EXTRASAMPLES
    int     EXTRASAMPLE_UNSPECIFIED
    int     EXTRASAMPLE_ASSOCALPHA
    int     EXTRASAMPLE_UNASSALPHA
    int TIFFTAG_SAMPLEFORMAT
    int     SAMPLEFORMAT_UINT
    int     SAMPLEFORMAT_INT
    int     SAMPLEFORMAT_IEEEFP
    int     SAMPLEFORMAT_VOID
    int     SAMPLEFORMAT_COMPLEXINT
    int     SAMPLEFORMAT_COMPLEXIEEEFP
    int TIFFTAG_SMINSAMPLEVALUE
    int TIFFTAG_SMAXSAMPLEVALUE
    int TIFFTAG_CLIPPATH
    int TIFFTAG_XCLIPPATHUNITS
    int TIFFTAG_YCLIPPATHUNITS
    int TIFFTAG_INDEXED
    int TIFFTAG_JPEGTABLES
    int TIFFTAG_OPIPROXY
    int TIFFTAG_GLOBALPARAMETERSIFD
    int TIFFTAG_PROFILETYPE
    int     PROFILETYPE_UNSPECIFIED
    int     PROFILETYPE_G3_FAX
    int TIFFTAG_FAXPROFILE
    int     FAXPROFILE_S
    int     FAXPROFILE_F
    int     FAXPROFILE_J
    int     FAXPROFILE_C
    int     FAXPROFILE_L
    int     FAXPROFILE_M
    int TIFFTAG_CODINGMETHODS
    int     CODINGMETHODS_T4_1D
    int     CODINGMETHODS_T4_2D
    int     CODINGMETHODS_T6
    int     CODINGMETHODS_T85
    int     CODINGMETHODS_T42
    int     CODINGMETHODS_T43
    int TIFFTAG_VERSIONYEAR
    int TIFFTAG_MODENUMBER
    int TIFFTAG_DECODE
    int TIFFTAG_IMAGEBASECOLOR
    int TIFFTAG_T82OPTIONS
    int TIFFTAG_JPEGPROC
    int     JPEGPROC_BASELINE
    int     JPEGPROC_LOSSLESS
    int TIFFTAG_JPEGIFOFFSET
    int TIFFTAG_JPEGIFBYTECOUNT
    int TIFFTAG_JPEGRESTARTINTERVAL
    int TIFFTAG_JPEGLOSSLESSPREDICTORS
    int TIFFTAG_JPEGPOINTTRANSFORM
    int TIFFTAG_JPEGQTABLES
    int TIFFTAG_JPEGDCTABLES
    int TIFFTAG_JPEGACTABLES
    int TIFFTAG_YCBCRCOEFFICIENTS
    int TIFFTAG_YCBCRSUBSAMPLING
    int TIFFTAG_YCBCRPOSITIONING
    int     YCBCRPOSITION_CENTERED
    int     YCBCRPOSITION_COSITED
    int TIFFTAG_REFERENCEBLACKWHITE
    int TIFFTAG_STRIPROWCOUNTS
    int TIFFTAG_XMLPACKET
    int TIFFTAG_OPIIMAGEID
    int TIFFTAG_TIFFANNOTATIONDATA
    int TIFFTAG_REFPTS
    int TIFFTAG_REGIONTACKPOINT
    int TIFFTAG_REGIONWARPCORNERS
    int TIFFTAG_REGIONAFFINE
    int TIFFTAG_MATTEING
    int TIFFTAG_DATATYPE
    int TIFFTAG_IMAGEDEPTH
    int TIFFTAG_TILEDEPTH
    int TIFFTAG_PIXAR_IMAGEFULLWIDTH
    int TIFFTAG_PIXAR_IMAGEFULLLENGTH
    int TIFFTAG_PIXAR_TEXTUREFORMAT
    int TIFFTAG_PIXAR_WRAPMODES
    int TIFFTAG_PIXAR_FOVCOT
    int TIFFTAG_PIXAR_MATRIX_WORLDTOSCREEN
    int TIFFTAG_PIXAR_MATRIX_WORLDTOCAMERA
    int TIFFTAG_WRITERSERIALNUMBER
    int TIFFTAG_CFAREPEATPATTERNDIM
    int TIFFTAG_CFAPATTERN
    int TIFFTAG_COPYRIGHT
    int	TIFFTAG_MD_FILETAG
    int	TIFFTAG_MD_SCALEPIXEL
    int	TIFFTAG_MD_COLORTABLE
    int	TIFFTAG_MD_LABNAME
    int	TIFFTAG_MD_SAMPLEINFO
    int	TIFFTAG_MD_PREPDATE
    int	TIFFTAG_MD_PREPTIME
    int	TIFFTAG_MD_FILEUNITS
    int TIFFTAG_RICHTIFFIPTC
    int	TIFFTAG_INGR_PACKET_DATA_TAG
    int	TIFFTAG_INGR_FLAG_REGISTERS
    int	TIFFTAG_IRASB_TRANSORMATION_MATRIX
    int	TIFFTAG_MODELTIEPOINTTAG
    int TIFFTAG_IT8SITE
    int TIFFTAG_IT8COLORSEQUENCE
    int TIFFTAG_IT8HEADER
    int TIFFTAG_IT8RASTERPADDING
    int TIFFTAG_IT8BITSPERRUNLENGTH
    int TIFFTAG_IT8BITSPEREXTENDEDRUNLENGTH
    int TIFFTAG_IT8COLORTABLE
    int TIFFTAG_IT8IMAGECOLORINDICATOR
    int TIFFTAG_IT8BKGCOLORINDICATOR
    int TIFFTAG_IT8IMAGECOLORVALUE
    int TIFFTAG_IT8BKGCOLORVALUE
    int TIFFTAG_IT8PIXELINTENSITYRANGE
    int TIFFTAG_IT8TRANSPARENCYINDICATOR
    int TIFFTAG_IT8COLORCHARACTERIZATION
    int TIFFTAG_IT8HCUSAGE
    int TIFFTAG_IT8TRAPINDICATOR
    int TIFFTAG_IT8CMYKEQUIVALENT
    int TIFFTAG_FRAMECOUNT
    int TIFFTAG_MODELTRANSFORMATIONTAG
    int TIFFTAG_PHOTOSHOP
    int TIFFTAG_EXIFIFD
    int TIFFTAG_ICCPROFILE
    int TIFFTAG_IMAGELAYER
    int TIFFTAG_JBIGOPTIONS
    int TIFFTAG_GPSIFD
    int TIFFTAG_FAXRECVPARAMS
    int TIFFTAG_FAXSUBADDRESS
    int TIFFTAG_FAXRECVTIME
    int TIFFTAG_FAXDCS
    int TIFFTAG_STONITS
    int TIFFTAG_FEDEX_EDR
    int TIFFTAG_IMAGESOURCEDATA
    int TIFFTAG_INTEROPERABILITYIFD
    int	TIFFTAG_GDAL_METADATA
    int	TIFFTAG_GDAL_NODATA
    int	TIFFTAG_OCE_SCANJOB_DESCRIPTION
    int	TIFFTAG_OCE_APPLICATION_SELECTOR
    int	TIFFTAG_OCE_IDENTIFICATION_NUMBER
    int	TIFFTAG_OCE_IMAGELOGIC_CHARACTERISTICS
    int TIFFTAG_LERC_PARAMETERS
    int TIFFTAG_DNGVERSION
    int TIFFTAG_DNGBACKWARDVERSION
    int TIFFTAG_UNIQUECAMERAMODEL
    int TIFFTAG_LOCALIZEDCAMERAMODEL
    int TIFFTAG_CFAPLANECOLOR
    int TIFFTAG_CFALAYOUT
    int TIFFTAG_LINEARIZATIONTABLE
    int TIFFTAG_BLACKLEVELREPEATDIM
    int TIFFTAG_BLACKLEVEL
    int TIFFTAG_BLACKLEVELDELTAH
    int TIFFTAG_BLACKLEVELDELTAV
    int TIFFTAG_WHITELEVEL
    int TIFFTAG_DEFAULTSCALE
    int TIFFTAG_DEFAULTCROPORIGIN
    int TIFFTAG_DEFAULTCROPSIZE
    int TIFFTAG_COLORMATRIX1
    int TIFFTAG_COLORMATRIX2
    int TIFFTAG_CAMERACALIBRATION1
    int TIFFTAG_CAMERACALIBRATION2
    int TIFFTAG_REDUCTIONMATRIX1
    int TIFFTAG_REDUCTIONMATRIX2
    int TIFFTAG_ANALOGBALANCE
    int TIFFTAG_ASSHOTNEUTRAL
    int TIFFTAG_ASSHOTWHITEXY
    int TIFFTAG_BASELINEEXPOSURE
    int TIFFTAG_BASELINENOISE
    int TIFFTAG_BASELINESHARPNESS
    int TIFFTAG_BAYERGREENSPLIT
    int TIFFTAG_LINEARRESPONSELIMIT
    int TIFFTAG_CAMERASERIALNUMBER
    int TIFFTAG_LENSINFO
    int TIFFTAG_CHROMABLURRADIUS
    int TIFFTAG_ANTIALIASSTRENGTH
    int TIFFTAG_SHADOWSCALE
    int TIFFTAG_DNGPRIVATEDATA
    int TIFFTAG_MAKERNOTESAFETY
    int TIFFTAG_CALIBRATIONILLUMINANT1
    int TIFFTAG_CALIBRATIONILLUMINANT2
    int TIFFTAG_BESTQUALITYSCALE
    int TIFFTAG_RAWDATAUNIQUEID
    int TIFFTAG_ORIGINALRAWFILENAME
    int TIFFTAG_ORIGINALRAWFILEDATA
    int TIFFTAG_ACTIVEAREA
    int TIFFTAG_MASKEDAREAS
    int TIFFTAG_ASSHOTICCPROFILE
    int TIFFTAG_ASSHOTPREPROFILEMATRIX
    int TIFFTAG_CURRENTICCPROFILE
    int TIFFTAG_CURRENTPREPROFILEMATRIX
    int TIFFTAG_RPCCOEFFICIENT
    int	TIFFTAG_ALIAS_LAYER_METADATA
    int TIFFTAG_TIFF_RSID
    int TIFFTAG_GEO_METADATA
    int TIFFTAG_EXTRACAMERAPROFILES
    int TIFFTAG_DCSHUESHIFTVALUES
    # pseudo tags
    int TIFFTAG_FAXMODE
    int     FAXMODE_CLASSIC
    int     FAXMODE_NORTC
    int     FAXMODE_NOEOL
    int     FAXMODE_BYTEALIGN
    int     FAXMODE_WORDALIGN
    int     FAXMODE_CLASSF_NORTC
    int TIFFTAG_JPEGQUALITY
    int TIFFTAG_JPEGCOLORMODE
    int     JPEGCOLORMODE_RAW
    int     JPEGCOLORMODE_RGB
    int TIFFTAG_JPEGTABLESMODE
    int     JPEGTABLESMODE_QUANT
    int     JPEGTABLESMODE_HUFF
    int TIFFTAG_FAXFILLFUNC
    int TIFFTAG_PIXARLOGDATAFMT
    int     PIXARLOGDATAFMT_8BIT
    int     PIXARLOGDATAFMT_8BITABGR
    int     PIXARLOGDATAFMT_11BITLOG
    int     PIXARLOGDATAFMT_12BITPICIO
    int     PIXARLOGDATAFMT_16BIT
    int     PIXARLOGDATAFMT_FLOAT
    int TIFFTAG_DCSIMAGERTYPE
    int     DCSIMAGERMODEL_M3
    int     DCSIMAGERMODEL_M5
    int     DCSIMAGERMODEL_M6
    int     DCSIMAGERFILTER_IR
    int     DCSIMAGERFILTER_MONO
    int     DCSIMAGERFILTER_CFA
    int     DCSIMAGERFILTER_OTHER
    int TIFFTAG_DCSINTERPMODE
    int     DCSINTERPMODE_NORMAL
    int     DCSINTERPMODE_PREVIEW
    int TIFFTAG_DCSBALANCEARRAY
    int TIFFTAG_DCSCORRECTMATRIX
    int TIFFTAG_DCSGAMMA
    int TIFFTAG_DCSTOESHOULDERPTS
    int TIFFTAG_DCSCALIBRATIONFD
    int TIFFTAG_ZIPQUALITY
    int TIFFTAG_PIXARLOGQUALITY
    int TIFFTAG_DCSCLIPRECTANGLE
    int TIFFTAG_SGILOGDATAFMT
    int     SGILOGDATAFMT_FLOAT
    int     SGILOGDATAFMT_16BIT
    int     SGILOGDATAFMT_RAW
    int     SGILOGDATAFMT_8BIT
    int TIFFTAG_SGILOGENCODE
    int     SGILOGENCODE_NODITHER
    int     SGILOGENCODE_RANDITHER
    int TIFFTAG_LZMAPRESET
    int TIFFTAG_PERSAMPLE
    int     PERSAMPLE_MERGED
    int     PERSAMPLE_MULTI
    int TIFFTAG_ZSTD_LEVEL
    int TIFFTAG_LERC_VERSION
    int     LERC_VERSION_2_4
    int TIFFTAG_LERC_ADD_COMPRESSION
    int     LERC_ADD_COMPRESSION_NONE
    int     LERC_ADD_COMPRESSION_DEFLATE
    int     LERC_ADD_COMPRESSION_ZSTD
    int TIFFTAG_LERC_MAXZERROR
    int TIFFTAG_WEBP_LEVEL
    int TIFFTAG_WEBP_LOSSLESS

    int EXIFTAG_EXPOSURETIME
    int EXIFTAG_FNUMBER
    int EXIFTAG_EXPOSUREPROGRAM
    int EXIFTAG_SPECTRALSENSITIVITY
    int EXIFTAG_ISOSPEEDRATINGS
    int EXIFTAG_PHOTOGRAPHICSENSITIVITY
    int EXIFTAG_OECF
    int EXIFTAG_EXIFVERSION
    int EXIFTAG_DATETIMEORIGINAL
    int EXIFTAG_DATETIMEDIGITIZED
    int EXIFTAG_COMPONENTSCONFIGURATION
    int EXIFTAG_COMPRESSEDBITSPERPIXEL
    int EXIFTAG_SHUTTERSPEEDVALUE
    int EXIFTAG_APERTUREVALUE
    int EXIFTAG_BRIGHTNESSVALUE
    int EXIFTAG_EXPOSUREBIASVALUE
    int EXIFTAG_MAXAPERTUREVALUE
    int EXIFTAG_SUBJECTDISTANCE
    int EXIFTAG_METERINGMODE
    int EXIFTAG_LIGHTSOURCE
    int EXIFTAG_FLASH
    int EXIFTAG_FOCALLENGTH
    int EXIFTAG_SUBJECTAREA
    int EXIFTAG_MAKERNOTE
    int EXIFTAG_USERCOMMENT
    int EXIFTAG_SUBSECTIME
    int EXIFTAG_SUBSECTIMEORIGINAL
    int EXIFTAG_SUBSECTIMEDIGITIZED
    int EXIFTAG_FLASHPIXVERSION
    int EXIFTAG_COLORSPACE
    int EXIFTAG_PIXELXDIMENSION
    int EXIFTAG_PIXELYDIMENSION
    int EXIFTAG_RELATEDSOUNDFILE
    int EXIFTAG_FLASHENERGY
    int EXIFTAG_SPATIALFREQUENCYRESPONSE
    int EXIFTAG_FOCALPLANEXRESOLUTION
    int EXIFTAG_FOCALPLANEYRESOLUTION
    int EXIFTAG_FOCALPLANERESOLUTIONUNIT
    int EXIFTAG_SUBJECTLOCATION
    int EXIFTAG_EXPOSUREINDEX
    int EXIFTAG_SENSINGMETHOD
    int EXIFTAG_FILESOURCE
    int EXIFTAG_SCENETYPE
    int EXIFTAG_CFAPATTERN
    int EXIFTAG_CUSTOMRENDERED
    int EXIFTAG_EXPOSUREMODE
    int EXIFTAG_WHITEBALANCE
    int EXIFTAG_DIGITALZOOMRATIO
    int EXIFTAG_FOCALLENGTHIN35MMFILM
    int EXIFTAG_SCENECAPTURETYPE
    int EXIFTAG_GAINCONTROL
    int EXIFTAG_CONTRAST
    int EXIFTAG_SATURATION
    int EXIFTAG_SHARPNESS
    int EXIFTAG_DEVICESETTINGDESCRIPTION
    int EXIFTAG_SUBJECTDISTANCERANGE
    int EXIFTAG_IMAGEUNIQUEID
    int EXIFTAG_SENSITIVITYTYPE
    int EXIFTAG_STANDARDOUTPUTSENSITIVITY
    int EXIFTAG_RECOMMENDEDEXPOSUREINDEX
    int EXIFTAG_ISOSPEED
    int EXIFTAG_ISOSPEEDLATITUDEYYY
    int EXIFTAG_ISOSPEEDLATITUDEZZZ
    int EXIFTAG_OFFSETTIME
    int EXIFTAG_OFFSETTIMEORIGINAL
    int EXIFTAG_OFFSETTIMEDIGITIZED
    int EXIFTAG_TEMPERATURE
    int EXIFTAG_HUMIDITY
    int EXIFTAG_PRESSURE
    int EXIFTAG_WATERDEPTH
    int EXIFTAG_ACCELERATION
    int EXIFTAG_CAMERAELEVATIONANGLE
    int EXIFTAG_CAMERAOWNERNAME
    int EXIFTAG_BODYSERIALNUMBER
    int EXIFTAG_LENSSPECIFICATION
    int EXIFTAG_LENSMAKE
    int EXIFTAG_LENSMODEL
    int EXIFTAG_LENSSERIALNUMBER
    int EXIFTAG_GAMMA
    int EXIFTAG_COMPOSITEIMAGE
    int EXIFTAG_SOURCEIMAGENUMBEROFCOMPOSITEIMAGE
    int EXIFTAG_SOURCEEXPOSURETIMESOFCOMPOSITEIMAGE

    # EXIF-GPS tags  (Version 2.31, July 2016)
    int GPSTAG_VERSIONID
    int GPSTAG_LATITUDEREF
    int GPSTAG_LATITUDE
    int GPSTAG_LONGITUDEREF
    int GPSTAG_LONGITUDE
    int GPSTAG_ALTITUDEREF
    int GPSTAG_ALTITUDE
    int GPSTAG_TIMESTAMP
    int GPSTAG_SATELLITES
    int GPSTAG_STATUS
    int GPSTAG_MEASUREMODE
    int GPSTAG_DOP
    int GPSTAG_SPEEDREF
    int GPSTAG_SPEED
    int GPSTAG_TRACKREF
    int GPSTAG_TRACK
    int GPSTAG_IMGDIRECTIONREF
    int GPSTAG_IMGDIRECTION
    int GPSTAG_MAPDATUM
    int GPSTAG_DESTLATITUDEREF
    int GPSTAG_DESTLATITUDE
    int GPSTAG_DESTLONGITUDEREF
    int GPSTAG_DESTLONGITUDE
    int GPSTAG_DESTBEARINGREF
    int GPSTAG_DESTBEARING
    int GPSTAG_DESTDISTANCEREF
    int GPSTAG_DESTDISTANCE
    int GPSTAG_PROCESSINGMETHOD
    int GPSTAG_AREAINFORMATION
    int GPSTAG_DATESTAMP
    int GPSTAG_DIFFERENTIAL
