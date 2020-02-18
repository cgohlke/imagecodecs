# giflib.pxd
# cython: language_level = 3

# Cython declarations for the `giflib 5.2.1` library.
# http://giflib.sourceforge.net

cdef extern from 'gif_lib.h':

    int GIFLIB_MAJOR
    int GIFLIB_MINOR
    int GIFLIB_RELEASE

    int GIF_ERROR
    int GIF_OK

    char* GIF_STAMP
    char* GIF87_STAMP
    char* GIF89_STAMP
    int GIF_STAMP_LEN
    int GIF_VERSION_POS

    int CONTINUE_EXT_FUNC_CODE
    int COMMENT_EXT_FUNC_CODE
    int GRAPHICS_EXT_FUNC_CODE
    int PLAINTEXT_EXT_FUNC_CODE
    int APPLICATION_EXT_FUNC_CODE

    int DISPOSAL_UNSPECIFIED
    int DISPOSE_DO_NOT
    int DISPOSE_BACKGROUND
    int DISPOSE_PREVIOUS
    int NO_TRANSPARENT_COLOR

    int E_GIF_SUCCEEDED
    int E_GIF_ERR_OPEN_FAILED
    int E_GIF_ERR_WRITE_FAILED
    int E_GIF_ERR_HAS_SCRN_DSCR
    int E_GIF_ERR_HAS_IMAG_DSCR
    int E_GIF_ERR_NO_COLOR_MAP
    int E_GIF_ERR_DATA_TOO_BIG
    int E_GIF_ERR_NOT_ENOUGH_MEM
    int E_GIF_ERR_DISK_IS_FULL
    int E_GIF_ERR_CLOSE_FAILED
    int E_GIF_ERR_NOT_WRITEABLE

    int D_GIF_SUCCEEDED
    int D_GIF_ERR_OPEN_FAILED
    int D_GIF_ERR_READ_FAILED
    int D_GIF_ERR_NOT_GIF_FILE
    int D_GIF_ERR_NO_SCRN_DSCR
    int D_GIF_ERR_NO_IMAG_DSCR
    int D_GIF_ERR_NO_COLOR_MAP
    int D_GIF_ERR_WRONG_RECORD
    int D_GIF_ERR_DATA_TOO_BIG
    int D_GIF_ERR_NOT_ENOUGH_MEM
    int D_GIF_ERR_CLOSE_FAILED
    int D_GIF_ERR_NOT_READABLE
    int D_GIF_ERR_IMAGE_DEFECT
    int D_GIF_ERR_EOF_TOO_SOON

    # define GIF_ASPECT_RATIO(n) ((n)+15.0/64.0)

    ctypedef unsigned char GifPixelType
    ctypedef unsigned char* GifRowType
    ctypedef unsigned char GifByteType
    ctypedef unsigned int GifPrefixType
    ctypedef int GifWord

    ctypedef struct GifColorType:
        GifByteType Red
        GifByteType Green
        GifByteType Blue

    ctypedef struct ColorMapObject:
        int ColorCount
        int BitsPerPixel
        bint SortFlag
        GifColorType* Colors

    ctypedef struct GifImageDesc:
        GifWord Left
        GifWord Top
        GifWord Width
        GifWord Height
        bint Interlace
        ColorMapObject* ColorMap

    ctypedef struct ExtensionBlock:
        int ByteCount
        GifByteType* Bytes
        int Function

    ctypedef struct SavedImage:
        GifImageDesc ImageDesc
        GifByteType* RasterBits
        int ExtensionBlockCount
        ExtensionBlock* ExtensionBlocks

    ctypedef struct GifFileType:
        GifWord SWidth
        GifWord SHeight
        GifWord SColorResolution
        GifWord SBackGroundColor
        GifByteType AspectByte
        ColorMapObject* SColorMap
        int ImageCount
        GifImageDesc Image
        SavedImage* SavedImages
        int ExtensionBlockCount
        ExtensionBlock* ExtensionBlocks
        int Error
        void* UserData
        void* Private

    ctypedef int (*InputFunc) (GifFileType*, GifByteType*, int) nogil

    ctypedef int (*OutputFunc) (GifFileType*, const GifByteType*, int) nogil

    ctypedef struct GraphicsControlBlock:
        int DisposalMode
        bint UserInputFlag
        int DelayTime
        int TransparentColor

    ctypedef enum GifRecordType:
        UNDEFINED_RECORD_TYPE
        SCREEN_DESC_RECORD_TYPE
        IMAGE_DESC_RECORD_TYPE
        EXTENSION_RECORD_TYPE
        TERMINATE_RECORD_TYPE

    extern const char* GifErrorString(int ErrorCode) nogil

    # encoding

    GifFileType* EGifOpenFileName(
        const char* GifFileName,
        const bint GifTestExistence,
        int* Error) nogil

    GifFileType* EGifOpenFileHandle(
        const int GifFileHandle,
        int* Error) nogil

    GifFileType* EGifOpen(
        void* userPtr,
        OutputFunc writeFunc,
        int* Error) nogil

    int EGifCloseFile(
        GifFileType* GifFile,
        int* ErrorCode) nogil

    int EGifSpew(GifFileType* ifFile) nogil

    const char* EGifGetGifVersion(GifFileType* GifFile) nogil

    # encoding legacy

    int EGifPutScreenDesc(
        GifFileType* GifFile,
        const int GifWidth,
        const int GifHeight,
        const int GifColorRes,
        const int GifBackGround,
        const ColorMapObject* GifColorMap) nogil

    int EGifPutImageDesc(GifFileType* GifFile,
        const int GifLeft,
        const int GifTop,
        const int GifWidth,
        const int GifHeight,
        const bint GifInterlace,
        const ColorMapObject* GifColorMap) nogil

    void EGifSetGifVersion(
        GifFileType* GifFile,
        const bint gif89) nogil

    int EGifPutLine(
        GifFileType* GifFile,
        GifPixelType* GifLine,
        int GifLineLen) nogil

    int EGifPutPixel(
        GifFileType* GifFile,
        const GifPixelType GifPixel) nogil

    int EGifPutComment(
        GifFileType* GifFile,
        const char* GifComment) nogil

    int EGifPutExtensionLeader(
        GifFileType* GifFile,
        const int GifExtCode) nogil

    int EGifPutExtensionBlock(
        GifFileType* GifFile,
        const int GifExtLen,
        const void* GifExtension) nogil

    int EGifPutExtensionTrailer(GifFileType* GifFile) nogil

    int EGifPutExtension(
        GifFileType* GifFile,
        const int GifExtCode,
        const int GifExtLen,
        const void* GifExtension) nogil

    int EGifPutCode(
        GifFileType* GifFile,
        int GifCodeSize,
        const GifByteType* GifCodeBlock) nogil

    int EGifPutCodeNext(
        GifFileType* GifFile,
        const GifByteType* GifCodeBlock) nogil

    # decoding

    GifFileType* DGifOpenFileName(
        const char* GifFileName,
        int* Error) nogil

    GifFileType* DGifOpenFileHandle(
        int GifFileHandle,
        int* Error) nogil

    int DGifSlurp(GifFileType* GifFile) nogil

    GifFileType* DGifOpen(
        void* userPtr,
        InputFunc readFunc,
        int* Error) nogil

    int DGifCloseFile(GifFileType* GifFile, int* ErrorCode) nogil

    # decoding legacy
    int DGifGetScreenDesc(GifFileType* GifFile) nogil

    int DGifGetRecordType(
        GifFileType* GifFile,
        GifRecordType* GifType) nogil

    int DGifGetImageHeader(GifFileType* GifFile) nogil

    int DGifGetImageDesc(GifFileType* GifFile) nogil

    int DGifGetLine(
        GifFileType* GifFile,
        GifPixelType* GifLine,
        int GifLineLen) nogil

    int DGifGetPixel(
        GifFileType* GifFile,
        GifPixelType GifPixel) nogil

    int DGifGetExtension(
        GifFileType* GifFile,
        int* GifExtCode,
        GifByteType** GifExtension) nogil

    int DGifGetExtensionNext(
        GifFileType* GifFile,
        GifByteType** GifExtension) nogil

    int DGifGetCode(
        GifFileType* GifFile,
        int* GifCodeSize,
        GifByteType** GifCodeBlock) nogil

    int DGifGetCodeNext(
        GifFileType* GifFile,
        GifByteType** GifCodeBlock) nogil

    int DGifGetLZCodes(
        GifFileType* GifFile,
        int* GifCode) nogil

    const char* DGifGetGifVersion(GifFileType* GifFile) nogil

    # from gif_alloc.c

    ColorMapObject* GifMakeMapObject(
        int ColorCount,
        const GifColorType* ColorMap) nogil

    void GifFreeMapObject(ColorMapObject* Object) nogil

    ColorMapObject* GifUnionColorMap(
        const ColorMapObject* ColorIn1,
        const ColorMapObject* ColorIn2,
        GifPixelType* ColorTransIn2) nogil

    int GifBitSize(int n) nogil

    # slurp mode

    void GifApplyTranslation(
        SavedImage* Image,
        GifPixelType* Translation) nogil

    int GifAddExtensionBlock(
        int* ExtensionBlock_Count,
        ExtensionBlock** ExtensionBlocks,
        int Function,
        unsigned int Len,
        unsigned char* ExtData) nogil

    void GifFreeExtensions(
        int* ExtensionBlock_Count,
        ExtensionBlock** ExtensionBlocks) nogil

    SavedImage* GifMakeSavedImage(
        GifFileType* GifFile,
        const SavedImage* CopyFrom) nogil

    void GifFreeSavedImages(GifFileType* GifFile) nogil

    # GIF89 graphics control blocks

    int DGifExtensionToGCB(
        const size_t GifExtensionLength,
        const GifByteType* GifExtension,
        GraphicsControlBlock* GCB) nogil

    size_t EGifGCBToExtension(
        const GraphicsControlBlock* GCB,
        GifByteType* GifExtension) nogil

    int DGifSavedExtensionToGCB(
        GifFileType* GifFile,
        int ImageIndex,
        GraphicsControlBlock* GCB) nogil

    int EGifGCBToSavedExtension(
        const GraphicsControlBlock* GCB,
        GifFileType* GifFile,
        int ImageIndex) nogil

    # internal utility font

    int GIF_FONT_WIDTH
    int GIF_FONT_HEIGHT

    # extern const unsigned char GifAsciiTable8x8[][GIF_FONT_WIDTH]

    void GifDrawText8x8(
        SavedImage* Image,
        const int x,
        const int y,
        const char* legend,
        const int color) nogil

    void GifDrawBox(
        SavedImage* Image,
        const int x,
        const int y,
        const int w,
        const int d,
        const int color) nogil

    void GifDrawRectangle(
        SavedImage* Image,
        const int x,
        const int y,
        const int w,
        const int d,
        const int color) nogil

    void GifDrawBoxedText8x8(
        SavedImage* Image,
        const int x,
        const int y,
        const char* legend,
        const int border,
        const int bg,
        const int fg) nogil
