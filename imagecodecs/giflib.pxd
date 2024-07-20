# imagecodecs/giflib.pxd
# cython: language_level = 3

# Cython declarations for the `giflib 5.2.2` library.
# http://giflib.sourceforge.net

cdef extern from 'gif_lib.h' nogil:

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

    int GIF_ASPECT_RATIO(int n)  # ((n)+15.0/64.0)

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

    ctypedef int (*InputFunc)(
        GifFileType*,
        GifByteType*,
        int
    ) nogil

    ctypedef int (*OutputFunc)(
        GifFileType*,
        const GifByteType*,
        int
    ) nogil

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

    extern const char* GifErrorString(
        int ErrorCode
    )

    # encoding

    GifFileType* EGifOpenFileName(
        const char* GifFileName,
        const bint GifTestExistence,
        int* Error
    )

    GifFileType* EGifOpenFileHandle(
        const int GifFileHandle,
        int* Error
    )

    GifFileType* EGifOpen(
        void* userPtr,
        OutputFunc writeFunc,
        int* Error
    )

    int EGifCloseFile(
        GifFileType* GifFile,
        int* ErrorCode
    )

    int EGifSpew(
        GifFileType* ifFile
    )

    const char* EGifGetGifVersion(
        GifFileType* GifFile
    )

    # encoding legacy

    int EGifPutScreenDesc(
        GifFileType* GifFile,
        const int GifWidth,
        const int GifHeight,
        const int GifColorRes,
        const int GifBackGround,
        const ColorMapObject* GifColorMap
    )

    int EGifPutImageDesc(
        GifFileType* GifFile,
        const int GifLeft,
        const int GifTop,
        const int GifWidth,
        const int GifHeight,
        const bint GifInterlace,
        const ColorMapObject* GifColorMap
    )

    void EGifSetGifVersion(
        GifFileType* GifFile,
        const bint gif89
    )

    int EGifPutLine(
        GifFileType* GifFile,
        GifPixelType* GifLine,
        int GifLineLen
    )

    int EGifPutPixel(
        GifFileType* GifFile,
        const GifPixelType GifPixel
    )

    int EGifPutComment(
        GifFileType* GifFile,
        const char* GifComment
    )

    int EGifPutExtensionLeader(
        GifFileType* GifFile,
        const int GifExtCode
    )

    int EGifPutExtensionBlock(
        GifFileType* GifFile,
        const int GifExtLen,
        const void* GifExtension
    )

    int EGifPutExtensionTrailer(
        GifFileType* GifFile
    )

    int EGifPutExtension(
        GifFileType* GifFile,
        const int GifExtCode,
        const int GifExtLen,
        const void* GifExtension
    )

    int EGifPutCode(
        GifFileType* GifFile,
        int GifCodeSize,
        const GifByteType* GifCodeBlock
    )

    int EGifPutCodeNext(
        GifFileType* GifFile,
        const GifByteType* GifCodeBlock
    )

    # decoding

    GifFileType* DGifOpenFileName(
        const char* GifFileName,
        int* Error
    )

    GifFileType* DGifOpenFileHandle(
        int GifFileHandle,
        int* Error
    )

    int DGifSlurp(
        GifFileType* GifFile
    )

    GifFileType* DGifOpen(
        void* userPtr,
        InputFunc readFunc,
        int* Error
    )

    int DGifCloseFile(
        GifFileType* GifFile,
        int* ErrorCode
    )

    # decoding legacy
    int DGifGetScreenDesc(
        GifFileType* GifFile
    )

    int DGifGetRecordType(
        GifFileType* GifFile,
        GifRecordType* GifType
    )

    int DGifGetImageHeader(
        GifFileType* GifFile
    )

    int DGifGetImageDesc(
        GifFileType* GifFile
    )

    int DGifGetLine(
        GifFileType* GifFile,
        GifPixelType* GifLine,
        int GifLineLen
    )

    int DGifGetPixel(
        GifFileType* GifFile,
        GifPixelType GifPixel
    )

    int DGifGetExtension(
        GifFileType* GifFile,
        int* GifExtCode,
        GifByteType** GifExtension
    )

    int DGifGetExtensionNext(
        GifFileType* GifFile,
        GifByteType** GifExtension
    )

    int DGifGetCode(
        GifFileType* GifFile,
        int* GifCodeSize,
        GifByteType** GifCodeBlock
    )

    int DGifGetCodeNext(
        GifFileType* GifFile,
        GifByteType** GifCodeBlock
    )

    int DGifGetLZCodes(
        GifFileType* GifFile,
        int* GifCode
    )

    const char* DGifGetGifVersion(
        GifFileType* GifFile
    )

    # from gif_alloc.c

    ColorMapObject* GifMakeMapObject(
        int ColorCount,
        const GifColorType* ColorMap
    )

    void GifFreeMapObject(
        ColorMapObject* Object
    )

    ColorMapObject* GifUnionColorMap(
        const ColorMapObject* ColorIn1,
        const ColorMapObject* ColorIn2,
        GifPixelType* ColorTransIn2
    )

    int GifBitSize(
        int n
    )

    # slurp mode

    void GifApplyTranslation(
        SavedImage* Image,
        GifPixelType* Translation
    )

    int GifAddExtensionBlock(
        int* ExtensionBlock_Count,
        ExtensionBlock** ExtensionBlocks,
        int Function,
        unsigned int Len,
        unsigned char* ExtData
    )

    void GifFreeExtensions(
        int* ExtensionBlock_Count,
        ExtensionBlock** ExtensionBlocks
    )

    SavedImage* GifMakeSavedImage(
        GifFileType* GifFile,
        const SavedImage* CopyFrom
    )

    void GifFreeSavedImages(
        GifFileType* GifFile
    )

    # GIF89 graphics control blocks

    int DGifExtensionToGCB(
        const size_t GifExtensionLength,
        const GifByteType* GifExtension,
        GraphicsControlBlock* GCB
    )

    size_t EGifGCBToExtension(
        const GraphicsControlBlock* GCB,
        GifByteType* GifExtension
    )

    int DGifSavedExtensionToGCB(
        GifFileType* GifFile,
        int ImageIndex,
        GraphicsControlBlock* GCB
    )

    int EGifGCBToSavedExtension(
        const GraphicsControlBlock* GCB,
        GifFileType* GifFile,
        int ImageIndex
    )

    # internal utility font

    int GIF_FONT_WIDTH
    int GIF_FONT_HEIGHT

    # extern const unsigned char GifAsciiTable8x8[][GIF_FONT_WIDTH]

    void GifDrawText8x8(
        SavedImage* Image,
        const int x,
        const int y,
        const char* legend,
        const int color
    )

    void GifDrawBox(
        SavedImage* Image,
        const int x,
        const int y,
        const int w,
        const int d,
        const int color
    )

    void GifDrawRectangle(
        SavedImage* Image,
        const int x,
        const int y,
        const int w,
        const int d,
        const int color
    )

    void GifDrawBoxedText8x8(
        SavedImage* Image,
        const int x,
        const int y,
        const char* legend,
        const int border,
        const int bg,
        const int fg
    )
