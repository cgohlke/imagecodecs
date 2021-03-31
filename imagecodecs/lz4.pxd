# imagecodecs/lz4.pxd
# cython: language_level = 3

# Cython declarations for the `lz4 1.9.3` library.
# https://github.com/lz4/lz4

cdef extern from 'lz4.h':

    int LZ4_VERSION_MAJOR
    int LZ4_VERSION_MINOR
    int LZ4_VERSION_RELEASE
    int LZ4_VERSION_NUMBER

    char* LZ4_VERSION_STRING

    int LZ4_versionNumber() nogil

    const char* LZ4_versionString() nogil

    # Tuning parameter

    int LZ4_MEMORY_USAGE

    # Simple Functions

    int LZ4_compress_default(
        const char* src,
        char* dst,
        int srcSize,
        int dstCapacity
    ) nogil

    int LZ4_decompress_safe(
        const char* src,
        char* dst,
        int compressedSize,
        int dstCapacity
    ) nogil

    # Advanced Functions

    int LZ4_MAX_INPUT_SIZE

    int LZ4_COMPRESSBOUND(
        int isize
    ) nogil

    int LZ4_compressBound(
        int inputSize
    ) nogil

    int LZ4_compress_fast(
        const char* src,
        char* dst,
        int srcSize,
        int dstCapacity,
        int acceleration
    ) nogil

    int LZ4_sizeofState() nogil

    int LZ4_compress_fast_extState(
        void* state,
        const char* src,
        char* dst,
        int srcSize,
        int dstCapacity,
        int acceleration
    ) nogil

    int LZ4_compress_destSize(
        const char* src,
        char* dst,
        int* srcSizePtr,
        int targetDstSize
    ) nogil

    int LZ4_decompress_safe_partial(
        const char* src,
        char* dst,
        int srcSize,
        int targetOutputSize,
        int dstCapacity
    ) nogil

    # Streaming Compression Functions

    ctypedef union LZ4_stream_t:
        pass

    LZ4_stream_t* LZ4_createStream() nogil

    int LZ4_freeStream(
        LZ4_stream_t* streamPtr
    ) nogil

    void LZ4_resetStream_fast(
        LZ4_stream_t* streamPtr
    ) nogil

    int LZ4_loadDict(
        LZ4_stream_t* streamPtr,
        const char* dictionary,
        int dictSize
    ) nogil

    int LZ4_compress_fast_continue(
        LZ4_stream_t* streamPtr,
        const char* src,
        char* dst,
        int srcSize,
        int dstCapacity,
        int acceleration
    ) nogil

    int LZ4_saveDict(
        LZ4_stream_t* streamPtr,
        char* safeBuffer,
        int maxDictSize
    ) nogil

    # Streaming Decompression Functions

    ctypedef union LZ4_streamDecode_t:
        pass

    LZ4_streamDecode_t* LZ4_createStreamDecode() nogil

    int LZ4_freeStreamDecode(
        LZ4_streamDecode_t* LZ4_stream
    ) nogil

    int LZ4_setStreamDecode(
        LZ4_streamDecode_t* LZ4_streamDecode,
        const char* dictionary,
        int dictSize
    ) nogil

    int LZ4_decoderRingBufferSize(
        int maxBlockSize
    ) nogil

    int LZ4_DECODER_RING_BUFFER_SIZE(
        int maxBlockSize
    ) nogil

    int LZ4_decompress_safe_continue(
        LZ4_streamDecode_t* LZ4_streamDecode,
        const char* src,
        char* dst,
        int srcSize,
        int dstCapacity
    ) nogil

    int LZ4_decompress_safe_usingDict(
        const char* src,
        char* dst,
        int srcSize,
        int dstCapcity,
        const char* dictStart,
        int dictSize
    ) nogil


cdef extern from 'lz4hc.h':

    int LZ4HC_CLEVEL_MIN
    int LZ4HC_CLEVEL_DEFAULT
    int LZ4HC_CLEVEL_OPT_MIN
    int LZ4HC_CLEVEL_MAX

    # Block Compression

    int LZ4_compress_HC(
        const char* src,
        char* dst,
        int srcSize,
        int dstCapacity,
        int compressionLevel
    ) nogil

    int LZ4_sizeofStateHC() nogil

    int LZ4_compress_HC_extStateHC(
        void* stateHC,
        const char* src,
        char* dst,
        int srcSize,
        int maxDstSize,
        int compressionLevel
    ) nogil

    int LZ4_compress_HC_destSize(
        void* stateHC,
        const char* src,
        char* dst,
        int* srcSizePtr,
        int targetDstSize,
        int compressionLevel
    ) nogil

    # Streaming Compression

    ctypedef union LZ4_streamHC_t:
        pass

    LZ4_streamHC_t* LZ4_createStreamHC() nogil

    int LZ4_freeStreamHC(
        LZ4_streamHC_t* streamHCPtr
    ) nogil

    void LZ4_resetStreamHC_fast(
        LZ4_streamHC_t* streamHCPtr,
        int compressionLevel
    ) nogil

    int LZ4_loadDictHC(
        LZ4_streamHC_t* streamHCPtr,
        const char* dictionary,
        int dictSize
    ) nogil

    int LZ4_compress_HC_continue(
        LZ4_streamHC_t* streamHCPtr,
        const char* src,
        char* dst,
        int srcSize,
        int maxDstSize
    ) nogil

    int LZ4_compress_HC_continue_destSize(
        LZ4_streamHC_t* LZ4_streamHCPtr,
        const char* src,
        char* dst,
        int* srcSizePtr,
        int targetDstSize
    ) nogil

    int LZ4_saveDictHC(
        LZ4_streamHC_t* streamHCPtr,
        char* safeBuffer,
        int maxDictSize
    ) nogil


cdef extern from 'lz4frame.h':

    # Error management

    ctypedef size_t LZ4F_errorCode_t

    unsigned LZ4F_isError(
        LZ4F_errorCode_t code
    ) nogil

    const char* LZ4F_getErrorName(
        LZ4F_errorCode_t code
    ) nogil

    # Frame compression types

    ctypedef enum LZ4F_blockSizeID_t:
        LZ4F_default
        LZ4F_max64KB
        LZ4F_max256KB
        LZ4F_max1MB
        LZ4F_max4MB

    ctypedef enum LZ4F_blockMode_t:
        LZ4F_blockLinked
        LZ4F_blockIndependent

    ctypedef enum LZ4F_contentChecksum_t:
        LZ4F_noContentChecksum
        LZ4F_contentChecksumEnabled

    ctypedef enum LZ4F_blockChecksum_t:
        LZ4F_noBlockChecksum
        LZ4F_blockChecksumEnabled

    ctypedef enum LZ4F_frameType_t:
        LZ4F_frame
        LZ4F_skippableFrame

    ctypedef struct LZ4F_frameInfo_t:
        LZ4F_blockSizeID_t  blockSizeID
        LZ4F_blockMode_t  blockMode
        LZ4F_contentChecksum_t contentChecksumFlag
        LZ4F_frameType_t frameType
        unsigned long long contentSize
        unsigned dictID
        LZ4F_blockChecksum_t blockChecksumFlag

    # define LZ4F_INIT_FRAMEINFO :
    #    LZ4F_default,
    #    LZ4F_blockLinked,
    #    LZ4F_noContentChecksum,
    #    LZ4F_frame,
    #    0ULL,
    #    0U,
    #    LZ4F_noBlockChecksum

    ctypedef struct LZ4F_preferences_t:
        LZ4F_frameInfo_t frameInfo
        int compressionLevel
        unsigned autoFlush
        unsigned favorDecSpeed
        unsigned reserved[3]

    # define LZ4F_INIT_PREFERENCES :
    #     LZ4F_INIT_FRAMEINFO,
    #     0,
    #     0u,
    #     0u,
    #     0u, 0u, 0u

    # Simple compression function

    int LZ4F_compressionLevel_max() nogil

    size_t LZ4F_compressFrameBound(
        size_t srcSize,
        const LZ4F_preferences_t* preferencesPtr
    ) nogil

    size_t LZ4F_compressFrame(
        void* dstBuffer,
        size_t dstCapacity,
        const void* srcBuffer,
        size_t srcSize,
        const LZ4F_preferences_t* preferencesPtr
    ) nogil

    # Advanced compression functions

    ctypedef struct LZ4F_cctx:
        pass

    ctypedef LZ4F_cctx* LZ4F_compressionContext_t

    ctypedef struct LZ4F_compressOptions_t:
        unsigned stableSrc
        unsigned reserved[3]

    # Resource Management

    int LZ4F_VERSION

    unsigned LZ4F_getVersion() nogil

    LZ4F_errorCode_t LZ4F_createCompressionContext(
        LZ4F_cctx** cctxPtr,
        unsigned version
    ) nogil

    LZ4F_errorCode_t LZ4F_freeCompressionContext(
        LZ4F_cctx* cctx
    ) nogil

    # Compression

    int LZ4F_HEADER_SIZE_MIN
    int LZ4F_HEADER_SIZE_MAX
    int LZ4F_BLOCK_HEADER_SIZE
    int LZ4F_BLOCK_CHECKSUM_SIZE
    int LZ4F_CONTENT_CHECKSUM_SIZE

    size_t LZ4F_compressBegin(
        LZ4F_cctx* cctx,
        void* dstBuffer,
        size_t dstCapacity,
        const LZ4F_preferences_t* prefsPtr
    ) nogil

    size_t LZ4F_compressBound(
        size_t srcSize,
        const LZ4F_preferences_t* prefsPtr
    ) nogil

    size_t LZ4F_compressUpdate(
        LZ4F_cctx* cctx,
        void* dstBuffer,
        size_t dstCapacity,
        const void* srcBuffer,
        size_t srcSize,
        const LZ4F_compressOptions_t* cOptPtr
    ) nogil

    size_t LZ4F_flush(
        LZ4F_cctx* cctx,
        void* dstBuffer,
        size_t dstCapacity,
        const LZ4F_compressOptions_t* cOptPtr
    ) nogil

    size_t LZ4F_compressEnd(
        LZ4F_cctx* cctx,
        void* dstBuffer,
        size_t dstCapacity,
        const LZ4F_compressOptions_t* cOptPtr
    ) nogil

    # Decompression functions

    ctypedef struct LZ4F_dctx:
        pass

    ctypedef LZ4F_dctx* LZ4F_decompressionContext_t

    ctypedef struct LZ4F_decompressOptions_t:
        unsigned stableDst
        unsigned reserved[3]

    LZ4F_errorCode_t LZ4F_createDecompressionContext(
        LZ4F_dctx** dctxPtr,
        unsigned version
    ) nogil

    LZ4F_errorCode_t LZ4F_freeDecompressionContext(
        LZ4F_dctx* dctx
    ) nogil

    # Streaming decompression functions

    int LZ4F_MIN_SIZE_TO_KNOW_HEADER_LENGTH

    size_t LZ4F_headerSize(
        const void* src,
        size_t srcSize
    ) nogil

    size_t LZ4F_getFrameInfo(
        LZ4F_dctx* dctx,
        LZ4F_frameInfo_t* frameInfoPtr,
        const void* srcBuffer,
        size_t* srcSizePtr
    ) nogil

    size_t LZ4F_decompress(
        LZ4F_dctx* dctx,
        void* dstBuffer,
        size_t* dstSizePtr,
        const void* srcBuffer,
        size_t* srcSizePtr,
        const LZ4F_decompressOptions_t* dOptPtr
    ) nogil

    void LZ4F_resetDecompressionContext(
        LZ4F_dctx* dctx
    ) nogil
