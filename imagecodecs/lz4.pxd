# imagecodecs/lz4.pxd
# cython: language_level = 3

# Cython declarations for the `lz4 1.10.0` library.
# https://github.com/lz4/lz4

cdef extern from 'lz4.h' nogil:

    int LZ4_VERSION_MAJOR
    int LZ4_VERSION_MINOR
    int LZ4_VERSION_RELEASE
    int LZ4_VERSION_NUMBER

    char* LZ4_VERSION_STRING

    int LZ4_versionNumber()

    const char* LZ4_versionString()

    # Tuning parameter

    int LZ4_MEMORY_USAGE

    # Simple Functions

    int LZ4_compress_default(
        const char* src,
        char* dst,
        int srcSize,
        int dstCapacity
    )

    int LZ4_decompress_safe(
        const char* src,
        char* dst,
        int compressedSize,
        int dstCapacity
    )

    # Advanced Functions

    int LZ4_MAX_INPUT_SIZE

    int LZ4_COMPRESSBOUND(
        int isize
    )

    int LZ4_compressBound(
        int inputSize
    )

    int LZ4_compress_fast(
        const char* src,
        char* dst,
        int srcSize,
        int dstCapacity,
        int acceleration
    )

    int LZ4_sizeofState()

    int LZ4_compress_fast_extState(
        void* state,
        const char* src,
        char* dst,
        int srcSize,
        int dstCapacity,
        int acceleration
    )

    int LZ4_compress_destSize(
        const char* src,
        char* dst,
        int* srcSizePtr,
        int targetDstSize
    )

    int LZ4_decompress_safe_partial(
        const char* src,
        char* dst,
        int srcSize,
        int targetOutputSize,
        int dstCapacity
    )

    # Streaming Compression Functions

    ctypedef union LZ4_stream_t:
        pass

    LZ4_stream_t* LZ4_createStream()

    int LZ4_freeStream(
        LZ4_stream_t* streamPtr
    )

    void LZ4_resetStream_fast(
        LZ4_stream_t* streamPtr
    )

    int LZ4_loadDict(
        LZ4_stream_t* streamPtr,
        const char* dictionary,
        int dictSize
    )

    int LZ4_loadDictSlow(
        LZ4_stream_t* streamPtr,
        const char* dictionary,
        int dictSize
    )

    void LZ4_attach_dictionary(
        LZ4_stream_t* workingStream,
        const LZ4_stream_t* dictionaryStream
    )

    int LZ4_compress_fast_continue(
        LZ4_stream_t* streamPtr,
        const char* src,
        char* dst,
        int srcSize,
        int dstCapacity,
        int acceleration
    )

    int LZ4_saveDict(
        LZ4_stream_t* streamPtr,
        char* safeBuffer,
        int maxDictSize
    )

    # Streaming Decompression Functions

    ctypedef union LZ4_streamDecode_t:
        pass

    LZ4_streamDecode_t* LZ4_createStreamDecode()

    int LZ4_freeStreamDecode(
        LZ4_streamDecode_t* LZ4_stream
    )

    int LZ4_setStreamDecode(
        LZ4_streamDecode_t* LZ4_streamDecode,
        const char* dictionary,
        int dictSize
    )

    int LZ4_decoderRingBufferSize(
        int maxBlockSize
    )

    int LZ4_DECODER_RING_BUFFER_SIZE(
        int maxBlockSize
    )

    int LZ4_decompress_safe_continue(
        LZ4_streamDecode_t* LZ4_streamDecode,
        const char* src,
        char* dst,
        int srcSize,
        int dstCapacity
    )

    int LZ4_decompress_safe_usingDict(
        const char* src,
        char* dst,
        int srcSize,
        int dstCapcity,
        const char* dictStart,
        int dictSize
    )

    int LZ4_decompress_safe_partial_usingDict(
        const char* src,
        char* dst,
        int compressedSize,
        int targetOutputSize,
        int maxOutputSize,
        const char* dictStart,
        int dictSize
    )

    int LZ4_decompress_safe_partial_usingDict(
        const char* src,
        char* dst,
        int compressedSize,
        int targetOutputSize,
        int maxOutputSize,
        const char* dictStart,
        int dictSize
    )


cdef extern from 'lz4hc.h' nogil:

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
    )

    int LZ4_sizeofStateHC()

    int LZ4_compress_HC_extStateHC(
        void* stateHC,
        const char* src,
        char* dst,
        int srcSize,
        int maxDstSize,
        int compressionLevel
    )

    int LZ4_compress_HC_destSize(
        void* stateHC,
        const char* src,
        char* dst,
        int* srcSizePtr,
        int targetDstSize,
        int compressionLevel
    )

    # Streaming Compression

    ctypedef union LZ4_streamHC_t:
        pass

    LZ4_streamHC_t* LZ4_createStreamHC()

    int LZ4_freeStreamHC(
        LZ4_streamHC_t* streamHCPtr
    )

    void LZ4_resetStreamHC_fast(
        LZ4_streamHC_t* streamHCPtr,
        int compressionLevel
    )

    int LZ4_loadDictHC(
        LZ4_streamHC_t* streamHCPtr,
        const char* dictionary,
        int dictSize
    )

    int LZ4_compress_HC_continue(
        LZ4_streamHC_t* streamHCPtr,
        const char* src,
        char* dst,
        int srcSize,
        int maxDstSize
    )

    int LZ4_compress_HC_continue_destSize(
        LZ4_streamHC_t* LZ4_streamHCPtr,
        const char* src,
        char* dst,
        int* srcSizePtr,
        int targetDstSize
    )

    int LZ4_saveDictHC(
        LZ4_streamHC_t* streamHCPtr,
        char* safeBuffer,
        int maxDictSize
    )

    void LZ4_attach_HC_dictionary(
        LZ4_streamHC_t* working_stream,
        const LZ4_streamHC_t* dictionary_stream
    )


cdef extern from 'lz4frame.h' nogil:

    # Error management

    ctypedef size_t LZ4F_errorCode_t

    unsigned LZ4F_isError(
        LZ4F_errorCode_t code
    )

    const char* LZ4F_getErrorName(
        LZ4F_errorCode_t code
    )

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
        unsigned[3] reserved

    # define LZ4F_INIT_PREFERENCES :
    #     LZ4F_INIT_FRAMEINFO,
    #     0,
    #     0u,
    #     0u,
    #     0u, 0u, 0u

    # Simple compression function

    size_t LZ4F_compressFrame(
        void* dstBuffer,
        size_t dstCapacity,
        const void* srcBuffer,
        size_t srcSize,
        const LZ4F_preferences_t* preferencesPtr
    )

    int LZ4F_compressionLevel_max()

    size_t LZ4F_compressFrameBound(
        size_t srcSize,
        const LZ4F_preferences_t* preferencesPtr
    )

    size_t LZ4F_compressFrame(
        void* dstBuffer,
        size_t dstCapacity,
        const void* srcBuffer,
        size_t srcSize,
        const LZ4F_preferences_t* preferencesPtr
    )

    # Advanced compression functions

    ctypedef struct LZ4F_cctx:
        pass

    ctypedef LZ4F_cctx* LZ4F_compressionContext_t

    ctypedef struct LZ4F_compressOptions_t:
        unsigned stableSrc
        unsigned[3] reserved

    # Resource Management

    int LZ4F_VERSION

    unsigned LZ4F_getVersion()

    LZ4F_errorCode_t LZ4F_createCompressionContext(
        LZ4F_cctx** cctxPtr,
        unsigned version
    )

    LZ4F_errorCode_t LZ4F_freeCompressionContext(
        LZ4F_cctx* cctx
    )

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
    )

    size_t LZ4F_compressBound(
        size_t srcSize,
        const LZ4F_preferences_t* prefsPtr
    )

    size_t LZ4F_compressUpdate(
        LZ4F_cctx* cctx,
        void* dstBuffer,
        size_t dstCapacity,
        const void* srcBuffer,
        size_t srcSize,
        const LZ4F_compressOptions_t* cOptPtr
    )

    size_t LZ4F_flush(
        LZ4F_cctx* cctx,
        void* dstBuffer,
        size_t dstCapacity,
        const LZ4F_compressOptions_t* cOptPtr
    )

    size_t LZ4F_compressEnd(
        LZ4F_cctx* cctx,
        void* dstBuffer,
        size_t dstCapacity,
        const LZ4F_compressOptions_t* cOptPtr
    )

    # Decompression functions

    ctypedef struct LZ4F_dctx:
        pass

    ctypedef LZ4F_dctx* LZ4F_decompressionContext_t

    ctypedef struct LZ4F_decompressOptions_t:
        unsigned stableDst
        unsigned[3] reserved

    LZ4F_errorCode_t LZ4F_createDecompressionContext(
        LZ4F_dctx** dctxPtr,
        unsigned version
    )

    LZ4F_errorCode_t LZ4F_freeDecompressionContext(
        LZ4F_dctx* dctx
    )

    # Streaming decompression functions

    int LZ4F_MIN_SIZE_TO_KNOW_HEADER_LENGTH

    size_t LZ4F_headerSize(
        const void* src,
        size_t srcSize
    )

    size_t LZ4F_getFrameInfo(
        LZ4F_dctx* dctx,
        LZ4F_frameInfo_t* frameInfoPtr,
        const void* srcBuffer,
        size_t* srcSizePtr
    )

    size_t LZ4F_decompress(
        LZ4F_dctx* dctx,
        void* dstBuffer,
        size_t* dstSizePtr,
        const void* srcBuffer,
        size_t* srcSizePtr,
        const LZ4F_decompressOptions_t* dOptPtr
    )

    void LZ4F_resetDecompressionContext(
        LZ4F_dctx* dctx
    )

    # Dictionary compression API

    size_t LZ4F_compressBegin_usingDict(
        LZ4F_cctx* cctx,
        void* dstBuffer,
        size_t dstCapacity,
        const void* dictBuffer,
        size_t dictSize,
        const LZ4F_preferences_t* prefsPtr
    )

    size_t LZ4F_decompress_usingDict(
        LZ4F_dctx* dctxPtr,
        void* dstBuffer,
        size_t* dstSizePtr,
        const void* srcBuffer,
        size_t* srcSizePtr,
        const void* dict,
        size_t dictSize,
        const LZ4F_decompressOptions_t* decompressOptionsPtr
    )

    ctypedef struct LZ4F_CDict:
        pass

    LZ4F_CDict* LZ4F_createCDict(
        const void* dictBuffer,
        size_t dictSize
    )

    void LZ4F_freeCDict(
        LZ4F_CDict* CDict
    )

    LZ4F_compressFrame_usingCDict(
        LZ4F_cctx* cctx,
        void* dst,
        size_t dstCapacity,
        const void* src,
        size_t srcSize,
        const LZ4F_CDict* cdict,
        const LZ4F_preferences_t* preferencesPtr
    )

    size_t LZ4F_compressBegin_usingCDict(
        LZ4F_cctx* cctx,
        void* dstBuffer,
        size_t dstCapacity,
        const LZ4F_CDict* cdict,
        const LZ4F_preferences_t* prefsPtr
    )
