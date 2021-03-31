# imagecodecs/zstd.pxd
# cython: language_level = 3

# Cython declarations for the `zstd 1.4.8` library (aka Zstandard).
# https://github.com/facebook/zstd

cdef extern from 'zstd.h':

    int ZSTD_VERSION_MAJOR
    int ZSTD_VERSION_MINOR
    int ZSTD_VERSION_RELEASE
    int ZSTD_VERSION_NUMBER

    const char* ZSTD_VERSION_STRING

    const char* ZSTD_versionString() nogil

    unsigned ZSTD_versionNumber() nogil

    int ZSTD_CLEVEL_DEFAULT
    int ZSTD_CONTENTSIZE_UNKNOWN
    int ZSTD_CONTENTSIZE_ERROR
    int ZSTD_MAGICNUMBER
    int ZSTD_MAGIC_DICTIONARY
    int ZSTD_MAGIC_SKIPPABLE_START
    int ZSTD_MAGIC_SKIPPABLE_MASK
    int ZSTD_BLOCKSIZELOG_MAX
    int ZSTD_BLOCKSIZE_MAX

    size_t ZSTD_compress(
        void* dst,
        size_t dstCapacity,
        const void* src,
        size_t srcSize,
        int compressionLevel
    ) nogil

    size_t ZSTD_decompress(
        void* dst,
        size_t dstCapacity,
        const void* src,
        size_t compressedSize
    ) nogil

    unsigned long long ZSTD_getFrameContentSize(
        const void *src,
        size_t srcSize
    ) nogil

    unsigned long long ZSTD_getDecompressedSize(
        const void* src,
        size_t srcSize
        ) nogil

    size_t ZSTD_findFrameCompressedSize(
        const void* src,
        size_t srcSize
        ) nogil

    int ZSTD_COMPRESSBOUND(srcSize) nogil

    size_t ZSTD_compressBound(
        size_t srcSize
    ) nogil

    unsigned ZSTD_isError(
        size_t code
    ) nogil

    const char* ZSTD_getErrorName(
        size_t code
    ) nogil

    int ZSTD_minCLevel() nogil

    int ZSTD_maxCLevel() nogil

    ctypedef struct ZSTD_CCtx:
        pass

    ZSTD_CCtx* ZSTD_createCCtx() nogil

    size_t ZSTD_freeCCtx(
        ZSTD_CCtx* cctx
    ) nogil

    size_t ZSTD_compressCCtx(
        ZSTD_CCtx* cctx,
        void* dst,
        size_t dstCapacity,
        const void* src,
        size_t srcSize,
        int compressionLevel
    ) nogil

    ctypedef struct ZSTD_DCtx:
        pass

    ZSTD_DCtx* ZSTD_createDCtx() nogil

    size_t ZSTD_freeDCtx(
        ZSTD_DCtx* dctx
    ) nogil

    size_t ZSTD_decompressDCtx(
        ZSTD_DCtx* dctx,
        void* dst,
        size_t dstCapacity,
        const void* src,
        size_t srcSize
    ) nogil

    ctypedef enum ZSTD_strategy:
        ZSTD_fast
        ZSTD_dfast
        ZSTD_greedy
        ZSTD_lazy
        ZSTD_lazy2
        ZSTD_btlazy2
        ZSTD_btopt
        ZSTD_btultra
        ZSTD_btultra2

    ctypedef enum ZSTD_cParameter:
        ZSTD_c_compressionLevel
        ZSTD_c_windowLog
        ZSTD_c_hashLog
        ZSTD_c_chainLog
        ZSTD_c_searchLog
        ZSTD_c_minMatch
        ZSTD_c_targetLength
        ZSTD_c_strategy
        ZSTD_c_enableLongDistanceMatching
        ZSTD_c_ldmHashLog
        ZSTD_c_ldmMinMatch
        ZSTD_c_ldmBucketSizeLog
        ZSTD_c_ldmHashRateLog
        ZSTD_c_contentSizeFlag
        ZSTD_c_checksumFlag
        ZSTD_c_dictIDFlag
        ZSTD_c_nbWorkers
        ZSTD_c_jobSize
        ZSTD_c_overlapLog
        ZSTD_c_experimentalParam1
        ZSTD_c_experimentalParam2
        ZSTD_c_experimentalParam3
        ZSTD_c_experimentalParam4
        ZSTD_c_experimentalParam5
        ZSTD_c_experimentalParam6
        ZSTD_c_experimentalParam7
        ZSTD_c_experimentalParam8
        ZSTD_c_experimentalParam9
        ZSTD_c_experimentalParam10
        ZSTD_c_experimentalParam11
        ZSTD_c_experimentalParam12

    ctypedef struct ZSTD_bounds:
        size_t error
        int lowerBound
        int upperBound

    ZSTD_bounds ZSTD_cParam_getBounds(
        ZSTD_cParameter cParam
    ) nogil

    size_t ZSTD_CCtx_setParameter(
        ZSTD_CCtx* cctx,
        ZSTD_cParameter param,
        int value
    ) nogil

    size_t ZSTD_CCtx_setPledgedSrcSize(
        ZSTD_CCtx* cctx,
        unsigned long long pledgedSrcSize
    ) nogil

    ctypedef enum ZSTD_ResetDirective:
        ZSTD_reset_session_only
        ZSTD_reset_parameters
        ZSTD_reset_session_and_parameters

    size_t ZSTD_CCtx_reset(
        ZSTD_CCtx* cctx,
        ZSTD_ResetDirective reset
    ) nogil

    size_t ZSTD_compress2(
        ZSTD_CCtx* cctx,
        void* dst,
        size_t dstCapacity,
        const void* src,
        size_t srcSize
    ) nogil

    ctypedef enum ZSTD_dParameter:
        ZSTD_d_windowLogMax
        ZSTD_d_experimentalParam1
        ZSTD_d_experimentalParam2
        ZSTD_d_experimentalParam3

    ZSTD_bounds ZSTD_dParam_getBounds(
        ZSTD_dParameter dParam
    ) nogil

    size_t ZSTD_DCtx_setParameter(
        ZSTD_DCtx* dctx,
        ZSTD_dParameter param,
        int value
    ) nogil

    size_t ZSTD_DCtx_reset(
        ZSTD_DCtx* dctx,
        ZSTD_ResetDirective reset
    ) nogil

    ctypedef struct ZSTD_inBuffer:
        const void* src
        size_t size
        size_t pos

    ctypedef struct ZSTD_outBuffer:
        void* dst
        size_t size
        size_t pos

    ctypedef ZSTD_CCtx ZSTD_CStream

    ZSTD_CStream* ZSTD_createCStream() nogil

    size_t ZSTD_freeCStream(
        ZSTD_CStream* zcs
    ) nogil

    ctypedef enum ZSTD_EndDirective:
        ZSTD_e_continue
        ZSTD_e_flush
        ZSTD_e_end

    size_t ZSTD_compressStream2(
        ZSTD_CCtx* cctx,
        ZSTD_outBuffer* output,
        ZSTD_inBuffer* input,
        ZSTD_EndDirective endOp
    ) nogil

    size_t ZSTD_CStreamInSize() nogil

    size_t ZSTD_CStreamOutSize() nogil

    size_t ZSTD_initCStream(
        ZSTD_CStream* zcs,
        int compressionLevel
    ) nogil

    size_t ZSTD_compressStream(
        ZSTD_CStream* zcs,
        ZSTD_outBuffer* output,
        ZSTD_inBuffer* input
    ) nogil

    size_t ZSTD_flushStream(
        ZSTD_CStream* zcs,
        ZSTD_outBuffer* output
    ) nogil

    size_t ZSTD_endStream(
        ZSTD_CStream* zcs,
        ZSTD_outBuffer* output
    ) nogil

    ctypedef ZSTD_DCtx ZSTD_DStream

    ZSTD_DStream* ZSTD_createDStream() nogil

    size_t ZSTD_freeDStream(
        ZSTD_DStream* zds
    ) nogil

    size_t ZSTD_initDStream(
        ZSTD_DStream* zds
    ) nogil

    size_t ZSTD_decompressStream(
        ZSTD_DStream* zds,
        ZSTD_outBuffer* output,
        ZSTD_inBuffer* input
    ) nogil

    size_t ZSTD_DStreamInSize() nogil

    size_t ZSTD_DStreamOutSize() nogil

    size_t ZSTD_compress_usingDict(
        ZSTD_CCtx* ctx,
        void* dst,
        size_t dstCapacity,
        const void* src,
        size_t srcSize,
        const void* dict,
        size_t dictSize,
        int compressionLevel
    ) nogil

    size_t ZSTD_decompress_usingDict(
        ZSTD_DCtx* dctx,
        void* dst,
        size_t dstCapacity,
        const void* src,
        size_t srcSize,
        const void* dict,
        size_t dictSize
    ) nogil

    ctypedef struct ZSTD_CDict:
        pass

    ZSTD_CDict* ZSTD_createCDict(
        const void* dictBuffer,
        size_t dictSize,
        int compressionLevel
    ) nogil

    size_t ZSTD_freeCDict(
        ZSTD_CDict* CDict
    ) nogil

    size_t ZSTD_compress_usingCDict(
        ZSTD_CCtx* cctx,
        void* dst,
        size_t dstCapacity,
        const void* src,
        size_t srcSize,
        const ZSTD_CDict* cdict
    ) nogil

    ctypedef struct ZSTD_DDict:
        pass

    ZSTD_DDict* ZSTD_createDDict(
        const void* dictBuffer,
        size_t dictSize
    ) nogil

    size_t ZSTD_freeDDict(
        ZSTD_DDict* ddict
    ) nogil

    size_t ZSTD_decompress_usingDDict(
        ZSTD_DCtx* dctx,
        void* dst,
        size_t dstCapacity,
        const void* src,
        size_t srcSize,
        const ZSTD_DDict* ddict
    ) nogil

    unsigned ZSTD_getDictID_fromDict(
        const void* dict,
        size_t dictSize
    ) nogil

    unsigned ZSTD_getDictID_fromDDict(
        const ZSTD_DDict* ddict
    ) nogil

    unsigned ZSTD_getDictID_fromFrame(
        const void* src,
        size_t srcSize
    ) nogil

    size_t ZSTD_CCtx_loadDictionary(
        ZSTD_CCtx* cctx,
        const void* dict,
        size_t dictSize
    ) nogil

    size_t ZSTD_CCtx_refCDict(
        ZSTD_CCtx* cctx,
        const ZSTD_CDict* cdict
    ) nogil

    size_t ZSTD_CCtx_refPrefix(
        ZSTD_CCtx* cctx,
        const void* prefix,
        size_t prefixSize
    ) nogil

    size_t ZSTD_DCtx_loadDictionary(
        ZSTD_DCtx* dctx,
        const void* dict,
        size_t dictSize
    ) nogil

    size_t ZSTD_DCtx_refDDict(
        ZSTD_DCtx* dctx,
        const ZSTD_DDict* ddict
    ) nogil

    size_t ZSTD_DCtx_refPrefix(
        ZSTD_DCtx* dctx,
        const void* prefix,
        size_t prefixSize
    ) nogil

    size_t ZSTD_sizeof_CCtx(
        const ZSTD_CCtx* cctx
    ) nogil

    size_t ZSTD_sizeof_DCtx(
        const ZSTD_DCtx* dctx
    ) nogil

    size_t ZSTD_sizeof_CStream(
        const ZSTD_CStream* zcs
    ) nogil

    size_t ZSTD_sizeof_DStream(
        const ZSTD_DStream* zds
    ) nogil

    size_t ZSTD_sizeof_CDict(
        const ZSTD_CDict* cdict
    ) nogil

    size_t ZSTD_sizeof_DDict(
        const ZSTD_DDict* ddict
    ) nogil

    #endif

    #if defined(ZSTD_STATIC_LINKING_ONLY) && !defined(ZSTD_H_ZSTD_STATIC_LINKING_ONLY)
    int ZSTD_H_ZSTD_STATIC_LINKING_ONLY

    int ZSTD_FRAMEHEADERSIZE_PREFIX(format) nogil
    int ZSTD_FRAMEHEADERSIZE_MIN(format) nogil
    int ZSTD_FRAMEHEADERSIZE_MAX
    int ZSTD_SKIPPABLEHEADERSIZE
    int ZSTD_WINDOWLOG_MAX_32
    int ZSTD_WINDOWLOG_MAX_64
    int ZSTD_WINDOWLOG_MAX
    int ZSTD_WINDOWLOG_MIN
    int ZSTD_HASHLOG_MAX
    int ZSTD_HASHLOG_MIN
    int ZSTD_CHAINLOG_MAX_32
    int ZSTD_CHAINLOG_MAX_64
    int ZSTD_CHAINLOG_MAX
    int ZSTD_CHAINLOG_MIN
    int ZSTD_SEARCHLOG_MAX
    int ZSTD_SEARCHLOG_MIN
    int ZSTD_MINMATCH_MAX
    int ZSTD_MINMATCH_MIN
    int ZSTD_TARGETLENGTH_MAX
    int ZSTD_TARGETLENGTH_MIN
    int ZSTD_STRATEGY_MIN
    int ZSTD_STRATEGY_MAX
    int ZSTD_OVERLAPLOG_MIN
    int ZSTD_OVERLAPLOG_MAX
    int ZSTD_WINDOWLOG_LIMIT_DEFAULT
    int ZSTD_LDM_HASHLOG_MIN
    int ZSTD_LDM_HASHLOG_MAX
    int ZSTD_LDM_MINMATCH_MIN
    int ZSTD_LDM_MINMATCH_MAX
    int ZSTD_LDM_BUCKETSIZELOG_MIN
    int ZSTD_LDM_BUCKETSIZELOG_MAX
    int ZSTD_LDM_HASHRATELOG_MIN
    int ZSTD_LDM_HASHRATELOG_MAX
    int ZSTD_TARGETCBLOCKSIZE_MIN
    int ZSTD_TARGETCBLOCKSIZE_MAX
    int ZSTD_SRCSIZEHINT_MIN
    int ZSTD_SRCSIZEHINT_MAX
    int ZSTD_HASHLOG3_MAX

    ctypedef struct ZSTD_CCtx_params:
        pass

    ctypedef struct ZSTD_Sequence:
        unsigned int offset
        unsigned int litLength
        unsigned int matchLength
        unsigned int rep

    ctypedef struct ZSTD_compressionParameters:
        unsigned windowLog
        unsigned chainLog
        unsigned hashLog
        unsigned searchLog
        unsigned minMatch
        unsigned targetLength
        ZSTD_strategy strategy

    ctypedef struct ZSTD_frameParameters:
        int contentSizeFlag
        int checksumFlag
        int noDictIDFlag

    ctypedef struct ZSTD_parameters:
        ZSTD_compressionParameters cParams
        ZSTD_frameParameters fParams

    ctypedef enum ZSTD_dictContentType_e:
        ZSTD_dct_auto
        ZSTD_dct_rawContent
        ZSTD_dct_fullDict

    ctypedef enum ZSTD_dictLoadMethod_e:
        ZSTD_dlm_byCopy
        ZSTD_dlm_byRef

    ctypedef enum ZSTD_format_e:
        ZSTD_f_zstd1
        ZSTD_f_zstd1_magicless

    ctypedef enum ZSTD_forceIgnoreChecksum_e:
        ZSTD_d_validateChecksum
        ZSTD_d_ignoreChecksum

    ctypedef enum ZSTD_dictAttachPref_e:
        ZSTD_dictDefaultAttach
        ZSTD_dictForceAttach
        ZSTD_dictForceCopy
        ZSTD_dictForceLoad

    ctypedef enum ZSTD_literalCompressionMode_e:
        ZSTD_lcm_auto
        ZSTD_lcm_huffman
        ZSTD_lcm_uncompressed

    unsigned long long ZSTD_findDecompressedSize(
        const void* src,
        size_t srcSize
    ) nogil

    unsigned long long ZSTD_decompressBound(
        const void* src,
        size_t srcSize
    ) nogil

    size_t ZSTD_frameHeaderSize(
        const void* src,
        size_t srcSize
    ) nogil

    ctypedef enum ZSTD_sequenceFormat_e:
        ZSTD_sf_noBlockDelimiters
        ZSTD_sf_explicitBlockDelimiters


    size_t ZSTD_generateSequences(
        ZSTD_CCtx* zc,
        ZSTD_Sequence* outSeqs,
        size_t outSeqsSize,
        const void* src,
        size_t srcSize
    ) nogil

    size_t ZSTD_mergeBlockDelimiters(
        ZSTD_Sequence* sequences,
        size_t seqsSize
    ) nogil

    size_t ZSTD_compressSequences(
        ZSTD_CCtx* const cctx,
        void* dst,
        size_t dstSize,
        const ZSTD_Sequence* inSeqs,
        size_t inSeqsSize,
        const void* src,
        size_t srcSize
    ) nogil

    size_t ZSTD_estimateCCtxSize(
        int compressionLevel
    ) nogil

    size_t ZSTD_estimateCCtxSize_usingCParams(
        ZSTD_compressionParameters cParams
    ) nogil

    size_t ZSTD_estimateCCtxSize_usingCCtxParams(
        const ZSTD_CCtx_params* params
    ) nogil

    size_t ZSTD_estimateDCtxSize() nogil

    size_t ZSTD_estimateCStreamSize(
        int compressionLevel
    ) nogil

    size_t ZSTD_estimateCStreamSize_usingCParams(
        ZSTD_compressionParameters cParams
    ) nogil

    size_t ZSTD_estimateCStreamSize_usingCCtxParams(
        const ZSTD_CCtx_params* params
    ) nogil

    size_t ZSTD_estimateDStreamSize(
        size_t windowSize
    ) nogil

    size_t ZSTD_estimateDStreamSize_fromFrame(
        const void* src,
        size_t srcSize
    ) nogil

    size_t ZSTD_estimateCDictSize(
        size_t dictSize,
        int compressionLevel
    ) nogil

    size_t ZSTD_estimateCDictSize_advanced(
        size_t dictSize,
        ZSTD_compressionParameters cParams,
        ZSTD_dictLoadMethod_e dictLoadMethod
    ) nogil

    size_t ZSTD_estimateDDictSize(
        size_t dictSize,
        ZSTD_dictLoadMethod_e dictLoadMethod
    ) nogil

    ZSTD_CCtx* ZSTD_initStaticCCtx(
        void* workspace,
        size_t workspaceSize
    ) nogil

    ZSTD_CStream* ZSTD_initStaticCStream(
        void* workspace,
        size_t workspaceSize
    ) nogil

    ZSTD_DCtx* ZSTD_initStaticDCtx(
        void* workspace,
        size_t workspaceSize
    ) nogil

    ZSTD_DStream* ZSTD_initStaticDStream(
        void* workspace,
        size_t workspaceSize
    ) nogil

    const ZSTD_CDict* ZSTD_initStaticCDict(
        void* workspace,
        size_t workspaceSize,
        const void* dict,
        size_t dictSize,
        ZSTD_dictLoadMethod_e dictLoadMethod,
        ZSTD_dictContentType_e dictContentType,
        ZSTD_compressionParameters cParams
    ) nogil

    const ZSTD_DDict* ZSTD_initStaticDDict(
        void* workspace,
        size_t workspaceSize,
        const void* dict,
        size_t dictSize,
        ZSTD_dictLoadMethod_e dictLoadMethod,
        ZSTD_dictContentType_e dictContentType
    ) nogil

    ctypedef void* (*ZSTD_allocFunction) (
        void* opaque,
        size_t size
    ) nogil

    ctypedef void (*ZSTD_freeFunction) (
        void* opaque,
        void* address
    ) nogil

    ctypedef struct ZSTD_customMem:
        ZSTD_allocFunction customAlloc
        ZSTD_freeFunction customFree
        void* opaque

    # ZSTD_customMem const ZSTD_defaultCMem = { NULL, NULL, NULL }

    ZSTD_CCtx* ZSTD_createCCtx_advanced(
        ZSTD_customMem customMem
    ) nogil

    ZSTD_CStream* ZSTD_createCStream_advanced(
        ZSTD_customMem customMem
    ) nogil

    ZSTD_DCtx* ZSTD_createDCtx_advanced(
        ZSTD_customMem customMem
    ) nogil

    ZSTD_DStream* ZSTD_createDStream_advanced(
        ZSTD_customMem customMem
    ) nogil

    ZSTD_CDict* ZSTD_createCDict_advanced(
        const void* dict,
        size_t dictSize,
        ZSTD_dictLoadMethod_e dictLoadMethod,
        ZSTD_dictContentType_e dictContentType,
        ZSTD_compressionParameters cParams,
        ZSTD_customMem customMem
    ) nogil

    ctypedef struct ZSTD_threadPool:
        pass

    ZSTD_threadPool* ZSTD_createThreadPool(
        size_t numThreads
    ) nogil

    void ZSTD_freeThreadPool(
        ZSTD_threadPool* pool
    ) nogil

    size_t ZSTD_CCtx_refThreadPool(
        ZSTD_CCtx* cctx,
        ZSTD_threadPool* pool
    ) nogil

    ZSTD_CDict* ZSTD_createCDict_advanced2(
        const void* dict,
        size_t dictSize,
        ZSTD_dictLoadMethod_e dictLoadMethod,
        ZSTD_dictContentType_e dictContentType,
        const ZSTD_CCtx_params* cctxParams,
        ZSTD_customMem customMem
    ) nogil

    ZSTD_DDict* ZSTD_createDDict_advanced(
        const void* dict,
        size_t dictSize,
        ZSTD_dictLoadMethod_e dictLoadMethod,
        ZSTD_dictContentType_e dictContentType,
        ZSTD_customMem customMem
    ) nogil

    ZSTD_CDict* ZSTD_createCDict_byReference(
        const void* dictBuffer,
        size_t dictSize,
        int compressionLevel
    ) nogil

    unsigned ZSTD_getDictID_fromCDict(
        const ZSTD_CDict* cdict
    ) nogil

    ZSTD_compressionParameters ZSTD_getCParams(
        int compressionLevel,
        unsigned long long estimatedSrcSize,
        size_t dictSize
    ) nogil

    ZSTD_parameters ZSTD_getParams(
        int compressionLevel,
        unsigned long long estimatedSrcSize,
        size_t dictSize
    ) nogil

    size_t ZSTD_checkCParams(
        ZSTD_compressionParameters params
    ) nogil

    ZSTD_compressionParameters ZSTD_adjustCParams(
        ZSTD_compressionParameters cPar,
        unsigned long long srcSize,
        size_t dictSize
    ) nogil

    size_t ZSTD_compress_advanced(
        ZSTD_CCtx* cctx,
        void* dst,
        size_t dstCapacity,
        const void* src,
        size_t srcSize,
        const void* dict,
        size_t dictSize,
        ZSTD_parameters params
    ) nogil

    size_t ZSTD_compress_usingCDict_advanced(
        ZSTD_CCtx* cctx,
        void* dst,
        size_t dstCapacity,
        const void* src,
        size_t srcSize,
        const ZSTD_CDict* cdict,
        ZSTD_frameParameters fParams
    ) nogil

    size_t ZSTD_CCtx_loadDictionary_byReference(
        ZSTD_CCtx* cctx,
        const void* dict,
        size_t dictSize
    ) nogil

    size_t ZSTD_CCtx_loadDictionary_advanced(
        ZSTD_CCtx* cctx,
        const void* dict,
        size_t dictSize,
        ZSTD_dictLoadMethod_e dictLoadMethod,
        ZSTD_dictContentType_e dictContentType
    ) nogil

    size_t ZSTD_CCtx_refPrefix_advanced(
        ZSTD_CCtx* cctx,
        const void* prefix,
        size_t prefixSize,
        ZSTD_dictContentType_e dictContentType
    ) nogil

    int ZSTD_c_rsyncable
    int ZSTD_c_format
    int ZSTD_c_forceMaxWindow
    int ZSTD_c_forceAttachDict
    int ZSTD_c_literalCompressionMode
    int ZSTD_c_targetCBlockSize
    int ZSTD_c_srcSizeHint
    int ZSTD_c_enableDedicatedDictSearch
    int ZSTD_c_stableInBuffer
    int ZSTD_c_stableOutBuffer
    int ZSTD_c_blockDelimiters
    int ZSTD_c_validateSequences

    size_t ZSTD_CCtx_getParameter(
        ZSTD_CCtx* cctx,
        ZSTD_cParameter param,
        int* value
    ) nogil

    ZSTD_CCtx_params* ZSTD_createCCtxParams() nogil

    size_t ZSTD_freeCCtxParams(
        ZSTD_CCtx_params* params
    ) nogil

    size_t ZSTD_CCtxParams_reset(
        ZSTD_CCtx_params* params
    ) nogil

    size_t ZSTD_CCtxParams_init(
        ZSTD_CCtx_params* cctxParams,
        int compressionLevel
    ) nogil

    size_t ZSTD_CCtxParams_init_advanced(
        ZSTD_CCtx_params* cctxParams,
        ZSTD_parameters params
    ) nogil

    size_t ZSTD_CCtxParams_setParameter(
        ZSTD_CCtx_params* params,
        ZSTD_cParameter param,
        int value
    ) nogil

    size_t ZSTD_CCtxParams_getParameter(
        ZSTD_CCtx_params* params,
        ZSTD_cParameter param,
        int* value
    ) nogil

    size_t ZSTD_CCtx_setParametersUsingCCtxParams(
        ZSTD_CCtx* cctx,
        const ZSTD_CCtx_params* params
    ) nogil

    size_t ZSTD_compressStream2_simpleArgs(
        ZSTD_CCtx* cctx,
        void* dst,
        size_t dstCapacity,
        size_t* dstPos,
        const void* src,
        size_t srcSize,
        size_t* srcPos,
        ZSTD_EndDirective endOp
    ) nogil

    unsigned ZSTD_isFrame(
        const void* buffer,
        size_t size
    ) nogil

    ZSTD_DDict* ZSTD_createDDict_byReference(
        const void* dictBuffer,
        size_t dictSize
    ) nogil

    size_t ZSTD_DCtx_loadDictionary_byReference(
        ZSTD_DCtx* dctx,
        const void* dict,
        size_t dictSize
    ) nogil

    size_t ZSTD_DCtx_loadDictionary_advanced(
        ZSTD_DCtx* dctx,
        const void* dict,
        size_t dictSize,
        ZSTD_dictLoadMethod_e dictLoadMethod,
        ZSTD_dictContentType_e dictContentType
    ) nogil

    size_t ZSTD_DCtx_refPrefix_advanced(
        ZSTD_DCtx* dctx,
        const void* prefix,
        size_t prefixSize,
        ZSTD_dictContentType_e dictContentType
    ) nogil

    size_t ZSTD_DCtx_setMaxWindowSize(
        ZSTD_DCtx* dctx,
        size_t maxWindowSize
    ) nogil

    size_t ZSTD_DCtx_getParameter(
        ZSTD_DCtx* dctx,
        ZSTD_dParameter param,
        int* value
    ) nogil

    int ZSTD_d_format
    int ZSTD_d_stableOutBuffer
    int ZSTD_d_forceIgnoreChecksum

    size_t ZSTD_DCtx_setFormat(
        ZSTD_DCtx* dctx,
        ZSTD_format_e format
    ) nogil

    size_t ZSTD_decompressStream_simpleArgs(
        ZSTD_DCtx* dctx,
        void* dst,
        size_t dstCapacity,
        size_t* dstPos,
        const void* src,
        size_t srcSize,
        size_t* srcPos
    ) nogil

    size_t ZSTD_initCStream_srcSize(
        ZSTD_CStream* zcs,
        int compressionLevel,
        unsigned long long pledgedSrcSize
    ) nogil

    size_t ZSTD_initCStream_usingDict(
        ZSTD_CStream* zcs,
        const void* dict,
        size_t dictSize,
        int compressionLevel
    ) nogil

    size_t ZSTD_initCStream_advanced(
        ZSTD_CStream* zcs,
        const void* dict,
        size_t dictSize,
        ZSTD_parameters params,
        unsigned long long pledgedSrcSize
    ) nogil

    size_t ZSTD_initCStream_usingCDict(
        ZSTD_CStream* zcs,
        const ZSTD_CDict* cdict
    ) nogil

    size_t ZSTD_initCStream_usingCDict_advanced(
        ZSTD_CStream* zcs,
        const ZSTD_CDict* cdict,
        ZSTD_frameParameters fParams,
        unsigned long long pledgedSrcSize
    ) nogil

    size_t ZSTD_resetCStream(
        ZSTD_CStream* zcs,
        unsigned long long pledgedSrcSize
    ) nogil

    ctypedef struct ZSTD_frameProgression:
        unsigned long long ingested
        unsigned long long consumed
        unsigned long long produced
        unsigned long long flushed
        unsigned currentJobID
        unsigned nbActiveWorkers

    ZSTD_frameProgression ZSTD_getFrameProgression(
        const ZSTD_CCtx* cctx
    ) nogil

    size_t ZSTD_toFlushNow(
        ZSTD_CCtx* cctx
    ) nogil

    size_t ZSTD_initDStream_usingDict(
        ZSTD_DStream* zds,
        const void* dict,
        size_t dictSize
    ) nogil

    size_t ZSTD_initDStream_usingDDict(
        ZSTD_DStream* zds,
        const ZSTD_DDict* ddict
    ) nogil

    size_t ZSTD_resetDStream(
        ZSTD_DStream* zds) nogil

    size_t ZSTD_compressBegin(
        ZSTD_CCtx* cctx,
        int compressionLevel
    ) nogil

    size_t ZSTD_compressBegin_usingDict(
        ZSTD_CCtx* cctx,
        const void* dict,
        size_t dictSize,
        int compressionLevel
    ) nogil

    size_t ZSTD_compressBegin_advanced(
        ZSTD_CCtx* cctx,
        const void* dict,
        size_t dictSize,
        ZSTD_parameters params,
        unsigned long long pledgedSrcSize
    ) nogil

    size_t ZSTD_compressBegin_usingCDict(
        ZSTD_CCtx* cctx,
        const ZSTD_CDict* cdict
    ) nogil

    size_t ZSTD_compressBegin_usingCDict_advanced(
        ZSTD_CCtx* const cctx,
        const ZSTD_CDict* const cdict,
        ZSTD_frameParameters fParams,
        unsigned long long pledgedSrcSize
    ) nogil

    size_t ZSTD_copyCCtx(
        ZSTD_CCtx* cctx,
        const ZSTD_CCtx* preparedCCtx,
        unsigned long long pledgedSrcSize
    ) nogil

    size_t ZSTD_compressContinue(
        ZSTD_CCtx* cctx,
        void* dst,
        size_t dstCapacity,
        const void* src,
        size_t srcSize
    ) nogil

    size_t ZSTD_compressEnd(
        ZSTD_CCtx* cctx,
        void* dst,
        size_t dstCapacity,
        const void* src,
        size_t srcSize
    ) nogil

    ctypedef enum ZSTD_frameType_e:
        ZSTD_frame
        ZSTD_skippableFrame

    ctypedef struct ZSTD_frameHeader:
        unsigned long long frameContentSize
        unsigned long long windowSize
        unsigned blockSizeMax
        ZSTD_frameType_e frameType
        unsigned headerSize
        unsigned dictID
        unsigned checksumFlag

    size_t ZSTD_getFrameHeader(
        ZSTD_frameHeader* zfhPtr,
        const void* src,
        size_t srcSize
    ) nogil

    size_t ZSTD_getFrameHeader_advanced(
        ZSTD_frameHeader* zfhPtr,
        const void* src,
        size_t srcSize,
        ZSTD_format_e format
    ) nogil

    size_t ZSTD_decodingBufferSize_min(
        unsigned long long windowSize,
         unsigned long long frameContentSize
    ) nogil

    size_t ZSTD_decompressBegin(
        ZSTD_DCtx* dctx) nogil

    size_t ZSTD_decompressBegin_usingDict(
        ZSTD_DCtx* dctx,
        const void* dict,
        size_t dictSize
    ) nogil

    size_t ZSTD_decompressBegin_usingDDict(
        ZSTD_DCtx* dctx,
        const ZSTD_DDict* ddict
    ) nogil

    size_t ZSTD_nextSrcSizeToDecompress(
        ZSTD_DCtx* dctx
    ) nogil

    size_t ZSTD_decompressContinue(
        ZSTD_DCtx* dctx,
        void* dst,
        size_t dstCapacity,
        const void* src,
        size_t srcSize
    ) nogil

    void ZSTD_copyDCtx(
        ZSTD_DCtx* dctx,
        const ZSTD_DCtx* preparedDCtx
    ) nogil

    ctypedef enum ZSTD_nextInputType_e:
        ZSTDnit_frameHeader
        ZSTDnit_blockHeader
        ZSTDnit_block
        ZSTDnit_lastBlock
        ZSTDnit_checksum
        ZSTDnit_skippableFrame

    ZSTD_nextInputType_e ZSTD_nextInputType(
        ZSTD_DCtx* dctx
    ) nogil

    size_t ZSTD_getBlockSize(
        const ZSTD_CCtx* cctx) nogil

    size_t ZSTD_compressBlock(
        ZSTD_CCtx* cctx,
        void* dst,
        size_t dstCapacity,
        const void* src,
        size_t srcSize
    ) nogil

    size_t ZSTD_decompressBlock(
        ZSTD_DCtx* dctx,
        void* dst,
        size_t dstCapacity,
        const void* src,
        size_t srcSize
    ) nogil

    size_t ZSTD_insertBlock(
        ZSTD_DCtx* dctx,
        const void* blockStart,
        size_t blockSize
    ) nogil
