# imagecodecs/brotli.pxd
# cython: language_level = 3

# Cython declarations for the `Brotli 1.1.0` library.
# https://github.com/google/brotli

from libc.stdint cimport uint8_t, uint32_t

cdef extern from 'brotli/types.h' nogil:

    ctypedef int BROTLI_BOOL

    BROTLI_BOOL BROTLI_TRUE
    BROTLI_BOOL BROTLI_FALSE

    ctypedef void* (*brotli_alloc_func)(
        void* opaque,
        size_t size
    ) nogil

    ctypedef void (*brotli_free_func)(
        void* opaque,
        void* address
    ) nogil


cdef extern from 'brotli/decode.h' nogil:

    ctypedef enum BrotliDecoderErrorCode:
        pass

    ctypedef struct BrotliDecoderState:
        pass

    ctypedef enum BrotliDecoderResult:
        BROTLI_DECODER_RESULT_ERROR
        BROTLI_DECODER_RESULT_SUCCESS
        BROTLI_DECODER_RESULT_NEEDS_MORE_INPUT
        BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT

    ctypedef enum BrotliDecoderParameter:
        BROTLI_DECODER_PARAM_DISABLE_RING_BUFFER_REALLOCATION
        BROTLI_DECODER_PARAM_LARGE_WINDOW

    BROTLI_BOOL BrotliDecoderSetParameter(
        BrotliDecoderState* state,
        BrotliDecoderParameter param,
        uint32_t value
    )

    ctypedef struct BrotliSharedDictionaryType:
        pass

    BROTLI_BOOL BrotliDecoderAttachDictionary(
        BrotliDecoderState* state,
        BrotliSharedDictionaryType type_,
        size_t data_size,
        const uint8_t *data
    )

    BrotliDecoderState* BrotliDecoderCreateInstance(
        brotli_alloc_func alloc_func,
        brotli_free_func free_func,
        void* opaque
    )

    void BrotliDecoderDestroyInstance(
        BrotliDecoderState* state
    )

    BrotliDecoderResult BrotliDecoderDecompress(
        size_t encoded_size,
        const uint8_t* encoded_buffer,
        size_t* decoded_size,
        uint8_t* decoded_buffer
    )

    BrotliDecoderResult BrotliDecoderDecompressStream(
        BrotliDecoderState* state,
        size_t* available_in,
        const uint8_t** next_in,
        size_t* available_out,
        uint8_t** next_out,
        size_t* total_out
    )

    BROTLI_BOOL BrotliDecoderHasMoreOutput(
        const BrotliDecoderState* state
    )

    const uint8_t* BrotliDecoderTakeOutput(
        BrotliDecoderState* state,
        size_t* size
    )

    BROTLI_BOOL BrotliDecoderIsUsed(
        const BrotliDecoderState* state
    )

    BROTLI_BOOL BrotliDecoderIsFinished(
        const BrotliDecoderState* state
    )

    BrotliDecoderErrorCode BrotliDecoderGetErrorCode(
        const BrotliDecoderState* state
    )

    const char* BrotliDecoderErrorString(
        BrotliDecoderErrorCode c
    )

    uint32_t BrotliDecoderVersion()

    ctypedef void (*brotli_decoder_metadata_start_func)(
        void* opaque,
        size_t size
    ) nogil

    ctypedef void (*brotli_decoder_metadata_chunk_func)(
        void* opaque,
        const uint8_t* data,
        size_t size
    ) nogil

    void BrotliDecoderSetMetadataCallbacks(
        BrotliDecoderState* state,
        brotli_decoder_metadata_start_func start_func,
        brotli_decoder_metadata_chunk_func chunk_func, void* opaque
    )


cdef extern from 'brotli/encode.h' nogil:

    int BROTLI_MIN_WINDOW_BITS
    int BROTLI_MAX_WINDOW_BITS
    int BROTLI_LARGE_MAX_WINDOW_BITS
    int BROTLI_MIN_INPUT_BLOCK_BITS
    int BROTLI_MAX_INPUT_BLOCK_BITS
    int BROTLI_MIN_QUALITY
    int BROTLI_MAX_QUALITY

    int BROTLI_DEFAULT_QUALITY
    int BROTLI_DEFAULT_WINDOW
    int BROTLI_DEFAULT_MODE

    ctypedef enum BrotliEncoderMode:
        BROTLI_MODE_GENERIC
        BROTLI_MODE_TEXT
        BROTLI_MODE_FONT

    ctypedef enum BrotliEncoderOperation:
        BROTLI_OPERATION_PROCESS
        BROTLI_OPERATION_FLUSH
        BROTLI_OPERATION_FINISH
        BROTLI_OPERATION_EMIT_METADATA

    ctypedef enum BrotliEncoderParameter:
        BROTLI_PARAM_MODE
        BROTLI_PARAM_QUALITY
        BROTLI_PARAM_LGWIN
        BROTLI_PARAM_LGBLOCK
        BROTLI_PARAM_DISABLE_LITERAL_CONTEXT_MODELING
        BROTLI_PARAM_SIZE_HINT
        BROTLI_PARAM_LARGE_WINDOW
        BROTLI_PARAM_NPOSTFIX
        BROTLI_PARAM_NDIRECT

    ctypedef struct BrotliEncoderState:
        pass

    BROTLI_BOOL BrotliEncoderSetParameter(
        BrotliEncoderState* state,
        BrotliEncoderParameter param,
        uint32_t value
    )

    BrotliEncoderState* BrotliEncoderCreateInstance(
        brotli_alloc_func alloc_func,
        brotli_free_func free_func,
        void* opaque
    )

    void BrotliEncoderDestroyInstance(
        BrotliEncoderState* state
    )

    ctypedef struct BrotliEncoderPreparedDictionary:
        pass

    BrotliEncoderPreparedDictionary* BrotliEncoderPrepareDictionary(
        BrotliSharedDictionaryType type_,
        size_t data_size,
        const uint8_t* data,
        int quality,
        brotli_alloc_func alloc_func,
        brotli_free_func free_func,
        void* opaque
    )

    void BrotliEncoderDestroyPreparedDictionary(
        BrotliEncoderPreparedDictionary* dictionary
    )

    BROTLI_BOOL BrotliEncoderAttachPreparedDictionary(
        BrotliEncoderState* state,
        const BrotliEncoderPreparedDictionary* dictionary
    )

    size_t BrotliEncoderMaxCompressedSize(
        size_t input_size
    )

    BROTLI_BOOL BrotliEncoderCompress(
        int quality,
        int lgwin,
        BrotliEncoderMode mode,
        size_t input_size,
        const uint8_t* input_buffer,
        size_t* encoded_size,
        uint8_t* encoded_buffer
    )

    BROTLI_BOOL BrotliEncoderCompressStream(
        BrotliEncoderState* state,
        BrotliEncoderOperation op,
        size_t* available_in,
        const uint8_t** next_in,
        size_t* available_out,
        uint8_t** next_out,
        size_t* total_out
    )

    BROTLI_BOOL BrotliEncoderIsFinished(
        BrotliEncoderState* state
    )

    BROTLI_BOOL BrotliEncoderHasMoreOutput(
        BrotliEncoderState* state
    )

    const uint8_t* BrotliEncoderTakeOutput(
        BrotliEncoderState* state,
        size_t* size
    )

    size_t BrotliEncoderEstimatePeakMemoryUsage(
        int quality,
        int lgwin,
        size_t input_size
    )

    size_t BrotliEncoderGetPreparedDictionarySize(
        const BrotliEncoderPreparedDictionary* dictionary
    )

    uint32_t BrotliEncoderVersion(
    )
