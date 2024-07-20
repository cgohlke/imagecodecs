# imagecodecs/libaec.pxd
# cython: language_level = 3

# Cython declarations for the `libaec 1.1.2` library.
# https://gitlab.dkrz.de/k202009/libaec

cdef extern from 'libaec.h' nogil:

    int AEC_VERSION_MAJOR
    int AEC_VERSION_MINOR
    int AEC_VERSION_PATCH
    char* AEC_VERSION_STR

    int AEC_DATA_SIGNED
    int AEC_DATA_3BYTE
    int AEC_DATA_MSB
    int AEC_DATA_PREPROCESS
    int AEC_RESTRICTED
    int AEC_PAD_RSI
    int AEC_NOT_ENFORCE

    int AEC_OK
    int AEC_CONF_ERROR
    int AEC_STREAM_ERROR
    int AEC_DATA_ERROR
    int AEC_MEM_ERROR
    int AEC_RSI_OFFSETS_ERROR

    int AEC_NO_FLUSH
    int AEC_FLUSH

    struct aec_stream:
        unsigned char* next_in
        size_t avail_in
        size_t total_in
        unsigned char* next_out
        size_t avail_out
        size_t total_out
        unsigned int bits_per_sample
        unsigned int block_size
        unsigned int rsi
        unsigned int flags
        # internal_state* state

    int aec_encode_init(
        aec_stream* strm
    )

    int aec_encode_enable_offsets(
        aec_stream *strm
    )

    int aec_encode_count_offsets(
        aec_stream *strm,
        size_t *rsi_offsets_count
    )

    int aec_encode_get_offsets(
        aec_stream *strm,
        size_t *rsi_offsets,
        size_t rsi_offsets_count
    )

    int aec_buffer_seek(
        aec_stream *strm,
        size_t offset
    )

    int aec_encode_c 'aec_encode' (
        aec_stream* strm,
        int flush
    )

    int aec_encode_end(
        aec_stream* strm
    )

    int aec_decode_init(
        aec_stream* strm
    )

    int aec_decode_enable_offsets(
        aec_stream *strm
    )

    int aec_decode_count_offsets(
        aec_stream *strm,
        size_t *rsi_offsets_count
    )

    int aec_decode_get_offsets(
        aec_stream *strm,
        size_t *rsi_offsets,
        size_t rsi_offsets_count
    )

    int aec_decode_c 'aec_decode' (
        aec_stream* strm,
        int flush
    )

    int aec_decode_range(
        aec_stream *strm,
        const size_t *rsi_offsets,
        size_t rsi_offsets_count,
        size_t pos,
        size_t size
    )

    int aec_decode_end(
        aec_stream* strm
    )

    int aec_buffer_encode(
        aec_stream* strm
    )

    int aec_buffer_decode(
        aec_stream* strm
    )
