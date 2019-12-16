# -*- coding: utf-8 -*-
# liblzma.pxd
# cython: language_level = 3

# Cython declarations for the `liblzma 5.2.4` library.
# https://github.com/xz-mirror/xz

from libc.stdint cimport uint8_t, uint32_t, uint64_t

cdef extern from 'lzma.h':

    int LZMA_VERSION_MAJOR
    int LZMA_VERSION_MINOR
    int LZMA_VERSION_PATCH

    int LZMA_CONCATENATED
    int LZMA_STREAM_HEADER_SIZE

    ctypedef uint64_t lzma_vli

    ctypedef struct lzma_stream_flags:
        uint32_t version
        lzma_vli backward_size

    ctypedef struct lzma_index:
        pass

    ctypedef struct lzma_allocator:
        pass

    ctypedef struct lzma_internal:
        pass

    ctypedef enum lzma_reserved_enum:
        LZMA_RESERVED_ENUM

    ctypedef enum lzma_check:
        LZMA_CHECK_NONE
        LZMA_CHECK_CRC32
        LZMA_CHECK_CRC64
        LZMA_CHECK_SHA256

    ctypedef struct lzma_stream:
        uint8_t *next_in
        size_t avail_in
        uint64_t total_in
        uint8_t *next_out
        size_t avail_out
        uint64_t total_out
        lzma_allocator *allocator
        lzma_internal *internal
        void *reserved_ptr1
        void *reserved_ptr2
        void *reserved_ptr3
        void *reserved_ptr4
        uint64_t reserved_int1
        uint64_t reserved_int2
        size_t reserved_int3
        size_t reserved_int4
        lzma_reserved_enum reserved_enum1
        lzma_reserved_enum reserved_enum2

    ctypedef enum lzma_action:
        LZMA_RUN
        LZMA_SYNC_FLUSH
        LZMA_FULL_FLUSH
        LZMA_FULL_BARRIER
        LZMA_FINISH

    ctypedef enum lzma_ret:
        LZMA_OK
        LZMA_STREAM_END
        LZMA_NO_CHECK
        LZMA_UNSUPPORTED_CHECK
        LZMA_GET_CHECK
        LZMA_MEM_ERROR
        LZMA_MEMLIMIT_ERROR
        LZMA_FORMAT_ERROR
        LZMA_OPTIONS_ERROR
        LZMA_DATA_ERROR
        LZMA_BUF_ERROR
        LZMA_PROG_ERROR

    lzma_ret lzma_easy_encoder(
        lzma_stream *strm,
        uint32_t preset,
        lzma_check check) nogil

    lzma_ret lzma_stream_decoder(
        lzma_stream *strm,
        uint64_t memlimit,
        uint32_t flags) nogil

    lzma_ret lzma_stream_footer_decode(
        lzma_stream_flags *options,
        const uint8_t *in_) nogil

    lzma_ret lzma_index_buffer_decode(
        lzma_index **i,
        uint64_t *memlimit,
        const lzma_allocator *allocator,
        const uint8_t *in_,
        size_t *in_pos,
        size_t in_size) nogil

    lzma_ret lzma_code(lzma_stream *strm, lzma_action action) nogil
    void lzma_end(lzma_stream *strm) nogil
    size_t lzma_stream_buffer_bound(size_t uncompressed_size) nogil
    lzma_vli lzma_index_uncompressed_size(const lzma_index *i) nogil
    lzma_index* lzma_index_init(const lzma_allocator *allocator) nogil
    void lzma_index_end(lzma_index *i, const lzma_allocator *allocator) nogil
