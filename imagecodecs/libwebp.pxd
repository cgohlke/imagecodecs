# -*- coding: utf-8 -*-
# libwebp.pxd
# cython: language_level = 3

# Cython declarations for the `libwebp 1.0.3` library.
# https://github.com/webmproject/libwebp

from libc.stdint cimport uint8_t, uint32_t

cdef extern from 'webp/decode.h':

    ctypedef enum VP8StatusCode:
        VP8_STATUS_OK
        VP8_STATUS_OUT_OF_MEMORY
        VP8_STATUS_INVALID_PARAM
        VP8_STATUS_BITSTREAM_ERROR
        VP8_STATUS_UNSUPPORTED_FEATURE
        VP8_STATUS_SUSPENDED
        VP8_STATUS_USER_ABORT
        VP8_STATUS_NOT_ENOUGH_DATA

    ctypedef struct WebPBitstreamFeatures:
        int width
        int height
        int has_alpha
        int has_animation
        int format
        uint32_t[5] pad

    int WebPGetDecoderVersion() nogil

    int WebPGetInfo(
        const uint8_t *data,
        size_t data_size,
        int *width,
        int *height) nogil

    VP8StatusCode WebPGetFeatures(
        const uint8_t *data,
        size_t data_size,
        WebPBitstreamFeatures *features) nogil

    uint8_t* WebPDecodeRGBAInto(
        const uint8_t *data,
        size_t data_size,
        uint8_t *output_buffer,
        size_t output_buffer_size,
        int output_stride) nogil

    uint8_t* WebPDecodeRGBInto(
        const uint8_t *data,
        size_t data_size,
        uint8_t *output_buffer,
        size_t output_buffer_size,
        int output_stride) nogil

    uint8_t* WebPDecodeYUVInto(
        const uint8_t *data,
        size_t data_size,
        uint8_t *luma,
        size_t luma_size,
        int luma_stride,
        uint8_t *u,
        size_t u_size,
        int u_stride,
        uint8_t *v,
        size_t v_size,
        int v_stride) nogil


cdef extern from 'webp/encode.h':

    int WEBP_MAX_DIMENSION

    int WebPGetEncoderVersion() nogil
    void WebPFree(void *ptr) nogil

    size_t WebPEncodeRGB(
        const uint8_t *rgb,
        int width,
        int height,
        int stride,
        float quality_factor,
        uint8_t **output) nogil

    size_t WebPEncodeRGBA(
        const uint8_t *rgba,
        int width,
        int height,
        int stride,
        float quality_factor,
        uint8_t **output) nogil

    size_t WebPEncodeLosslessRGB(
        const uint8_t *rgb,
        int width,
        int height,
        int stride,
        uint8_t **output) nogil

    size_t WebPEncodeLosslessRGBA(
        const uint8_t *rgba,
        int width,
        int height,
        int stride,
        uint8_t **output) nogil
