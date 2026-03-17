# imagecodecs/wic.pxd

# Cython declarations for the `wic.cpp` library.

from libc.stddef cimport size_t
from libc.stdint cimport int32_t, uint8_t, uint32_t


cdef extern from 'wic.h' nogil:

    ctypedef struct wic_decode_result_t:
        uint8_t* data
        uint32_t width
        uint32_t height
        uint32_t stride
        uint32_t components
        uint32_t bpc
        uint32_t frame_count

    int32_t wic_factory_init()
    void wic_factory_destroy()

    int32_t wic_get_info(
        const uint8_t* src,
        size_t srcsize,
        uint32_t* width,
        uint32_t* height,
        uint32_t* components,
        uint32_t* bpc,
        uint32_t* frame_count
    )

    int32_t wic_copy_pixels(
        const uint8_t* src,
        size_t srcsize,
        uint32_t frame_index,
        uint8_t* dst,
        uint32_t dst_stride,
        size_t dst_size
    )

    int32_t wic_decode_impl "wic_decode_" (
        const uint8_t* src,
        size_t srcsize,
        uint32_t frame_index,
        wic_decode_result_t* result
    )

    int32_t wic_check_impl "wic_check_" (
        const uint8_t* src,
        size_t srcsize,
    )

    void wic_decode_free(uint8_t* data)

    int WIC_FORMAT_BMP
    int WIC_FORMAT_PNG
    int WIC_FORMAT_JPEG
    int WIC_FORMAT_TIFF
    int WIC_FORMAT_GIF
    int WIC_FORMAT_WMP
    int WIC_FORMAT_HEIF
    int WIC_FORMAT_WEBP

    int32_t wic_encode_impl "wic_encode_" (
        const uint8_t* src,
        uint32_t width,
        uint32_t height,
        uint32_t components,
        uint32_t bpc,
        int32_t format,
        int32_t quality,
        uint8_t** dst,
        size_t* dstsize,
    )

    void wic_encode_free(uint8_t* data)

    const char* wic_version_string()
