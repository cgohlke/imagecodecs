# -*- coding: utf-8 -*-
# libpng.pxd
# cython: language_level = 3

# Cython declarations for the `libpng 1.6.37` library.
# https://github.com/glennrp/libpng

cdef extern from 'png.h':

    char* PNG_LIBPNG_VER_STRING

    int PNG_COLOR_TYPE_GRAY
    int PNG_COLOR_TYPE_PALETTE
    int PNG_COLOR_TYPE_RGB
    int PNG_COLOR_TYPE_RGB_ALPHA
    int PNG_COLOR_TYPE_GRAY_ALPHA
    int PNG_INTERLACE_NONE
    int PNG_COMPRESSION_TYPE_DEFAULT
    int PNG_FILTER_TYPE_DEFAULT
    int PNG_ZBUF_SIZE

    ctypedef struct png_struct:
        pass

    ctypedef struct png_info:
        pass

    ctypedef size_t png_size_t
    ctypedef unsigned int png_uint_32
    ctypedef unsigned char *png_bytep
    ctypedef unsigned char *png_const_bytep
    ctypedef unsigned char **png_bytepp
    ctypedef char *png_charp
    ctypedef char *png_const_charp
    ctypedef void *png_voidp
    ctypedef png_struct *png_structp
    ctypedef png_struct *png_structrp
    ctypedef png_struct *png_const_structp
    ctypedef png_struct *png_const_structrp
    ctypedef png_struct **png_structpp
    ctypedef png_info *png_infop
    ctypedef png_info *png_inforp
    ctypedef png_info *png_const_infop
    ctypedef png_info *png_const_inforp
    ctypedef png_info **png_infopp
    ctypedef void(*png_error_ptr)(png_structp, png_const_charp)
    ctypedef void(*png_rw_ptr)(png_structp, png_bytep, size_t)
    ctypedef void(*png_flush_ptr)(png_structp)
    ctypedef void(*png_read_status_ptr)(png_structp, png_uint_32, int)
    ctypedef void(*png_write_status_ptr)(png_structp, png_uint_32, int)

    int png_sig_cmp(
        png_const_bytep sig,
        size_t start,
        size_t num_to_check) nogil

    void png_set_sig_bytes(png_structrp png_ptr, int num_bytes) nogil

    png_uint_32 png_get_IHDR(
        png_const_structrp png_ptr,
        png_const_inforp info_ptr,
        png_uint_32 *width,
        png_uint_32 *height,
        int *bit_depth,
        int *color_type,
        int *interlace_method,
        int *compression_method,
        int *filter_method) nogil

    void png_set_IHDR(
        png_const_structrp png_ptr,
        png_inforp info_ptr,
        png_uint_32 width,
        png_uint_32 height,
        int bit_depth,
        int color_type,
        int interlace_method,
        int compression_method,
        int filter_method) nogil

    void png_read_row(
        png_structrp png_ptr,
        png_bytep row,
        png_bytep display_row) nogil

    void png_write_row(png_structrp png_ptr, png_const_bytep row) nogil
    void png_read_image(png_structrp png_ptr, png_bytepp image) nogil
    void png_write_image(png_structrp png_ptr, png_bytepp image) nogil
    png_infop png_create_info_struct(const png_const_structrp png_ptr) nogil

    png_structp png_create_write_struct(
        png_const_charp user_png_ver,
        png_voidp error_ptr,
        png_error_ptr error_fn,
        png_error_ptr warn_fn) nogil

    png_structp png_create_read_struct(
        png_const_charp user_png_ver,
        png_voidp error_ptr,
        png_error_ptr error_fn,
        png_error_ptr warn_fn) nogil

    void png_destroy_write_struct(
        png_structpp png_ptr_ptr,
        png_infopp info_ptr_ptr) nogil

    void png_destroy_read_struct(
        png_structpp png_ptr_ptr,
        png_infopp info_ptr_ptr,
        png_infopp end_info_ptr_ptr) nogil

    void png_set_write_fn(
        png_structrp png_ptr,
        png_voidp io_ptr,
        png_rw_ptr write_data_fn,
        png_flush_ptr output_flush_fn) nogil

    void png_set_read_fn(
        png_structrp png_ptr,
        png_voidp io_ptr,
        png_rw_ptr read_data_fn) nogil

    png_voidp png_get_io_ptr(png_const_structrp png_ptr) nogil
    void png_set_palette_to_rgb(png_structrp png_ptr) nogil
    void png_set_expand_gray_1_2_4_to_8(png_structrp png_ptr) nogil
    void png_read_info(png_structrp png_ptr, png_inforp info_ptr) nogil
    void png_write_info(png_structrp png_ptr, png_const_inforp info_ptr) nogil
    void png_write_end(png_structrp png_ptr, png_inforp info_ptr) nogil
    void png_read_update_info(png_structrp png_ptr, png_inforp info_ptr) nogil
    void png_set_expand_16(png_structrp png_ptr) nogil
    void png_set_swap(png_structrp png_ptr) nogil
    void png_set_compression_level(png_structrp png_ptr, int level) nogil
