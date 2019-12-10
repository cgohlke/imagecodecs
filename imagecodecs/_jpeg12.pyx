# -*- coding: utf-8 -*-
# _jpeg12.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2018-2019, Christoph Gohlke
# Copyright (c) 2018-2019, The Regents of the University of California
# Produced at the Laboratory for Fluorescence Dynamics.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""JPEG 12-bit codec for the imagecodecs package.

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics. University of California, Irvine

:License: BSD 3-Clause

:Version: 2019.12.10

"""

__version__ = '2019.12.10'

import numbers
import numpy

cimport cython
cimport numpy

from cpython.bytearray cimport PyByteArray_FromStringAndSize
from cpython.bytes cimport PyBytes_FromStringAndSize
from cython.operator cimport dereference as deref

from libc.string cimport memset, memcpy
from libc.stdlib cimport malloc, free
from libc.setjmp cimport setjmp, longjmp, jmp_buf
from libc.stdint cimport uint8_t, uint16_t

numpy.import_array()


# JPEG 12-bit #################################################################

cdef extern from 'jpeglib.h':

    int JPEG_LIB_VERSION
    int LIBJPEG_TURBO_VERSION
    int LIBJPEG_TURBO_VERSION_NUMBER

    ctypedef void noreturn_t
    ctypedef int boolean
    ctypedef char JOCTET
    ctypedef unsigned int JDIMENSION
    ctypedef unsigned short JSAMPLE
    ctypedef JSAMPLE* JSAMPROW
    ctypedef JSAMPROW* JSAMPARRAY

    ctypedef enum J_COLOR_SPACE:
        JCS_UNKNOWN
        JCS_GRAYSCALE
        JCS_RGB
        JCS_YCbCr
        JCS_CMYK
        JCS_YCCK
        JCS_EXT_RGB
        JCS_EXT_RGBX
        JCS_EXT_BGR
        JCS_EXT_BGRX
        JCS_EXT_XBGR
        JCS_EXT_XRGB
        JCS_EXT_RGBA
        JCS_EXT_BGRA
        JCS_EXT_ABGR
        JCS_EXT_ARGB
        JCS_RGB565

    ctypedef enum J_DITHER_MODE:
        JDITHER_NONE
        JDITHER_ORDERED
        JDITHER_FS

    ctypedef enum J_DCT_METHOD:
        JDCT_ISLOW
        JDCT_IFAST
        JDCT_FLOAT

    struct jpeg_source_mgr:
        pass

    struct jpeg_destination_mgr:
        pass

    struct jpeg_error_mgr:
        int msg_code
        const char** jpeg_message_table
        noreturn_t error_exit(jpeg_common_struct*)
        void output_message(jpeg_common_struct*)

    struct jpeg_common_struct:
        jpeg_error_mgr* err

    struct jpeg_component_info:
        int component_id
        int component_index
        int h_samp_factor
        int v_samp_factor

    struct jpeg_decompress_struct:
        jpeg_error_mgr* err
        void* client_data
        jpeg_source_mgr* src
        JDIMENSION image_width
        JDIMENSION image_height
        JDIMENSION output_width
        JDIMENSION output_height
        JDIMENSION output_scanline
        J_COLOR_SPACE jpeg_color_space
        J_COLOR_SPACE out_color_space
        J_DCT_METHOD dct_method
        J_DITHER_MODE dither_mode
        boolean buffered_image
        boolean raw_data_out
        boolean do_fancy_upsampling
        boolean do_block_smoothing
        boolean quantize_colors
        boolean two_pass_quantize
        unsigned int scale_num
        unsigned int scale_denom
        int num_components
        int out_color_components
        int output_components
        int rec_outbuf_height
        int desired_number_of_colors
        int actual_number_of_colors
        int data_precision
        double output_gamma

    struct jpeg_compress_struct:
        jpeg_error_mgr* err
        void* client_data
        jpeg_destination_mgr *dest
        JDIMENSION image_width
        JDIMENSION image_height
        int input_components
        J_COLOR_SPACE in_color_space
        J_COLOR_SPACE jpeg_color_space
        double input_gamma
        int data_precision
        int num_components
        int smoothing_factor
        boolean optimize_coding
        JDIMENSION next_scanline
        boolean progressive_mode
        jpeg_component_info *comp_info
        # JPEG_LIB_VERSION >= 70
        # unsigned int scale_num
        # unsigned int scale_denom
        # JDIMENSION jpeg_width
        # JDIMENSION jpeg_height
        # boolean do_fancy_downsampling

    jpeg_error_mgr* jpeg_std_error(jpeg_error_mgr*) nogil

    void jpeg_create_decompress(jpeg_decompress_struct*) nogil

    void jpeg_destroy_decompress(jpeg_decompress_struct*) nogil

    int jpeg_read_header(jpeg_decompress_struct*, boolean) nogil

    boolean jpeg_start_decompress(jpeg_decompress_struct*) nogil

    boolean jpeg_finish_decompress(jpeg_decompress_struct*) nogil

    JDIMENSION jpeg_read_scanlines(
        jpeg_decompress_struct*,
        JSAMPARRAY,
        JDIMENSION) nogil

    void jpeg_mem_src(
        jpeg_decompress_struct*,
        unsigned char*,
        unsigned long) nogil

    void jpeg_mem_dest(
        jpeg_compress_struct*,
        unsigned char**,
        unsigned long*) nogil

    void jpeg_create_compress(jpeg_compress_struct*) nogil

    void jpeg_destroy_compress(jpeg_compress_struct*) nogil

    void jpeg_set_defaults(jpeg_compress_struct*) nogil

    void jpeg_set_quality(jpeg_compress_struct*, int, boolean) nogil

    void jpeg_start_compress(jpeg_compress_struct*, boolean) nogil

    void jpeg_finish_compress(jpeg_compress_struct* cinfo) nogil

    JDIMENSION jpeg_write_scanlines(
        jpeg_compress_struct*,
        JSAMPARRAY,
        JDIMENSION) nogil


ctypedef struct my_error_mgr:
    jpeg_error_mgr pub
    jmp_buf setjmp_buffer


cdef void my_error_exit(jpeg_common_struct* cinfo):
    cdef my_error_mgr* error = <my_error_mgr*> deref(cinfo).err
    longjmp(deref(error).setjmp_buffer, 1)


cdef void my_output_message(jpeg_common_struct* cinfo):
    pass


def _jcs_colorspace(colorspace):
    """Return JCS colorspace value from user input."""
    return {
        'GRAY': JCS_GRAYSCALE,
        'GRAYSCALE': JCS_GRAYSCALE,
        'MINISWHITE': JCS_GRAYSCALE,
        'MINISBLACK': JCS_GRAYSCALE,
        'RGB': JCS_RGB,
        'RGBA': JCS_EXT_RGBA,
        'CMYK': JCS_CMYK,
        'YCCK': JCS_YCCK,
        'YCBCR': JCS_YCbCr,
        'UNKNOWN': JCS_UNKNOWN,
        None: JCS_UNKNOWN,
        JCS_UNKNOWN: JCS_UNKNOWN,
        JCS_GRAYSCALE: JCS_GRAYSCALE,
        JCS_RGB: JCS_RGB,
        JCS_YCbCr: JCS_YCbCr,
        JCS_CMYK: JCS_CMYK,
        JCS_YCCK: JCS_YCCK,
        JCS_EXT_RGB: JCS_EXT_RGB,
        JCS_EXT_RGBX: JCS_EXT_RGBX,
        JCS_EXT_BGR: JCS_EXT_BGR,
        JCS_EXT_BGRX: JCS_EXT_BGRX,
        JCS_EXT_XBGR: JCS_EXT_XBGR,
        JCS_EXT_XRGB: JCS_EXT_XRGB,
        JCS_EXT_RGBA: JCS_EXT_RGBA,
        JCS_EXT_BGRA: JCS_EXT_BGRA,
        JCS_EXT_ABGR: JCS_EXT_ABGR,
        JCS_EXT_ARGB: JCS_EXT_ARGB,
        JCS_RGB565: JCS_RGB565,
        }.get(colorspace, JCS_UNKNOWN)


def _jcs_colorspace_samples(colorspace):
    """Return expected number of samples in colorspace."""
    three = (3,)
    four = (4,)
    return {
        JCS_UNKNOWN: (1, 2, 3, 4),
        JCS_GRAYSCALE: (1,),
        JCS_RGB: three,
        JCS_YCbCr: three,
        JCS_CMYK: four,
        JCS_YCCK: four,
        JCS_EXT_RGB: three,
        JCS_EXT_RGBX: four,
        JCS_EXT_BGR: three,
        JCS_EXT_BGRX: four,
        JCS_EXT_XBGR: four,
        JCS_EXT_XRGB: four,
        JCS_EXT_RGBA: four,
        JCS_EXT_BGRA: four,
        JCS_EXT_ABGR: four,
        JCS_EXT_ARGB: four,
        JCS_RGB565: three,
        }[colorspace]


class Jpeg12Error(RuntimeError):
    """JPEG 12-bit Exceptions."""


def jpeg12_encode(data, level=None, colorspace=None, outcolorspace=None,
                  subsampling=None, optimize=None, smoothing=None, out=None):
    """Return JPEG 12-bit image from numpy array.

    """
    cdef:
        numpy.ndarray src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        ssize_t srcsize = src.size * src.itemsize
        ssize_t rowstride = src.strides[0]
        int samples = <int>src.shape[2] if src.ndim == 3 else 1
        int quality = _default_level(level, 90, 0, 100)
        my_error_mgr err
        jpeg_compress_struct cinfo
        JSAMPROW rowpointer
        J_COLOR_SPACE in_color_space = JCS_UNKNOWN
        J_COLOR_SPACE jpeg_color_space = JCS_UNKNOWN
        unsigned long outsize = 0
        unsigned char* outbuffer = NULL
        const char* msg
        int h_samp_factor = 0
        int v_samp_factor = 0
        int smoothing_factor = _default_level(smoothing, -1, 0, 100)
        int optimize_coding = -1 if optimize is None else 1 if optimize else 0

    if data is out:
        raise ValueError('cannot encode in-place')

    if not (data.dtype == numpy.uint16
            and data.ndim in (2, 3)
            # and data.size * data.itemsize < 2**31-1  # limit to 2 GB
            and samples in (1, 3, 4)
            and data.strides[data.ndim-1] == data.itemsize
            and (data.ndim == 2 or data.strides[1] == samples*data.itemsize)):
        raise ValueError('invalid input shape, strides, or dtype')

    if not _check_12bit(data):
        # values larger than 12-bit cause segfault
        raise ValueError('all data values must be < 4096')

    if colorspace is None:
        if samples == 1:
            in_color_space = JCS_GRAYSCALE
        elif samples == 3:
            in_color_space = JCS_RGB
        # elif samples == 4:
        #     in_color_space = JCS_CMYK
        else:
            in_color_space = JCS_UNKNOWN
    else:
        in_color_space = _jcs_colorspace(colorspace)
        if samples not in _jcs_colorspace_samples(in_color_space):
            raise ValueError('invalid input shape')

    jpeg_color_space = _jcs_colorspace(outcolorspace)

    if jpeg_color_space == JCS_YCbCr and subsampling is not None:
        if subsampling in ('444', (1, 1)):
            h_samp_factor = 1
            v_samp_factor = 1
        elif subsampling in ('422', (2, 1)):
            h_samp_factor = 2
            v_samp_factor = 1
        elif subsampling in ('420', (2, 2)):
            h_samp_factor = 2
            v_samp_factor = 2
        elif subsampling in ('411', (4, 1)):
            h_samp_factor = 4
            v_samp_factor = 1
        elif subsampling in ('440', (1, 2)):
            h_samp_factor = 1
            v_samp_factor = 2
        else:
            raise ValueError('invalid subsampling')

    out, dstsize, out_given, out_type = _parse_output(out)

    if out is None and dstsize > 0:
        if out_type is bytes:
            out = PyBytes_FromStringAndSize(NULL, dstsize)
        else:
            out = PyByteArray_FromStringAndSize(NULL, dstsize)

    if out is not None:
        dst = out
        dstsize = dst.size * dst.itemsize
        outsize = <unsigned long>dstsize
        outbuffer = <unsigned char*>&dst[0]

    with nogil:
        cinfo.err = jpeg_std_error(&err.pub)
        err.pub.error_exit = my_error_exit
        err.pub.output_message = my_output_message

        if setjmp(err.setjmp_buffer):
            jpeg_destroy_compress(&cinfo)
            msg = err.pub.jpeg_message_table[err.pub.msg_code]
            raise Jpeg12Error(msg.decode('utf-8'))

        jpeg_create_compress(&cinfo)

        cinfo.image_height = <JDIMENSION>src.shape[0]
        cinfo.image_width = <JDIMENSION>src.shape[1]
        cinfo.input_components = samples

        if in_color_space != JCS_UNKNOWN:
            cinfo.in_color_space = in_color_space
        if jpeg_color_space != JCS_UNKNOWN:
            cinfo.jpeg_color_space = jpeg_color_space

        jpeg_set_defaults(&cinfo)
        jpeg_mem_dest(&cinfo, &outbuffer, &outsize)  # must call after defaults
        jpeg_set_quality(&cinfo, quality, 1)

        if smoothing_factor >= 0:
            cinfo.smoothing_factor = smoothing_factor
        if optimize_coding >= 0:
            cinfo.optimize_coding = <boolean>optimize_coding
        if h_samp_factor != 0:
            cinfo.comp_info[0].h_samp_factor = h_samp_factor
            cinfo.comp_info[0].v_samp_factor = v_samp_factor
            cinfo.comp_info[1].h_samp_factor = 1
            cinfo.comp_info[1].v_samp_factor = 1
            cinfo.comp_info[2].h_samp_factor = 1
            cinfo.comp_info[2].v_samp_factor = 1

        # TODO: add option to use or return JPEG tables

        jpeg_start_compress(&cinfo, 1)

        while cinfo.next_scanline < cinfo.image_height:
            rowpointer = <JSAMPROW>(<char*>src.data
                                    + cinfo.next_scanline * rowstride)
            jpeg_write_scanlines(&cinfo, &rowpointer, 1)

        jpeg_finish_compress(&cinfo)
        jpeg_destroy_compress(&cinfo)

    if out is None or outbuffer != <unsigned char*>&dst[0]:
        if out_type is bytes:
            out = PyBytes_FromStringAndSize(<const char*>outbuffer,
                                            <ssize_t>outsize)
        else:
            out = PyByteArray_FromStringAndSize(<const char*>outbuffer,
                                                <ssize_t>outsize)
        free(outbuffer)
    elif outsize < dstsize:
        if out_given:
            out = memoryview(out)[:outsize]
        else:
            out = out[:outsize]

    return out


def jpeg12_decode(data, tables=None, colorspace=None, outcolorspace=None,
                  shape=None, out=None):
    """Decode JPEG 12-bit image to numpy array.

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        const uint8_t[::1] tables_
        unsigned long tablesize = 0
        ssize_t srcsize = src.size
        ssize_t dstsize
        ssize_t rowstride
        int numlines
        my_error_mgr err
        jpeg_decompress_struct cinfo
        JSAMPROW rowpointer
        J_COLOR_SPACE jpeg_color_space
        J_COLOR_SPACE out_color_space
        JDIMENSION width = 0
        JDIMENSION height = 0
        const char *msg

    if data is out:
        raise ValueError('cannot decode in-place')

    if srcsize > 2**32-1:
        # limit to 4 GB
        raise ValueError('data too large')

    jpeg_color_space = _jcs_colorspace(colorspace)
    if outcolorspace is None:
        out_color_space = jpeg_color_space
    else:
        out_color_space = _jcs_colorspace(outcolorspace)

    if tables is not None:
        tables_ = tables
        tablesize = tables_.size

    if shape is not None and (shape[0] >= 65500 or shape[1] >= 65500):
        # enable decoding of large (JPEG_MAX_DIMENSION <= 2^20) JPEG
        # when using a patched jibjpeg-turbo
        height = <JDIMENSION>shape[0]
        width = <JDIMENSION>shape[1]

    with nogil:

        cinfo.err = jpeg_std_error(&err.pub)
        err.pub.error_exit = my_error_exit
        err.pub.output_message = my_output_message
        if setjmp(err.setjmp_buffer):
            jpeg_destroy_decompress(&cinfo)
            msg = err.pub.jpeg_message_table[err.pub.msg_code]
            raise Jpeg12Error(msg.decode('utf-8'))

        jpeg_create_decompress(&cinfo)
        cinfo.do_fancy_upsampling = True
        if width > 0:
            cinfo.image_width = width
            cinfo.image_height = height

        if tablesize > 0:
            jpeg_mem_src(&cinfo, &tables_[0], tablesize)
            jpeg_read_header(&cinfo, 0)

        jpeg_mem_src(&cinfo, &src[0], <unsigned long>srcsize)
        jpeg_read_header(&cinfo, 1)

        if jpeg_color_space != JCS_UNKNOWN:
            cinfo.jpeg_color_space = jpeg_color_space
        if out_color_space != JCS_UNKNOWN:
            cinfo.out_color_space = out_color_space

        jpeg_start_decompress(&cinfo)

        with gil:
            # if (cinfo.output_components not in
            #         _jcs_colorspace_samples(out_color_space)):
            #     raise ValueError('invalid output shape')

            shape = cinfo.output_height, cinfo.output_width
            if cinfo.output_components > 1:
                shape += cinfo.output_components,

            out = _create_array(out, shape, numpy.uint16)  # TODO: strides
            dst = out
            dstsize = dst.size * dst.itemsize
            rowstride = dst.strides[0] // dst.itemsize

        memset(<void *>dst.data, 0, dstsize)
        rowpointer = <JSAMPROW>dst.data
        while cinfo.output_scanline < cinfo.output_height:
            jpeg_read_scanlines(&cinfo, &rowpointer, 1)
            rowpointer += rowstride

        jpeg_finish_decompress(&cinfo)
        jpeg_destroy_decompress(&cinfo)

    return out


def jpeg12_version():
    """Return JPEG 12-bit version string."""
    return 'jpeg12 %.1f' % (JPEG_LIB_VERSION / 10.0)


def jpeg_turbo_version():
    """Return JPEG version string."""
    jpeg_turbo_version = str(LIBJPEG_TURBO_VERSION_NUMBER)
    return 'jpeg12_turbo %i.%i.%i' % (
        int(jpeg_turbo_version[:1]),
        int(jpeg_turbo_version[3:4]),
        int(jpeg_turbo_version[6:]))


###############################################################################

cdef _create_array(out, shape, dtype, strides=None):
    """Return numpy array of shape and dtype from output argument."""
    if out is None or isinstance(out, numbers.Integral):
        out = numpy.empty(shape, dtype)
    elif isinstance(out, numpy.ndarray):
        if out.shape != shape:
            raise ValueError('invalid output shape')
        if out.itemsize != numpy.dtype(dtype).itemsize:
            raise ValueError('invalid output dtype')
        if strides is not None:
            for i, j in zip(strides, out.strides):
                if i is not None and i != j:
                    raise ValueError('invalid output strides')
        elif not numpy.PyArray_ISCONTIGUOUS(out):
            raise ValueError('output is not contiguous')
    else:
        dstsize = 1
        for i in shape:
            dstsize *= i
        out = numpy.frombuffer(out, dtype, dstsize)
        out.shape = shape
    return out


cdef _parse_output(out, ssize_t out_size=-1, out_given=False, out_type=bytes):
    """Return out, out_size, out_given, out_type from output argument."""
    if out is None:
        pass
    elif out is bytes:
        out = None
        out_type = bytes
    elif out is bytearray:
        out = None
        out_type = bytearray
    elif isinstance(out, numbers.Integral):
        out_size = out
        out = None
    else:
        # out_size = len(out)
        # out_type = type(out)
        out_given = True
    return out, out_size, out_given, out_type


def _default_level(level, default, smallest, largest):
    """Return compression level in range."""
    if level is None:
        level = default
    if largest is not None:
        level = min(level, largest)
    if smallest is not None:
        level = max(level, smallest)
    return level


def _check_12bit(numpy.ndarray data, uint16_t upper=4095):
    """Return if all values are below 2^12."""
    cdef:
        numpy.flatiter srciter
        uint8_t* srcptr = NULL
        ssize_t srcsize = 0
        ssize_t srcstride = 0
        ssize_t i
        int axis = -1
        int ret = 1

    srciter = numpy.PyArray_IterAllButAxis(data, &axis)
    srcsize = data.shape[axis]
    srcstride = data.strides[axis]

    with nogil:
        while numpy.PyArray_ITER_NOTDONE(srciter) and ret != 0:
            srcptr = <uint8_t*>numpy.PyArray_ITER_DATA(srciter)
            for i in range(srcsize):
                if (<uint16_t*>srcptr)[0] > upper:
                    ret = 0
                    break
                srcptr += srcstride
            numpy.PyArray_ITER_NEXT(srciter)

    return bool(ret)
