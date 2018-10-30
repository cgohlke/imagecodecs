# -*- coding: utf-8 -*-
# _jpeg12.pyx
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2018, Christoph Gohlke
# Copyright (c) 2018, The Regents of the University of California
# Produced at the Laboratory for Fluorescence Dynamics.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
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

:Version: 2018.10.28

"""

__version__ = '2018.10.28'

import numpy

cimport cython
cimport numpy

from cython.operator cimport dereference as deref
from libc.setjmp cimport setjmp, longjmp, jmp_buf
from libc.stdint cimport uint8_t
from libc.string cimport memset

numpy.import_array()


# JPEG 12-bit #################################################################

cdef extern from 'jpeglib.h':
    ctypedef void noreturn_t
    ctypedef int boolean
    ctypedef unsigned int JDIMENSION
    ctypedef unsigned short JSAMPLE
    ctypedef JSAMPLE* JSAMPROW
    ctypedef JSAMPROW* JSAMPARRAY

    ctypedef enum J_COLOR_SPACE:
        JCS_UNKNOWN,
        JCS_GRAYSCALE,
        JCS_RGB,
        JCS_YCbCr,
        JCS_CMYK,
        JCS_YCCK

    ctypedef enum J_DITHER_MODE:
        JDITHER_NONE,
        JDITHER_ORDERED,
        JDITHER_FS

    ctypedef enum J_DCT_METHOD:
        JDCT_ISLOW,
        JDCT_IFAST,
        JDCT_FLOAT

    struct jpeg_source_mgr:
        pass

    struct jpeg_error_mgr:
        int msg_code
        char** jpeg_message_table
        noreturn_t error_exit(jpeg_common_struct*)
        void output_message(jpeg_common_struct*)

    struct jpeg_common_struct:
        jpeg_error_mgr* err

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

    jpeg_error_mgr* jpeg_std_error(jpeg_error_mgr*) nogil

    void jpeg_create_decompress(jpeg_decompress_struct*) nogil

    void jpeg_destroy_decompress(jpeg_decompress_struct*) nogil

    int jpeg_read_header(jpeg_decompress_struct*, boolean) nogil

    boolean jpeg_start_decompress(jpeg_decompress_struct*) nogil

    boolean jpeg_finish_decompress(jpeg_decompress_struct*) nogil

    JDIMENSION jpeg_read_scanlines(jpeg_decompress_struct*,
                                   JSAMPARRAY,
                                   JDIMENSION) nogil

    void jpeg_mem_src(jpeg_decompress_struct*,
                      unsigned char*,
                      unsigned long) nogil


ctypedef struct my_error_mgr:
    jpeg_error_mgr pub
    jmp_buf setjmp_buffer


cdef void my_error_exit(jpeg_common_struct* cinfo):
    cdef my_error_mgr* error = <my_error_mgr*> deref(cinfo).err
    longjmp(deref(error).setjmp_buffer, 1)


cdef void my_output_message(jpeg_common_struct* cinfo):
    pass


class Jpeg12Error(RuntimeError):
    """JPEG Exceptions."""
    pass


def jpeg12_encode(*args, **kwargs):
    """Not implemented."""
    # TODO: JPEG 12-bit encoding
    raise NotImplementedError('jpeg12_encode')


def jpeg12_decode(data, tables=None, colorspace=None, outcolorspace=None,
                  out=None):
    """Decode JPEG 12-bit image to numpy array.

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        const uint8_t[::1] tables_
        ssize_t dstsize
        int numlines
        my_error_mgr err
        jpeg_decompress_struct cinfo
        JSAMPROW samples

    if data is out:
        raise ValueError('cannot decode in-place')

    cinfo.err = jpeg_std_error(&err.pub)
    err.pub.error_exit = my_error_exit
    err.pub.output_message = my_output_message
    if setjmp(err.setjmp_buffer):
        jpeg_destroy_decompress(&cinfo)
        raise Jpeg12Error(
            err.pub.jpeg_message_table[err.pub.msg_code].decode('utf-8'))

    jpeg_create_decompress(&cinfo)
    cinfo.do_fancy_upsampling = True

    if tables is not None:
        tables_ = tables
        jpeg_mem_src(&cinfo, &tables_[0], tables_.size)
        jpeg_read_header(&cinfo, 0)

    jpeg_mem_src(&cinfo, &src[0], src.size)
    jpeg_read_header(&cinfo, 1)

    if colorspace is None:
        pass
    elif colorspace == 'RGB':
        cinfo.jpeg_color_space = JCS_RGB
        cinfo.out_color_space = JCS_RGB
    elif colorspace == 'YCBCR':
        cinfo.jpeg_color_space = JCS_YCbCr
        cinfo.out_color_space = JCS_YCbCr
    elif colorspace in ('MINISBLACK', 'MINISWHITE', 'GRAYSCALE'):
        cinfo.jpeg_color_space = JCS_GRAYSCALE
        cinfo.out_color_space = JCS_GRAYSCALE
    elif colorspace == 'CMYK':
        cinfo.jpeg_color_space = JCS_CMYK
        cinfo.out_color_space = JCS_CMYK
    elif colorspace == 'YCCK':
        cinfo.jpeg_color_space = JCS_YCCK
        cinfo.out_color_space = JCS_YCCK

    if outcolorspace is None:
        pass
    elif outcolorspace == 'RGB':
        cinfo.out_color_space = JCS_RGB
    elif outcolorspace == 'YCBCR':
        cinfo.out_color_space = JCS_YCbCr
    elif outcolorspace in ('MINISBLACK', 'MINISWHITE', 'GRAYSCALE'):
        cinfo.out_color_space = JCS_GRAYSCALE
    elif outcolorspace == 'CMYK':
        cinfo.out_color_space = JCS_CMYK
    elif outcolorspace == 'YCCK':
        cinfo.out_color_space = JCS_YCCK

    jpeg_start_decompress(&cinfo)

    shape = cinfo.output_height, cinfo.output_width
    if cinfo.output_components > 1:
        shape += cinfo.output_components,

    out = _create_array(out, shape, numpy.uint16)
    dst = out
    dstsize = dst.size * dst.itemsize

    with nogil:
        memset(<void *>dst.data, 0, dstsize)
        samples = <JSAMPROW>dst.data
        while cinfo.output_scanline < cinfo.output_height:
            numlines = jpeg_read_scanlines(&cinfo, <JSAMPARRAY> &samples, 1)
            samples += numlines * cinfo.output_width * cinfo.output_components
        jpeg_finish_decompress(&cinfo)
        jpeg_destroy_decompress(&cinfo)

    return out


###############################################################################

cdef _create_array(out, shape, dtype, strides=None):
    """Return numpy array of shape and dtype from output argument."""
    if out is None:
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
