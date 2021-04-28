# imagecodecs/_jpeg8.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2018-2021, Christoph Gohlke
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

"""JPEG 8-bit codec for the imagecodecs package."""

__version__ = '2021.4.28'

include '_shared.pxi'

# DEF HAVE_LIBJPEG_TURBO = True

IF HAVE_LIBJPEG_TURBO:
    from libjpeg_turbo cimport *
ELSE:
    from libjpeg cimport *

from cython.operator cimport dereference as deref

from libc.setjmp cimport setjmp, longjmp, jmp_buf


class JPEG8:
    """JPEG 8-bit Constants."""

    CS_UNKNOWN = JCS_UNKNOWN
    CS_GRAYSCALE = JCS_GRAYSCALE
    CS_RGB = JCS_RGB
    CS_YCbCr = JCS_YCbCr
    CS_CMYK = JCS_CMYK
    CS_YCCK = JCS_YCCK
    IF HAVE_LIBJPEG_TURBO:
        CS_EXT_RGB = JCS_EXT_RGB
        CS_EXT_RGBX = JCS_EXT_RGBX
        CS_EXT_BGR = JCS_EXT_BGR
        CS_EXT_BGRX = JCS_EXT_BGRX
        CS_EXT_XBGR = JCS_EXT_XBGR
        CS_EXT_XRGB = JCS_EXT_XRGB
        CS_EXT_RGBA = JCS_EXT_RGBA
        CS_EXT_BGRA = JCS_EXT_BGRA
        CS_EXT_ABGR = JCS_EXT_ABGR
        CS_EXT_ARGB = JCS_EXT_ARGB
        CS_RGB565 = JCS_RGB565


class Jpeg8Error(RuntimeError):
    """JPEG 8-bit Exceptions."""


def jpeg8_version():
    """Return libjpeg library version string."""
    IF HAVE_LIBJPEG_TURBO:
        ver = str(LIBJPEG_TURBO_VERSION_NUMBER)
        return 'libjpeg_turbo {}.{}.{}/{:.1f}'.format(
            int(ver[:1]), int(ver[3:4]), int(ver[6:]), JPEG_LIB_VERSION / 10.0
        )
    ELSE:
        return f'libjpeg {JPEG_LIB_VERSION_MAJOR}.{JPEG_LIB_VERSION_MINOR}'


def jpeg8_check(const uint8_t[::1] data):
    """Return True if data likely contains a JPEG image."""
    sig = bytes(data[:10])
    return (
        sig[:4] == b'\xFF\xD8\xFF\xDB'
        or sig[:4] == b'\xFF\xD8\xFF\xEE'
        or sig[:4] == b'\xFF\xD8\xFF\xC3'
        or (sig[:3] == b'\xFF\xD8\xFF' and sig[6:10] == b'JFIF')
        or (sig[:3] == b'\xFF\xD8\xFF' and sig[6:10] == b'Exif')
    )


def jpeg8_encode(
    data,
    level=None,
    colorspace=None,
    outcolorspace=None,
    subsampling=None,
    optimize=None,
    smoothing=None,
    out=None
):
    """Return JPEG 8-bit image from numpy array.

    """
    cdef:
        numpy.ndarray src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        ssize_t srcsize = src.nbytes
        ssize_t rowstride = src.strides[0]
        int samples = <int> src.shape[2] if src.ndim == 3 else 1
        int quality = _default_value(level, 95, 0, 100)
        my_error_mgr err
        jpeg_compress_struct cinfo
        JSAMPROW rowpointer
        J_COLOR_SPACE in_color_space = JCS_UNKNOWN
        J_COLOR_SPACE jpeg_color_space = JCS_UNKNOWN
        unsigned long outsize = 0
        unsigned char* outbuffer = NULL
        char msg[200]  # JMSG_LENGTH_MAX
        int h_samp_factor = 0
        int v_samp_factor = 0
        int smoothing_factor = _default_value(smoothing, -1, 0, 100)
        int optimize_coding = -1 if optimize is None else 1 if optimize else 0

    if data is out:
        raise ValueError('cannot encode in-place')

    if not (
        src.dtype == numpy.uint8
        and src.ndim in (2, 3)
        # src.nbytes < 2 ** 31 and  # limit to 2 GB
        and samples in (1, 3, 4)
        and src.strides[src.ndim-1] == src.itemsize
        and (src.ndim == 2 or src.strides[1] == samples * src.itemsize)
    ):
        raise ValueError('invalid input shape, strides, or dtype')

    if colorspace is None:
        if samples == 1:
            in_color_space = JCS_GRAYSCALE
        elif samples == 3:
            in_color_space = JCS_RGB
        # elif samples == 4:
        #     in_color_space = JCS_EXT_RGBA
        #     in_color_space = JCS_CMYK
        else:
            # libjpeg-turbo does not currently support alpha channels.
            # JCS_UNKNOWN seems to preserve the 4th channel.
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

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None and dstsize > 0:
        out = _create_output(outtype, dstsize)

    if out is not None:
        dst = out
        dstsize = dst.nbytes
        outsize = <unsigned long> dstsize
        outbuffer = <unsigned char*> &dst[0]

    with nogil:
        cinfo.err = jpeg_std_error(&err.pub)
        err.pub.error_exit = my_error_exit
        err.pub.output_message = my_output_message

        if setjmp(err.setjmp_buffer):
            # msg = err.pub.jpeg_message_table[err.pub.msg_code]
            msg[0] = b'\x00'
            err.pub.format_message(<jpeg_common_struct*> &cinfo, &msg[0])
            jpeg_destroy_compress(&cinfo)
            raise Jpeg8Error(msg.decode())

        jpeg_create_compress(&cinfo)

        cinfo.image_height = <JDIMENSION> src.shape[0]
        cinfo.image_width = <JDIMENSION> src.shape[1]
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
            cinfo.optimize_coding = <boolean> optimize_coding
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
            rowpointer = <JSAMPROW> (
                <char*> src.data + cinfo.next_scanline * rowstride
            )
            jpeg_write_scanlines(&cinfo, &rowpointer, 1)

        jpeg_finish_compress(&cinfo)
        jpeg_destroy_compress(&cinfo)

    if out is None or outbuffer != <unsigned char*> &dst[0]:
        # outbuffer was allocated in jpeg_mem_dest
        out = _create_output(
            outtype, <ssize_t> outsize, <const char*> outbuffer
        )
        free(outbuffer)
        return out

    del dst
    return _return_output(out, dstsize, outsize, outgiven)


def jpeg8_decode(
    data,
    index=None,
    tables=None,
    colorspace=None,
    outcolorspace=None,
    shape=None,
    out=None
):
    """Decode JPEG 8-bit image to numpy array.

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
        char msg[200]  # JMSG_LENGTH_MAX

    if data is out:
        raise ValueError('cannot decode in-place')

    if srcsize >= 2 ** 32:
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
        height = <JDIMENSION> shape[0]
        width = <JDIMENSION> shape[1]

    with nogil:

        cinfo.err = jpeg_std_error(&err.pub)
        err.pub.error_exit = my_error_exit
        err.pub.output_message = my_output_message
        if setjmp(err.setjmp_buffer):
            # msg = err.pub.jpeg_message_table[err.pub.msg_code]
            msg[0] = b'\x00'
            err.pub.format_message(<jpeg_common_struct*> &cinfo, &msg[0])
            jpeg_destroy_decompress(&cinfo)
            raise Jpeg8Error(msg.decode())

        jpeg_create_decompress(&cinfo)
        cinfo.do_fancy_upsampling = True
        if width > 0:
            cinfo.image_width = width
            cinfo.image_height = height

        if tablesize > 0:
            jpeg_mem_src(&cinfo, &tables_[0], tablesize)
            jpeg_read_header(&cinfo, 0)

        jpeg_mem_src(&cinfo, &src[0], <unsigned long> srcsize)
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

            out = _create_array(out, shape, numpy.uint8)  # TODO: allow strides
            dst = out
            dstsize = dst.nbytes
            rowstride = dst.strides[0]

        memset(<void*> dst.data, 0, dstsize)
        rowpointer = <JSAMPROW> dst.data
        while cinfo.output_scanline < cinfo.output_height:
            jpeg_read_scanlines(&cinfo, &rowpointer, 1)
            rowpointer += rowstride

        jpeg_finish_decompress(&cinfo)
        jpeg_destroy_decompress(&cinfo)

    return out


ctypedef struct my_error_mgr:
    jpeg_error_mgr pub
    jmp_buf setjmp_buffer


cdef void my_error_exit(jpeg_common_struct* cinfo) nogil:
    cdef:
        my_error_mgr* error = <my_error_mgr*> deref(cinfo).err

    longjmp(deref(error).setjmp_buffer, 1)


cdef void my_output_message(jpeg_common_struct* cinfo) nogil:
    pass


def _jcs_colorspace(colorspace):
    """Return JCS colorspace value from user input."""
    IF HAVE_LIBJPEG_TURBO:
        jcs = {
            None: JCS_UNKNOWN,
            'UNKNOWN': JCS_UNKNOWN,
            'GRAY': JCS_GRAYSCALE,
            'GRAYSCALE': JCS_GRAYSCALE,
            'MINISWHITE': JCS_GRAYSCALE,
            'MINISBLACK': JCS_GRAYSCALE,
            'RGB': JCS_RGB,
            'CMYK': JCS_CMYK,
            'YCCK': JCS_YCCK,
            'YCBCR': JCS_YCbCr,
            'RGBA': JCS_EXT_RGBA,
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
        }
    ELSE:
        jcs = {
            None: JCS_UNKNOWN,
            'UNKNOWN': JCS_UNKNOWN,
            'GRAY': JCS_GRAYSCALE,
            'GRAYSCALE': JCS_GRAYSCALE,
            'MINISWHITE': JCS_GRAYSCALE,
            'MINISBLACK': JCS_GRAYSCALE,
            'RGB': JCS_RGB,
            'CMYK': JCS_CMYK,
            'YCCK': JCS_YCCK,
            'YCBCR': JCS_YCbCr,
            JCS_UNKNOWN: JCS_UNKNOWN,
            JCS_GRAYSCALE: JCS_GRAYSCALE,
            JCS_RGB: JCS_RGB,
            JCS_YCbCr: JCS_YCbCr,
            JCS_CMYK: JCS_CMYK,
            JCS_YCCK: JCS_YCCK,
        }
    return jcs.get(colorspace, JCS_UNKNOWN)


def _jcs_colorspace_samples(colorspace):
    """Return expected number of samples in colorspace."""
    three = (3,)
    four = (4,)
    IF HAVE_LIBJPEG_TURBO:
        jcs = {
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
        }
    ELSE:
        jcs = {
            JCS_UNKNOWN: (1, 2, 3, 4),
            JCS_GRAYSCALE: (1,),
            JCS_RGB: three,
            JCS_YCbCr: three,
            JCS_CMYK: four,
            JCS_YCCK: four,
        }
    return jcs[colorspace]
