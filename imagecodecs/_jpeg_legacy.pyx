# imagecodecs/_jpeg_legacy.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: freethreading_compatible = True

# Copyright (c) 2018-2026, Christoph Gohlke
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

"""JPEG (Joint Photographic Experts Group) codec for the imagecodecs package.

This fallback implementation works with libjpeg v8 and libjpeg-turbo < 3.

"""

include '_shared.pxi'

from cython.operator cimport dereference as deref
from libc.setjmp cimport jmp_buf, longjmp, setjmp
from libjpeg cimport *


class JPEG:
    """JPEG codec constants."""

    available = True
    legacy = True
    all_precisions = False

    class CS(enum.IntEnum):
        """JPEG codec color spaces."""

        UNKNOWN = JCS_UNKNOWN
        GRAYSCALE = JCS_GRAYSCALE
        RGB = JCS_RGB
        YCbCr = JCS_YCbCr
        CMYK = JCS_CMYK
        YCCK = JCS_YCCK
        # libjpeg-turbo constants
        EXT_RGB = 6
        EXT_RGBX = 7
        EXT_BGR = 8
        EXT_BGRX = 9
        EXT_XBGR = 10
        EXT_XRGB = 11
        EXT_RGBA = 12
        EXT_BGRA = 13
        EXT_ABGR = 14
        EXT_ARGB = 15
        # RGB565 = 16


class JpegError(RuntimeError):
    """JPEG codec exceptions."""


def jpeg_version():
    """Return libjpeg library version string."""
    return f'libjpeg {JPEG_LIB_VERSION // 10}.{JPEG_LIB_VERSION % 10}'


def jpeg_check(const uint8_t[::1] data not None, /):
    """Return whether data is JPEG encoded image or None if unknown."""
    return bool(data.shape[0] > 4 and data[0] == 0xFF and data[1] == 0xD8)


def jpeg_encode(
    data,
    /,
    level=None,
    *,
    colorspace=None,
    outcolorspace=None,
    subsampling=None,
    optimize=None,
    smoothing=None,
    lossless=None,
    predictor=None,
    bitspersample=None,
    iccprofile=None,
    validate=None,  # for compatibility
    out=None,
):
    """Return JPEG encoded image.

    If the encoded image does not fit into a provided `out` buffer, a new
    bytes object is returned and the content of `out` is undefined.

    Parameters:
        data:
            Image to encode.
            Must be uint8 array of shape (height, width) or
            (height, width, samples) with 1 to 4 samples.
        level:
            Quality of lossy compression in range 0 to 100.
            Defaults to 95.
        colorspace:
            Color space of `data`.
            For example, 'GRAY', 'RGB', 'CMYK', 'YCbCr', or 'YCCK'.
            Defaults to 'GRAY' for 1 sample, 'RGB' for 3 samples, and
            unknown otherwise.
        outcolorspace:
            Color space of encoded JPEG components.
            Defaults to 'GRAY' for grayscale input, same as `colorspace`
            for CMYK or YCCK input, 'YCbCr' for other known color spaces,
            and unknown otherwise.
        subsampling:
            Chroma subsampling: '444', '422', '420', '411', '440', or
            (horizontal, vertical) factors of the luma component.
            Applies only to YCbCr and YCCK output color spaces.
        optimize:
            Enable optimized entropy coding when True.
        smoothing:
            Smoothing factor in range 0 to 100. Disabled by default.
        lossless:
            Not supported by legacy codec.
        predictor:
            Unused.
        bitspersample:
            Data precision in bits. Only 8 is supported.
        iccprofile:
            Not supported by legacy codec.
        validate:
            Unused. For compatibility only.
        out:
            Optional output buffer.

    Returns:
        JPEG encoded image as bytes or in `out`.

    Raises:
        ValueError:
            If data shape, dtype, or parameters are invalid or not
            supported by legacy codec.
        JpegError:
            If encoding fails.

    """
    cdef:
        numpy.ndarray src = _array_input(data, out, dtype=None, contig=False)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize, rowstride
        int quality = _default_value(level, 95, 0, 100)
        my_error_mgr err
        jpeg_compress_struct cinfo
        JSAMPROW rowpointer8
        J_COLOR_SPACE in_color_space = JCS_UNKNOWN
        J_COLOR_SPACE jpeg_color_space = JCS_UNKNOWN
        unsigned long outsize = 0
        unsigned char* outbuffer = NULL
        unsigned char* dstbuffer = NULL
        char[200] msg  # JMSG_LENGTH_MAX
        int h_samp_factor = 0
        int v_samp_factor = 0
        int smoothing_factor = _default_value(smoothing, -1, 0, 100)
        int optimize_coding = -1 if optimize is None else 1 if optimize else 0
        imagelayout_t layout

    if src.dtype.char == 'H':
        raise ValueError('dtype uint16 not supported by legacy JPEG codec')
    if lossless:
        raise ValueError('lossless mode not supported by legacy JPEG codec')
    if bitspersample not in {None, 8}:
        raise ValueError('bitspersample not supported by legacy JPEG codec')
    if iccprofile is not None:
        raise ValueError('ICC profiles not supported by legacy JPEG codec')

    _image_layout(
        IC_UINT | IC_SZ1 | IC_GRAY | IC_RGB,
        src.ndim,
        src.shape,
        src.dtype,
        None,  # photometric
        None,  # bitspersample
        None,  # planar
        None,  # frames
        None,  # volumetric
        None,  # extrasample
        &layout,
    )

    if not (
        layout.height > 0
        and src.strides[src.ndim - 1] == src.itemsize
        and (src.ndim == 2 or src.strides[1] == layout.samples * src.itemsize)
    ):
        raise ValueError('invalid data shape, strides, or dtype')

    rowstride = src.strides[0]

    if colorspace is not None:
        in_color_space = _jcs_colorspace(colorspace, True)
        if not (
            _JCS_COLORSPACE_SAMPLES[in_color_space]
            & (1 << <int> layout.samples)
        ):
            raise ValueError('invalid data shape')
    elif layout.samples == 1:
        in_color_space = JCS_GRAYSCALE
    elif layout.samples == 3:
        in_color_space = JCS_RGB
    # elif layout.samples == 4:
    #     in_color_space = JCS_EXT_RGBA
    #     in_color_space = JCS_CMYK
    else:
        # libjpeg-turbo does not currently support alpha channels.
        # JCS_UNKNOWN seems to preserve the 4th channel.
        in_color_space = JCS_UNKNOWN

    if in_color_space == JCS_GRAYSCALE:
        jpeg_color_space = JCS_GRAYSCALE
    elif outcolorspace is not None:
        jpeg_color_space = _jcs_colorspace(outcolorspace, True)
    elif in_color_space == JCS_CMYK or in_color_space == JCS_YCCK:
        jpeg_color_space = in_color_space
    elif in_color_space != JCS_UNKNOWN:
        # same as jpeg_default_colorspace
        jpeg_color_space = JCS_YCbCr
    else:
        jpeg_color_space = JCS_UNKNOWN

    if subsampling is not None:
        if not isinstance(subsampling, str):
            subsampling = tuple(subsampling)
        if subsampling in {'444', (1, 1)}:
            h_samp_factor = 1
            v_samp_factor = 1
        elif subsampling in {'422', (2, 1)}:
            h_samp_factor = 2
            v_samp_factor = 1
        elif subsampling in {'420', (2, 2)}:
            h_samp_factor = 2
            v_samp_factor = 2
        elif subsampling in {'411', (4, 1)}:
            h_samp_factor = 4
            v_samp_factor = 1
        elif subsampling in {'440', (1, 2)}:
            h_samp_factor = 1
            v_samp_factor = 2
        else:
            raise ValueError(f'invalid {subsampling=!r}')
        if jpeg_color_space != JCS_YCbCr and jpeg_color_space != JCS_YCCK:
            # only chroma components can be subsampled
            h_samp_factor = 0
            v_samp_factor = 0

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None and dstsize > 0:
        out = _create_output(outtype, dstsize)

    if out is not None:
        dst = out
        dstsize = dst.shape[0]
        outsize = <unsigned long> dst.nbytes  # validates overflow
        outbuffer = <unsigned char*> dst._data
        dstbuffer = outbuffer

    with nogil:
        cinfo.err = jpeg_std_error(&err.pub)
        cinfo.dest = NULL
        err.pub.error_exit = my_error_exit
        err.pub.output_message = my_output_message

        if setjmp(err.setjmp_buffer):
            # msg = err.pub.jpeg_message_table[err.pub.msg_code]
            msg[0] = b'\x00'
            err.pub.format_message(<jpeg_common_struct*> &cinfo, &msg[0])
            if cinfo.dest != NULL:
                # set outbuffer to the buffer currently used by jpeg_mem_dest
                cinfo.dest.term_destination(&cinfo)
            jpeg_destroy_compress(&cinfo)
            if outbuffer != dstbuffer:
                free(outbuffer)
            raise JpegError(msg.decode())

        jpeg_create_compress(&cinfo)

        cinfo.image_height = <JDIMENSION> layout.height
        cinfo.image_width = <JDIMENSION> layout.width
        cinfo.input_components = <int> layout.samples

        if in_color_space != JCS_UNKNOWN:
            cinfo.in_color_space = in_color_space

        jpeg_set_defaults(&cinfo)

        jpeg_set_colorspace(&cinfo, jpeg_color_space)
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
            if cinfo.num_components == 4:
                cinfo.comp_info[3].h_samp_factor = 1
                cinfo.comp_info[3].v_samp_factor = 1

        # TODO: add option to use or return JPEG tables

        jpeg_start_compress(&cinfo, 1)

        while cinfo.next_scanline < cinfo.image_height:
            rowpointer8 = <JSAMPROW> (
                <char*> src.data + cinfo.next_scanline * rowstride
            )
            jpeg_write_scanlines(&cinfo, &rowpointer8, 1)

        jpeg_finish_compress(&cinfo)
        jpeg_destroy_compress(&cinfo)

    if out is None or outbuffer != <unsigned char*> dst._data:
        # outbuffer was allocated in jpeg_mem_dest
        try:
            if out is not None and not outgiven:
                raise ValueError(f'output too small {dstsize} < {outsize}')
            out = _create_output(
                outtype, <ssize_t> outsize, <const char*> outbuffer
            )
        finally:
            free(outbuffer)
        return out

    del dst
    return _return_output(out, dstsize, outsize, outgiven)


def jpeg_decode(
    data,
    /,
    *,
    tables=None,
    header=None,
    colorspace=None,
    outcolorspace=None,
    fancyupsampling=None,
    shape=None,
    scale=None,
    out=None,
    bitspersample=None,  # TODO: remove unused parameter
    bint _info=False,
):
    """Return decoded JPEG image.

    Parameters:
        data:
            JPEG encoded image, or entropy-coded data following `header`.
        tables:
            JPEG tables-only stream, for example, from TIFF JPEGTables tag.
        header:
            JPEG header to prepend to `data`. An EOI marker is appended.
        colorspace:
            Color space of encoded JPEG components.
            For example, 'RGB', 'YCbCr', 'CMYK', or 'YCCK'.
            Overrides the color space detected from markers.
        outcolorspace:
            Color space of decoded image.
            For example, 'GRAY', 'RGB', or 'CMYK'.
            Defaults to `colorspace` if specified, else library default.
        fancyupsampling:
            Enable smooth upsampling of chroma components.
            Defaults to True.
        shape:
            Image (height, width).
            Only used to decode images larger than 65500 pixels with a
            patched libjpeg-turbo.
        scale:
            Scale denominator, or (numerator, denominator) tuple.
            For example, 2 decodes at half size.
        out:
            Optional output array.
        bitspersample:
            Unused.

    Returns:
        Decoded image as uint8 array of shape (height, width) or
        (height, width, samples).

    Raises:
        ValueError:
            If parameters are invalid or data precision exceeds 8 bits.
        JpegError:
            If decoding fails.

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src
        const uint8_t[::1] tables_
        unsigned long tablesize = 0
        size_t srcsize
        ssize_t rowstride
        my_error_mgr err
        jpeg_decompress_struct cinfo
        JSAMPROW rowpointer8
        J_COLOR_SPACE jpeg_color_space
        J_COLOR_SPACE out_color_space
        JDIMENSION width = 0
        JDIMENSION height = 0
        char[200] msg  # JMSG_LENGTH_MAX
        int do_fancy_upsampling = 1
        unsigned int scale_num = 1
        unsigned int scale_denom = 1

    if data is out:
        raise ValueError('cannot transform in-place')

    if header is not None:
        data = b''.join((header, bytes(data), b'\xff\xd9'))

    src = data
    srcsize = <size_t> src.shape[0]

    if srcsize > ULONG_MAX:
        raise ValueError(f'{srcsize=} > {ULONG_MAX}')

    if colorspace is None:
        jpeg_color_space = JCS_UNKNOWN
    else:
        jpeg_color_space = _jcs_colorspace(colorspace)

    if outcolorspace is None:
        out_color_space = jpeg_color_space
    else:
        out_color_space = _jcs_colorspace(outcolorspace)

    if fancyupsampling is not None:
        do_fancy_upsampling = 1 if fancyupsampling else 0

    if tables is not None:
        tables_ = tables
        tablesize = <unsigned long> tables_.shape[0]
        if tables_.shape[0] > INT32_MAX:
            raise ValueError(f'table size > {INT32_MAX}')

    if shape is not None and (shape[0] >= 65500 or shape[1] >= 65500):
        # enable decoding of large (JPEG_MAX_DIMENSION <= 2^20) JPEG
        # when using a patched jibjpeg-turbo
        height = <JDIMENSION> shape[0]
        width = <JDIMENSION> shape[1]

    if scale is not None:
        if isinstance(scale, tuple | list):
            scale_num, scale_denom = scale[0], scale[1]
        else:
            scale_num, scale_denom = 1, int(scale)
        if scale_num < 1 or scale_denom < 1:
            raise ValueError(f'invalid {scale=}')

    with nogil:
        cinfo.err = jpeg_std_error(&err.pub)
        err.pub.error_exit = my_error_exit
        err.pub.output_message = my_output_message
        if setjmp(err.setjmp_buffer):
            # msg = err.pub.jpeg_message_table[err.pub.msg_code]
            msg[0] = b'\x00'
            err.pub.format_message(<jpeg_common_struct*> &cinfo, &msg[0])
            jpeg_destroy_decompress(&cinfo)
            raise JpegError(msg.decode())

        jpeg_create_decompress(&cinfo)
        if width > 0:
            cinfo.image_width = width
            cinfo.image_height = height

        if tablesize > 0:
            jpeg_mem_src(
                &cinfo, <const unsigned char*> tables_._data, tablesize
            )
            jpeg_read_header(&cinfo, 0)  # == JPEG_HEADER_TABLES_ONLY

        jpeg_mem_src(
            &cinfo, <const unsigned char*> src._data, <unsigned long> srcsize
        )

        if _info:
            jpeg_save_markers(&cinfo, JPEG_APP0 + 2, 0xFFFF)
        jpeg_read_header(&cinfo, 1)

        cinfo.do_fancy_upsampling = do_fancy_upsampling

        if jpeg_color_space != JCS_UNKNOWN:
            cinfo.jpeg_color_space = jpeg_color_space
        if out_color_space != JCS_UNKNOWN:
            cinfo.out_color_space = out_color_space

        cinfo.scale_num = scale_num
        cinfo.scale_denom = scale_denom

        if cinfo.data_precision > 8:
            raise ValueError(
                f'data_precision={cinfo.data_precision} '
                'not supported by legacy JPEG codec'
            )

        if _info:
            # header fields reflect the first scan
            jpeg_calc_output_dimensions(&cinfo)
            with gil:
                try:
                    info = _jpeg_info(&cinfo)
                finally:
                    jpeg_destroy_decompress(&cinfo)
                return info

        jpeg_start_decompress(&cinfo)

        with gil:
            # if (cinfo.output_components not in
            #     _jcs_colorspace_samples(out_color_space)):
            #     raise ValueError('invalid output shape')

            if cinfo.output_components > 1:
                shape = (
                    cinfo.output_height,
                    cinfo.output_width,
                    cinfo.output_components
                )
            else:
                shape = cinfo.output_height, cinfo.output_width

            # TODO: allow strides
            out = _create_array(out, shape, numpy.uint8)
            dst = out
            rowstride = dst.strides[0]

        rowpointer8 = <JSAMPROW> dst.data
        while cinfo.output_scanline < cinfo.output_height:
            jpeg_read_scanlines(&cinfo, &rowpointer8, 1)
            rowpointer8 += rowstride

        jpeg_finish_decompress(&cinfo)
        jpeg_destroy_decompress(&cinfo)

    return out


def jpeg_info(
    data,
    /,
    *,
    tables=None,
    header=None,
    colorspace=None,
    outcolorspace=None,
    fancyupsampling=None,
    shape=None,
    scale=None,
    out=None,
    bint _info=False,
):
    """Return information about JPEG image.

    Parameters are the same as for :py:func:`jpeg_decode`.

    Returns:
        Dict of image properties, such as shape, dtype, bitspersample,
        colorspace, iccprofile, subsampling, and coding mode.

    """
    return jpeg_decode(
        data,
        tables=tables,
        header=header,
        colorspace=colorspace,
        outcolorspace=outcolorspace,
        fancyupsampling=fancyupsampling,
        shape=shape,
        scale=scale,
        out=out,
        _info=True,
    )


def jpeg_transcode(
    data,
    /,
    *,
    tables=None,
    header=None,
    colorspace=None,
    tiles=None,
    crop=None,
    optimize=None,
    progressive=None,
    copy=None,
    restart=None,
    grayscale=None,
    arithmetic=None,
    out=None,
):
    """Return losslessly transcoded JPEG image."""
    raise NotImplementedError(
        'jpeg_transcode not supported by legacy JPEG codec'
    )


def jpeg_merge(
    data,
    /,
    *,
    tables=None,
    header=None,
    colorspace=None,
    tiles=None,
):
    """Return complete JPEG stream from parts."""
    raise NotImplementedError('jpeg_merge not supported by legacy JPEG codec')


ctypedef struct my_error_mgr:
    jpeg_error_mgr pub
    jmp_buf setjmp_buffer


cdef void my_error_exit(jpeg_common_struct* cinfo) noexcept nogil:
    cdef:
        my_error_mgr* error = <my_error_mgr*> deref(cinfo).err

    longjmp(deref(error).setjmp_buffer, 1)


cdef void my_output_message(jpeg_common_struct* cinfo) noexcept nogil:
    pass


cdef dict _jpeg_info(jpeg_decompress_struct* cinfo):
    # Return information from decompressor after jpeg_read_header
    cdef:
        int i

    if cinfo.output_components > 1:
        shape = (
            cinfo.output_height,
            cinfo.output_width,
            cinfo.output_components
        )
    else:
        shape = cinfo.output_height, cinfo.output_width

    subsampling = None
    if cinfo.num_components >= 3 and (
        cinfo.jpeg_color_space == JCS_YCbCr
        or cinfo.jpeg_color_space == JCS_YCCK
    ):
        subsampling = (
            cinfo.comp_info[0].h_samp_factor,
            cinfo.comp_info[0].v_samp_factor
        )
        for i in range(1, cinfo.num_components):
            if (
                cinfo.comp_info[i].h_samp_factor != 1
                or cinfo.comp_info[i].v_samp_factor != 1
            ):
                subsampling = None
                break

    return {
        '_codec': 'jpeg',
        'shape': shape,
        'dtype': numpy.uint8,
        'bitspersample': int(cinfo.data_precision),
        'colorspace': _JCS_COLORSPACE_NAME.get(int(cinfo.jpeg_color_space)),
        'iccprofile': _jpeg_iccprofile(cinfo),
        'image_height': int(cinfo.image_height),
        'image_width': int(cinfo.image_width),
        'num_components': int(cinfo.num_components),
        'subsampling': subsampling,
        'progressive': bool(cinfo.progressive_mode),
        'arithmetic': bool(cinfo.arith_code),
        'lossless': None,
        'predictor': None,
        'pointtransform': None,
        'restartinterval': int(cinfo.restart_interval),
    }


cdef object _jpeg_iccprofile(jpeg_decompress_struct* cinfo):
    # Return reassembled ICC_PROFILE APP2 chunks or None
    cdef:
        jpeg_saved_marker_ptr marker = cinfo.marker_list
        int seq
        dict chunks = {}

    while marker != NULL:
        if (
            marker.marker == JPEG_APP0 + 2
            and marker.data_length >= 14
            and marker.data[0] == 0x49  # I
            and marker.data[1] == 0x43  # C
            and marker.data[2] == 0x43  # C
            and marker.data[3] == 0x5F  # _
            and marker.data[4] == 0x50  # P
            and marker.data[5] == 0x52  # R
            and marker.data[6] == 0x4F  # O
            and marker.data[7] == 0x46  # F
            and marker.data[8] == 0x49  # I
            and marker.data[9] == 0x4C  # L
            and marker.data[10] == 0x45  # E
            and marker.data[11] == 0x00
        ):
            seq = marker.data[12]
            chunks[seq] = bytes(
                (<unsigned char*> marker.data)[14:marker.data_length]
            )
        marker = marker.next

    return b''.join(chunks[k] for k in sorted(chunks)) if chunks else None


cdef dict _JCS_COLORSPACE_NAME = {
    JCS_UNKNOWN: 'UNKNOWN',
    JCS_GRAYSCALE: 'GRAYSCALE',
    JCS_RGB: 'RGB',
    JCS_YCbCr: 'YCbCr',
    JCS_CMYK: 'CMYK',
    JCS_YCCK: 'YCCK',
}

cdef dict _JCS_COLORSPACE_SAMPLES = {
    # bitmask: bit N is set when N samples is valid
    JCS_UNKNOWN: 0b11110,  # 1, 2, 3, 4
    JCS_GRAYSCALE: 0b00010,  # 1
    JCS_RGB: 0b01000,  # 3
    JCS_YCbCr: 0b01000,  # 3
    JCS_CMYK: 0b10000,  # 4
    JCS_YCCK: 0b10000,  # 4
    6: 0b01000,  # 3
    7: 0b10000,  # 4
    8: 0b01000,  # 3
    9: 0b10000,  # 4
    10: 0b10000,  # 4
    11: 0b10000,  # 4
    12: 0b10000,  # 4
    13: 0b10000,  # 4
    14: 0b10000,  # 4
    15: 0b10000,  # 4
}

cdef dict _JCS_COLORSPACE_MAP = {
    None: JCS_UNKNOWN,
    'UNKNOWN': JCS_UNKNOWN,
    'GRAY': JCS_GRAYSCALE,
    'GRAYSCALE': JCS_GRAYSCALE,
    'MINISWHITE': JCS_GRAYSCALE,
    'MINISBLACK': JCS_GRAYSCALE,
    'RGB': JCS_RGB,
    'CMYK': JCS_CMYK,
    'SEPARATED': JCS_CMYK,
    'YCCK': JCS_YCCK,
    'YCBCR': JCS_YCbCr,
    'RGBA': 12,
    'BGR': 8,
    'BGRA': 13,
    'ARGB': 15,
    'ABGR': 14,
    'RGBX': 7,
    'BGRX': 9,
    'XBGR': 10,
    'XRGB': 11,
    # 'RGB565': 16,
    JCS_UNKNOWN: JCS_UNKNOWN,
    JCS_GRAYSCALE: JCS_GRAYSCALE,
    JCS_RGB: JCS_RGB,
    JCS_YCbCr: JCS_YCbCr,
    JCS_CMYK: JCS_CMYK,
    JCS_YCCK: JCS_YCCK,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 12,
    13: 13,
    14: 14,
    15: 15,
    # 16: 16,
}


cdef J_COLOR_SPACE _jcs_colorspace(colorspace, bint strict=False):
    """Return JCS colorspace value from user input."""
    if isinstance(colorspace, str):
        colorspace = colorspace.upper()
    if strict and colorspace not in _JCS_COLORSPACE_MAP:
        raise ValueError(f'invalid {colorspace=!r}')
    return _JCS_COLORSPACE_MAP.get(colorspace, JCS_UNKNOWN)
