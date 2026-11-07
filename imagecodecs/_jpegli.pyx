# imagecodecs/_jpegli.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: freethreading_compatible = True

# Copyright (c) 2026, Christoph Gohlke
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

"""JPEGLI (improved JPEG encoder and decoder) codec for imagecodecs."""

include '_shared.pxi'

from cython.operator cimport dereference as deref
from jpegli cimport *
from libc.setjmp cimport jmp_buf, longjmp, setjmp


class JPEGLI:
    """JPEGLI codec constants."""

    available = True

    class CS(enum.IntEnum):
        """JPEGLI codec color spaces."""

        UNKNOWN = JCS_UNKNOWN
        GRAYSCALE = JCS_GRAYSCALE
        RGB = JCS_RGB
        YCbCr = JCS_YCbCr
        CMYK = JCS_CMYK
        YCCK = JCS_YCCK
        EXT_RGB = JCS_EXT_RGB
        EXT_RGBX = JCS_EXT_RGBX
        EXT_BGR = JCS_EXT_BGR
        EXT_BGRX = JCS_EXT_BGRX
        EXT_XBGR = JCS_EXT_XBGR
        EXT_XRGB = JCS_EXT_XRGB
        EXT_RGBA = JCS_EXT_RGBA
        EXT_BGRA = JCS_EXT_BGRA
        EXT_ABGR = JCS_EXT_ABGR
        EXT_ARGB = JCS_EXT_ARGB


class JpegliError(RuntimeError):
    """JPEGLI codec exceptions."""


def jpegli_version():
    """Return jpegli library version string."""
    # jpegli does not define a version macro
    return 'jpegli 0.12.0/{:.1f}'.format(JPEG_LIB_VERSION / 10.0)


def jpegli_check(const uint8_t[::1] data not None, /):
    """Return whether data is JPEG encoded image or None if unknown."""
    return bool(data.shape[0] > 4 and data[0] == 0xFF and data[1] == 0xD8)


def jpegli_encode(
    data,
    /,
    level=None,
    *,
    distance=None,
    colorspace=None,
    outcolorspace=None,
    subsampling=None,
    optimize=None,
    smoothing=None,
    progressive=None,
    adaptive=None,
    standardtables=None,
    iccprofile=None,
    out=None,
):
    """Return JPEG encoded image.

    Data type of `data` must be uint8, uint16, or float32 in range [0, 1].
    The encoded image is always 8-bit JPEG. Full-range uint16 or float32
    data avoid rounding to 8-bit before the floating-point DCT.

    Parameters:
        data:
            Image to encode.
        level:
            Quality in range [0, 100]. Defaults to 90.
        distance:
            Butteraugli distance. Overrides `level` if specified.
        colorspace:
            Color space of `data`.
        outcolorspace:
            Color space of JPEG image.
        subsampling:
            Chroma subsampling: '444', '422', '420', '411', or '440'.
        optimize:
            Optimize entropy encoding.
        smoothing:
            Input smoothing factor in range [0, 100].
        progressive:
            Progressive level. True is 2, False is 0 (sequential).
            Defaults to 0.
        adaptive:
            Use adaptive quantization. Defaults to True.
        standardtables:
            Use standard quantization tables from Annex K of JPEG standard.
        iccprofile:
            ICC profile to embed.
        out:
            Optional output buffer.

    If the encoded image does not fit into a provided `out` buffer, a new
    bytes object is returned and the content of `out` is undefined.

    """
    cdef:
        numpy.ndarray src = _array_input(
            data, out, dtype=None, contig=False, empty=False, native=False
        )
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize, rowstride
        int quality = _default_value(level, 90, 0, 100)
        float distance_ = 0.0 if distance is None else distance
        bint use_distance = distance is not None
        my_error_mgr err
        jpeg_compress_struct cinfo
        JSAMPROW rowpointer
        J_COLOR_SPACE in_color_space = JCS_UNKNOWN
        J_COLOR_SPACE jpeg_color_space = JCS_UNKNOWN
        JpegliDataType data_type
        JpegliEndianness endianness = JPEGLI_NATIVE_ENDIAN
        unsigned long outsize = 0
        unsigned char* outbuffer = NULL
        unsigned char* dstbuffer = NULL
        char[200] msg  # JMSG_LENGTH_MAX
        int h_samp_factor = 0
        int v_samp_factor = 0
        int smoothing_factor = _default_value(smoothing, -1, 0, 100)
        int optimize_coding = -1 if optimize is None else 1 if optimize else 0
        int progressive_level = -1
        int adaptive_quantization = (
            -1 if adaptive is None else 1 if adaptive else 0
        )
        bint standard_tables = bool(standardtables)
        const uint8_t[::1] icc_
        unsigned int iccsize = 0
        imagelayout_t layout

    _image_layout(
        IC_UINT
        | IC_FLOAT
        | IC_SZ1
        | IC_SZ2
        | IC_SZ4
        | IC_GRAY
        | IC_RGB
        | IC_ALPHA
        | IC_BE,
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

    if layout.sampleformat == IC_SF_UINT and layout.itemsize == 1:
        data_type = JPEGLI_TYPE_UINT8
    elif layout.sampleformat == IC_SF_UINT and layout.itemsize == 2:
        data_type = JPEGLI_TYPE_UINT16
    elif layout.sampleformat == IC_SF_FLOAT and layout.itemsize == 4:
        data_type = JPEGLI_TYPE_FLOAT
    else:
        raise ValueError(f'invalid data {src.dtype=}')

    if src.dtype.byteorder == '>':
        endianness = JPEGLI_BIG_ENDIAN

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
    else:
        # JCS_UNKNOWN preserves the alpha channel
        in_color_space = JCS_UNKNOWN

    if in_color_space == JCS_GRAYSCALE:
        jpeg_color_space = JCS_GRAYSCALE
    elif outcolorspace is not None:
        jpeg_color_space = _jcs_colorspace(outcolorspace, True)
    elif in_color_space == JCS_CMYK or in_color_space == JCS_YCCK:
        jpeg_color_space = in_color_space
    elif in_color_space != JCS_UNKNOWN:
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

    if progressive is not None:
        if isinstance(progressive, bool):
            progressive_level = 2 if progressive else 0
        else:
            progressive_level = int(progressive)
            if progressive_level < 0:
                raise ValueError(f'invalid {progressive=!r}')

    if iccprofile is not None:
        icc_ = iccprofile
        if <size_t> icc_.shape[0] > UINT32_MAX:
            raise ValueError('iccprofile too large')
        iccsize = <unsigned int> icc_.shape[0]

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
        cinfo.err = jpegli_std_error(&err.pub)
        cinfo.dest = NULL
        err.pub.error_exit = my_error_exit
        err.pub.output_message = my_output_message

        if setjmp(err.setjmp_buffer):
            msg[0] = b'\x00'
            err.pub.format_message(<jpeg_common_struct*> &cinfo, &msg[0])
            if cinfo.dest != NULL:
                # set outbuffer to the buffer currently used by jpegli_mem_dest
                cinfo.dest.term_destination(&cinfo)
            jpegli_destroy_compress(&cinfo)
            if outbuffer != dstbuffer:
                free(outbuffer)
            raise JpegliError(msg.decode())

        jpegli_create_compress(&cinfo)

        cinfo.image_height = <JDIMENSION> layout.height
        cinfo.image_width = <JDIMENSION> layout.width
        cinfo.input_components = <int> layout.samples

        if in_color_space != JCS_UNKNOWN:
            cinfo.in_color_space = in_color_space

        jpegli_set_input_format(&cinfo, data_type, endianness)
        if standard_tables:
            # must be called before jpegli_set_defaults
            jpegli_use_standard_quant_tables(&cinfo)

        jpegli_set_defaults(&cinfo)
        jpegli_set_colorspace(&cinfo, jpeg_color_space)
        jpegli_mem_dest(&cinfo, &outbuffer, &outsize)
        if use_distance:
            jpegli_set_distance(&cinfo, distance_, 1)
        else:
            jpegli_set_quality(&cinfo, quality, 1)

        if smoothing_factor >= 0:
            cinfo.smoothing_factor = smoothing_factor
        if optimize_coding >= 0:
            cinfo.optimize_coding = <boolean> optimize_coding
        if progressive_level >= 0:
            jpegli_set_progressive_level(&cinfo, progressive_level)
        if adaptive_quantization >= 0:
            jpegli_enable_adaptive_quantization(
                &cinfo, <boolean> adaptive_quantization
            )
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

        jpegli_start_compress(&cinfo, 1)

        if iccsize > 0:
            jpegli_write_icc_profile(
                &cinfo, <const JOCTET*> icc_._data, iccsize
            )

        while cinfo.next_scanline < cinfo.image_height:
            rowpointer = <JSAMPROW> (
                <char*> src.data + cinfo.next_scanline * rowstride
            )
            jpegli_write_scanlines(&cinfo, &rowpointer, 1)

        jpegli_finish_compress(&cinfo)
        jpegli_destroy_compress(&cinfo)

    if out is None or outbuffer != <unsigned char*> dst._data:
        # outbuffer was allocated in jpegli_mem_dest
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


def jpegli_decode(
    data,
    /,
    *,
    tables=None,
    header=None,
    colorspace=None,
    outcolorspace=None,
    fancyupsampling=None,
    scale=None,
    dtype=None,
    out=None,
):
    """Return decoded JPEG image.

    Parameters:
        data:
            JPEG encoded image.
        tables:
            JPEG tables-only stream, for example, from TIFF JPEGTables tag.
        header:
            JPEG header to prepend to `data`.
        colorspace:
            Color space of JPEG image.
        outcolorspace:
            Color space of decoded image.
        fancyupsampling:
            Use fancy upsampling of chroma components.
        scale:
            Output scaling factor as (numerator, denominator) or denominator.
        dtype:
            Data type of decoded image: uint8 (default), uint16, or float32.
            Floating point samples are in range [0, 1].
            uint16 and float32 retain more precision than 8-bit samples.
        out:
            Optional output array.

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
        JSAMPROW rowpointer
        J_COLOR_SPACE jpeg_color_space
        J_COLOR_SPACE out_color_space
        JpegliDataType data_type = JPEGLI_TYPE_UINT8
        JpegliEndianness endianness = JPEGLI_NATIVE_ENDIAN
        char[200] msg  # JMSG_LENGTH_MAX
        int do_fancy_upsampling = 1
        unsigned int scale_num = 1
        unsigned int scale_denom = 1

    if data is out:
        raise ValueError('cannot transform in-place')

    if header is not None:
        data = b''.join((header, data, b'\xff\xd9'))

    src = data
    srcsize = <size_t> src.shape[0]

    if srcsize > ULONG_MAX:
        raise ValueError(f'{srcsize=} > {ULONG_MAX}')

    dtype = numpy.dtype('uint8' if dtype is None else dtype)
    if dtype.kind == 'u' and dtype.itemsize == 1:
        data_type = JPEGLI_TYPE_UINT8
    elif dtype.kind == 'u' and dtype.itemsize == 2:
        data_type = JPEGLI_TYPE_UINT16
    elif dtype.kind == 'f' and dtype.itemsize == 4:
        data_type = JPEGLI_TYPE_FLOAT
    else:
        raise ValueError(f'invalid {dtype=}')
    if dtype.byteorder == '>':
        endianness = JPEGLI_BIG_ENDIAN

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

    if scale is not None:
        if isinstance(scale, tuple | list):
            scale_num, scale_denom = scale[0], scale[1]
        else:
            scale_num, scale_denom = 1, int(scale)
        if scale_num < 1 or scale_denom < 1:
            raise ValueError(f'invalid {scale=}')

    with nogil:
        cinfo.err = jpegli_std_error(&err.pub)
        err.pub.error_exit = my_error_exit
        err.pub.output_message = my_output_message
        if setjmp(err.setjmp_buffer):
            msg[0] = b'\x00'
            err.pub.format_message(<jpeg_common_struct*> &cinfo, &msg[0])
            jpegli_destroy_decompress(&cinfo)
            raise JpegliError(msg.decode())

        jpegli_create_decompress(&cinfo)

        if tablesize > 0:
            jpegli_mem_src(
                &cinfo, <const unsigned char*> tables_._data, tablesize
            )
            jpegli_read_header(&cinfo, 0)  # == JPEG_HEADER_TABLES_ONLY

        jpegli_mem_src(
            &cinfo, <const unsigned char*> src._data, <unsigned long> srcsize
        )
        jpegli_read_header(&cinfo, 1)

        cinfo.do_fancy_upsampling = do_fancy_upsampling

        if jpeg_color_space != JCS_UNKNOWN:
            cinfo.jpeg_color_space = jpeg_color_space
        if out_color_space != JCS_UNKNOWN:
            cinfo.out_color_space = out_color_space

        cinfo.scale_num = scale_num
        cinfo.scale_denom = scale_denom

        jpegli_set_output_format(&cinfo, data_type, endianness)
        jpegli_start_decompress(&cinfo)

        with gil:
            if cinfo.output_components > 1:
                shape = (
                    cinfo.output_height,
                    cinfo.output_width,
                    cinfo.output_components
                )
            else:
                shape = cinfo.output_height, cinfo.output_width
            try:
                out = _create_array(out, shape, dtype)
            except BaseException:
                jpegli_destroy_decompress(&cinfo)
                raise
            dst = out
            rowstride = dst.strides[0]

        rowpointer = <JSAMPROW> dst.data
        while cinfo.output_scanline < cinfo.output_height:
            jpegli_read_scanlines(&cinfo, &rowpointer, 1)
            rowpointer += rowstride

        jpegli_finish_decompress(&cinfo)
        jpegli_destroy_decompress(&cinfo)

    return out


def jpegli_transcode(
    data,
    /,
    *,
    optimize=None,
    progressive=None,
    copy=None,
    restart=None,
    grayscale=None,
    out=None,
):
    """Return losslessly transcoded JPEG image.

    Arithmetic coded, lossless, and 12-bit JPEG are not supported.

    Parameters:
        data:
            JPEG-encoded input.
        optimize:
            Enable optimized entropy coding when True.
        progressive:
            Progressive level. True is 2, False is 0 (sequential).
            Defaults to 0.
        copy:
            Markers to copy: 'all', 'comments', 'icc', or 'none'.
            Defaults to 'all'.
        restart:
            Set restart interval in rows when greater than zero.
        grayscale:
            Convert three-component YCbCr image to grayscale by
            discarding chroma data. This changes image content.
        out:
            Optional output buffer.

    Returns:
        Transcoded JPEG image as bytes or in `out`.

    Raises:
        ValueError:
            If `copy` is not a supported marker policy or input is too large.
        JpegliError:
            If the input cannot be decoded or encoding fails.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst
        size_t srcsize = <size_t> src.shape[0]
        ssize_t dstsize
        my_error_mgr err
        jpeg_decompress_struct srcinfo
        jpeg_compress_struct dstinfo
        jpeg_saved_marker_ptr marker
        jvirt_barray_ptr* src_coef_arrays = NULL
        unsigned long outsize = 0
        unsigned char* outbuffer = NULL
        unsigned char* dstbuffer = NULL
        char[200] msg  # JMSG_LENGTH_MAX
        int m, sv_quant_tbl_no
        int optimize_coding = (
            -1 if optimize is None else 1 if optimize else 0
        )
        int progressive_level = -1
        int copy_option = 3  # default: all
        int restart_rows = 0 if restart is None else int(restart)
        int grayscale_flag = 0 if not grayscale else 1

    if srcsize > ULONG_MAX:
        raise ValueError(f'{srcsize=} > {ULONG_MAX}')

    if progressive is not None:
        if isinstance(progressive, bool):
            progressive_level = 2 if progressive else 0
        else:
            progressive_level = int(progressive)
            if progressive_level < 0:
                raise ValueError(f'invalid {progressive=!r}')

    if copy is not None:
        if copy == 'all':
            copy_option = 3
        elif copy == 'comments':
            copy_option = 1
        elif copy == 'icc':
            copy_option = 2
        elif copy == 'none':
            copy_option = 0
        else:
            raise ValueError(f'invalid {copy=}')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None and dstsize > 0:
        out = _create_output(outtype, dstsize)

    if out is not None:
        dst = out
        dstsize = dst.shape[0]
        outsize = <unsigned long> dst.nbytes
        outbuffer = <unsigned char*> dst._data
        dstbuffer = outbuffer

    with nogil:
        # jpegli_destroy is a no-op if mem is NULL
        srcinfo.mem = NULL
        dstinfo.mem = NULL
        dstinfo.dest = NULL

        srcinfo.err = jpegli_std_error(&err.pub)
        err.pub.error_exit = my_error_exit
        err.pub.output_message = my_output_message
        dstinfo.err = srcinfo.err  # share one error manager and setjmp buffer

        if setjmp(err.setjmp_buffer):
            msg[0] = b'\x00'
            err.pub.format_message(<jpeg_common_struct*> &srcinfo, &msg[0])
            if dstinfo.dest != NULL:
                # set outbuffer to the buffer currently used by jpegli_mem_dest
                dstinfo.dest.term_destination(&dstinfo)
            jpegli_destroy_decompress(&srcinfo)
            jpegli_destroy_compress(&dstinfo)
            if outbuffer != dstbuffer:
                free(outbuffer)
            raise JpegliError(msg.decode())

        jpegli_create_decompress(&srcinfo)
        jpegli_create_compress(&dstinfo)

        if copy_option == 1:
            jpegli_save_markers(&srcinfo, JPEG_COM, 0xFFFF)
        elif copy_option == 2:
            jpegli_save_markers(&srcinfo, JPEG_APP0 + 2, 0xFFFF)
        elif copy_option == 3:
            jpegli_save_markers(&srcinfo, JPEG_COM, 0xFFFF)
            m = 0
            while m < 16:
                jpegli_save_markers(&srcinfo, JPEG_APP0 + m, 0xFFFF)
                m += 1

        jpegli_mem_src(
            &srcinfo, <const unsigned char*> src._data, <unsigned long> srcsize
        )
        jpegli_read_header(&srcinfo, 1)

        src_coef_arrays = jpegli_read_coefficients(&srcinfo)
        jpegli_copy_critical_parameters(&srcinfo, &dstinfo)

        if grayscale_flag:
            if (
                dstinfo.jpeg_color_space == JCS_YCbCr
                and dstinfo.num_components == 3
            ):
                sv_quant_tbl_no = dstinfo.comp_info[0].quant_tbl_no
                jpegli_set_colorspace(&dstinfo, JCS_GRAYSCALE)
                # jpegli_set_colorspace resets quant table. restore from source
                dstinfo.comp_info[0].quant_tbl_no = sv_quant_tbl_no
            elif dstinfo.num_components == 1:
                dstinfo.comp_info[0].h_samp_factor = 1
                dstinfo.comp_info[0].v_samp_factor = 1
        if optimize_coding >= 0:
            dstinfo.optimize_coding = <boolean> optimize_coding
        if restart_rows > 0:
            dstinfo.restart_in_rows = restart_rows
        if progressive_level >= 0:
            jpegli_set_progressive_level(&dstinfo, progressive_level)

        jpegli_mem_dest(&dstinfo, &outbuffer, &outsize)
        jpegli_write_coefficients(&dstinfo, src_coef_arrays)

        marker = srcinfo.marker_list
        while marker != NULL:
            if copy_option == 0:
                marker = marker.next
                continue
            if copy_option == 1:
                if marker.marker != JPEG_COM:
                    marker = marker.next
                    continue
            elif copy_option == 2:
                if (
                    marker.marker != JPEG_APP0 + 2
                    or marker.data_length < 12
                    or memcmp(marker.data, b'ICC_PROFILE\x00', 12) != 0
                ):
                    marker = marker.next
                    continue
            else:
                # all: suppress duplicate JFIF/Adobe markers
                if dstinfo.write_JFIF_header:
                    if (
                        marker.marker == JPEG_APP0
                        and marker.data_length >= 5
                        and marker.data[0] == 0x4A  # 'J'
                        and marker.data[1] == 0x46  # 'F'
                        and marker.data[2] == 0x49  # 'I'
                        and marker.data[3] == 0x46  # 'F'
                        and marker.data[4] == 0
                    ):
                        marker = marker.next
                        continue
                if dstinfo.write_Adobe_marker:
                    if (
                        marker.marker == JPEG_APP0 + 14
                        and marker.data_length >= 5
                        and marker.data[0] == 0x41  # 'A'
                        and marker.data[1] == 0x64  # 'd'
                        and marker.data[2] == 0x6F  # 'o'
                        and marker.data[3] == 0x62  # 'b'
                        and marker.data[4] == 0x65  # 'e'
                    ):
                        marker = marker.next
                        continue
            jpegli_write_marker(
                &dstinfo, marker.marker, marker.data, marker.data_length
            )
            marker = marker.next

        jpegli_finish_compress(&dstinfo)
        jpegli_destroy_compress(&dstinfo)
        # srcinfo must outlive dstinfo finish (coef arrays in srcinfo memory)
        jpegli_finish_decompress(&srcinfo)
        jpegli_destroy_decompress(&srcinfo)

    if out is None or outbuffer != <unsigned char*> dst._data:
        # outbuffer was allocated by jpegli_mem_dest
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


ctypedef struct my_error_mgr:
    jpeg_error_mgr pub
    jmp_buf setjmp_buffer


cdef void my_error_exit(jpeg_common_struct* cinfo) noexcept nogil:
    cdef:
        my_error_mgr* error = <my_error_mgr*> deref(cinfo).err

    longjmp(deref(error).setjmp_buffer, 1)


cdef void my_output_message(jpeg_common_struct* cinfo) noexcept nogil:
    pass


cdef dict _JCS_COLORSPACE_SAMPLES = {
    # bitmask: bit N is set when N samples is valid
    JCS_UNKNOWN: 0b11110,  # 1, 2, 3, 4
    JCS_GRAYSCALE: 0b00010,  # 1
    JCS_RGB: 0b01000,  # 3
    JCS_YCbCr: 0b01000,  # 3
    JCS_CMYK: 0b10000,  # 4
    JCS_YCCK: 0b10000,  # 4
    JCS_EXT_RGB: 0b01000,  # 3
    JCS_EXT_RGBX: 0b10000,  # 4
    JCS_EXT_BGR: 0b01000,  # 3
    JCS_EXT_BGRX: 0b10000,  # 4
    JCS_EXT_XBGR: 0b10000,  # 4
    JCS_EXT_XRGB: 0b10000,  # 4
    JCS_EXT_RGBA: 0b10000,  # 4
    JCS_EXT_BGRA: 0b10000,  # 4
    JCS_EXT_ABGR: 0b10000,  # 4
    JCS_EXT_ARGB: 0b10000,  # 4
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
    'RGBA': JCS_EXT_RGBA,
    'BGR': JCS_EXT_BGR,
    'BGRA': JCS_EXT_BGRA,
    'ARGB': JCS_EXT_ARGB,
    'ABGR': JCS_EXT_ABGR,
    'RGBX': JCS_EXT_RGBX,
    'BGRX': JCS_EXT_BGRX,
    'XBGR': JCS_EXT_XBGR,
    'XRGB': JCS_EXT_XRGB,
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
}


cdef J_COLOR_SPACE _jcs_colorspace(colorspace, bint strict=False):
    """Return JCS colorspace value from user input."""
    if isinstance(colorspace, str):
        colorspace = colorspace.upper()
    if strict and colorspace not in _JCS_COLORSPACE_MAP:
        raise ValueError(f'invalid {colorspace=!r}')
    return _JCS_COLORSPACE_MAP.get(colorspace, JCS_UNKNOWN)
