# imagecodecs/_jpeg2k.pyx
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

"""JPEG2K (JPEG 2000) codec for the imagecodecs package."""

include '_shared.pxi'

from libc.math cimport log
from openjpeg cimport *


class JPEG2K:
    """JPEG2K codec constants."""

    available = True

    class CODEC(enum.IntEnum):
        """JPEG2K codec file formats."""

        JP2 = OPJ_CODEC_JP2
        J2K = OPJ_CODEC_J2K
        # JPT = OPJ_CODEC_JPT
        # JPP = OPJ_CODEC_JPP
        # JPX = OPJ_CODEC_JPX

    class CLRSPC(enum.IntEnum):
        """JPEG2K codec color spaces."""

        UNSPECIFIED = OPJ_CLRSPC_UNSPECIFIED
        SRGB = OPJ_CLRSPC_SRGB
        GRAY = OPJ_CLRSPC_GRAY
        SYCC = OPJ_CLRSPC_SYCC
        EYCC = OPJ_CLRSPC_EYCC
        CMYK = OPJ_CLRSPC_CMYK

    class EXTRA(enum.IntEnum):
        """JPEG2K extra sample types."""

        UNSPECIFIED = IC_EXTRA_UNSPECIFIED
        ASSOCALPHA = IC_EXTRA_ASSOCALPHA
        UNASSALPHA = IC_EXTRA_UNASSALPHA


class Jpeg2kError(RuntimeError):
    """JPEG2K codec exceptions."""

    def __init__(self, msg, error=None):
        if error:
            msg = f'{msg}: {error.decode(errors="replace").strip()}'
        super().__init__(msg)


def jpeg2k_version():
    """Return OpenJPEG library version string."""
    return 'openjpeg ' + opj_version().decode()


def jpeg2k_check(const uint8_t[::1] data not None, /):
    """Return whether data is JPEG2K encoded image or None if unknown."""
    return (
        # JP2/JPX
        _startswith(data, b'\x00\x00\x00\x0c\x6a\x50\x20\x20\x0d\x0a\x87\x0a')
        or _startswith(data, b'\xff\x4f\xff\x51')  # J2K
        or _startswith(data, b'\x0d\x0a\x87\x0a')  # JP2
    )


def jpeg2k_encode(
    data,
    /,
    level=None,  # quality, psnr, level < 1 or > 1000 map to quality=0
    *,
    codecformat=None,
    colorspace=None,  # input interpretation, like photometric in other codecs
    extrasample=None,
    planar=None,
    tile=None,  # not implemented
    bitspersample=None,
    resolutions=None,
    reversible=None,
    mct=True,  # multiple component transform: rgb->ycc
    verbose=None,
    numthreads=None,
    out=None,
):
    """Return JPEG2K encoded image.

    WIP: currently only single-tile, single-quality-layer formats are supported

    Level values < 1 or > 1000 select lossless compression with the
    reversible 5/3 wavelet, unless `reversible` is explicitly False, in which
    case the irreversible 9/7 wavelet is used and the result is not lossless.

    Precision is limited to 26 bits. 32-bit integer data must fit into the
    26-bit range. This is not checked. Values outside are silently corrupted.

    If `colorspace` is specified, it determines the number of color samples.
    Remaining samples are extra samples. A single extra sample is assumed to
    be unassociated alpha unless `extrasample` is specified.
    OpenJPEG writes associated alpha as unassociated alpha.
    For example, encode 4-sample uint8 data as gray with 3 unspecified extra
    samples using ``colorspace='gray', extrasample='unspecified'``, or as RGB
    with 1 unspecified extra sample using ``extrasample='unspecified'``.

    Parameters:
        data:
            Image to encode.
            Must be 8, 16, or 32-bit integer array of shape (height, width)
            or (height, width, samples).
        level:
            Quality of lossy compression as PSNR in dB in range 1 to 1000.
            Other values select lossless compression.
            Defaults to lossless compression.
        codecformat:
            File format: JPEG2K.CODEC or its name, 'JP2' or 'J2K'.
            Defaults to 'JP2'.
        colorspace:
            Color space of `data`: JPEG2K.CLRSPC or its name.
            Defaults to 'SRGB' for 3 or more samples, else 'GRAY'.
        extrasample:
            Type of extra samples: JPEG2K.EXTRA or its name.
        planar:
            Data is in planar configuration (samples, height, width).
        tile:
            Not implemented.
        bitspersample:
            Number of significant bits per sample in range 1 to 26.
            Defaults to bit depth of data type, limited to 26.
        resolutions:
            Maximum number of resolution levels.
            Limited by image size. Defaults to 6.
        reversible:
            Use reversible 5/3 wavelet when True, else irreversible 9/7.
            Defaults to True for lossless and False for lossy compression.
        mct:
            Apply multiple component transform RGB to YCC to 3-sample
            sRGB images. Defaults to True.
        verbose:
            Verbosity level: 1 logs errors, 2 also warnings, 3 also info.
        numthreads:
            Number of threads to use.
            0 uses half the number of CPUs.
            Defaults to 1.
        out:
            Optional output buffer.

    Returns:
        JPEG2K encoded image as bytes or in `out`.

    Raises:
        ValueError:
            If data shape, dtype, or parameters are invalid.
        NotImplementedError:
            If `tile` is specified.
        Jpeg2kError:
            If encoding fails.

    """
    cdef:
        numpy.ndarray src = _array_input(data, out)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        ssize_t srcsize = src.nbytes
        ssize_t itemsize = src.itemsize
        ssize_t byteswritten, size, i
        memopj_t memopj
        j2k_msg_t msg
        opj_codec_t* codec = NULL
        opj_image_t* image = NULL
        opj_stream_t* stream = NULL
        opj_image_cmptparm_t* cmptparms = NULL
        opj_cparameters_t parameters
        OPJ_CODEC_FORMAT codec_format
        OPJ_BOOL ret = OPJ_TRUE
        OPJ_COLOR_SPACE color_space = OPJ_CLRSPC_UNSPECIFIED
        OPJ_UINT32 sgnd, prec, width, height, samples
        OPJ_UINT16 alpha = 0
        int verbosity = int(verbose) if verbose else 0
        # int tile_width = 0
        # int tile_height = 0
        float quality = _default_value(level, 0, 0, None)
        int numresolution = _default_value(resolutions, 6, 1, OPJ_J2K_MAXRLVLS)
        int num_threads = _default_threads(numthreads)
        int irreversible = 0 if reversible else 1
        bint tcp_mct = bool(mct)
        imagelayout_t layout

    if colorspace is None:
        photometric = None
    else:
        color_space = _enum_value(
            colorspace, JPEG2K.CLRSPC, OPJ_CLRSPC_UNSPECIFIED
        )
        photometric = _CLRSPC_PHOTOMETRIC.get(color_space)

    _image_layout(
        IC_UINT
        | IC_SINT
        | IC_SZ1
        | IC_SZ2
        | IC_SZ4
        | IC_GRAY
        | IC_RGB
        | IC_CMYK
        | IC_YCBCR
        | IC_ALPHA
        | IC_EXTRA
        | IC_PLANAR
        | IC_BPS,
        src.ndim,
        src.shape,
        src.dtype,
        photometric,
        bitspersample,
        planar,
        None,  # frames
        None,  # volumetric
        _enum_value(extrasample, JPEG2K.EXTRA, None),
        &layout,
    )

    if <uint64_t> srcsize > UINT32_MAX:
        raise ValueError('image size must not exceed 4 GB')

    if quality < 1 or quality > 1000:
        # lossless
        quality = 0.0
        if reversible is None:
            irreversible = 0

    # use container format by default
    codec_format = _enum_value(codecformat, JPEG2K.CODEC, OPJ_CODEC_JP2)

    sgnd = 1 if src.dtype.kind == 'i' else 0
    prec = <OPJ_UINT32> layout.bitspersample
    width = <OPJ_UINT32> layout.width
    height = <OPJ_UINT32> layout.height
    samples = <OPJ_UINT32> layout.samples

    if prec > 26:
        # OpenJPEG effectively supports only up to 26-bit precision
        if bitspersample is not None:
            raise ValueError(f'{bitspersample=} > 26 not supported')
        prec = 26

    if tile:
        # tile_height, tile_width = tile
        # if width % tile_width or height % tile_height:
        #     raise ValueError('invalid tiles')
        raise NotImplementedError('writing tiles not implemented yet')  # TODO

    if samples > 4095:
        # TODO: check this limit
        raise ValueError(f'invalid number of {samples=}')

    if colorspace is None:
        if layout.photometric == IC_PHOTO_RGB:
            color_space = OPJ_CLRSPC_SRGB
        else:
            color_space = OPJ_CLRSPC_GRAY

    if layout.extratype == IC_EXTRA_ASSOCALPHA:
        alpha = 2
    elif layout.extratype == IC_EXTRA_UNASSALPHA:
        alpha = 1

    memset(&msg, 0, sizeof(j2k_msg_t))
    msg.verbose = verbosity

    # create memory stream
    memopj.data = NULL
    memopj.size = 0
    memopj.offset = 0
    memopj.written = 0
    memopj.owner = 0

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = _align_ssize_t(srcsize // (2 if quality == 0.0 else 4))
            memopj.data = <OPJ_UINT8*> malloc(dstsize)
            if memopj.data == NULL:
                raise MemoryError('failed to allocate output buffer')
            memopj.size = dstsize
            memopj.owner = 1
        else:
            out = _create_output(outtype, dstsize)

    if out is not None:
        dst = out
        dstsize = dst.shape[0]
        memopj.data = <OPJ_UINT8*> dst._data
        memopj.size = dstsize

    try:
        with nogil:
            stream = opj_memstream_create(&memopj, OPJ_FALSE)
            if stream == NULL:
                raise Jpeg2kError('opj_memstream_create failed')

            # create image
            cmptparms = <opj_image_cmptparm_t*> calloc(
                samples, sizeof(opj_image_cmptparm_t)
            )
            if cmptparms == NULL:
                raise MemoryError('failed to allocate cmptparms')
            for i in range(<ssize_t> samples):
                cmptparms[i].dx = 1  # subsampling
                cmptparms[i].dy = 1
                cmptparms[i].h = height
                cmptparms[i].w = width
                cmptparms[i].x0 = 0
                cmptparms[i].y0 = 0
                cmptparms[i].prec = prec
                # cmptparms[i].bpp = prec  # redundant, not required
                cmptparms[i].sgnd = sgnd

            # if tile_height > 0:
            #    image = opj_image_tile_create(samples, cmptparms, color_space)
            #    if image == NULL:
            #        raise Jpeg2kError('opj_image_tile_create failed')

            image = opj_image_create(samples, cmptparms, color_space)
            if image == NULL:
                raise Jpeg2kError('opj_image_create failed')

            image.x0 = 0
            image.y0 = 0
            image.x1 = width
            image.y1 = height
            image.color_space = color_space
            image.numcomps = samples

            if alpha > 0 and layout.extraindex < samples:
                image.comps[layout.extraindex].alpha = alpha

            # Set encoding parameters to default values, that means:
            # Lossless
            # 1 tile
            # Size of precinct : 2^15 x 2^15 (means 1 precinct)
            # Size of code-block : 64 x 64
            # Number of resolutions: 6
            # No SOP marker in the codestream
            # No EPH marker in the codestream
            # No sub-sampling in x or y direction
            # No mode switch activated
            # Progression order: LRCP
            # No index file
            # No ROI upshifted
            # No offset of the origin of the image
            # No offset of the origin of the tiles
            # Reversible DWT 5-3
            opj_set_default_encoder_parameters(&parameters)

            # single quality layer
            parameters.tcp_numlayers = 1

            # number of resolutions depends on image size
            parameters.numresolution = <int> (
                (log(<double> min(width, height)) / log(2)) - 2
            )
            parameters.numresolution = min(
                max(parameters.numresolution, 1), numresolution
            )

            # multiple component transform: rgb->ycc
            if samples == 3 and color_space == OPJ_CLRSPC_SRGB:
                parameters.tcp_mct = <char> tcp_mct

            if quality == 0.0:
                # lossless
                parameters.irreversible = irreversible
                parameters.cp_disto_alloc = 1
                parameters.tcp_rates[0] = 0
            else:
                # lossy
                parameters.irreversible = irreversible

                # progression order resolution-position-component-layer
                parameters.prog_order = OPJ_RPCL

                # code block width and height
                # parameters.cblockw_init = 64
                # parameters.cblockh_init = 64

                # fixed quality
                parameters.cp_fixed_quality = 1
                parameters.tcp_distoratio[0] = quality

                # fixed rate
                # parameters.cp_disto_alloc = 1
                # parameters.tcp_rates[0] = quality

                # precinct width and height
                parameters.res_spec = parameters.numresolution
                for i in range(parameters.res_spec):
                    parameters.prcw_init[i] = 256
                    parameters.prch_init[i] = 256
                parameters.csty = 1

            # if tile_height > 0:
            #     parameters.tile_size_on = OPJ_TRUE
            #     parameters.cp_tx0 = 0
            #     parameters.cp_ty0 = 0
            #     parameters.cp_tdy = tile_height
            #     parameters.cp_tdx = tile_width

            # create and setup encoder
            codec = opj_create_compress(codec_format)
            if codec == NULL:
                raise Jpeg2kError('opj_create_compress failed')

            opj_set_error_handler(
                codec, <opj_msg_callback> j2k_error_callback, &msg
            )
            if verbosity > 1:
                opj_set_warning_handler(
                    codec, <opj_msg_callback> j2k_warning_callback, NULL
                )
                if verbosity > 2:
                    opj_set_info_handler(
                        codec, <opj_msg_callback> j2k_info_callback, NULL
                    )

            ret = opj_setup_encoder(codec, &parameters, image)
            if ret == OPJ_FALSE:
                raise Jpeg2kError('opj_setup_encoder failed', msg.error)

            if num_threads != 1 and opj_has_thread_support():
                if num_threads == 0:
                    num_threads = opj_get_num_cpus() // 2
                    num_threads = 1 if num_threads < 1 else num_threads
                ret = opj_codec_set_threads(codec, num_threads)
                if ret == OPJ_FALSE:
                    raise Jpeg2kError('opj_codec_set_threads failed')

            size = <ssize_t> width * <ssize_t> height
            if itemsize == 1:
                if sgnd:
                    _opj_fill_image_comps(
                        <int8_t*> src.data,
                        image.comps,
                        size,
                        <ssize_t> samples,
                        not layout.planar
                    )
                else:
                    _opj_fill_image_comps(
                        <uint8_t*> src.data,
                        image.comps,
                        size,
                        <ssize_t> samples,
                        not layout.planar
                    )
            elif itemsize == 2:
                if sgnd:
                    _opj_fill_image_comps(
                        <int16_t*> src.data,
                        image.comps,
                        size,
                        <ssize_t> samples,
                        not layout.planar
                    )
                else:
                    _opj_fill_image_comps(
                        <uint16_t*> src.data,
                        image.comps,
                        size,
                        <ssize_t> samples,
                        not layout.planar
                    )
            else:
                _opj_fill_image_comps(
                    <int32_t*> src.data,
                    image.comps,
                    size,
                    <ssize_t> samples,
                    not layout.planar
                )

            ret = opj_start_compress(codec, image, stream)
            if ret == OPJ_FALSE:
                raise Jpeg2kError('opj_start_compress failed', msg.error)

            ret = opj_encode(codec, stream)
            if ret == OPJ_FALSE:
                raise Jpeg2kError('opj_encode failed', msg.error)

            ret = opj_end_compress(codec, stream)
            if ret == OPJ_FALSE:
                raise Jpeg2kError('opj_end_compress failed', msg.error)

            # superfluous
            if memopj.written > memopj.size:
                raise Jpeg2kError('output buffer too small')

            byteswritten = <ssize_t> memopj.written

    except Exception:
        if memopj.owner:
            free(memopj.data)
        raise
    finally:
        if stream != NULL:
            opj_stream_destroy(stream)
        if codec != NULL:
            opj_destroy_codec(codec)
        if image != NULL:
            opj_image_destroy(image)
        if cmptparms != NULL:
            free(cmptparms)

    if out is None:
        try:
            out = _create_output(
                outtype, byteswritten, <const char*> memopj.data
            )
        finally:
            free(memopj.data)
        return out

    del dst
    return _return_output(out, dstsize, byteswritten, outgiven)


def jpeg2k_decode(
    data,
    /,
    *,
    planar=None,
    skipres=None,
    verbose=None,
    numthreads=None,
    out=None,
    bint _info=False,
):
    """Return decoded JPEG2K image.

    YCC images are converted to RGB. ICC profiles and CIELab are applied.

    Parameters:
        data:
            JPEG2K encoded image, J2K codestream or JP2 file.
        planar:
            Return image in planar configuration (samples, height, width)
            when True.
        skipres:
            Number of highest resolution levels to discard.
            Image dimensions are reduced by factor 2**skipres.
        verbose:
            Verbosity level: 1 logs errors, 2 also warnings, 3 also info.
        numthreads:
            Number of threads to use.
            0 uses half the number of CPUs.
            Defaults to 1.
        out:
            Optional output array.

    Returns:
        Decoded image as 8, 16, or 32-bit integer array of shape
        (height, width), (height, width, samples), or
        (samples, height, width).

    Raises:
        NotImplementedError:
            If components differ in dtype or subsampling.
        Jpeg2kError:
            If data is not JPEG2K or decoding fails.

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t itemsize
        memopj_t memopj
        j2k_msg_t msg
        opj_codec_t* codec = NULL
        opj_image_t* image = NULL
        opj_stream_t* stream = NULL
        opj_image_comp_t* comp = NULL
        opj_dparameters_t parameters
        OPJ_BOOL ret = OPJ_FALSE
        OPJ_CODEC_FORMAT codecformat
        OPJ_UINT32 sgnd, prec, width, height
        OPJ_UINT32 cp_reduce = 0 if skipres is None else <OPJ_UINT32> skipres
        ssize_t i, bandsize, samples
        int num_threads = _default_threads(numthreads)
        int verbosity = int(verbose) if verbose else 0
        bint contig = not planar

    if data is out:
        raise ValueError('cannot transform in-place')

    if _startswith(src, b'\xff\x4f\xff\x51'):
        codecformat = OPJ_CODEC_J2K
    elif _startswith(src, b'\x0d\x0a\x87\x0a'):
        codecformat = OPJ_CODEC_JP2
    elif _startswith(
        src, b'\x00\x00\x00\x0c\x6a\x50\x20\x20\x0d\x0a\x87\x0a'
    ):
        # OPJ_CODEC_JPX support not implemented in OpenJPEG
        # if b'jpx ' in bytes(sig[12:24]):
        #     codecformat = OPJ_CODEC_JPX
        # else:
        #     codecformat = OPJ_CODEC_JP2
        codecformat = OPJ_CODEC_JP2
    else:
        raise Jpeg2kError('not a J2K or JP2 data stream')

    memset(&msg, 0, sizeof(j2k_msg_t))
    msg.verbose = verbosity

    try:
        memopj.data = <OPJ_UINT8*> src._data
        memopj.size = src.shape[0]
        memopj.offset = 0
        memopj.written = 0
        memopj.owner = 0

        with nogil:
            stream = opj_memstream_create(&memopj, OPJ_TRUE)
            if stream == NULL:
                raise Jpeg2kError('opj_memstream_create failed')

            codec = opj_create_decompress(codecformat)
            if codec == NULL:
                raise Jpeg2kError('opj_create_decompress failed')

            opj_set_error_handler(
                codec, <opj_msg_callback> j2k_error_callback, &msg
            )
            if verbosity > 1:
                opj_set_warning_handler(
                    codec, <opj_msg_callback> j2k_warning_callback, NULL
                )
                if verbosity > 2:
                    opj_set_info_handler(
                        codec, <opj_msg_callback> j2k_info_callback, NULL
                    )

            opj_set_default_decoder_parameters(&parameters)
            parameters.cp_reduce = cp_reduce

            ret = opj_setup_decoder(codec, &parameters)
            if ret == OPJ_FALSE:
                raise Jpeg2kError('opj_setup_decoder failed')

            if num_threads != 1 and opj_has_thread_support():
                if num_threads == 0:
                    num_threads = opj_get_num_cpus() // 2
                    num_threads = 1 if num_threads < 1 else num_threads
                ret = opj_codec_set_threads(codec, num_threads)
                if ret == OPJ_FALSE:
                    raise Jpeg2kError('opj_codec_set_threads failed')

            ret = opj_read_header(stream, codec, &image)
            if ret == OPJ_FALSE:
                raise Jpeg2kError('opj_read_header failed', msg.error)
            if image.numcomps == 0 or image.comps == NULL:
                raise Jpeg2kError('image has no components')

            if _info:
                with gil:
                    return _jpeg2k_info(
                        codec, image, codecformat, cp_reduce, not contig
                    )

            # ret = opj_set_decode_area(
            #     codec,
            #     image,
            #     <OPJ_INT32> parameters.DA_x0,
            #     <OPJ_INT32> parameters.DA_y0,
            #     <OPJ_INT32> parameters.DA_x1,
            #     <OPJ_INT32> parameters.DA_y1
            # )
            # if ret == OPJ_FALSE:
            #     raise Jpeg2kError('opj_set_decode_area failed')

            ret = opj_decode(codec, stream, image)
            if ret != OPJ_FALSE:
                ret = opj_end_decompress(codec, stream)
            if ret == OPJ_FALSE:
                raise Jpeg2kError(
                    'opj_decode or opj_end_decompress failed', msg.error
                )

            # handle subsampling and color profiles
            # This heuristic tries to auto-detect SYCC but may be fragile.
            # Assume that if component 1 has different subsampling,
            # it must be SYCC.
            if (
                image.color_space != OPJ_CLRSPC_SYCC
                and image.numcomps == 3
                and image.comps[0].dx == image.comps[0].dy
                and image.comps[1].dx != 1
            ):
                image.color_space = OPJ_CLRSPC_SYCC
            elif image.numcomps <= 2:
                image.color_space = OPJ_CLRSPC_GRAY

            if image.color_space == OPJ_CLRSPC_SYCC:
                color_sycc_to_rgb(image)
            # icc_profile_buf is freed by opj_image_destroy
            if image.icc_profile_buf:
                if image.icc_profile_len:
                    color_apply_icc_profile(image)
                else:
                    color_cielab_to_rgb(image)

            comp = &image.comps[0]
            sgnd = comp.sgnd
            prec = comp.prec
            height = comp.h
            width = comp.w
            samples = <ssize_t> image.numcomps
            itemsize = (prec + 7) // 8

            for i in range(samples):
                comp = &image.comps[i]
                if comp.sgnd != sgnd or comp.prec != prec:
                    raise NotImplementedError('components dtype mismatch')
                    # TODO: support upcast
                    # use scale_component
                if comp.w != width or comp.h != height:
                    raise NotImplementedError('subsampling not supported')
                    # TODO: support upsampling
                    # use upsample_image_components in opj_decompress.c
            if itemsize == 3:
                itemsize = 4
            elif itemsize < 1 or itemsize > 4:
                raise Jpeg2kError(f'unsupported {itemsize=}')

        dtype = '{}{}'.format('i' if sgnd else 'u', itemsize)
        if samples == 1:
            shape = int(height), int(width)
            contig = 0
        elif contig:
            shape = int(height), int(width), int(samples)
        else:
            shape = int(samples), int(height), int(width)

        out = _create_array(out, shape, dtype)
        dst = out

        with nogil:
            comp = image.comps
            bandsize = <ssize_t> height * <ssize_t> width
            if itemsize == 1:
                if sgnd:
                    _opj_copy_image_comps(
                        <int8_t*> dst.data, comp, bandsize, samples, contig
                    )
                else:
                    _opj_copy_image_comps(
                        <uint8_t*> dst.data, comp, bandsize, samples, contig
                    )
            elif itemsize == 2:
                if sgnd:
                    _opj_copy_image_comps(
                        <int16_t*> dst.data, comp, bandsize, samples, contig
                    )
                else:
                    _opj_copy_image_comps(
                        <uint16_t*> dst.data, comp, bandsize, samples, contig
                    )
            elif not contig:
                for i in range(samples):
                    memcpy(
                        <void*> &dst.data[i * bandsize * 4],
                        <const void*> comp[i].data,
                        bandsize * 4
                    )
            elif sgnd:
                _opj_copy_image_comps(
                    <int32_t*> dst.data, comp, bandsize, samples, contig
                )
            else:
                _opj_copy_image_comps(
                    <uint32_t*> dst.data, comp, bandsize, samples, contig
                )

    finally:
        if stream != NULL:
            opj_stream_destroy(stream)
        if codec != NULL:
            opj_destroy_codec(codec)
        if image != NULL:
            opj_image_destroy(image)

    return out


def jpeg2k_info(
    data,
    /,
    *,
    planar=None,
    skipres=None,
    verbose=None,
):
    """Return information about JPEG2K image.

    Parameters are the same as for :py:func:`jpeg2k_decode`.

    Returns:
        Dict of image properties.
        The shape and dtype match the output of `jpeg2k_decode` called
        with the same `planar` and `skipres` arguments.
        Other items can be passed to `jpeg2k_encode`.

    """
    return jpeg2k_decode(
        data,
        planar=planar,
        skipres=skipres,
        verbose=verbose,
        _info=True,
    )


cdef dict _jpeg2k_info(
    opj_codec_t* codec,
    opj_image_t* image,
    OPJ_CODEC_FORMAT codecformat,
    OPJ_UINT32 reduce,
    bint planar,
):
    """Return information from JPEG2K header."""
    cdef:
        opj_codestream_info_v2_t* cstr_info = NULL
        opj_tccp_info_t* tccp = NULL
        opj_image_comp_t* comp = &image.comps[0]
        OPJ_UINT32 width, height
        ssize_t i, itemsize
        ssize_t samples = <ssize_t> image.numcomps

    itemsize = (comp.prec + 7) // 8
    if itemsize == 3:
        itemsize = 4
    elif itemsize < 1 or itemsize > 4:
        raise Jpeg2kError(f'unsupported {itemsize=}')
    dtype = numpy.dtype(f'{"i" if comp.sgnd else "u"}{itemsize}')

    # component size after skipping resolutions, see opj_image_comp_t
    width = _ceildivpow2(
        _ceildiv(image.x1, comp.dx) - _ceildiv(image.x0, comp.dx), reduce
    )
    height = _ceildivpow2(
        _ceildiv(image.y1, comp.dy) - _ceildiv(image.y0, comp.dy), reduce
    )
    if samples == 1:
        shape = int(height), int(width)
    elif planar:
        shape = int(samples), int(height), int(width)
    else:
        shape = int(height), int(width), int(samples)

    subsampling = tuple(
        (int(image.comps[i].dy), int(image.comps[i].dx))
        for i in range(samples)
    )
    if all(s == (1, 1) for s in subsampling):
        subsampling = None

    iccprofile = None
    if image.icc_profile_buf != NULL and image.icc_profile_len > 0:
        iccprofile = (<char*> image.icc_profile_buf)[:image.icc_profile_len]

    cstr_info = opj_get_cstr_info(codec)
    try:
        if (
            cstr_info == NULL
            or cstr_info.m_default_tile_info.tccp_info == NULL
        ):
            raise Jpeg2kError('opj_get_cstr_info failed')
        tccp = &cstr_info.m_default_tile_info.tccp_info[0]
        # no extrasample: OpenJPEG applies the JP2 cdef box only on decode
        return {
            '_codec': 'jpeg2k',
            'shape': shape,
            'dtype': dtype,
            'codecformat': JPEG2K.CODEC(codecformat),
            'colorspace': _enum_or_int(image.color_space, JPEG2K.CLRSPC),
            'bitspersample': int(comp.prec),
            'resolutions': int(tccp.numresolutions),
            'reversible': tccp.qmfbid == 1,
            'mct': bool(cstr_info.m_default_tile_info.mct),
            'tile': (
                (int(cstr_info.tdy), int(cstr_info.tdx))
                if cstr_info.tw * cstr_info.th > 1
                else None
            ),
            'layers': int(cstr_info.m_default_tile_info.numlayers),
            'progression': _PROG_ORDER.get(
                cstr_info.m_default_tile_info.prg, None
            ),
            'subsampling': subsampling,
            'iccprofile': iccprofile,
        }
    finally:
        if cstr_info != NULL:
            opj_destroy_cstr_info(&cstr_info)


cdef inline OPJ_UINT32 _ceildiv(OPJ_UINT32 a, OPJ_UINT32 b) noexcept nogil:
    """Return a divided by b, rounded up."""
    return (a + b - 1) // b if b else 0


cdef inline OPJ_UINT32 _ceildivpow2(
    OPJ_UINT32 a, OPJ_UINT32 b
) noexcept nogil:
    """Return a divided by 2**b, rounded up."""
    return (a + (1 << b) - 1) >> b if b < 32 else 0


ctypedef fused data_t:
    int8_t
    uint8_t
    int16_t
    uint16_t
    int32_t
    uint32_t


cdef int _opj_copy_image_comps(
    data_t* dst,
    const opj_image_comp_t* comps,
    const ssize_t bandsize,
    const ssize_t samples,
    const bint contig,
) noexcept nogil:
    """Copy opj_image component data to buffer."""
    cdef:
        ssize_t i, j
        OPJ_INT32* band
        data_t* dstband

    if samples == 1:
        band = comps[0].data
        for j in range(bandsize):
            dst[j] = <data_t> band[j]
    elif contig:
        for i in range(samples):
            band = comps[i].data
            for j in range(bandsize):
                dst[j * samples + i] = <data_t> band[j]
    else:
        for i in range(samples):
            band = comps[i].data
            dstband = dst + i * bandsize
            for j in range(bandsize):
                dstband[j] = <data_t> band[j]


cdef int _opj_fill_image_comps(
    const data_t* src,
    opj_image_comp_t* comps,
    const ssize_t bandsize,
    const ssize_t samples,
    const bint contig,
) noexcept nogil:
    """Fill opj_image component data from buffer."""
    cdef:
        ssize_t i, j
        OPJ_INT32* band
        const data_t* srcband

    if samples == 1:
        band = comps[0].data
        for j in range(bandsize):
            band[j] = <OPJ_INT32> src[j]
    elif contig:
        for i in range(samples):
            band = comps[i].data
            for j in range(bandsize):
                band[j] = <OPJ_INT32> src[j * samples + i]
    else:
        for i in range(samples):
            band = comps[i].data
            srcband = src + i * bandsize
            for j in range(bandsize):
                band[j] = <OPJ_INT32> srcband[j]


ctypedef struct memopj_t:
    OPJ_UINT8* data
    OPJ_UINT64 size
    OPJ_UINT64 offset
    OPJ_UINT64 written
    int owner


cdef memopj_t* memopj_new(uint8_t* data, size_t size) noexcept nogil:
    """Return new memopj."""
    cdef:
        memopj_t* memopj = <memopj_t*> calloc(1, sizeof(memopj_t))

    if memopj == NULL:
        return NULL
    memopj.size = size
    memopj.written = 0
    memopj.offset = 0
    if data == NULL:
        memopj.owner = 1
        memopj.data = <OPJ_UINT8*> malloc(size)
    else:
        memopj.owner = 0
        memopj.data = <OPJ_UINT8*> data
    if memopj.data == NULL:
        free(memopj)
        return NULL
    return memopj


cdef OPJ_BOOL opj_mem_resize(
    memopj_t* memopj,
    OPJ_UINT64 newsize,
) noexcept nogil:
    """Reallocate buffer to at least fit newsize bytes."""
    cdef:
        OPJ_UINT8* tmp
        size_t alignedsize

    if newsize <= memopj.size:
        return OPJ_TRUE
    if not memopj.owner:
        return OPJ_FALSE
    # always over-allocate to avoid repeated reallocations
    newsize += newsize // 4
    if newsize > SIZE_MAX - 65536:
        return OPJ_FALSE
    alignedsize = _align_size_t(<size_t> newsize)
    tmp = <OPJ_UINT8*> realloc(<void*> memopj.data, alignedsize)
    if tmp == NULL:
        return OPJ_FALSE
    memopj.data = tmp
    memopj.size = alignedsize
    return OPJ_TRUE


cdef OPJ_SIZE_T opj_mem_read(
    void* dst,
    OPJ_SIZE_T size,
    void* data,
) noexcept nogil:
    """opj_stream_set_read_function."""
    cdef:
        memopj_t* memopj = <memopj_t*> data

    if memopj.offset >= memopj.size:
        return <OPJ_SIZE_T> -1
    if memopj.offset + size > memopj.size:
        size = <OPJ_SIZE_T> (memopj.size - memopj.offset)
    memcpy(<void*> dst, <const void*> &(memopj.data[memopj.offset]), size)
    memopj.offset += size
    return size


cdef OPJ_SIZE_T opj_mem_write(
    void* dst,
    OPJ_SIZE_T size,
    void* data,
) noexcept nogil:
    """opj_stream_set_write_function."""
    cdef:
        memopj_t* memopj = <memopj_t*> data

    if memopj.offset + size > memopj.size:
        if opj_mem_resize(memopj, memopj.offset + size) != OPJ_TRUE:
            return <OPJ_SIZE_T> -1  # error
    memcpy(<void*> &(memopj.data[memopj.offset]), <const void*> dst, size)
    memopj.offset += size
    if memopj.written < memopj.offset:
        memopj.written = memopj.offset
    return size


cdef OPJ_BOOL opj_mem_seek(
    OPJ_OFF_T offset,
    void* data,
) noexcept nogil:
    """opj_stream_set_seek_function."""
    cdef:
        memopj_t* memopj = <memopj_t*> data

    if offset < 0:
        return OPJ_FALSE

    # if offset > <OPJ_OFF_T> memopj.size:
    #     if opj_mem_resize(memopj, <OPJ_SIZE_T> offset) != OPJ_TRUE:
    #         memopj.offset = memopj.size
    #         if memopj.written < memopj.size:
    #             memopj.written = memopj.size
    #         return OPJ_TRUE
    # memopj.offset = <OPJ_SIZE_T> offset
    # if memopj.written < memopj.offset:
    #     memopj.written = memopj.offset

    # allow seek beyond memopj.size; subsequent read or write might fail
    memopj.offset = <OPJ_SIZE_T> offset
    return OPJ_TRUE


cdef OPJ_OFF_T opj_mem_skip(
    OPJ_OFF_T size,
    void* data,
) noexcept nogil:
    """opj_stream_set_skip_function."""
    cdef:
        memopj_t* memopj = <memopj_t*> data
    if size < 0:
        return <OPJ_OFF_T> -1

    # if memopj.offset + size > memopj.size:
    #     if opj_mem_resize(
    #         memopj, <OPJ_SIZE_T> memopj.offset + size
    #     ) != OPJ_TRUE:
    #         # return <OPJ_OFF_T> -1  # error
    #         size = memopj.size - memopj.offset
    #     memopj.offset = memopj.size
    # else:
    #     memopj.offset += size
    # if memopj.written < memopj.offset:
    #     memopj.written = memopj.offset

    # allow seek beyond memopj.size; subsequent read or write might fail
    memopj.offset += size
    return size


cdef void opj_mem_nop(void* data) noexcept nogil:
    """opj_stream_set_user_data."""


cdef opj_stream_t* opj_memstream_create(
    memopj_t* memopj,
    OPJ_BOOL isinput,
) noexcept nogil:
    """Return OPJ stream using memory as input or output."""
    cdef:
        opj_stream_t* stream = opj_stream_default_create(isinput)

    if stream == NULL:
        return NULL
    if isinput:
        opj_stream_set_read_function(stream, <opj_stream_read_fn> opj_mem_read)
    else:
        opj_stream_set_write_function(
            stream, <opj_stream_write_fn> opj_mem_write
        )
    opj_stream_set_seek_function(stream, <opj_stream_seek_fn> opj_mem_seek)
    opj_stream_set_skip_function(stream, <opj_stream_skip_fn> opj_mem_skip)
    opj_stream_set_user_data(
        stream, memopj, <opj_stream_free_user_data_fn> opj_mem_nop
    )
    opj_stream_set_user_data_length(stream, memopj.size)
    return stream


ctypedef struct j2k_msg_t:
    int verbose
    char error[256]


cdef void _j2k_log(const char* level, const char* msg) noexcept with gil:
    try:
        _log_warning(
            'JPEG2K %s: %s',
            level.decode(),
            msg.decode(errors='replace').strip(),
        )
    except Exception:
        pass


cdef void j2k_error_callback(
    const char* msg,
    void* client_data,
) noexcept nogil:
    cdef:
        j2k_msg_t* m = <j2k_msg_t*> client_data

    if m.error[0] == 0:
        strncpy(m.error, msg, 255)
    if m.verbose > 0:
        _j2k_log(b'error', msg)


cdef void j2k_warning_callback(
    const char* msg,
    void* client_data,
) noexcept nogil:
    _j2k_log(b'warning', msg)


cdef void j2k_info_callback(
    const char* msg,
    void* client_data,
) noexcept nogil:
    _j2k_log(b'info', msg)


_CLRSPC_PHOTOMETRIC = {
    OPJ_CLRSPC_GRAY: IC_PHOTO_GRAY,
    OPJ_CLRSPC_SRGB: IC_PHOTO_RGB,
    OPJ_CLRSPC_SYCC: IC_PHOTO_YCBCR,
    OPJ_CLRSPC_EYCC: IC_PHOTO_YCBCR,
    OPJ_CLRSPC_CMYK: IC_PHOTO_CMYK,
}

_PROG_ORDER = {
    OPJ_LRCP: 'LRCP',
    OPJ_RLCP: 'RLCP',
    OPJ_RPCL: 'RPCL',
    OPJ_PCRL: 'PCRL',
    OPJ_CPRL: 'CPRL',
}
