# imagecodecs/_jpegxr.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2016-2021, Christoph Gohlke
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

"""JPEG XR codec for the imagecodecs package.

The JPEG XR format is also known as HD Photo or Windows Media Photo.

"""

__version__ = '2021.2.26'

include '_shared.pxi'

from jxrlib cimport *


class JPEGXR:
    """JPEG XR Constants."""

    # Photometric Interpretation
    PI_W0 = PK_PI_W0
    PI_B0 = PK_PI_B0
    PI_RGB = PK_PI_RGB
    PI_RGBPalette = PK_PI_RGBPalette
    PI_TransparencyMask = PK_PI_TransparencyMask
    PI_CMYK = PK_PI_CMYK
    PI_YCbCr = PK_PI_YCbCr
    PI_CIELab = PK_PI_CIELab
    PI_NCH = PK_PI_NCH
    PI_RGBE = PK_PI_RGBE


class JpegxrError(RuntimeError):
    """JPEG XR Exceptions."""

    def __init__(self, func, err):
        msg = {
            WMP_errFail: 'WMP_errFail',
            WMP_errNotYetImplemented: 'WMP_errNotYetImplemented',
            WMP_errAbstractMethod: 'WMP_errAbstractMethod',
            WMP_errOutOfMemory: 'WMP_errOutOfMemory',
            WMP_errFileIO: 'WMP_errFileIO',
            WMP_errBufferOverflow: 'WMP_errBufferOverflow',
            WMP_errInvalidParameter: 'WMP_errInvalidParameter',
            WMP_errInvalidArgument: 'WMP_errInvalidArgument',
            WMP_errUnsupportedFormat: 'WMP_errUnsupportedFormat',
            WMP_errIncorrectCodecVersion: 'WMP_errIncorrectCodecVersion',
            WMP_errIndexNotFound: 'WMP_errIndexNotFound',
            WMP_errOutOfSequence: 'WMP_errOutOfSequence',
            WMP_errNotInitialized: 'WMP_errNotInitialized',
            WMP_errAlphaModeCannotBeTranscoded:
                'WMP_errAlphaModeCannotBeTranscoded',
            WMP_errIncorrectCodecSubVersion:
                'WMP_errIncorrectCodecSubVersion',
            WMP_errMustBeMultipleOf16LinesUntilLastCall:
                'WMP_errMustBeMultipleOf16LinesUntilLastCall',
            WMP_errPlanarAlphaBandedEncRequiresTempFile:
                'WMP_errPlanarAlphaBandedEncRequiresTempFile',
        }.get(err, f'unknown error {err!r}')
        msg = f'{func} returned {msg}'
        super().__init__(msg)


def jpegxr_version():
    """Return jxrlib library version string."""
    cdef:
        uint32_t ver = WMP_SDK_VERSION

    return f'jxrlib {ver >> 8}.{ver & 255}'


def jpegxr_check(data):
    """Return True if data likely contains a JPEG XR image."""


def jpegxr_encode(
    data,
    level=None,
    photometric=None,
    hasalpha=None,
    resolution=None,
    out=None
):
    """Return JPEG XR image from numpy array.

    """
    cdef:
        numpy.ndarray src = data
        numpy.dtype dtype = src.dtype
        const uint8_t[::1] dst  # must be const to write to bytes
        U8* outbuffer = NULL
        ssize_t dstsize
        ssize_t srcsize = src.nbytes
        size_t byteswritten = 0
        ssize_t samples
        int pi = jxr_encode_photometric(photometric)
        int alpha = 1 if hasalpha else 0
        float quality = 1.0 if level is None else level
        WMPStream* stream = NULL
        PKImageEncode* encoder = NULL
        PKPixelFormatGUID pixelformat
        PKPixelInfo pixelinfo
        float rx = 96.0
        float ry = 96.0
        I32 width
        I32 height
        U32 stride
        ERR err

    if data is out:
        raise ValueError('cannot encode in-place')

    if (
        dtype not in (
            numpy.bool8,
            numpy.uint8,
            numpy.uint16,
            numpy.float16,
            numpy.float32,
        )
        and src.ndim in (2, 3)
        and numpy.PyArray_ISCONTIGUOUS(src)
    ):
        raise ValueError('invalid data shape, strides, or dtype')

    if resolution:
        rx, ry = resolution

    width = <I32> src.shape[1]
    height = <I32> src.shape[0]
    stride = <U32> src.strides[0]
    samples = 1 if src.ndim == 2 else src.shape[2]

    if width < MB_WIDTH_PIXEL or height < MB_HEIGHT_PIXEL:
        raise ValueError('invalid data shape')

    if dtype == numpy.bool8:
        if src.ndim != 2:
            raise ValueError('invalid data shape, strides, or dtype')
        src = numpy.packbits(src, axis=-1)
        stride = <U32> src.strides[0]
        srcsize //= 8

    out, dstsize, outgiven, outtype = _parse_output(out)
    if out is None:
        if dstsize <= 0:
            dstsize = srcsize // 2
            dstsize = (((dstsize - 1) // 4096) + 1) * 4096
        elif dstsize < 4096:
            dstsize = 4096
        outbuffer = <U8*> malloc(dstsize)
        if outbuffer == NULL:
            raise MemoryError('failed to allocate ouput buffer')
    else:
        dst = out
        dstsize = dst.nbytes

    try:
        with nogil:
            pixelformat = jxr_encode_guid(dtype, samples, pi, &alpha)
            if IsEqualGUID(&pixelformat, &GUID_PKPixelFormatDontCare):
                raise ValueError('PKPixelFormatGUID not found')
            pixelinfo.pGUIDPixFmt = &pixelformat

            err = PixelFormatLookup(&pixelinfo, LOOKUP_FORWARD)
            if err:
                raise JpegxrError('PixelFormatLookup', err)

            if outbuffer == NULL:
                err = CreateWS_Memory(&stream, <void*> &dst[0], dstsize)
                if err:
                    raise JpegxrError('CreateWS_Memory', err)
                stream.Write = WriteWS_Memory
            else:
                err = CreateWS_Memory(&stream, <void*> outbuffer, dstsize)
                if err:
                    raise JpegxrError('CreateWS_Memory', err)
                stream.Write = WriteWS_Realloc
                stream.EOS = EOSWS_Realloc

            err = PKImageEncode_Create_WMP(&encoder)
            if err:
                raise JpegxrError('PKImageEncode_Create_WMP', err)

            err = encoder.Initialize(
                encoder,
                stream,
                &encoder.WMP.wmiSCP,
                sizeof(CWMIStrCodecParam)
            )
            if err:
                raise JpegxrError('PKImageEncode_Initialize', err)

            jxr_set_encoder(
                &encoder.WMP.wmiSCP,
                &pixelinfo,
                quality,
                alpha,
                pi
            )

            err = encoder.SetPixelFormat(encoder, pixelformat)
            if err:
                raise JpegxrError('PKImageEncode_SetPixelFormat', err)

            err = encoder.SetSize(encoder, width, height)
            if err:
                raise JpegxrError('PKImageEncode_SetSize', err)

            err = encoder.SetResolution(encoder, rx, ry)
            if err:
                raise JpegxrError('PKImageEncode_SetResolution', err)

            err = encoder.WritePixels(encoder, height, <U8*> src.data, stride)
            if err:
                raise JpegxrError('PKImageEncode_WritePixels', err)

            byteswritten = stream.state.buf.cbBufCount
            dstsize = stream.state.buf.cbBuf
            if outbuffer != NULL:
                outbuffer = stream.state.buf.pbBuf

    except Exception:
        if outbuffer != NULL:
            if stream != NULL:
                outbuffer = stream.state.buf.pbBuf
            free(outbuffer)
        raise
    finally:
        if encoder != NULL:
            PKImageEncode_Release(&encoder)
        elif stream != NULL:
            stream.Close(&stream)

    if outbuffer != NULL:
        out = _create_output(
            outtype, <ssize_t> byteswritten, <char*> outbuffer
        )
        free(outbuffer)
        return out

    del dst
    return _return_output(out, dstsize, byteswritten, outgiven)


def jpegxr_decode(data, index=None, fp2int=False, out=None):
    """Decode JPEG XR image to numpy array.

    fp2int: bool
        If True, return fixed point images as int16 or int32, else float32.

    """
    cdef:
        numpy.ndarray dst
        numpy.dtype dtype
        const uint8_t[::1] src = data
        PKImageDecode* decoder = NULL
        PKFormatConverter* converter = NULL
        PKPixelFormatGUID pixelformat
        PKRect rect
        I32 width
        I32 height
        U32 stride
        ERR err
        U8 alpha
        ssize_t srcsize = src.size
        ssize_t dstsize
        ssize_t samples
        int typenum
        bint fp2int_ = fp2int

    if data is out:
        raise ValueError('cannot decode in-place')

    try:
        with nogil:
            err = PKCodecFactory_CreateDecoderFromBytes(
                <void*> &src[0],
                srcsize,
                &decoder
            )
            if err:
                raise JpegxrError('PKCodecFactory_CreateDecoderFromBytes', err)

            err = PKImageDecode_GetSize(decoder, &width, &height)
            if err:
                raise JpegxrError('PKImageDecode_GetSize', err)

            err = PKImageDecode_GetPixelFormat(decoder, &pixelformat)
            if err:
                raise JpegxrError('PKImageDecode_GetPixelFormat', err)

            err = jxr_decode_guid(
                &pixelformat, &typenum, &samples, &alpha, fp2int_
            )
            if err:
                raise JpegxrError('jxr_decode_guid', err)
            decoder.WMP.wmiSCP.uAlphaMode = alpha

            err = PKCodecFactory_CreateFormatConverter(&converter)
            if err:
                raise JpegxrError('PKCodecFactory_CreateFormatConverter', err)

            err = PKFormatConverter_Initialize(
                converter,
                decoder,
                NULL,
                pixelformat
            )
            if err:
                raise JpegxrError('PKFormatConverter_Initialize', err)

            with gil:
                shape = height, width
                if samples > 1:
                    shape += samples,
                dtype = numpy.PyArray_DescrNewFromType(typenum)
                out = _create_array(out, shape, dtype)
                dst = out
                dstsize = dst.nbytes

            rect.X = 0
            rect.Y = 0
            rect.Width = <I32> dst.shape[1]
            rect.Height = <I32> dst.shape[0]
            stride = <U32> dst.strides[0]

            memset(<void*> dst.data, 0, dstsize)  # TODO: still necessary?
            # TODO: check alignment issues
            err = PKFormatConverter_Copy(
                converter,
                &rect,
                <U8*> dst.data,
                stride
            )
        if err:
            raise JpegxrError('PKFormatConverter_Copy', err)

    finally:
        if converter != NULL:
            PKFormatConverter_Release(&converter)
        if decoder != NULL:
            PKImageDecode_Release(&decoder)

    return out


cdef ERR WriteWS_Memory(WMPStream* pWS, const void* pv, size_t cb) nogil:
    """Relpacement for WriteWS_Memory to keep track of bytes written."""
    if pWS.state.buf.cbCur + cb < pWS.state.buf.cbCur:
        return WMP_errBufferOverflow
    if pWS.state.buf.cbBuf < pWS.state.buf.cbCur + cb:
        return WMP_errBufferOverflow

    memmove(pWS.state.buf.pbBuf + pWS.state.buf.cbCur, pv, cb)
    pWS.state.buf.cbCur += cb

    # keep track of bytes written
    if pWS.state.buf.cbCur > pWS.state.buf.cbBufCount:
        pWS.state.buf.cbBufCount = pWS.state.buf.cbCur

    return WMP_errSuccess


cdef ERR WriteWS_Realloc(WMPStream* pWS, const void* pv, size_t cb) nogil:
    """Relpacement for WriteWS_Memory to realloc buffers on overflow.

    Only use with buffers allocated by malloc.

    """
    cdef:
        size_t newsize = pWS.state.buf.cbCur + cb

    if newsize < pWS.state.buf.cbCur:
        return WMP_errBufferOverflow
    if pWS.state.buf.cbBuf < newsize:
        if newsize <= pWS.state.buf.cbBuf * 1.125:
            # moderate upsize: overallocate
            newsize = newsize + newsize // 8
            newsize = (((newsize-1) // 4096) + 1) * 4096
        else:
            # major upsize: resize to exact size
            newsize = newsize + 1
        pWS.state.buf.pbBuf = <U8*> realloc(
            <void*> pWS.state.buf.pbBuf,
            newsize)
        if pWS.state.buf.pbBuf == NULL:
            return WMP_errOutOfMemory
        pWS.state.buf.cbBuf = newsize

    memmove(pWS.state.buf.pbBuf + pWS.state.buf.cbCur, pv, cb)
    pWS.state.buf.cbCur += cb

    # keep track of bytes written
    if pWS.state.buf.cbCur >  pWS.state.buf.cbBufCount:
        pWS.state.buf.cbBufCount = pWS.state.buf.cbCur

    return WMP_errSuccess


cdef Bool EOSWS_Realloc(WMPStream* pWS) nogil:
    """Relpacement for EOSWS_Memory."""
    # return pWS.state.buf.cbBuf <= pWS.state.buf.cbCur
    return 1


cdef ERR PKCodecFactory_CreateDecoderFromBytes(
    void* bytes,
    size_t len,
    PKImageDecode** ppDecode
) nogil:
    """Create PKImageDecode from byte string."""
    cdef:
        char* pExt = NULL
        const PKIID* pIID = NULL
        WMPStream* stream = NULL
        PKImageDecode* decoder = NULL
        ERR err

    # get decode PKIID
    err = GetImageDecodeIID('.jxr', &pIID)
    if err:
        return err
    # create stream
    err = CreateWS_Memory(&stream, bytes, len)
    if err:
        return err
    # create decoder
    err = PKCodecFactory_CreateCodec(pIID, <void**> ppDecode)
    if err:
        return err
    # attach stream to decoder
    decoder = ppDecode[0]
    err = decoder.Initialize(decoder, stream)
    if err:
        return err
    decoder.fStreamOwner = 1
    return WMP_errSuccess


cdef ERR jxr_decode_guid(
    PKPixelFormatGUID* pixelformat,
    int* typenum,
    ssize_t* samples,
    U8* alpha,
    bint fp2int
) nogil:
    """Return dtype, samples, alpha from GUID.

    Change pixelformat to output format in-place.

    """
    alpha[0] = 0
    samples[0] = 1

    # bool
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormatBlackWhite):
        pixelformat[0] = GUID_PKPixelFormat8bppGray
        typenum[0] = numpy.NPY_BOOL
        return WMP_errSuccess

    # uint8
    typenum[0] = numpy.NPY_UINT8
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat8bppGray):
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat24bppRGB):
        samples[0] = 3
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat16bppRGB555):
        pixelformat[0] = GUID_PKPixelFormat24bppRGB
        samples[0] = 3
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat16bppRGB565):
        pixelformat[0] = GUID_PKPixelFormat24bppRGB
        samples[0] = 3
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat24bppBGR):
        pixelformat[0] = GUID_PKPixelFormat24bppRGB
        samples[0] = 3
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat32bppRGB):
        pixelformat[0] = GUID_PKPixelFormat24bppRGB
        samples[0] = 3
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat32bppBGRA):
        pixelformat[0] = GUID_PKPixelFormat32bppRGBA
        alpha[0] = 2
        samples[0] = 4
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat32bppRGBA):
        alpha[0] = 2
        samples[0] = 4
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat32bppPRGBA):
        alpha[0] = 2
        samples[0] = 4
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat32bppRGBE):
        samples[0] = 4
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat32bppCMYK):
        samples[0] = 4
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat40bppCMYKAlpha):
        alpha[0] = 2
        samples[0] = 5
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat24bpp3Channels):
        samples[0] = 3
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat32bpp4Channels):
        samples[0] = 4
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat40bpp5Channels):
        samples[0] = 5
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat48bpp6Channels):
        samples[0] = 6
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat56bpp7Channels):
        samples[0] = 7
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat64bpp8Channels):
        samples[0] = 8
        return WMP_errSuccess

    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat32bpp3ChannelsAlpha):
        alpha[0] = 2
        samples[0] = 4
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat40bpp4ChannelsAlpha):
        alpha[0] = 2
        samples[0] = 5
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat48bpp5ChannelsAlpha):
        alpha[0] = 2
        samples[0] = 6
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat56bpp6ChannelsAlpha):
        alpha[0] = 2
        samples[0] = 7
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat64bpp7ChannelsAlpha):
        alpha[0] = 2
        samples[0] = 8
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat72bpp8ChannelsAlpha):
        alpha[0] = 2
        samples[0] = 9
        return WMP_errSuccess

    # uint16
    typenum[0] = numpy.NPY_UINT16
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat16bppGray):
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat48bppRGB):
        samples[0] = 3
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat64bppRGBA):
        alpha[0] = 2
        samples[0] = 4
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat64bppPRGBA):
        alpha[0] = 2
        samples[0] = 4
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat64bppCMYK):
        samples[0] = 4
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat80bppCMYKAlpha):
        alpha[0] = 2
        samples[0] = 5
        return WMP_errSuccess

    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat32bppRGB101010):
        pixelformat[0] = GUID_PKPixelFormat48bppRGB
        samples[0] = 3
        return WMP_errSuccess

    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat48bpp3Channels):
        samples[0] = 3
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat64bpp4Channels):
        samples[0] = 4
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat80bpp5Channels):
        samples[0] = 5
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat96bpp6Channels):
        samples[0] = 6
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat112bpp7Channels):
        samples[0] = 7
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat128bpp8Channels):
        samples[0] = 8
        return WMP_errSuccess

    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat64bpp3ChannelsAlpha):
        alpha[0] = 2
        samples[0] = 4
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat80bpp4ChannelsAlpha):
        alpha[0] = 2
        samples[0] = 5
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat96bpp5ChannelsAlpha):
        alpha[0] = 2
        samples[0] = 6
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat112bpp6ChannelsAlpha):
        alpha[0] = 2
        samples[0] = 7
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat128bpp7ChannelsAlpha):
        alpha[0] = 2
        samples[0] = 8
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat144bpp8ChannelsAlpha):
        alpha[0] = 2
        samples[0] = 9
        return WMP_errSuccess

    # float32
    typenum[0] = numpy.NPY_FLOAT32
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat32bppGrayFloat):
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat96bppRGBFloat):
        samples[0] = 3
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat128bppRGBFloat):
        pixelformat[0] = GUID_PKPixelFormat96bppRGBFloat
        samples[0] = 3
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat128bppRGBAFloat):
        alpha[0] = 2
        samples[0] = 4
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat128bppPRGBAFloat):
        alpha[0] = 2
        samples[0] = 4
        return WMP_errSuccess

    # float16
    typenum[0] = numpy.NPY_FLOAT16
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat16bppGrayHalf):
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat48bppRGBHalf):
        samples[0] = 3
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat64bppRGBHalf):
        pixelformat[0] = GUID_PKPixelFormat48bppRGBHalf
        samples[0] = 3
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat64bppRGBAHalf):
        alpha[0] = 2
        samples[0] = 4
        return WMP_errSuccess

    if fp2int:
        # fixed to int
        typenum[0] = numpy.NPY_INT16
        if IsEqualGUID(pixelformat, &GUID_PKPixelFormat16bppGrayFixedPoint):
            return WMP_errSuccess
        if IsEqualGUID(pixelformat, &GUID_PKPixelFormat48bppRGBFixedPoint):
            samples[0] = 3
            return WMP_errSuccess
        if IsEqualGUID(pixelformat, &GUID_PKPixelFormat64bppRGBAFixedPoint):
            samples[0] = 4
            return WMP_errSuccess

        typenum[0] = numpy.NPY_INT32
        if IsEqualGUID(pixelformat, &GUID_PKPixelFormat32bppGrayFixedPoint):
            pixelformat[0] = GUID_PKPixelFormat32bppGrayFloat
            return WMP_errSuccess
        if IsEqualGUID(pixelformat, &GUID_PKPixelFormat96bppRGBFixedPoint):
            pixelformat[0] = GUID_PKPixelFormat96bppRGBFloat
            samples[0] = 3
            return WMP_errSuccess
        if IsEqualGUID(pixelformat, &GUID_PKPixelFormat128bppRGBAFixedPoint):
            pixelformat[0] = GUID_PKPixelFormat128bppRGBAFloat
            samples[0] = 4
            return WMP_errSuccess

    else:
        # fixed to float32
        typenum[0] = numpy.NPY_FLOAT32
        if IsEqualGUID(pixelformat, &GUID_PKPixelFormat16bppGrayFixedPoint):
            pixelformat[0] = GUID_PKPixelFormat32bppGrayFloat
            return WMP_errSuccess
        if IsEqualGUID(pixelformat, &GUID_PKPixelFormat48bppRGBFixedPoint):
            pixelformat[0] = GUID_PKPixelFormat96bppRGBFloat
            samples[0] = 3
            return WMP_errSuccess
        if IsEqualGUID(pixelformat, &GUID_PKPixelFormat64bppRGBAFixedPoint):
            pixelformat[0] = GUID_PKPixelFormat128bppRGBAFloat
            samples[0] = 4
            return WMP_errSuccess

        if IsEqualGUID(pixelformat, &GUID_PKPixelFormat32bppGrayFixedPoint):
            pixelformat[0] = GUID_PKPixelFormat32bppGrayFloat
            return WMP_errSuccess
        if IsEqualGUID(pixelformat, &GUID_PKPixelFormat96bppRGBFixedPoint):
            pixelformat[0] = GUID_PKPixelFormat96bppRGBFloat
            samples[0] = 3
            return WMP_errSuccess
        if IsEqualGUID(pixelformat, &GUID_PKPixelFormat128bppRGBAFixedPoint):
            pixelformat[0] = GUID_PKPixelFormat128bppRGBAFloat
            samples[0] = 4
            return WMP_errSuccess

    return WMP_errUnsupportedFormat


cdef PKPixelFormatGUID jxr_encode_guid(
    numpy.dtype dtype,
    ssize_t samples,
    int photometric,
    int* alpha
) nogil:
    """Return pixel format GUID from dtype, samples, and photometric."""
    cdef:
        int typenum = dtype.type_num

    if samples == 1:
        if typenum == numpy.NPY_UINT8:
            return GUID_PKPixelFormat8bppGray
        if typenum == numpy.NPY_UINT16:
            return GUID_PKPixelFormat16bppGray
        if typenum == numpy.NPY_FLOAT32:
            return GUID_PKPixelFormat32bppGrayFloat
        if typenum == numpy.NPY_FLOAT16:
            return GUID_PKPixelFormat16bppGrayHalf
        if typenum == numpy.NPY_BOOL:
            return GUID_PKPixelFormatBlackWhite
    if samples == 3:
        if typenum == numpy.NPY_UINT8:
            if photometric < 0 or photometric == PK_PI_RGB:
                return GUID_PKPixelFormat24bppRGB
            return GUID_PKPixelFormat24bpp3Channels
        if typenum == numpy.NPY_UINT16:
            if photometric < 0 or photometric == PK_PI_RGB:
                return GUID_PKPixelFormat48bppRGB
            return GUID_PKPixelFormat48bpp3Channels
        if typenum == numpy.NPY_FLOAT32:
            return GUID_PKPixelFormat96bppRGBFloat
        if typenum == numpy.NPY_FLOAT16:
            return GUID_PKPixelFormat48bppRGBHalf
    if samples == 4:
        if typenum == numpy.NPY_UINT8:
            if photometric < 0 or photometric == PK_PI_RGB:
                alpha[0] = 1
                return GUID_PKPixelFormat32bppRGBA
            if photometric == PK_PI_CMYK:
                return GUID_PKPixelFormat32bppCMYK
            if alpha:
                return GUID_PKPixelFormat32bpp3ChannelsAlpha
            return GUID_PKPixelFormat32bpp4Channels
        if typenum == numpy.NPY_UINT16:
            if photometric < 0 or photometric == PK_PI_RGB:
                alpha[0] = 1
                return GUID_PKPixelFormat64bppRGBA
            if photometric == PK_PI_CMYK:
                return GUID_PKPixelFormat64bppCMYK
            if alpha:
                return GUID_PKPixelFormat64bpp3ChannelsAlpha
            return GUID_PKPixelFormat64bpp4Channels
        alpha[0] = 1
        if typenum == numpy.NPY_FLOAT32:
            return GUID_PKPixelFormat128bppRGBAFloat
        if typenum == numpy.NPY_FLOAT16:
            return GUID_PKPixelFormat64bppRGBAHalf
    if samples == 5:
        if photometric == PK_PI_CMYK:
            alpha[0] = 1
            if typenum == numpy.NPY_UINT8:
                return GUID_PKPixelFormat40bppCMYKAlpha
            if typenum == numpy.NPY_UINT16:
                return GUID_PKPixelFormat80bppCMYKAlpha
        if alpha[0]:
            if typenum == numpy.NPY_UINT8:
                return GUID_PKPixelFormat40bpp4ChannelsAlpha
            if typenum == numpy.NPY_UINT16:
                return GUID_PKPixelFormat80bpp4ChannelsAlpha
        if typenum == numpy.NPY_UINT8:
            return GUID_PKPixelFormat40bpp5Channels
        if typenum == numpy.NPY_UINT16:
            return GUID_PKPixelFormat80bpp5Channels
    if samples == 6:
        if alpha[0]:
            if typenum == numpy.NPY_UINT8:
                return GUID_PKPixelFormat48bpp5ChannelsAlpha
            if typenum == numpy.NPY_UINT16:
                return GUID_PKPixelFormat96bpp5ChannelsAlpha
        if typenum == numpy.NPY_UINT8:
            return GUID_PKPixelFormat48bpp6Channels
        if typenum == numpy.NPY_UINT16:
            return GUID_PKPixelFormat96bpp6Channels
    if samples == 7:
        if alpha[0]:
            if typenum == numpy.NPY_UINT8:
                return GUID_PKPixelFormat56bpp6ChannelsAlpha
            if typenum == numpy.NPY_UINT16:
                return GUID_PKPixelFormat112bpp6ChannelsAlpha
        if typenum == numpy.NPY_UINT8:
            return GUID_PKPixelFormat56bpp7Channels
        if typenum == numpy.NPY_UINT16:
            return GUID_PKPixelFormat112bpp7Channels
    if samples == 8:
        if alpha[0]:
            if typenum == numpy.NPY_UINT8:
                return GUID_PKPixelFormat64bpp7ChannelsAlpha
            if typenum == numpy.NPY_UINT16:
                return GUID_PKPixelFormat128bpp7ChannelsAlpha
        if typenum == numpy.NPY_UINT8:
            return GUID_PKPixelFormat64bpp8Channels
        elif typenum == numpy.NPY_UINT16:
            return GUID_PKPixelFormat128bpp8Channels
    if samples == 9:
        alpha[0] = 1
        if typenum == numpy.NPY_UINT8:
            return GUID_PKPixelFormat72bpp8ChannelsAlpha
        if typenum == numpy.NPY_UINT16:
            return GUID_PKPixelFormat144bpp8ChannelsAlpha
    return GUID_PKPixelFormatDontCare


cdef int jxr_encode_photometric(photometric):
    """Return PK_PI value from photometric argument."""
    if photometric is None:
        return -1
    if isinstance(photometric, int):
        if photometric not in (-1, PK_PI_W0, PK_PI_B0, PK_PI_RGB, PK_PI_CMYK):
            raise ValueError('photometric interpretation not supported')
        return photometric
    photometric = photometric.upper()
    if photometric[:3] == 'RGB':
        return PK_PI_RGB
    if photometric == 'WHITEISZERO' or photometric == 'MINISWHITE':
        return PK_PI_W0
    if photometric in ('BLACKISZERO', 'MINISBLACK', 'GRAY'):
        return PK_PI_B0
    if photometric == 'CMYK' or photometric == 'SEPARATED':
        return PK_PI_CMYK
    # TODO: support more photometric modes
    # if photometric == 'YCBCR':
    #     return PK_PI_YCbCr
    # if photometric == 'CIELAB':
    #     return PK_PI_CIELab
    # if photometric == 'TRANSPARENCYMASK' or photometric == 'MASK':
    #     return PK_PI_TransparencyMask
    # if photometric == 'RGBPALETTE' or photometric == 'PALETTE':
    #     return PK_PI_RGBPalette
    raise ValueError('photometric interpretation not supported')


# Y, U, V, YHP, UHP, VHP
# optimized for PSNR
cdef int* DPK_QPS_420 = [
    66, 65, 70, 72, 72, 77, 59, 58, 63, 64, 63, 68, 52, 51, 57, 56, 56, 61, 48,
    48, 54, 51, 50, 55, 43, 44, 48, 46, 46, 49, 37, 37, 42, 38, 38, 43, 26, 28,
    31, 27, 28, 31, 16, 17, 22, 16, 17, 21, 10, 11, 13, 10, 10, 13, 5, 5, 6, 5,
    5, 6, 2, 2, 3, 2, 2, 2
]

cdef int* DPK_QPS_8 = [
    67, 79, 86, 72, 90, 98, 59, 74, 80, 64, 83, 89, 53, 68, 75, 57, 76, 83, 49,
    64, 71, 53, 70, 77, 45, 60, 67, 48, 67, 74, 40, 56, 62, 42, 59, 66, 33, 49,
    55, 35, 51, 58, 27, 44, 49, 28, 45, 50, 20, 36, 42, 20, 38, 44, 13, 27, 34,
    13, 28, 34, 7, 17, 21, 8, 17, 21, 2, 5, 6, 2, 5, 6
]

cdef int* DPK_QPS_16 = [
    197, 203, 210, 202, 207, 213, 174, 188, 193, 180, 189, 196, 152, 167, 173,
    156, 169, 174, 135, 152, 157, 137, 153, 158, 119, 137, 141, 119, 138, 142,
    102, 120, 125, 100, 120, 124, 82, 98, 104, 79, 98, 103, 60, 76, 81, 58, 76,
    81, 39, 52, 58, 36, 52, 58, 16, 27, 33, 14, 27, 33, 5, 8, 9, 4, 7, 8
]

cdef int* DPK_QPS_16f = [
    148, 177, 171, 165, 187, 191, 133, 155, 153, 147, 172, 181, 114, 133, 138,
    130, 157, 167, 97, 118, 120, 109, 137, 144, 76, 98, 103, 85, 115, 121, 63,
    86, 91, 62, 96, 99, 46, 68, 71, 43, 73, 75, 29, 48, 52, 27, 48, 51, 16, 30,
    35, 14, 29, 34, 8, 14, 17, 7,  13, 17, 3, 5, 7, 3, 5, 6
]

cdef int* DPK_QPS_32f = [
    194, 206, 209, 204, 211, 217, 175, 187, 196, 186, 193, 205, 157, 170, 177,
    167, 180, 190, 133, 152, 156, 144, 163, 168, 116, 138, 142, 117, 143, 148,
    98, 120, 123,  96, 123, 126, 80, 99, 102, 78, 99, 102, 65, 79, 84, 63, 79,
    84, 48, 61, 67, 45, 60, 66, 27, 41, 46, 24, 40, 45, 3, 22, 24,  2, 21, 22
]


cdef U8 jxr_quantization(int* qps, double quality, ssize_t i) nogil:
    """Return quantization from DPK_QPS table."""
    cdef:
        ssize_t qi = <ssize_t> (10.0 * quality)
        double qf = 10.0 * quality - <double> qi
        int* qps0 = qps + qi * 6
        int* qps1 = qps0 + 6

    return <U8> (<double> qps0[i] * (1.0 - qf) + <double> qps1[i] * qf + 0.5)


cdef ERR jxr_set_encoder(
    CWMIStrCodecParam* wmiscp,
    PKPixelInfo* pixelinfo,
    double quality,
    int alpha,
    int pi
) nogil:
    """Set encoder compression parameters from level argument and pixel format.

    Code and tables adapted from jxrlib's JxrEncApp.c.

    ImageQuality Q(BD==1) Q(BD==8)    Q(BD==16)   Q(BD==32F)  Subsample Overlap
    [0.0, 0.5)   8-IQ*5   (see table) (see table) (see table) 4:2:0     2
    [0.5, 1.0)   8-IQ*5   (see table) (see table) (see table) 4:4:4     1
    [1.0, 1.0]   1        1           1           1           4:4:4     0

    """
    cdef:
        int* qps

    # default: lossless, no tiles
    wmiscp.uiDefaultQPIndex = 1
    wmiscp.uiDefaultQPIndexAlpha = 1
    wmiscp.olOverlap = OL_NONE
    wmiscp.cfColorFormat = YUV_444
    wmiscp.sbSubband = SB_ALL
    wmiscp.bfBitstreamFormat = SPATIAL
    wmiscp.bProgressiveMode = 0
    wmiscp.cNumOfSliceMinus1H = 0
    wmiscp.cNumOfSliceMinus1V = 0
    wmiscp.uAlphaMode = 2 if alpha else 0
    # wmiscp.bdBitDepth = BD_LONG

    if pi == PK_PI_CMYK:
        wmiscp.cfColorFormat = CMYK

    if quality <= 0.0 or quality == 1.0 or quality >= 100.0:
        return WMP_errSuccess
    if quality > 1.0:
        quality /= 100.0
    if quality >= 1.0:
        return WMP_errSuccess
    if quality < 0.5:
        # overlap
        wmiscp.olOverlap = OL_TWO

    if quality < 0.5 and pixelinfo.uBitsPerSample <= 8 and pi != PK_PI_CMYK:
        # chroma sub-sampling
        wmiscp.cfColorFormat = YUV_420

    # bit depth
    if pixelinfo.bdBitDepth == BD_1:
        wmiscp.uiDefaultQPIndex = <U8> (8 - 5.0 * quality + 0.5)
    else:
        # remap [0.8, 0.866, 0.933, 1.0] to [0.8, 0.9, 1.0, 1.1]
        # to use 8-bit DPK QP table (0.933 == Photoshop JPEG 100)
        if (
            quality > 0.8
            and pixelinfo.bdBitDepth == BD_8
            and wmiscp.cfColorFormat != YUV_420
            and wmiscp.cfColorFormat != YUV_422
        ):
            quality = 0.8 + (quality - 0.8) * 1.5

        if wmiscp.cfColorFormat == YUV_420 or wmiscp.cfColorFormat == YUV_422:
            qps = DPK_QPS_420
        elif pixelinfo.bdBitDepth == BD_8:
            qps = DPK_QPS_8
        elif pixelinfo.bdBitDepth == BD_16:
            qps = DPK_QPS_16
        elif pixelinfo.bdBitDepth == BD_16F:
            qps = DPK_QPS_16f
        else:
            qps = DPK_QPS_32f

        wmiscp.uiDefaultQPIndex = jxr_quantization(qps, quality, 0)
        wmiscp.uiDefaultQPIndexU = jxr_quantization(qps, quality, 1)
        wmiscp.uiDefaultQPIndexV = jxr_quantization(qps, quality, 2)
        wmiscp.uiDefaultQPIndexYHP = jxr_quantization(qps, quality, 3)
        wmiscp.uiDefaultQPIndexUHP = jxr_quantization(qps, quality, 4)
        wmiscp.uiDefaultQPIndexVHP = jxr_quantization(qps, quality, 5)

    return WMP_errSuccess


cdef pixelformat_str(PKPixelFormatGUID* pf):
    """Return PKPixelFormatGUID as string."""
    return (
        'PKPixelFormatGUID '
        f'{(<unsigned long *> pf)[0]:#08x}, '
        f'{(<unsigned long *> pf)[1]:#08x}, '
        f'{(<unsigned long *> pf)[2]:#08x}, '
        f'{(<unsigned long *> pf)[3]:#08x}'
    )
