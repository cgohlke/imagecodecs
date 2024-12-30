# imagecodecs/_cms.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2021-2025, Christoph Gohlke
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

"""CMS codec for the imagecodecs package."""

include '_shared.pxi'

from lcms2 cimport *

from cpython.bytes cimport (
    PyBytes_Check, PyBytes_Size, PyBytes_AsString, PyBytes_FromStringAndSize
)


class CMS:
    """CMS codec constants."""

    available = True

    class INTENT(enum.IntEnum):
        """CMS codec intent types."""

        PERCEPTUAL = INTENT_PERCEPTUAL
        RELATIVE_COLORIMETRIC = INTENT_RELATIVE_COLORIMETRIC
        SATURATION = INTENT_SATURATION
        ABSOLUTE_COLORIMETRIC = INTENT_ABSOLUTE_COLORIMETRIC

    class FLAGS(enum.IntEnum):
        """CMS codec flags."""

        NOCACHE = cmsFLAGS_NOCACHE
        NOOPTIMIZE = cmsFLAGS_NOOPTIMIZE
        NULLTRANSFORM = cmsFLAGS_NULLTRANSFORM
        GAMUTCHECK = cmsFLAGS_GAMUTCHECK
        SOFTPROOFING = cmsFLAGS_SOFTPROOFING
        BLACKPOINTCOMPENSATION = cmsFLAGS_BLACKPOINTCOMPENSATION
        NOWHITEONWHITEFIXUP = cmsFLAGS_NOWHITEONWHITEFIXUP
        HIGHRESPRECALC = cmsFLAGS_HIGHRESPRECALC
        LOWRESPRECALC = cmsFLAGS_LOWRESPRECALC
        EIGHTBITS_DEVICELINK = cmsFLAGS_8BITS_DEVICELINK
        GUESSDEVICECLASS = cmsFLAGS_GUESSDEVICECLASS
        KEEP_SEQUENCE = cmsFLAGS_KEEP_SEQUENCE
        FORCE_CLUT = cmsFLAGS_FORCE_CLUT
        CLUT_POST_LINEARIZATION = cmsFLAGS_CLUT_POST_LINEARIZATION
        CLUT_PRE_LINEARIZATION = cmsFLAGS_CLUT_PRE_LINEARIZATION
        NONEGATIVES = cmsFLAGS_NONEGATIVES
        COPY_ALPHA = cmsFLAGS_COPY_ALPHA
        NODEFAULTRESOURCEDEF = cmsFLAGS_NODEFAULTRESOURCEDEF

    class PT(enum.IntEnum):
        """CMS codec pixel types."""

        GRAY = PT_GRAY
        RGB = PT_RGB
        CMY = PT_CMY
        CMYK = PT_CMYK
        YCBCR = PT_YCbCr
        YUV = PT_YUV
        XYZ = PT_XYZ
        LAB = PT_Lab
        YUVK = PT_YUVK
        HSV = PT_HSV
        HLS = PT_HLS
        YXY = PT_Yxy
        MCH1 = PT_MCH1
        MCH2 = PT_MCH2
        MCH3 = PT_MCH3
        MCH4 = PT_MCH4
        MCH5 = PT_MCH5
        MCH6 = PT_MCH6
        MCH7 = PT_MCH7
        MCH8 = PT_MCH8
        MCH9 = PT_MCH9
        MCH10 = PT_MCH10
        MCH11 = PT_MCH11
        MCH12 = PT_MCH12
        MCH13 = PT_MCH13
        MCH14 = PT_MCH14
        MCH15 = PT_MCH15


class CmsError(RuntimeError):
    """CMS codec exceptions."""


def cms_version():
    """Return Little-CMS library version string."""
    return 'lcms2 {}.{}.{}'.format(
        LCMS_VERSION // 1000,
        LCMS_VERSION % 1000 // 10,
        LCMS_VERSION % 1000 % 10,
    )


def cms_check(const uint8_t[::1] data):
    """Return whether data is ICC profile."""
    cdef:
        bytes sig = bytes(data[36:40])

    return sig == b'acsp'


def cms_transform(
    data,
    profile,
    outprofile,
    *,
    colorspace=None,  # pixeltype
    planar=None,
    outcolorspace=None,
    outplanar=None,
    outdtype=None,
    intent=None,
    flags=None,
    verbose=None,
    out=None
):
    """Return color-transformed array (experimental)."""
    cdef:
        numpy.ndarray src, dst
        cmsUInt32Number Intent = INTENT_PERCEPTUAL if not intent else intent
        cmsUInt32Number dwFlags = 0 if flags is None else flags
        cmsUInt32Number InputFormat = 0
        cmsUInt32Number OutputFormat = 0
        cmsHPROFILE hInProfile = NULL
        cmsHPROFILE hOutProfile = NULL
        cmsHTRANSFORM hTransform = NULL
        cmsUInt32Number numpixels

    data = numpy.ascontiguousarray(data)
    src = data

    # TODO: determine colorspace/pixeltype from profiles?

    InputFormat = _cms_format(
        data.shape,
        data.dtype,
        colorspace=colorspace,
        planar=planar
    )

    planar = T_PLANAR(InputFormat)
    numpixels = <cmsUInt32Number> (
        data.size // (T_CHANNELS(InputFormat) + T_EXTRA(InputFormat))
    )

    if out is not None and isinstance(out, numpy.ndarray):
        # assert outdtype is None
        OutputFormat = _cms_format(
            out.shape,
            out.dtype,
            colorspace=outcolorspace,
            planar=outplanar,
        )
    else:
        if outcolorspace is None:
            outcolorspace = colorspace
        if outplanar is None:
            outplanar = planar
        if outdtype is None:
            outdtype = data.dtype

        outshape = _cms_output_shape(
            InputFormat,
            data.shape,
            outcolorspace,
            outplanar
        )

        out = _create_array(out, outshape, outdtype)

        OutputFormat = _cms_format(
            out.shape,
            out.dtype,
            colorspace=outcolorspace,
            planar=outplanar,
        )

    outplanar = T_PLANAR(OutputFormat)

    if planar and data.ndim > 3:
        raise NotImplementedError  # TODO
    if outplanar and out.ndim > 3:
        raise NotImplementedError  # TODO
    if not out.data.contiguous:
        raise ValueError('output array not contiguous')

    if verbose:
        cmsSetLogErrorHandler(_cms_log_error_handler)

    dst = out

    try:
        if profile is None:
            hInProfile = cmsCreateNULLProfile()
        else:
            hInProfile = open_profile(profile)
            if hInProfile == NULL:
                raise CmsError('cmsOpenProfileFromMem returned NULL')

        if outprofile is None:
            hOutProfile = cmsCreateNULLProfile()
        else:
            hOutProfile = open_profile(outprofile)
            if hOutProfile == NULL:
                raise CmsError('cmsOpenProfileFromMem returned NULL')

        # cmsColorSpaceSignature ColorSpaceSignature
        # ColorSpaceSignature = cmsGetColorSpace(hProfile)

        with nogil:

            hTransform = cmsCreateTransform(
                hInProfile,
                InputFormat,
                hOutProfile,
                OutputFormat,
                Intent,
                dwFlags
            )
            if hTransform == NULL:
                raise CmsError('cmsCreateTransform returned NULL')

            cmsCloseProfile(hInProfile)
            hInProfile = NULL
            cmsCloseProfile(hOutProfile)
            hOutProfile = NULL

            # TODO: iterate over all but last three dimensions if planar
            cmsDoTransform(
                hTransform,
                <const void*> src.data,
                <void *> dst.data,
                numpixels
            )

            cmsDeleteTransform(hTransform)
            hTransform = NULL

    finally:
        if hInProfile != NULL:
            cmsCloseProfile(hInProfile)
        if hOutProfile != NULL:
            cmsCloseProfile(hOutProfile)
        if hTransform != NULL:
            cmsDeleteTransform(hTransform)
        if verbose:
            cmsSetLogErrorHandler(NULL)

    return out


cms_encode = cms_transform
cms_decode = cms_transform


@cython.boundscheck(True)
def cms_profile(
    profile,
    whitepoint=None,  # gray and rgb
    primaries=None,  # rgb
    transferfunction=None,  # gray and rgb
    gamma=None
):
    """Return ICC profile."""
    cdef:
        cmsHPROFILE hProfile = NULL
        cmsCIExyY WhitePoint
        cmsCIExyY* pWhitePoint = NULL
        cmsCIExyYTRIPLE Primaries
        cmsCIExyYTRIPLE* pPrimaries = NULL
        # cmsToneCurve TransferFunction
        cmsToneCurve* ppTransferFunction[3]
        cmsUInt32Number BytesNeeded
        cmsBool ret
        ssize_t tfcount = 0
        ssize_t tfentries = 0
        ssize_t i, j
        numpy.ndarray tf
        object out

    if whitepoint is not None:
        if len(whitepoint) == 3:
            # regular xyY
            WhitePoint.x = whitepoint[0]
            WhitePoint.y = whitepoint[1]
            WhitePoint.Y = whitepoint[2]
        elif len(whitepoint) == 2:
            # xy Y=1
            WhitePoint.x = whitepoint[0]
            WhitePoint.y = whitepoint[1]
            WhitePoint.Y = 1.0
        elif len(whitepoint) == 4:
            WhitePoint.x = whitepoint[0] / whitepoint[1]
            WhitePoint.y = whitepoint[2] / whitepoint[3]
            WhitePoint.Y = 1.0
        else:
            raise ValueError('invalid length of primaries')
        pWhitePoint = &WhitePoint

    if primaries is not None:
        if len(primaries) == 9:
            # regular xyY
            Primaries.Red.x = primaries[0]
            Primaries.Red.y = primaries[1]
            Primaries.Red.Y = primaries[2]
            Primaries.Green.x = primaries[3]
            Primaries.Green.y = primaries[4]
            Primaries.Green.Y = primaries[5]
            Primaries.Blue.x = primaries[6]
            Primaries.Blue.y = primaries[7]
            Primaries.Blue.Y = primaries[8]
        elif len(primaries) == 6:
            # xy Y=1
            Primaries.Red.x = primaries[0]
            Primaries.Red.y = primaries[1]
            Primaries.Red.Y = 1.0
            Primaries.Green.x = primaries[2]
            Primaries.Green.y = primaries[3]
            Primaries.Green.Y = 1.0
            Primaries.Blue.x = primaries[4]
            Primaries.Blue.y = primaries[5]
            Primaries.Blue.Y = 1.0
        elif len(primaries) == 12:
            # TIFF rational
            Primaries.Red.x = primaries[0] / primaries[1]
            Primaries.Red.y = primaries[2] / primaries[3]
            Primaries.Red.Y = 1.0
            Primaries.Green.x = primaries[4] / primaries[5]
            Primaries.Green.y = primaries[6] / primaries[7]
            Primaries.Green.Y = 1.0
            Primaries.Blue.x = primaries[8] / primaries[9]
            Primaries.Blue.y = primaries[10] / primaries[11]
            Primaries.Blue.Y = 1.0
        else:
            raise ValueError(f'invalid length of primaries {len(primaries)}')
        pPrimaries = &Primaries

    ppTransferFunction[0] = NULL
    ppTransferFunction[1] = NULL
    ppTransferFunction[2] = NULL

    if profile is not None:
        profile = profile.lower()
        if profile == 'gray':
            tfcount = 1
        elif profile == 'rgb':
            tfcount = 3

    try:
        if profile is None:
            pass
        elif gamma is not None:
            for i in range(tfcount):
                ppTransferFunction[i] = cmsBuildGamma(
                    <cmsContext> NULL, <cmsFloat64Number> gamma
                )
                if ppTransferFunction[i] == NULL:
                    raise CmsError('cmsBuildGamma returned NULL')
        elif transferfunction is not None:
            tf = numpy.ascontiguousarray(transferfunction)
            if (
                tf.dtype.char not in 'Hf'
                or tf.ndim not in {1, 2}
                or (tf.ndim == 2 and tf.shape[0] != tfcount)
            ):
                raise ValueError('invalid transferfunction shape or dtype')
            tfentries = tf.shape[tf.ndim - 1]

            for i in range(tfcount):
                j = 0 if tf.ndim == 1 else i
                if tf.dtype.char == 'H':
                    ppTransferFunction[i] = cmsBuildTabulatedToneCurve16(
                        <cmsContext> NULL,
                        <cmsUInt32Number> tfentries,
                        <const cmsUInt16Number*> &tf.data[j * 2 * tfentries]
                    )
                    if ppTransferFunction[i] == NULL:
                        raise CmsError(
                            'cmsBuildTabulatedToneCurve16 returned NULL'
                        )
                else:
                    # tf.dtype.char == 'f'
                    ppTransferFunction[i] = cmsBuildTabulatedToneCurveFloat(
                        <cmsContext> NULL,
                        <cmsUInt32Number> tfentries,
                        <const cmsFloat32Number*> &tf.data[j * 4 * tfentries]
                    )
                    if ppTransferFunction[i] == NULL:
                        raise CmsError(
                            'cmsBuildTabulatedToneCurveFloat returned NULL'
                        )

        if profile is None:
            hProfile = cmsCreateNULLProfile()
            if hProfile == NULL:
                raise CmsError('cmsCreateNULLProfile returned NULL')
        else:
            if profile == 'gray':
                hProfile = cmsCreateGrayProfile(
                    pWhitePoint, ppTransferFunction[0]
                )
                if hProfile == NULL:
                    raise CmsError('cmsCreateGrayProfile returned NULL')
            elif profile == 'rgb':
                if ppTransferFunction[0] == NULL:
                    hProfile = cmsCreateRGBProfile(
                        pWhitePoint, pPrimaries, NULL
                    )
                else:
                    hProfile = cmsCreateRGBProfile(
                        pWhitePoint, pPrimaries, ppTransferFunction
                    )
                if hProfile == NULL:
                    raise CmsError('cmsCreateRGBProfile returned NULL')
            elif profile == 'srgb':
                hProfile = cmsCreate_sRGBProfile()
                if hProfile == NULL:
                    raise CmsError('cmsCreate_sRGBProfile returned NULL')
            elif profile == 'xyz':
                hProfile = cmsCreateXYZProfile()
                if hProfile == NULL:
                    raise CmsError('cmsCreateXYZProfile returned NULL')
            elif profile == 'lab2':
                hProfile = cmsCreateLab2Profile(pWhitePoint)
                if hProfile == NULL:
                    raise CmsError('cmsCreateLab2Profile returned NULL')
            elif profile == 'lab4':
                hProfile = cmsCreateLab4Profile(pWhitePoint)
                if hProfile == NULL:
                    raise CmsError('cmsCreateLab4Profile returned NULL')
            elif profile == 'null':
                hProfile = cmsCreateNULLProfile()
                if hProfile == NULL:
                    raise CmsError('cmsCreateNULLProfile returned NULL')
            elif profile == 'adobergb':
                hProfile = adobe_rgb_compatible()

        if hProfile == NULL:
            raise ValueError(f'failed to create CMS {profile=!r}')

        ret = cmsSaveProfileToMem(hProfile, NULL, &BytesNeeded)
        if ret == 0:
            raise CmsError(f'cmsSaveProfileToMem returned {ret!r}')

        out = PyBytes_FromStringAndSize(NULL, <Py_ssize_t> BytesNeeded)
        if out is None:
            raise MemoryError('PyBytes_FromStringAndSize failed')

        ret = cmsSaveProfileToMem(
            hProfile, <void*> PyBytes_AsString(out), &BytesNeeded
        )
        if ret == 0:
            raise CmsError(f'cmsSaveProfileToMem returned {ret!r}')

    finally:
        if hProfile != NULL:
            cmsCloseProfile(hProfile)
        for i in range(tfcount):
            if ppTransferFunction[i] != NULL:
                cmsFreeToneCurve(ppTransferFunction[i])

    return out


def cms_profile_validate(profile, verbose=False):
    """Raise CmsError if ICC profile is invalid."""
    cdef:
        cmsHPROFILE hProfile = NULL

    if verbose:
        cmsSetLogErrorHandler(_cms_log_error_handler)
    hProfile = open_profile(profile)
    if verbose:
        cmsSetLogErrorHandler(NULL)
    if hProfile == NULL:
        raise CmsError('cmsOpenProfileFromMem returned NULL')
    cmsCloseProfile(hProfile)


@cython.wraparound(True)
def _cms_output_shape(
    cmsUInt32Number inputformat,
    inshape,
    str colorspace,
    cmsUInt32Number planar
):
    """Return shape of output array."""
    cdef:
        cmsUInt32Number inplanar, insamples
        cmsUInt32Number outchannels, outextrachannel, outsamples
        ssize_t ndim

    try:
        outchannels, outextrachannel = _CMS_FORMATS[colorspace.lower()][1:3]
    except (KeyError, AttributeError) as exc:
        raise ValueError(f'invalid output {colorspace=!r}') from exc

    inplanar = T_PLANAR(inputformat)
    insamples = T_CHANNELS(inputformat) + T_EXTRA(inputformat)
    outsamples = outchannels + outextrachannel
    ndim = len(inshape)

    if inplanar and (ndim < 3 or insamples < 2):
        raise RuntimeError(
            f'input planar with len(shape) {ndim} < 3 '
            f'or samples {insamples} < 2'
        )

    outshape = list(inshape)
    if inplanar and planar:
        # input and output planar
        outshape[-3] = int(outsamples)
    elif inplanar:
        # input planar (insamples > 1), output not
        del outshape[-3]
        if outsamples > 1:
            outshape += [int(outsamples)]
    elif planar:
        # output planar (outsamples > 1), input not
        if outsamples < 2:
            raise RuntimeError(f'output planar with samples {outsamples} < 2')
        if insamples > 1:
            del outshape[-1]
        outshape.insert(-2, int(outsamples))
    elif insamples > 1:
        if outsamples > 1:
            outshape[-1] = int(outsamples)
        else:
            del outshape[-1]
    elif outsamples > 1:
        outshape += [int(outsamples)]

    return tuple(outshape)


def _cms_format_decode(cmsUInt32Number cmsformat):
    """Return unpacked cms format number; for testing."""
    from collections import namedtuple

    CmsFormat = namedtuple(
        'CmsFormat',
        [
            'pixeltype',
            'dtype',
            'samples',
            'planar',
            'swap',
            'swapfirst',
            'isfloat',
            'itemsize',
            'channels',
            'extrachannels',
            'bigendian',
            'flavor',
            'optimized',
        ]
    )

    # can't use lcms T_FLOAT macro; it is redefined in Python structmember.h
    # isfloat = bool(T_FLOAT(cmsformat))
    isfloat = bool((((cmsformat) >> 22) & 1))
    itemsize = int(T_BYTES(cmsformat))
    if itemsize == 0:
        itemsize = 8
    dtype = f"{'f' if isfloat else 'u'}{itemsize}"
    if (T_ENDIAN16(cmsformat)):
        dtype = '>' + dtype

    return CmsFormat(
        pixeltype=CMS.PT(T_COLORSPACE(cmsformat)),
        dtype=dtype,
        samples=int(T_CHANNELS(cmsformat) + T_EXTRA(cmsformat)),
        planar=bool(T_PLANAR(cmsformat)),
        swap=bool(T_DOSWAP(cmsformat)),
        swapfirst=bool(T_SWAPFIRST(cmsformat)),
        isfloat=isfloat,
        itemsize=itemsize,
        channels=int(T_CHANNELS(cmsformat)),
        extrachannels=int(T_EXTRA(cmsformat)),
        bigendian=bool(T_ENDIAN16(cmsformat)),
        flavor=int(T_FLAVOR(cmsformat)),
        optimized=int(T_OPTIMIZED(cmsformat))
    )


@cython.wraparound(True)
def _cms_format(shape, dtype, colorspace=None, planar=None):
    """Return lcms format number.

    It's best to explicitly specify colorspace and planar to avoid ambiguities.
    By default, colorspace is gray unless dtype is u1 and shape is (i, j, 3|4).
    If planar is specified, colorpace is gray(a) for 1 or 2 samples or rgb(a)
    for 3 or 4 samples. Else raise a ValueError.
    If colorspace is specified, its number of components must match number of
    samples according to shape and planar. RGB and CMYK colorspaces are
    allowed to have one extra component.

    """
    cdef:
        cmsUInt32Number default = 0
        cmsUInt32Number pixeltype = 0
        cmsUInt32Number itemsize = 0
        cmsUInt32Number channels = 0
        cmsUInt32Number extrachannel = 0
        # cmsUInt32Number multichannel = 0
        cmsUInt32Number isplanar = 0
        cmsUInt32Number swap = 0
        cmsUInt32Number swapfirst = 0
        cmsUInt32Number endian = 0
        cmsUInt32Number flavor = 0
        cmsUInt32Number isfloat = 0
        cmsUInt32Number samples = 1
        ssize_t ndim = 1

    shape = tuple(shape)
    dtype = numpy.dtype(dtype)
    ndim = len(shape)
    itemsize = dtype.itemsize
    isfloat = dtype.kind == 'f'
    endian = 0 if dtype.isnative else 1

    if ndim < 2:
        raise ValueError('invalid shape')

    # TODO: float16 segfaulting
    if dtype.char not in 'BHdfe':
        raise ValueError(f'{dtype=!r} not supported')

    if itemsize == 8:
        itemsize = 0  # 8 would overflow the bitfield

    if ndim > 2:
        if planar is None:
            isplanar = 0  # do not autodetect planar configuration
        else:
            isplanar = 1 if planar else 0
        samples = <cmsUInt32Number> (shape[-3] if isplanar else shape[-1])

    if ndim == 2:
        pixeltype = PT_GRAY
        channels = 1
        isplanar = 0
        if planar:
            raise ValueError('planar not supported with 2D')
        if colorspace not in {None, 'gray', 'miniswhite', 'minisblack'}:
            raise ValueError('invalid colorspace for 2D')

    elif colorspace is None:

        if planar is None:
            if ndim == 3 and itemsize == 1 and (samples == 3 or samples == 4):
                # only uint8 RGB(A) is detected
                pixeltype = PT_RGB
                channels = 3
                if samples == 4:
                    extrachannel = 1
            else:
                pixeltype = PT_GRAY
                channels = 1
                samples = 1

        elif samples < 3:
            pixeltype = PT_GRAY
            channels = 1
            if samples == 2:
                extrachannel = 1  # GA
        elif samples < 5:
            pixeltype = PT_RGB
            channels = 3
            if samples == 4:
                extrachannel = 1  # RGBA
        # elif samples < 6:
        #     pixeltype = PT_CMYK
        #     channels = 4
        #     if samples == 5:
        #         extrachannel = 1  # CMYKA
        # elif isplanar = 0:
        #     pixeltype = PT_GRAY
        #     channels = 1
        #     samples = 1
        else:
            raise ValueError(
                f'can not guess colorspace for {samples} '
                f'{"planar" if isplanar else "contig"} samples'
            )

    else:
        # colorspace specified
        try:
            (
                pixeltype,
                channels,
                extrachannel,
                swap,
                swapfirst
            ) = _CMS_FORMATS[colorspace.lower()]
        except (KeyError, AttributeError) as exc:
            raise ValueError(f'invalid {colorspace=!r}') from exc

        if planar is None and ndim > 2:
            # determine isplanar
            if shape[-1] == channels + extrachannel:
                # exact match in last axis
                isplanar = 0
            elif pixeltype == PT_GRAY and extrachannel == 0:
                # gray volume
                samples = 1
                isplanar = 0
            elif (
                extrachannel == 0
                and (pixeltype == PT_RGB or pixeltype == PT_CMYK)
            ):
                # allow RGB -> RGBA, CMYK -> CMYKA
                # prefer over exact match in planar axis
                if shape[-1] == channels + 1:
                    extrachannel = 1
                elif shape[-3] == channels + 1:
                    isplanar = 1
                    extrachannel = 1
                elif shape[-3] == channels:
                    # exact match in planar axis
                    isplanar = 1
            elif shape[-3] == channels + extrachannel:
                # exact match in planar axis
                isplanar = 1

        if pixeltype == PT_GRAY and extrachannel == 0:
            samples = 1
            isplanar = 0
        else:
            samples = <cmsUInt32Number> (
                shape[-3] if isplanar else shape[-1]
            )

        if pixeltype == PT_GRAY and samples == 2:
            extrachannel = 1  # G -> GA
        elif pixeltype == PT_RGB and samples == 4:
            extrachannel = 1  # RGB -> RGBA
        elif pixeltype == PT_CMYK and samples == 5:
            extrachannel = 1  # CMYK -> CMYKA

        # TODO: support multichannel
        # elif pixeltype == PT_CMYK and samples > 5:
        #     if samples == 6:
        #         channels = 6
        #         pixeltype = PT_MCH6
        #     elif samples == 7:
        #         channels = 7
        #         pixeltype = PT_MCH7
        #     elif samples == 8:
        #         channels = 8
        #         pixeltype = PT_MCH8
        #     elif samples == 9:
        #         channels = 9
        #         pixeltype = PT_MCH9
        #     elif samples == 10:
        #         channels = 10
        #         pixeltype = PT_MCH10
        #     elif samples == 11:
        #         channels = 11
        #         pixeltype = PT_MCH11
        #     elif samples == 12:
        #         channels = 12
        #         pixeltype = PT_MCH12

        if pixeltype == PT_GRAY and extrachannel == 0:
            if planar:
                raise ValueError(
                    "planar can not be specified with colorspace 'gray' "
                    "and no extra samples"
                )
        elif samples != channels + extrachannel:
            raise ValueError(
                f'shape {shape!r} does not match {colorspace=!r} '
                f'with {channels + extrachannel} '
                f'{"planar" if isplanar else "contig"} samples'
            )

    return int(
        default
        | COLORSPACE_SH(pixeltype)
        | BYTES_SH(itemsize)
        | CHANNELS_SH(channels)
        | EXTRA_SH(extrachannel)
        | PLANAR_SH(isplanar)
        | ENDIAN16_SH(endian)
        | SWAPFIRST_SH(swapfirst)
        | DOSWAP_SH(swap)
        | FLAVOR_SH(flavor)
        | FLOAT_SH(isfloat)
    )


_CMS_FORMATS = {
    # colorspace -> pixeltype, channels, extrachannel, swap, swapfirst
    'minisblack': (PT_GRAY, 1, 0, 0, 0),
    'miniswhite': (PT_GRAY, 1, 0, 0, 0),
    'gray': (PT_GRAY, 1, 0, 0, 0),
    'graya': (PT_GRAY, 1, 1, 0, 0),
    'rgb': (PT_RGB, 3, 0, 0, 0),
    'rgba': (PT_RGB, 3, 1, 0, 0),
    'argb': (PT_RGB, 3, 1, 0, 1),
    'bgr': (PT_RGB, 3, 0, 1, 0),
    'abgr': (PT_RGB, 3, 1, 1, 0),
    'bgra': (PT_RGB, 3, 1, 1, 1),
    'cmy': (PT_CMY, 3, 0, 0, 0),
    'cmyk': (PT_CMYK, 4, 0, 0, 0),
    'cmyka': (PT_CMYK, 4, 1, 0, 0),
    'kymc': (PT_CMYK, 4, 0, 1, 0),
    'kcmy': (PT_CMYK, 4, 0, 0, 1),
    'lab': (PT_Lab, 3, 0, 0, 0),
    'laba': (PT_Lab, 3, 1, 0, 0),
    'alab': (PT_Lab, 3, 1, 0, 1),
    'labv2': (PT_LabV2, 3, 0, 0, 0),
    'alabv2': (PT_LabV2, 3, 1, 0, 1),
    'ycbcr': (PT_YCbCr, 3, 0, 0, 0),
    'xyz': (PT_XYZ, 3, 0, 0, 0),
    'hsv': (PT_HSV, 3, 0, 0, 0),
    'hls': (PT_HLS, 3, 0, 0, 0),
    'yxy': (PT_Yxy, 3, 0, 0, 0),
    'yuv': (PT_YUV, 3, 0, 0, 0),
    'luv': (PT_YUV, 3, 0, 0, 0),
    'yuvk': (PT_YUVK, 4, 0, 0, 0),
    'luvk': (PT_YUVK, 4, 0, 0, 0),
}


cdef cmsHPROFILE open_profile(object profile):
    """Return handle from CMS profile bytes."""
    cdef:
        cmsHPROFILE hProfile = NULL

    if not PyBytes_Check(profile):
        return NULL
    hProfile = cmsOpenProfileFromMem(
        <const void*> PyBytes_AsString(profile),
        <cmsUInt32Number> PyBytes_Size(profile)
    )
    return hProfile


cdef cmsHPROFILE adobe_rgb_compatible() nogil:
    """Return handle to Adobe RGB compatible CMS profile."""
    cdef:
        cmsHPROFILE hProfile = NULL
        cmsCIEXYZTRIPLE color
        cmsCIEXYZ black, d65
        cmsToneCurve* transferfunction = NULL
        cmsMLU *mlu = NULL
        cmsBool ret

    color.Red.X = 0.609741
    color.Red.Y = 0.311111
    color.Red.Z = 0.019470
    color.Green.X = 0.205276
    color.Green.Y = 0.625671
    color.Green.Z = 0.060867
    color.Blue.X = 0.149185
    color.Blue.Y = 0.063217
    color.Blue.Z = 0.744568

    d65.X = 0.95045
    d65.Y = 1.0
    d65.Z = 1.08905

    black.X = 0
    black.Y = 0
    black.Z = 0

    transferfunction = cmsBuildGamma(<cmsContext> NULL, 2.19921875)

    hProfile = cmsCreateProfilePlaceholder(<cmsContext> NULL)
    cmsSetProfileVersion(hProfile, 2.1)

    mlu = cmsMLUalloc(<cmsContext> NULL, 1)
    if mlu == NULL:
        cmsCloseProfile(hProfile)
        return NULL
    cmsMLUsetASCII(mlu, b'en\0', b'US\0', 'Public Domain')
    ret = cmsWriteTag(hProfile, cmsSigCopyrightTag, mlu)
    cmsMLUfree(mlu)

    mlu = cmsMLUalloc(<cmsContext> NULL, 1)
    if mlu == NULL:
        cmsCloseProfile(hProfile)
        return NULL
    cmsMLUsetASCII(mlu, b'en', b'US\0', 'Adobe RGB (compatible)')
    ret = cmsWriteTag(hProfile, cmsSigProfileDescriptionTag, mlu)
    cmsMLUfree(mlu)

    mlu = cmsMLUalloc(<cmsContext> NULL, 1)
    if mlu == NULL:
        cmsCloseProfile(hProfile)
        return NULL
    cmsMLUsetASCII(mlu, b'en\0', b'US\0', 'Imagecodecs')
    ret = cmsWriteTag(hProfile, cmsSigDeviceMfgDescTag, mlu)
    cmsMLUfree(mlu)

    mlu = cmsMLUalloc(<cmsContext> NULL, 1)
    if mlu == NULL:
        cmsCloseProfile(hProfile)
        return NULL
    cmsMLUsetASCII(mlu, b'en\0', b'US\0', 'Adobe RGB (compatible)')
    ret = cmsWriteTag(hProfile, cmsSigDeviceModelDescTag, mlu)
    cmsMLUfree(mlu)

    cmsSetDeviceClass(hProfile, cmsSigDisplayClass)
    cmsSetColorSpace(hProfile, cmsSigRgbData)
    cmsSetPCS(hProfile, cmsSigXYZData)

    ret = cmsWriteTag(hProfile, cmsSigMediaWhitePointTag, &d65)
    ret = cmsWriteTag(hProfile, cmsSigMediaBlackPointTag, &black)

    ret = cmsWriteTag(hProfile, cmsSigRedColorantTag, <void*> &color.Red)
    ret = cmsWriteTag(hProfile, cmsSigGreenColorantTag, <void*> &color.Green)
    ret = cmsWriteTag(hProfile, cmsSigBlueColorantTag, <void*> &color.Blue)

    ret = cmsWriteTag(hProfile, cmsSigRedTRCTag, <void*> transferfunction)

    cmsLinkTag(hProfile, cmsSigGreenTRCTag, cmsSigRedTRCTag)
    cmsLinkTag(hProfile, cmsSigBlueTRCTag, cmsSigRedTRCTag)

    return hProfile


cdef void _cms_log_error_handler(
    cmsContext ContextID,
    cmsUInt32Number ErrorCode,
    const char *Text
) noexcept with gil:
    _log_warning('CMS error: %s', Text.decode().strip())
