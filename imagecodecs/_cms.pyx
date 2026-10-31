# imagecodecs/_cms.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: freethreading_compatible = True

# Copyright (c) 2021-2026, Christoph Gohlke
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

"""CMS (Color Management System) codec for the imagecodecs package."""

include '_shared.pxi'

import cython

from cpython.bytes cimport PyBytes_AsString, PyBytes_FromStringAndSize
from lcms2 cimport *


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


def cms_check(const uint8_t[::1] data not None, /):
    """Return whether data is ICC profile or None if unknown."""
    if data.shape[0] < 40:
        return False
    if _startswith(data, b'acsp', 36):
        return True
    return None


def cms_info(
    profile,
    /,
):
    """Return information about ICC profile."""
    cdef:
        cmsHPROFILE hProfile = NULL
        cmsInt32Number nchannels
        cmsBool has_datetime
        cmsFloat64Number* chad
        cmsCIEXYZ* wp_xyz
        cmsCIEXYZ* r_xyz
        cmsCIEXYZ* g_xyz
        cmsCIEXYZ* b_xyz
        cmsCIEXYZ adapted
        cmsToneCurve* curve
        cmsFloat64Number gamma_val
        tm t
        int i

    hProfile = _cms_open_profile(profile)
    if hProfile == NULL:
        raise CmsError('cmsOpenProfileFromMem returned NULL')

    try:
        colorspace = _CMS_COLORSPACE_SIGNATURES.get(
            cmsGetColorSpace(hProfile)
        )
        pcs = _CMS_COLORSPACE_SIGNATURES.get(
            cmsGetPCS(hProfile)
        )
        deviceclass = _CMS_DEVICE_CLASSES.get(
            cmsGetDeviceClass(hProfile)
        )
        nchannels = cmsChannelsOfColorSpace(cmsGetColorSpace(hProfile))
        channels = int(nchannels) if nchannels >= 0 else None
        version = round(cmsGetProfileVersion(hProfile), 1)

        renderingintent = _CMS_RENDERING_INTENTS.get(
            cmsGetHeaderRenderingIntent(hProfile)
        )
        ismatrixshaper = bool(cmsIsMatrixShaper(hProfile))

        has_datetime = cmsGetHeaderCreationDateTime(hProfile, &t)
        if has_datetime:
            datetime = (
                f'{t.tm_year + 1900:04d}-{t.tm_mon + 1:02d}-{t.tm_mday:02d}'
                f'T{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}'
            )
        else:
            datetime = None

        # chromatic adaptation tag maps the adopted white to D50 (v4)
        chad = <cmsFloat64Number*> cmsReadTag(
            hProfile, cmsSigChromaticAdaptationTag
        )
        if chad != NULL:
            unadapt = numpy.linalg.inv(
                numpy.array([chad[i] for i in range(9)]).reshape(3, 3)
            )
        else:
            unadapt = None

        wp_xyz = <cmsCIEXYZ*> cmsReadTag(hProfile, cmsSigMediaWhitePointTag)
        if unadapt is not None:
            whitepoint = _cms_xyy(
                unadapt @ (cmsD50_XYZ().X, cmsD50_XYZ().Y, cmsD50_XYZ().Z)
            )
        elif wp_xyz != NULL:
            whitepoint = _cms_xyy((wp_xyz.X, wp_xyz.Y, wp_xyz.Z))
        else:
            whitepoint = None

        # primaries: unadapt colorant tags from D50 to adopted white
        primaries = None
        if colorspace == 'rgb':
            r_xyz = <cmsCIEXYZ*> cmsReadTag(hProfile, cmsSigRedColorantTag)
            g_xyz = <cmsCIEXYZ*> cmsReadTag(hProfile, cmsSigGreenColorantTag)
            b_xyz = <cmsCIEXYZ*> cmsReadTag(hProfile, cmsSigBlueColorantTag)
            if r_xyz == NULL or g_xyz == NULL or b_xyz == NULL:
                pass
            elif unadapt is not None:
                primaries = (
                    _cms_xyy(unadapt @ (r_xyz.X, r_xyz.Y, r_xyz.Z)),
                    _cms_xyy(unadapt @ (g_xyz.X, g_xyz.Y, g_xyz.Z)),
                    _cms_xyy(unadapt @ (b_xyz.X, b_xyz.Y, b_xyz.Z)),
                )
            elif wp_xyz != NULL:
                cmsAdaptToIlluminant(&adapted, cmsD50_XYZ(), wp_xyz, r_xyz)
                red = _cms_xyy((adapted.X, adapted.Y, adapted.Z))
                cmsAdaptToIlluminant(&adapted, cmsD50_XYZ(), wp_xyz, g_xyz)
                green = _cms_xyy((adapted.X, adapted.Y, adapted.Z))
                cmsAdaptToIlluminant(&adapted, cmsD50_XYZ(), wp_xyz, b_xyz)
                blue = _cms_xyy((adapted.X, adapted.Y, adapted.Z))
                primaries = (red, green, blue)

        # gamma from TRC tag; None if curve is not a pure power law
        if colorspace == 'gray':
            curve = <cmsToneCurve*> cmsReadTag(hProfile, cmsSigGrayTRCTag)
        elif colorspace == 'rgb':
            curve = <cmsToneCurve*> cmsReadTag(hProfile, cmsSigRedTRCTag)
        else:
            curve = NULL
        if curve != NULL:
            gamma_val = cmsEstimateGamma(curve, 0.01)
            gamma = round(gamma_val, 6) if gamma_val > 0 else None
        else:
            gamma = None

        description = _cms_info_ascii(hProfile, cmsInfoDescription)
        manufacturer = _cms_info_ascii(hProfile, cmsInfoManufacturer)
        model = _cms_info_ascii(hProfile, cmsInfoModel)
        copyright = _cms_info_ascii(hProfile, cmsInfoCopyright)

    finally:
        cmsCloseProfile(hProfile)

    return {
        '_codec': 'cms',
        'version': version,
        'deviceclass': deviceclass,
        'colorspace': colorspace,
        'channels': channels,
        'pcs': pcs,
        'renderingintent': renderingintent,
        'ismatrixshaper': ismatrixshaper,
        'datetime': datetime,
        'whitepoint': whitepoint,
        'primaries': primaries,
        'gamma': gamma,
        'description': description,
        'manufacturer': manufacturer,
        'model': model,
        'copyright': copyright,
    }


def cms_transform(
    data,
    /,
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
    out=None,
):
    """Return color-transformed array.

    Parameters:
        data:
            Image data to transform.
            Data type must be uint8, uint16, float16, float32, or float64.
            Samples are stored in the last axis (contiguous) or, if planar,
            in the third-to-last axis. 2D arrays are gray.
        profile:
            ICC profile of `data`, for example, returned by `cms_profile`.
            If None, use a NULL profile.
        outprofile:
            ICC profile of output.
            If None, use a NULL profile.
        colorspace:
            Color space and sample order of `data`, for example,
            ``'gray'``, ``'graya'``, ``'rgb'``, ``'rgba'``, ``'bgr'``,
            ``'cmyk'``, ``'lab'``, ``'xyz'``, or ``'miniswhite'``.
            By default, derived from color space of `profile`.
        planar:
            Samples of `data` are stored in third-to-last axis.
            By default, determined from shape of `data` and `colorspace`.
        outcolorspace:
            Color space and sample order of output.
            By default, derived from color space of `outprofile`, else
            `colorspace`.
        outplanar:
            Store output samples in third-to-last axis.
            By default, same as `planar`.
        outdtype:
            Data type of output.
            By default, same as `data`. Ignored if `out` is an array.
        intent:
            Rendering intent. By default, ``CMS.INTENT.PERCEPTUAL``.
        flags:
            Combination of ``CMS.FLAGS`` passed to ``cmsCreateTransform``.
        verbose:
            Log Little-CMS error messages.
        out:
            Writable, C-contiguous output array.
            Its shape must contain the same number of pixels as `data`.

    Returns:
        Color-transformed image.

    Raises:
        ValueError:
            Invalid data type, shape, color space, or output array.
        CmsError:
            Little-CMS failed to open profiles or create transform.

    """
    cdef:
        numpy.ndarray src
        numpy.ndarray dst
        cmsUInt32Number Intent = _enum_value(
            intent, CMS.INTENT, INTENT_PERCEPTUAL
        )
        cmsUInt32Number dwFlags = 0 if flags is None else flags
        cmsUInt32Number InputFormat = 0
        cmsUInt32Number OutputFormat = 0
        cmsContext ctx = NULL
        cmsHPROFILE hInProfile = NULL
        cmsHPROFILE hOutProfile = NULL
        cmsHTRANSFORM hTransform = NULL
        cmsUInt32Number numpixels
        ssize_t npixels, framepixels
        ssize_t nframes, src_frame_stride, dst_frame_stride
        char* src_ptr
        char* dst_ptr

    # lcms only byte-swaps 16-bit integers
    data = _array_input(data, out, None, True, False, True)
    src = data

    try:
        if verbose:
            ctx = _cms_verbose_context()

        if profile is None:
            hInProfile = cmsCreateNULLProfileTHR(ctx)
            if hInProfile == NULL:
                raise CmsError('cmsCreateNULLProfile returned NULL')
        else:
            hInProfile = _cms_open_profile(profile, ctx)
            if hInProfile == NULL:
                raise CmsError('cmsOpenProfileFromMem returned NULL')

        if outprofile is None:
            hOutProfile = cmsCreateNULLProfileTHR(ctx)
            if hOutProfile == NULL:
                raise CmsError('cmsCreateNULLProfile returned NULL')
        else:
            hOutProfile = _cms_open_profile(outprofile, ctx)
            if hOutProfile == NULL:
                raise CmsError('cmsOpenProfileFromMem returned NULL')

        # determine colorspace/pixeltype from profiles if not specified
        if colorspace is None:
            colorspace = _CMS_COLORSPACE_SIGNATURES.get(
                cmsGetColorSpace(hInProfile)
            )
        if outcolorspace is None:
            outcolorspace = _CMS_COLORSPACE_SIGNATURES.get(
                cmsGetColorSpace(hOutProfile)
            )

        InputFormat = _cms_format(data.shape, data.dtype, colorspace, planar)

        planar = T_PLANAR(InputFormat)
        npixels = src.size // (T_CHANNELS(InputFormat) + T_EXTRA(InputFormat))

        if out is not None and isinstance(out, numpy.ndarray):
            # assert outdtype is None
            OutputFormat = _cms_format(
                out.shape, out.dtype, outcolorspace, outplanar
            )
        else:
            if outcolorspace is None:
                outcolorspace = colorspace
            if outplanar is None:
                outplanar = planar
            if outdtype is None:
                outdtype = data.dtype

            outshape = _cms_output_shape(
                InputFormat, data.shape, outcolorspace, outplanar
            )

            out = _create_array(out, outshape, outdtype)

            OutputFormat = _cms_format(
                out.shape, out.dtype, outcolorspace, outplanar
            )

        outplanar = T_PLANAR(OutputFormat)

        dst = out
        if not dst.flags.c_contiguous:
            raise ValueError('output array not C-contiguous')
        if not dst.flags.writeable:
            raise ValueError('output array not writable')
        if dst.dtype.kind == 'f' and not dst.dtype.isnative:
            raise ValueError(f'non-native output {dst.dtype=} not supported')
        if (
            dst.size // (T_CHANNELS(OutputFormat) + T_EXTRA(OutputFormat))
            != npixels
        ):
            raise ValueError(
                f'output shape {out.shape!r} does not match {npixels} pixels'
            )

        if planar or outplanar:
            # planes are contiguous per frame of the last two axes
            if planar:
                framepixels = (
                    <ssize_t> src.shape[src.ndim - 2]
                    * src.shape[src.ndim - 1]
                )
            else:
                framepixels = (
                    <ssize_t> dst.shape[dst.ndim - 2]
                    * dst.shape[dst.ndim - 1]
                )
            if (
                outplanar
                and framepixels
                != <ssize_t> dst.shape[dst.ndim - 2] * dst.shape[dst.ndim - 1]
            ) or npixels % framepixels != 0:
                raise ValueError(
                    f'input shape {data.shape!r} and output shape '
                    f'{out.shape!r} frames do not match'
                )
            nframes = npixels // framepixels
        else:
            framepixels = npixels
            nframes = 1
        if <size_t> framepixels > <size_t> UINT32_MAX:
            raise ValueError(f'too many pixels per frame {framepixels}')
        numpixels = <cmsUInt32Number> framepixels
        src_frame_stride = src.nbytes // nframes
        dst_frame_stride = dst.nbytes // nframes

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

            src_ptr = src.data
            dst_ptr = dst.data
            for _i in range(nframes):
                cmsDoTransform(
                    hTransform,
                    <const void*> src_ptr,
                    <void*> dst_ptr,
                    numpixels
                )
                src_ptr += src_frame_stride
                dst_ptr += dst_frame_stride

            cmsDeleteTransform(hTransform)
            hTransform = NULL

    finally:
        if hInProfile != NULL:
            cmsCloseProfile(hInProfile)
        if hOutProfile != NULL:
            cmsCloseProfile(hOutProfile)
        if hTransform != NULL:
            cmsDeleteTransform(hTransform)
        if ctx != NULL:
            cmsDeleteContext(ctx)

    return out


# @cython.boundscheck(True)
def cms_profile(
    profile,
    /,
    *,
    whitepoint=None,  # gray and rgb
    primaries=None,  # rgb
    transferfunction=None,  # gray and rgb
    gamma=None,
):
    """Return ICC profile.

    Parameters:
        profile:
            Name of profile, case-insensitive:

            - ``'gray'``, ``'rgb'``: custom profile defined by `whitepoint`,
              `primaries`, and `transferfunction` or `gamma`.
            - ``'srgb'``: sRGB IEC61966-2.1.
            - ``'adobergb'``: Adobe RGB (1998) compatible, ICC v2.
            - ``'linearrgb'``: Rec. 709 primaries, D65, linear.
            - ``'displayp3'``: Display P3, D65, sRGB transfer function.
            - ``'rec2020'``: Rec. 2020, D65, Rec. 709 transfer function.
            - ``'sgray'``: gray, D65, sRGB transfer function.
            - ``'xyz'``: CIE XYZ.
            - ``'lab2'``, ``'lab4'``: CIE L*a*b* v2 or v4.
              White point defaults to D50.
            - ``'null'`` or None: output all zeros.

        whitepoint:
            White point of `'gray'`, `'rgb'`, `'lab2'`, or `'lab4'` profile.
            As CIE xyY (3 values), xy (2 values), or xy rationals (4 values).
            Defaults to D50.
        primaries:
            Red, green, and blue primaries of `'rgb'` profile.
            As CIE xyY (9 values), xy (6 values), or xy rationals (12 values).
            Requires `whitepoint`.
        transferfunction:
            Transfer function of `'gray'` or `'rgb'` profile.
            Either a named parametric curve, ``'srgb'`` or ``'rec709'``,
            or a uint16 or float32 lookup table of shape ``(entries,)``
            or, for `'rgb'`, ``(3, entries)``.
            Mutually exclusive with `gamma`.
        gamma:
            Gamma of `'gray'` or `'rgb'` profile transfer function.
            Mutually exclusive with `transferfunction`.

    Returns:
        ICC profile.

    Raises:
        ValueError:
            Invalid profile name or combination of parameters.
        CmsError:
            Little-CMS failed to create or serialize profile.

    """
    cdef:
        cmsHPROFILE hProfile = NULL
        cmsCIExyY WhitePoint
        cmsCIExyY* pWhitePoint = NULL
        cmsCIExyYTRIPLE Primaries
        cmsCIExyYTRIPLE* pPrimaries = NULL
        # cmsToneCurve TransferFunction
        (cmsToneCurve*)[3] ppTransferFunction
        cmsUInt32Number BytesNeeded
        cmsBool ret
        cmsInt32Number curvetype
        cmsFloat64Number[10] curveparams
        ssize_t tfcount = 0
        ssize_t tfentries = 0
        ssize_t i, j
        numpy.ndarray tf
        bytes description = None
        object out

    if profile is not None:
        profile = profile.lower()
        if profile in _CMS_NAMED_PROFILES:
            if (
                whitepoint is not None
                or primaries is not None
                or transferfunction is not None
                or gamma is not None
            ):
                raise ValueError(
                    f'{profile=!r} does not support whitepoint, primaries, '
                    'transferfunction, or gamma'
                )
            (
                profile,
                whitepoint,
                primaries,
                transferfunction,
                description,
            ) = _CMS_NAMED_PROFILES[profile]

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
            raise ValueError(f'invalid length of whitepoint {len(whitepoint)}')
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

    if profile == 'gray':
        tfcount = 1
    elif profile == 'rgb':
        tfcount = 3

    if whitepoint is not None and profile not in {
        'gray', 'rgb', 'lab2', 'lab4'
    }:
        raise ValueError(f'whitepoint not supported for {profile=!r}')

    if primaries is not None:
        if profile != 'rgb':
            raise ValueError(f'primaries not supported for {profile=!r}')
        if whitepoint is None:
            raise ValueError('primaries require whitepoint')

    if gamma is not None and transferfunction is not None:
        raise ValueError('gamma and transferfunction are mutually exclusive')

    if (gamma is not None or transferfunction is not None) and tfcount == 0:
        raise ValueError(
            f'gamma and transferfunction are only supported '
            f'for gray and rgb profiles, not {profile!r}'
        )

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
        elif (
            transferfunction is not None and isinstance(transferfunction, str)
        ):
            try:
                curvetype, params = _CMS_CURVES[transferfunction.lower()]
            except KeyError as exc:
                raise ValueError(f'invalid {transferfunction=!r}') from exc
            for i in range(10):
                curveparams[i] = params[i] if i < len(params) else 0.0
            for i in range(tfcount):
                ppTransferFunction[i] = cmsBuildParametricToneCurve(
                    <cmsContext> NULL, curvetype, curveparams
                )
                if ppTransferFunction[i] == NULL:
                    raise CmsError('cmsBuildParametricToneCurve returned NULL')
        elif transferfunction is not None:
            tf = numpy.asarray(transferfunction, order='C')
            if (
                tf.dtype.char not in 'Hf'
                or tf.ndim not in {1, 2}
                or (tf.ndim == 2 and tf.shape[0] != tfcount)
            ):
                raise ValueError('invalid transferfunction shape or dtype')
            if not tf.dtype.isnative:
                tf = tf.astype(tf.dtype.newbyteorder('='))
            tfentries = tf.shape[tf.ndim - 1]
            if tfentries < 2:
                raise ValueError(
                    f'transferfunction has {tfentries} < 2 entries'
                )

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
        elif profile == 'gray':
            hProfile = cmsCreateGrayProfile(pWhitePoint, ppTransferFunction[0])
            if hProfile == NULL:
                raise CmsError('cmsCreateGrayProfile returned NULL')
        elif profile == 'rgb':
            if ppTransferFunction[0] == NULL:
                hProfile = cmsCreateRGBProfile(pWhitePoint, pPrimaries, NULL)
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
            hProfile = _cms_adobe_rgb_compatible()
        elif profile == 'linearrgb':
            hProfile = _cms_linear_rgb()

        if hProfile == NULL:
            raise ValueError(f'failed to create CMS {profile=!r}')

        if description is not None and not _cms_write_mlu(
            hProfile, cmsSigProfileDescriptionTag, description
        ):
            raise CmsError('failed to write profile description')

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

        # constant creation date 2023-11-14 22:13:20 for reproducible output
        if BytesNeeded >= 36:
            memcpy(
                PyBytes_AsString(out) + 24,
                b'\x07\xe7\x00\x0b\x00\x0e\x00\x16\x00\x0d\x00\x14',
                12
            )

    finally:
        if hProfile != NULL:
            cmsCloseProfile(hProfile)
        for i in range(tfcount):
            if ppTransferFunction[i] != NULL:
                cmsFreeToneCurve(ppTransferFunction[i])

    return out


def cms_profile_validate(
    profile,
    /,
    *,
    verbose=None,
):
    """Raise CmsError if ICC profile is invalid."""
    cdef:
        cmsContext ctx = NULL
        cmsHPROFILE hProfile = NULL

    try:
        if verbose:
            ctx = _cms_verbose_context()
        hProfile = _cms_open_profile(profile, ctx)
        if hProfile == NULL:
            raise CmsError('cmsOpenProfileFromMem returned NULL')
        cmsCloseProfile(hProfile)
    finally:
        if ctx != NULL:
            cmsDeleteContext(ctx)


@cython.wraparound(True)
def _cms_output_shape(
    cmsUInt32Number inputformat,
    inshape,
    str colorspace,
    cmsUInt32Number planar,
    /,
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


def _cms_format_decode(cmsUInt32Number cmsformat, /):
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
def _cms_format(
    shape,
    dtype,
    colorspace=None,
    planar=None,
    /,
):
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

    if dtype.char not in 'BHdfe':
        raise ValueError(f'{dtype=!r} not supported')

    if colorspace is not None:
        try:
            colorspace = colorspace.lower()
        except AttributeError as exc:
            raise ValueError(f'invalid {colorspace=!r}') from exc

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
        if colorspace == 'miniswhite':
            flavor = 1

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

        if colorspace.lower() == 'miniswhite':
            flavor = 1

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


_CMS_COLORSPACE_SIGNATURES = {
    # cmsColorSpaceSignature -> colorspace string
    cmsSigRgbData: 'rgb',
    cmsSigGrayData: 'gray',
    cmsSigCmykData: 'cmyk',
    cmsSigLabData: 'lab',
    cmsSigXYZData: 'xyz',
    cmsSigYCbCrData: 'ycbcr',
    cmsSigHsvData: 'hsv',
    cmsSigHlsData: 'hls',
    cmsSigCmyData: 'cmy',
    cmsSigLuvData: 'luv',
    cmsSigYxyData: 'yxy',
    cmsSigMCH1Data: 'mch1',
    cmsSigMCH2Data: 'mch2',
    cmsSigMCH3Data: 'mch3',
    cmsSigMCH4Data: 'mch4',
    cmsSigMCH5Data: 'mch5',
    cmsSigMCH6Data: 'mch6',
    cmsSigMCH7Data: 'mch7',
    cmsSigMCH8Data: 'mch8',
    cmsSigMCH9Data: 'mch9',
}

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

_CMS_CURVES = {
    # name: lcms parametric curve type, parameters (g, a, b, c, d)
    'srgb': (4, (2.4, 1.0 / 1.055, 0.055 / 1.055, 1.0 / 12.92, 0.04045)),
    'rec709': (4, (1.0 / 0.45, 1.0 / 1.099, 0.099 / 1.099, 1.0 / 4.5, 0.081)),
}

_CMS_NAMED_PROFILES = {
    # name: profile, whitepoint, primaries, transferfunction, description
    'displayp3': (
        'rgb',
        (0.3127, 0.3290),
        (0.680, 0.320, 0.265, 0.690, 0.150, 0.060),
        'srgb',
        b'Display P3',
    ),
    'rec2020': (
        'rgb',
        (0.3127, 0.3290),
        (0.708, 0.292, 0.170, 0.797, 0.131, 0.046),
        'rec709',
        b'Rec. 2020',
    ),
    'sgray': ('gray', (0.3127, 0.3290), None, 'srgb', b'sGray'),
}

_CMS_DEVICE_CLASSES = {
    # cmsProfileClassSignature -> deviceclass string
    cmsSigInputClass: 'input',
    cmsSigDisplayClass: 'display',
    cmsSigOutputClass: 'output',
    cmsSigLinkClass: 'link',
    cmsSigAbstractClass: 'abstract',
    cmsSigColorSpaceClass: 'colorspace',
    cmsSigNamedColorClass: 'namedcolor',
}

_CMS_RENDERING_INTENTS = {
    # rendering intent integer -> string
    INTENT_PERCEPTUAL: 'perceptual',
    INTENT_RELATIVE_COLORIMETRIC: 'relative',
    INTENT_SATURATION: 'saturation',
    INTENT_ABSOLUTE_COLORIMETRIC: 'absolute',
}


cdef cmsHPROFILE _cms_open_profile(
    object profile,
    cmsContext ctx=NULL,
) except *:
    """Return handle from CMS profile bytes."""
    cdef:
        const uint8_t[::1] src = profile

    if src.shape[0] == 0 or <size_t> src.shape[0] > UINT32_MAX:
        return NULL
    return cmsOpenProfileFromMemTHR(
        ctx, <const void*> &src[0], <cmsUInt32Number> src.shape[0]
    )


cdef cmsContext _cms_verbose_context() except *:
    """Return new CMS context that logs errors."""
    cdef:
        cmsContext ctx = cmsCreateContext(NULL, NULL)

    if ctx == NULL:
        raise CmsError('cmsCreateContext returned NULL')
    cmsSetLogErrorHandlerTHR(ctx, _cms_log_error_handler)
    return ctx


cdef tuple _cms_xyy(XYZ):
    """Return rounded CIE xyY from XYZ."""
    cdef:
        cmsCIEXYZ xyz
        cmsCIExyY xyy

    xyz.X = XYZ[0]
    xyz.Y = XYZ[1]
    xyz.Z = XYZ[2]
    cmsXYZ2xyY(&xyy, &xyz)
    return (round(xyy.x, 6), round(xyy.y, 6), round(xyy.Y, 6))


cdef _cms_info_ascii(cmsHPROFILE hProfile, cmsInfoType info):
    """Return ASCII profile info string or None."""
    cdef:
        cmsUInt32Number size
        bytes buffer

    size = cmsGetProfileInfoASCII(
        hProfile, info, cmsNoLanguage, cmsNoLanguage, NULL, 0
    )
    if size == 0:
        return None
    buffer = PyBytes_FromStringAndSize(NULL, size)
    size = cmsGetProfileInfoASCII(
        hProfile,
        info,
        cmsNoLanguage,
        cmsNoLanguage,
        PyBytes_AsString(buffer),
        size,
    )
    return (
        buffer[:size].decode('ascii', errors='replace').rstrip('\x00').strip()
    ) or None


cdef cmsBool _cms_write_mlu(
    cmsHPROFILE hProfile,
    cmsTagSignature sig,
    const char* text,
) noexcept nogil:
    """Write en-US ASCII text tag to profile."""
    cdef:
        cmsMLU* mlu = cmsMLUalloc(<cmsContext> NULL, 1)
        cmsBool ret

    if mlu == NULL:
        return 0
    ret = (
        cmsMLUsetASCII(mlu, b'en', b'US', text)
        and cmsWriteTag(hProfile, sig, mlu)
    )
    cmsMLUfree(mlu)
    return ret


cdef cmsHPROFILE _cms_adobe_rgb_compatible() noexcept nogil:
    """Return handle to Adobe RGB compatible CMS profile."""
    cdef:
        cmsHPROFILE hProfile = NULL
        cmsCIEXYZTRIPLE color
        cmsCIEXYZ black, d65
        cmsToneCurve* transferfunction = NULL
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

    hProfile = cmsCreateProfilePlaceholder(<cmsContext> NULL)
    if hProfile == NULL:
        return NULL

    cmsSetProfileVersion(hProfile, 2.1)

    ret = (
        _cms_write_mlu(hProfile, cmsSigCopyrightTag, 'Public Domain')
        and _cms_write_mlu(
            hProfile, cmsSigProfileDescriptionTag, 'Adobe RGB (compatible)'
        )
        and _cms_write_mlu(hProfile, cmsSigDeviceMfgDescTag, 'Imagecodecs')
        and _cms_write_mlu(
            hProfile, cmsSigDeviceModelDescTag, 'Adobe RGB (compatible)'
        )
    )

    cmsSetDeviceClass(hProfile, cmsSigDisplayClass)
    cmsSetColorSpace(hProfile, cmsSigRgbData)
    cmsSetPCS(hProfile, cmsSigXYZData)

    ret = (
        ret
        and cmsWriteTag(hProfile, cmsSigMediaWhitePointTag, &d65)
        and cmsWriteTag(hProfile, cmsSigMediaBlackPointTag, &black)
        and cmsWriteTag(
            hProfile, cmsSigRedColorantTag, <void*> &color.Red
        )
        and cmsWriteTag(
            hProfile, cmsSigGreenColorantTag, <void*> &color.Green
        )
        and cmsWriteTag(
            hProfile, cmsSigBlueColorantTag, <void*> &color.Blue
        )
    )
    if not ret:
        cmsCloseProfile(hProfile)
        return NULL

    transferfunction = cmsBuildGamma(<cmsContext> NULL, 2.19921875)
    if transferfunction == NULL:
        cmsCloseProfile(hProfile)
        return NULL
    ret = cmsWriteTag(hProfile, cmsSigRedTRCTag, <void*> transferfunction)
    cmsFreeToneCurve(transferfunction)

    ret = (
        ret
        and cmsLinkTag(hProfile, cmsSigGreenTRCTag, cmsSigRedTRCTag)
        and cmsLinkTag(hProfile, cmsSigBlueTRCTag, cmsSigRedTRCTag)
    )
    if not ret:
        cmsCloseProfile(hProfile)
        return NULL

    return hProfile


cdef cmsHPROFILE _cms_linear_rgb() noexcept nogil:
    """Return handle to linear RGB CMS profile."""
    cdef:
        cmsHPROFILE hProfile = NULL
        cmsCIExyY whitepoint
        cmsCIExyYTRIPLE primaries
        (cmsToneCurve*)[3] transferfunction

    # ITU-R Recommendation 709 Primaries
    primaries.Red.x = 0.6400
    primaries.Red.y = 0.3300
    primaries.Red.Y = 1.0
    primaries.Green.x = 0.3000
    primaries.Green.y = 0.6000
    primaries.Green.Y = 1.0
    primaries.Blue.x = 0.1500
    primaries.Blue.y = 0.0600
    primaries.Blue.Y = 1.0

    # CIE Standard Illuminant D65
    whitepoint.x = 0.3127
    whitepoint.y = 0.3290
    whitepoint.Y = 1.0

    transferfunction[0] = cmsBuildGamma(<cmsContext> NULL, 1.0)
    if transferfunction[0] == NULL:
        return NULL
    transferfunction[1] = transferfunction[2] = transferfunction[0]

    hProfile = cmsCreateRGBProfileTHR(
        <cmsContext> NULL, &whitepoint, &primaries, transferfunction
    )
    cmsFreeToneCurve(transferfunction[0])
    if hProfile == NULL:
        return NULL

    if not _cms_write_mlu(
        hProfile, cmsSigProfileDescriptionTag, 'Linear RGB'
    ):
        cmsCloseProfile(hProfile)
        return NULL

    return hProfile


cdef void _cms_log_error_handler(
    cmsContext ContextID,
    cmsUInt32Number ErrorCode,
    const char* Text,
) noexcept with gil:
    try:
        _log_warning('CMS error: %s', Text.decode().strip())
    except Exception:
        pass
