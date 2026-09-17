# imagecodecs/_htj2k.pyx
# distutils: language = c++
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: freethreading_compatible = True

# Copyright (c) 2025-2026, Christoph Gohlke
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

"""HTJ2K (High-Throughput JPEG 2000) codec for the imagecodecs package."""

include '_shared.pxi'

from libcpp cimport bool as cppbool
from openjph cimport *

set_message_level(OJPH_MSG_NO_MSG)


class HTJ2K:
    """HTJ2K codec constants."""

    available = True

    class TILEPART(enum.IntFlag):
        """HTJ2K tile parts divisions."""

        RESOLUTIONS = 1
        COMPONENTS = 2


class Htj2kError(RuntimeError):
    """HTJ2K codec exceptions."""


def htj2k_init(verbose=None):
    """Initialize HTJ2K codec."""
    cdef:
        OJPH_MSG_LEVEL msglevel = OJPH_MSG_NO_MSG

    if verbose is not None:
        msglevel = min(
            OJPH_MSG_NO_MSG,
            max(OJPH_MSG_ALL_MSG, OJPH_MSG_NO_MSG - int(verbose))
        )
        set_message_level(msglevel)


def htj2k_version():
    """Return OpenJPH library version string."""
    return (
        f'openjph {OPENJPH_VERSION_MAJOR}.{OPENJPH_VERSION_MINOR}.'
        f'{OPENJPH_VERSION_PATCH}'
    )


def htj2k_check(const uint8_t[::1] data, /):
    """Return whether data is HTJ2K encoded or None if unknown."""
    cdef:
        bytes sig = bytes(data[:12])

    # TODO: is there a more specific JPH signature?
    return (
        sig == b'\x00\x00\x00\x0c\x6a\x50\x20\x20\x0d\x0a\x87\x0a'  # JP2
        or sig[:4] == b'\xff\x4f\xff\x51'  # J2K
        # or sig[:4] == b'\x0d\x0a\x87\x0a'  # JP2 ?
    )


def htj2k_encode(
    data,
    /,
    level=None,  # maps to qstep [0.0-1.0] or qfactor (1.0-100.0]
    *,
    rgb=None,  # set color transform for non-planar
    planar=None,
    bitspersample=None,
    tile=None,
    resolutions=None,
    reversible=None,
    tlm=None,  # tile length marker
    tilepart=None,  # 1: resolutions, 2: components, 3: both
    block_size=None,  # (width, height) of code-block
    prog_order=None,  # 'LRCP', 'RLCP', 'RPCL', 'PCRL', 'CPRL'
    profile=None,  # 'IMF' or 'BROADCAST'
    out=None,
):
    """Return HTJ2K encoded image.

    Experimental: float32 input is accepted but its bits are encoded as
    signed 32-bit integers with an NLT type 3 marker. Lossless roundtrip
    requires reversible=True. Lossy encoding is possible but quantization
    errors in bit-pattern space translate to large errors in float values.

    """
    cdef:
        numpy.ndarray src = _array_input(data, out)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        ssize_t srcsize = src.nbytes
        codestream* cs = NULL
        mem_outfile* file = NULL
        size_t out_len
        ssize_t itemsize = src.itemsize
        ui32 tile_w, tile_h, c
        ui32 decompositions = 0 if resolutions is None else resolutions
        float qfactor = 0.0
        float qstep = 0.0
        bint is_signed = src.dtype.kind == 'i'
        bint is_float = src.dtype.kind == 'f'
        bint is_reversible = reversible
        bint tlm_needed = tlm
        bint at_resolutions = False
        bint at_components = False
        bint color_transform = False
        char* prog_order_str = NULL
        char* profile_str = NULL
        ui32 block_w = 0
        ui32 block_h = 0
        ui32 min_dim = 0
        ui32 max_decomp = 0
        int ret
        imagelayout_t layout

    # htj2k-specific planar auto-detection:
    # if planar is not explicitly set and the array looks like (S, H, W)
    # (i.e. shape[2] > 4 and shape[0] <= 4), treat as planar
    if planar is None and src.ndim == 3:
        if src.shape[2] > 4 and src.shape[0] <= 4:
            planar = True

    _image_layout(
        IC_UINT
        | IC_SINT
        | IC_FLOAT  # limited support
        | IC_SZ1
        | IC_SZ2
        | IC_SZ4
        | IC_GRAY
        | IC_RGB
        | IC_ALPHA
        | IC_EXTRA
        | IC_PLANAR
        | IC_BPS,
        src.ndim,
        src.shape,
        src.dtype,
        None,  # photometric
        bitspersample,
        planar,
        None,  # frames
        None,  # volumetric
        None,  # extrasample
        &layout,
    )

    if is_float:
        if itemsize != 4:
            raise ValueError('only float32 is supported')
        is_signed = True  # float32 bits are encoded as signed 32-bit integers

    if tile is None:
        tile_w = 0
        tile_h = 0
        if srcsize > UINT32_MAX:
            raise ValueError('single tile size must not exceed 4 GB')
    else:
        tile_w, tile_h = tile

    if tilepart is not None:
        tilepart = _enum_value(tilepart, HTJ2K.TILEPART)
        at_resolutions = tilepart & 1  # HTJ2K.TILEPART.RESOLUTIONS
        at_components = tilepart & 2  # HTJ2K.TILEPART.COMPONENTS

    if level is not None:
        if level <= 1.0:
            qstep = _default_value(level, 0.0, 0.0, 1.0)
        else:
            qfactor = _default_value(level, 0.0, 1.0, 100.0)

    if qstep < 0.00001:
        qstep = 0.0
    if reversible is None:
        is_reversible = qstep == 0.0 and qfactor == 0.0
    if is_float and not is_reversible and qstep <= 0.0:
        # minimum step for irreversible float32
        qstep = 1.0 / 16384.0

    if layout.samples > 1 and not layout.planar and not is_float:
        if (
            rgb or
            (rgb is None and (layout.samples == 3 or layout.samples == 4))
        ):
            color_transform = True

    if profile is not None:
        profile = profile.upper().encode()
        profile_str = profile

    if prog_order is not None:
        prog_order = prog_order.upper().encode()
        prog_order_str = prog_order

    if block_size is not None:
        block_w, block_h = block_size

    try:
        with nogil:
            file = new mem_outfile()
            file.open()

            cs = new codestream()
            cs.set_planar(layout.planar)
            cs.set_tilepart_divisions(at_resolutions, at_components)
            cs.request_tlm_marker(tlm_needed)
            if profile_str != NULL:
                cs.set_profile(profile_str)
            cs.access_siz().set_image_extent(
                point(<ui32> layout.width, <ui32> layout.height)
            )
            if tile_w != 0 and tile_h != 0:
                cs.access_siz().set_tile_size(size(tile_w, tile_h))
            cs.access_siz().set_num_components(<ui32> layout.samples)
            for c in range(<ui32> layout.samples):
                cs.access_siz().set_component(
                    c, point(1, 1), <ui32> layout.bitspersample, is_signed
                )
            cs.access_cod().set_reversible(is_reversible)
            if decompositions > 0:
                min_dim = (
                    <ui32> layout.width
                    if layout.width < layout.height
                    else <ui32> layout.height
                )
                if tile_w != 0 and tile_h != 0:
                    if tile_w < min_dim:
                        min_dim = tile_w
                    if tile_h < min_dim:
                        min_dim = tile_h
                max_decomp = 0
                while min_dim > 1:
                    min_dim >>= 1
                    max_decomp += 1
                if decompositions > max_decomp:
                    decompositions = max_decomp
                cs.access_cod().set_num_decomposition(decompositions)
            if block_w != 0 and block_h != 0:
                cs.access_cod().set_block_dims(block_w, block_h)
            if color_transform:
                cs.access_cod().set_color_transform(True)
            if prog_order_str != NULL:
                cs.access_cod().set_progression_order(prog_order_str)
            if not is_reversible:
                if qfactor > 0.0:
                    cs.access_qcd().set_qfactor(qfactor)
                elif qstep > 0.0:
                    cs.access_qcd().set_irrev_quant(qstep)
            if is_float:
                cs.access_nlt().set_nonlinear_transform(
                    <ui32> 65535,  # ALL_COMPS
                    <ui8> 3  # OJPH_NLT_BINARY_COMPLEMENT_NLT
                )

            cs.write_headers(file, NULL, 0)

            if itemsize == 1:
                if is_signed:
                    ret = _copy_to_codestream(
                        <int8_t*> src.data,
                        cs,
                        layout.height,
                        layout.width,
                        layout.samples,
                        layout.planar,
                    )
                else:
                    ret = _copy_to_codestream(
                        <uint8_t*> src.data,
                        cs,
                        layout.height,
                        layout.width,
                        layout.samples,
                        layout.planar,
                    )
            elif itemsize == 2:
                if is_signed:
                    ret = _copy_to_codestream(
                        <int16_t*> src.data,
                        cs,
                        layout.height,
                        layout.width,
                        layout.samples,
                        layout.planar,
                    )
                else:
                    ret = _copy_to_codestream(
                        <uint16_t*> src.data,
                        cs,
                        layout.height,
                        layout.width,
                        layout.samples,
                        layout.planar,
                    )
            elif is_float:
                ret = _copy_to_codestream(
                    <float*> src.data,
                    cs,
                    layout.height,
                    layout.width,
                    layout.samples,
                    layout.planar,
                )
            elif is_signed:
                ret = _copy_to_codestream(
                    <int32_t*> src.data,
                    cs,
                    layout.height,
                    layout.width,
                    layout.samples,
                    layout.planar,
                )
            else:
                ret = _copy_to_codestream(
                    <uint32_t*> src.data,
                    cs,
                    layout.height,
                    layout.width,
                    layout.samples,
                    layout.planar,
                )
            cs.flush()
            cs.close()

            out_len = file.get_used_size()

        out, dstsize, outgiven, outtype = _parse_output(out)
        if out is None:
            out = _create_output(outtype, out_len)
        dst = out
        dstsize = dst.shape[0]
        if dstsize < <ssize_t> out_len:
            raise ValueError('output too small')
        memcpy(<void*> dst._data, <const void*> file.get_data(), out_len)

        file.close()

    except Htj2kError:
        raise
    except RuntimeError as exc:
        raise Htj2kError(str(exc)) from exc

    finally:
        if cs != NULL:
            del cs
        if file != NULL:
            del file

    del dst
    return _return_output(out, dstsize, out_len, outgiven)


def htj2k_decode(
    data,
    /,
    *,
    planar=None,
    skipres=None,
    resilient=False,
    out=None,
):
    """Return decoded HTJ2K image.

    Images are returned as float32 if they were encoded with a nonlinear
    transform. Otherwise, the dtype is determined by the bit depth and
    signedness of the encoded image.

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t srcsize = src.shape[0]
        codestream* cs = NULL
        mem_infile* file = NULL
        ui32 skipped_res_for_data = 0
        ui32 skipped_res_for_recon = 0
        ui32 bit_depth, c
        ssize_t height, width, samples, itemsize
        bint is_rgb, is_signed, is_planar, to_planar
        bint is_resilient = resilient
        bint is_float = False
        cppbool signed_nlt = 0
        ui8 nl_type = 0
        ui8 bd_nlt = 0
        int ret

    if data is out:
        raise ValueError('cannot transform in-place')

    if skipres is not None:
        try:
            skipped_res_for_data, skipped_res_for_recon = skipres
        except Exception:
            skipped_res_for_data = skipped_res_for_recon = skipres

    try:
        with nogil:
            file = new mem_infile()
            file.open(<ui8*> src._data, <size_t> srcsize)

            cs = new codestream()
            if is_resilient:
                cs.enable_resilience()
            cs.read_headers(file)
            cs.restrict_input_resolution(
                skipped_res_for_data, skipped_res_for_recon
            )
            cs.create()

            # height = cs.access_siz().get_recon_height().y
            # width = cs.access_siz().get_image_extent().x
            height = cs.access_siz().get_recon_height(0)
            width = cs.access_siz().get_recon_width(0)
            samples = cs.access_siz().get_num_components()
            bit_depth = cs.access_siz().get_bit_depth(0)
            is_signed = cs.access_siz().is_signed(0)
            is_rgb = cs.access_cod().is_using_color_transform()
            is_planar = cs.is_planar()
            if cs.access_nlt().get_nonlinear_transform(
                0, bd_nlt, signed_nlt, nl_type
            ):
                is_float = nl_type == <ui8> 3  # OJPH_NLT_BINARY_COMPLEMENT_NLT

            # TODO: is this necessary/correct?
            # TODO: find dtype compatible with all components?
            for c in range(<ui32> samples):
                if (
                    bit_depth != cs.access_siz().get_bit_depth(c)
                    or is_signed != <bint> cs.access_siz().is_signed(c)
                    or cs.access_siz().get_downsampling(c).y != 1
                    or cs.access_siz().get_downsampling(c).x != 1
                ):
                    raise ValueError(
                        f'{bit_depth=} != {cs.access_siz().get_bit_depth(c)}, '
                        f'{is_signed=} != {cs.access_siz().is_signed(c)}, '
                        'or downsampling=('
                        f'{cs.access_siz().get_downsampling(c).y}, '
                        f'{cs.access_siz().get_downsampling(c).x}) != (1, 1)'
                        f'for component {c}'
                    )

        if planar is None:
            to_planar = False if is_rgb else is_planar
        else:
            to_planar = planar

        if samples == 1:
            shape = int(height), int(width)
        elif to_planar:
            shape = int(samples), int(height), int(width)
        else:
            shape = int(height), int(width), int(samples)

        if bit_depth <= 0 or bit_depth > 32:
            raise ValueError(f'{bit_depth=} not supported')
        if bit_depth <= 8:
            itemsize = 1
        elif bit_depth <= 16:
            itemsize = 2
        else:
            itemsize = 4
        if is_float:
            dtype = f'f{itemsize}'
        elif is_signed:
            dtype = f'i{itemsize}'
        else:
            dtype = f'u{itemsize}'

        out = _create_array(out, shape, dtype, None, True)
        dst = out

        with nogil:
            if itemsize == 1:
                if is_signed:
                    ret = _copy_from_codestream(
                        <int8_t*> dst.data,
                        cs,
                        height,
                        width,
                        samples,
                        is_planar,
                        to_planar,
                    )
                else:
                    ret = _copy_from_codestream(
                        <uint8_t*> dst.data,
                        cs,
                        height,
                        width,
                        samples,
                        is_planar,
                        to_planar,
                    )
            elif itemsize == 2:
                if is_signed:
                    ret = _copy_from_codestream(
                        <int16_t*> dst.data,
                        cs,
                        height,
                        width,
                        samples,
                        is_planar,
                        to_planar,
                    )
                else:
                    ret = _copy_from_codestream(
                        <uint16_t*> dst.data,
                        cs,
                        height,
                        width,
                        samples,
                        is_planar,
                        to_planar,
                    )
            elif is_float:
                ret = _copy_from_codestream(
                    <float*> dst.data,
                    cs,
                    height,
                    width,
                    samples,
                    is_planar,
                    to_planar,
                )
            elif is_signed:
                ret = _copy_from_codestream(
                    <int32_t*> dst.data,
                    cs,
                    height,
                    width,
                    samples,
                    is_planar,
                    to_planar,
                )
            else:
                ret = _copy_from_codestream(
                    <uint32_t*> dst.data,
                    cs,
                    height,
                    width,
                    samples,
                    is_planar,
                    to_planar,
                )
            cs.close()

    except Htj2kError:
        raise
    except RuntimeError as exc:
        raise Htj2kError(str(exc)) from exc

    finally:
        if cs != NULL:
            del cs
        if file != NULL:
            del file

    return out


ctypedef fused data_t:
    int8_t
    uint8_t
    int16_t
    uint16_t
    int32_t
    uint32_t
    float


cdef int _copy_to_codestream(
    const data_t* src,
    codestream* cs,
    const ssize_t height,
    const ssize_t width,
    const ssize_t samples,
    const bint is_planar,
) except -1 nogil:
    """Copy buffer to ojph codestream."""
    cdef:
        line_buf* line = NULL
        ssize_t j
        ui32 c = 0

    line = cs.exchange(NULL, c)
    if line.i32 == NULL or line.flags != LFT_INTEGER | LFT_32BIT:
        raise Htj2kError('unexpected line buffer')

    if is_planar or samples == 1:
        for c in range(<ui32> samples):
            for _i in range(height):
                for j in range(_min(width, line.size)):
                    if data_t is float:
                        memcpy(&line.i32[j], &src[j], 4)
                    else:
                        line.i32[j] = <si32> src[j]
                line = cs.exchange(line, c)
                src += width
    else:
        for _i in range(height):
            for c in range(<ui32> samples):
                for j in range(_min(width, line.size)):
                    if data_t is float:
                        memcpy(&line.i32[j], &src[j * samples + c], 4)
                    else:
                        line.i32[j] = <si32> src[j * samples + c]
                line = cs.exchange(line, c)
            src += width * samples

    return 0


cdef int _copy_from_codestream(
    data_t* dst,
    codestream* cs,
    const ssize_t height,
    const ssize_t width,
    const ssize_t samples,
    const bint is_planar,
    const bint to_planar,
) except -1 nogil:
    """Copy ojph codestream to buffer."""
    cdef:
        line_buf* line = NULL
        ssize_t i, j, stride, row_base
        ui32 c = 0

    if samples == 1:
        for _i in range(height):
            line = cs.pull(c)
            # if not line.flags & flags:  # or c >= samples
            #     return -1
            for j in range(_min(width, line.size)):
                if data_t is float:
                    memcpy(&dst[j], &line.i32[j], 4)
                else:
                    dst[j] = <data_t> line.i32[j]
            dst += width

    elif is_planar and to_planar:
        for c in range(<ui32> samples):
            for _i in range(height):
                line = cs.pull(c)
                for j in range(_min(width, line.size)):
                    if data_t is float:
                        memcpy(&dst[j], &line.i32[j], 4)
                    else:
                        dst[j] = <data_t> line.i32[j]
                dst += width

    elif is_planar and not to_planar:
        stride = width * samples
        for c in range(<ui32> samples):
            for i in range(height):
                line = cs.pull(c)
                row_base = i * stride + c
                for j in range(_min(width, line.size)):
                    if data_t is float:
                        memcpy(&dst[row_base + j * samples], &line.i32[j], 4)
                    else:
                        dst[row_base + j * samples] = <data_t> line.i32[j]

    elif not is_planar and not to_planar:
        stride = width * samples
        for _i in range(height):
            for c in range(<ui32> samples):
                line = cs.pull(c)
                for j in range(_min(width, line.size)):
                    if data_t is float:
                        memcpy(&dst[j * samples + c], &line.i32[j], 4)
                    else:
                        dst[j * samples + c] = <data_t> line.i32[j]
            dst += stride

    elif not is_planar and to_planar:
        stride = height * width
        for i in range(height):
            for c in range(<ui32> samples):
                line = cs.pull(c)
                row_base = c * stride + i * width
                for j in range(_min(width, line.size)):
                    if data_t is float:
                        memcpy(&dst[row_base + j], &line.i32[j], 4)
                    else:
                        dst[row_base + j] = <data_t> line.i32[j]

    return 0


cdef inline ssize_t _min(ssize_t a, size_t b) noexcept nogil:
    return a if <ssize_t> b >= a else <ssize_t> b
