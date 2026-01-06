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
        or sig[:4] == b'\x0d\x0a\x87\x0a'  # JP2
    )


def htj2k_encode(
    data,
    /,
    level=None,  # quantization_step
    *,
    rgb=None,  # set color transform for non-planar
    planar=None,
    tile=None,
    resolutions=None,
    reversible=None,
    tlm=None,  # tile length marker
    tilepart=None,  # 1: resolutions, 2: components, 3: both
    out=None,
):
    """Return HTJ2K encoded image."""
    cdef:
        numpy.ndarray src = numpy.ascontiguousarray(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        ssize_t srcsize = src.nbytes
        codestream* cs = NULL
        mem_outfile* file = NULL
        size_t out_len
        ssize_t itemsize = src.itemsize
        ui32 bitdepth = src.itemsize * 8
        ui32 height, width, samples, tile_w, tile_h, c
        ui32 decompositions = 0 if resolutions is None else resolutions
        float quantization_step = _default_value(level, 0.0, 0.0, 1.0)
        bint is_signed = src.dtype.kind == 'i'
        bint is_reversible = reversible
        bint tlm_needed = tlm
        bint at_resolutions = False
        bint at_components = False
        bint color_transform = False
        bint is_planar = False
        char* profile = NULL  # *IMF* and BROADCAST
        char* prog_order = NULL  # LRCP, RLCP, *RPCL*, PCRL, CPRL
        int ret

    if not (src.dtype.char in 'bBhHiIlL' and src.ndim in {2, 3}):
        raise ValueError(f'invalid data ndim={src.ndim} or dtype={src.dtype}')

    height = <ui32> src.shape[0]
    width = <ui32> src.shape[1]
    samples = 1 if src.ndim == 2 else <ui32> src.shape[2]

    if tile is None:
        tile_w = 0
        tile_h = 0
        if srcsize > UINT32_MAX:
            raise ValueError('single tile size must not exceed 4 GB')
    else:
        tile_w, tile_h = tile

    if tilepart is not None:
        at_resolutions = tilepart & 1  # HTJ2K.TILEPART.RESOLUTIONS
        at_components = tilepart & 2  # HTJ2K.TILEPART.COMPONENTS

    if quantization_step < 0.00001:
        quantization_step = 0.0
    if reversible is None:
        is_reversible = quantization_step <= 0.0

    if samples > 1:
        if planar or (planar is None and samples > 4 and height <= 4):
            # separate bands
            is_planar = True
            samples = <ui32> src.shape[0]
            height = <ui32> src.shape[1]
            width = <ui32> src.shape[2]
        elif rgb or (rgb is None and (samples == 3 or samples == 4)):
            color_transform = True

    try:
        with nogil:
            file = new mem_outfile()
            file.open()

            cs = new codestream()
            cs.set_planar(is_planar)
            cs.set_tilepart_divisions(at_resolutions, at_components)
            cs.request_tlm_marker(tlm_needed)
            if profile != NULL:
                cs.set_profile(profile)
            cs.access_siz().set_image_extent(point(width, height))
            if tile_w != 0 and tile_h != 0:
                cs.access_siz().set_tile_size(size(tile_w, tile_h))
            cs.access_siz().set_num_components(<ui32> samples)
            for c in range(samples):
                cs.access_siz().set_component(
                    c, point(1, 1), bitdepth, is_signed
                )
            cs.access_cod().set_reversible(is_reversible)
            if decompositions > 0:
                cs.access_cod().set_num_decomposition(decompositions)
            if color_transform:
                cs.access_cod().set_color_transform(True)
            if prog_order != NULL:
                cs.access_cod().set_progression_order(prog_order)
            if not is_reversible and quantization_step > 0.0:
                cs.access_qcd().set_irrev_quant(quantization_step)

            cs.write_headers(file, NULL, 0)

            if itemsize == 1:
                if is_signed:
                    ret = _copy_to_codestream(
                        <int8_t*> src.data,
                        cs,
                        height,
                        width,
                        samples,
                        is_planar,
                    )
                else:
                    ret = _copy_to_codestream(
                        <uint8_t*> src.data,
                        cs,
                        height,
                        width,
                        samples,
                        is_planar,
                    )
            elif itemsize == 2:
                if is_signed:
                    ret = _copy_to_codestream(
                        <int16_t*> src.data,
                        cs,
                        height,
                        width,
                        samples,
                        is_planar,
                    )
                else:
                    ret = _copy_to_codestream(
                        <uint16_t*> src.data,
                        cs,
                        height,
                        width,
                        samples,
                        is_planar,
                    )
            else:
                if is_signed:
                    ret = _copy_to_codestream(
                        <int32_t*> src.data,
                        cs,
                        height,
                        width,
                        samples,
                        is_planar,
                    )
                else:
                    ret = _copy_to_codestream(
                        <uint32_t*> src.data,
                        cs,
                        height,
                        width,
                        samples,
                        is_planar,
                    )
            if ret != 0:
                raise Htj2kError(f'_copy_to_codestream() returned {ret}')

            cs.flush()
            cs.close()

            out_len = file.get_used_size()

        out, dstsize, outgiven, outtype = _parse_output(out)
        if out is None:
            out = _create_output(outtype, out_len)
        dst = out
        dstsize = dst.nbytes
        if dstsize < <ssize_t> out_len:
            raise ValueError('output too small')
        memcpy(<void*> &dst[0], <const void*> file.get_data(), out_len)

        file.close()

    except RuntimeError as exc:
        # TODO: improve error handling
        raise Htj2kError('OpenJPH error') from exc

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
    """Return decoded HTJ2K image."""
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t srcsize = src.size
        codestream* cs = NULL
        mem_infile* file = NULL
        ui32 skipped_res_for_data = 0
        ui32 skipped_res_for_recon = 0
        ui32 bit_depth, c
        ssize_t height, width, samples, itemsize
        bint is_rgb, is_signed, is_planar, to_planar
        bint is_resilient = resilient
        int ret

    if skipres is not None:
        try:
            skipped_res_for_data, skipped_res_for_recon = skipres
        except Exception:
            skipped_res_for_data = skipped_res_for_recon = skipres

    try:
        with nogil:
            file = new mem_infile()
            file.open(<ui8*> &src[0], <size_t> srcsize)

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

            # TODO: is this necessary/correct?
            # TODO: find dtype compatible with all components?
            for c in range(samples):
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
        dtype = f'i{itemsize}' if is_signed else f'u{itemsize}'

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
            else:
                if is_signed:
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
            if ret != 0:
                raise Htj2kError(f'_copy_from_codestream returned {ret}')
            cs.close()

    except RuntimeError as exc:
        # TODO: improve error handling
        raise Htj2kError('OpenJPH error') from exc

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


cdef int _copy_to_codestream(
    const data_t* src,
    codestream* cs,
    const ssize_t height,
    const ssize_t width,
    const ssize_t samples,
    const bint is_planar,
) noexcept nogil:
    """Copy buffer to ojph codestream."""
    cdef:
        line_buf* line = NULL
        ssize_t i, j
        ui32 c = 0

    line = cs.exchange(NULL, c)
    if line.i32 == NULL or line.flags != LFT_INTEGER | LFT_32BIT:
        return -1

    if is_planar or samples == 1:
        for c in range(samples):
            for i in range(height):
                for j in range(_min(width, line.size)):
                    line.i32[j] = <si32> src[j]
                line = cs.exchange(line, c)
                src += width
    else:
        for i in range(height):
            for c in range(samples):
                for j in range(_min(width, line.size)):
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
) noexcept nogil:
    """Copy ojph codestream to buffer."""
    cdef:
        line_buf* line = NULL
        ssize_t i, j, stride
        ui32 c = 0

    if samples == 1:
        for i in range(height):
            line = cs.pull(c)
            # if not line.flags & flags:  # or c >= samples
            #     return -1
            for j in range(_min(width, line.size)):
                dst[j] = <data_t> line.i32[j]
            dst += width

    elif is_planar and to_planar:
        for c in range(samples):
            for i in range(height):
                line = cs.pull(c)
                for j in range(_min(width, line.size)):
                    dst[j] = <data_t> line.i32[j]
                dst += width

    elif is_planar and not to_planar:
        stride = width * samples
        for c in range(samples):
            for i in range(height):
                line = cs.pull(c)
                for j in range(_min(width, line.size)):
                    dst[i * stride + j * samples + c] = <data_t> line.i32[j]

    elif not is_planar and not to_planar:
        stride = width * samples
        for i in range(height):
            for c in range(samples):
                line = cs.pull(c)
                for j in range(_min(width, line.size)):
                    dst[j * samples + c] = <data_t> line.i32[j]
            dst += stride

    elif not is_planar and to_planar:
        stride = height * width
        for i in range(height):
            for c in range(samples):
                line = cs.pull(c)
                for j in range(_min(width, line.size)):
                    dst[c * stride + i * width + j] = <data_t> line.i32[j]

    return 0


cdef inline ssize_t _min(ssize_t a, size_t b) noexcept nogil:
    return a if <ssize_t> b >= a else <ssize_t> b
