# imagecodecs/tests/test_image_layout.py

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

"""Unittests for the imagecodecs._shared.image_layout function."""

import numpy
import pytest

from imagecodecs._shared import (
    IC,
    ExtraSample,
    ImageLayout,
    Photometric,
    SampleFormat,
    image_layout,
)


def test_image_layout_0d():
    """Test that 0-dimensional array raises."""
    with pytest.raises(ValueError, match='0-dimensional'):
        image_layout((), numpy.uint8)


def test_image_layout_1d():
    """Test 1D array -> single scanline, gray."""
    r = image_layout((100,), numpy.uint8)
    assert isinstance(r, ImageLayout)
    assert r.height == 0
    assert r.width == 100
    assert r.samples == 1
    assert r.photometric == Photometric.GRAY
    assert r.frames == 1
    assert r.depth == 1


def test_image_layout_1d_photo_hint():
    """Test 1D array with photometric hint: gray ok, multi-sample rejects."""
    r = image_layout((100,), numpy.uint8, photometric='gray')
    assert r.photometric == Photometric.GRAY
    with pytest.raises(ValueError, match=r'samples.*less than'):
        image_layout((100,), numpy.uint8, photometric='rgb')


def test_image_layout_2d_gray():
    """Test 2D array -> height x width, gray."""
    r = image_layout((64, 128), numpy.uint8)
    assert r.height == 64
    assert r.width == 128
    assert r.samples == 1
    assert r.photometric == Photometric.GRAY
    assert r.frames == 1
    assert r.depth == 1


def test_image_layout_3d_rgb():
    """Test (H, W, 3) -> RGB."""
    r = image_layout((48, 64, 3), numpy.uint8)
    assert r.height == 48
    assert r.width == 64
    assert r.samples == 3
    assert r.photometric == Photometric.RGB
    assert r.extracount == 0


def test_image_layout_3d_rgb_autodetect():
    """Test RGB auto-detect for allowed dtypes and sample counts."""
    # uint8 (H,W,3) -> RGB
    r = image_layout((48, 64, 3), dtype=numpy.uint8)
    assert r.photometric == Photometric.RGB

    # uint16 (H,W,4) -> RGBA
    r = image_layout((48, 64, 4), dtype=numpy.uint16)
    assert r.photometric == Photometric.RGB

    # float32 (H,W,3) -> RGB
    r = image_layout((48, 64, 3), dtype=numpy.float32)
    assert r.photometric == Photometric.RGB

    # float64 (H,W,4) -> RGBA
    r = image_layout((48, 64, 4), dtype=numpy.float64)
    assert r.photometric == Photometric.RGB

    # float16 (H,W,3) -> RGB
    r = image_layout((48, 64, 3), dtype=numpy.float16)
    assert r.photometric == Photometric.RGB

    # >4 samples with extrasample -> RGB
    r = image_layout(
        (48, 64, 5), dtype=numpy.uint8, extrasample='unassociated'
    )
    assert r.photometric == Photometric.RGB
    assert r.extracount == 2

    # >4 samples WITHOUT extrasample -> gray (ambiguous)
    r = image_layout(
        (2, 48, 64, 5),
        dtype=numpy.uint8,
        caps=IC.UINT | IC.SZ1 | IC.GRAY | IC.EXTRA | IC.FRAMES,
    )
    assert r.photometric == Photometric.GRAY
    assert r.samples == 5


def test_image_layout_3d_no_rgb_autodetect():
    """Test dtypes that should NOT auto-detect as RGB."""
    # int16 (H,W,3) -> gray (sint excluded)
    r = image_layout(
        (48, 64, 3),
        dtype=numpy.int16,
        caps=IC.SINT | IC.SZ2 | IC.GRAY | IC.EXTRA | IC.FRAMES,
    )
    assert r.photometric == Photometric.GRAY
    assert r.samples == 3

    # uint32 (H,W,3) -> gray (uint itemsize > 2)
    r = image_layout(
        (48, 64, 3),
        dtype=numpy.uint32,
        caps=IC.UINT | IC.SZ4 | IC.GRAY | IC.EXTRA | IC.FRAMES,
    )
    assert r.photometric == Photometric.GRAY
    assert r.samples == 3

    # complex64 (H,W,3) -> gray
    r = image_layout(
        (48, 64, 3),
        dtype=numpy.complex64,
        caps=IC.COMPLEX | IC.SZ8 | IC.GRAY | IC.EXTRA | IC.FRAMES,
    )
    assert r.photometric == Photometric.GRAY
    assert r.samples == 3

    # bool (H,W,3) -> gray
    r = image_layout(
        (48, 64, 3),
        dtype=numpy.bool_,
        caps=IC.BOOL | IC.SZ1 | IC.GRAY | IC.EXTRA | IC.FRAMES,
    )
    assert r.photometric == Photometric.GRAY
    assert r.samples == 3


def test_image_layout_3d_rgba():
    """Test (H, W, 4) -> RGB + 1 extra."""
    r = image_layout((48, 64, 4), numpy.uint8)
    assert r.photometric == Photometric.RGB
    assert r.samples == 4
    assert r.extracount == 1
    assert r.extratype == ExtraSample.UNASSALPHA
    assert r.extraindex == 3


def test_image_layout_3d_gray_alpha():
    """Test (H, W, 2) -> gray + alpha, not mistaken for frames."""
    r = image_layout((48, 64, 2), numpy.uint8)
    assert r.photometric == Photometric.GRAY
    assert r.samples == 2
    assert r.extracount == 1
    assert r.extratype == ExtraSample.UNASSALPHA
    assert r.extraindex == 1
    assert r.frames == 1


def test_image_layout_3d_gray_single():
    """Test (H, W, 1) -> gray, no extra."""
    r = image_layout((48, 64, 1), numpy.uint8)
    assert r.photometric == Photometric.GRAY
    assert r.samples == 1
    assert r.extracount == 0
    assert r.frames == 1


def test_image_layout_3d_frames_hint():
    """Test (N, H, W) with frames=True -> N frames, gray."""
    r = image_layout((10, 48, 64), numpy.uint8, frames=True)
    assert r.frames == 10
    assert r.height == 48
    assert r.width == 64
    assert r.samples == 1
    assert r.photometric == Photometric.GRAY


def test_image_layout_3d_gray_photo_hint():
    """Test (H, W, 3) with photometric=GRAY and extrasample -> gray+extras."""
    r = image_layout(
        (48, 64, 3),
        numpy.uint8,
        photometric='gray',
        extrasample='unassociated',
    )
    assert r.photometric == Photometric.GRAY
    assert r.samples == 3
    assert r.extracount == 2


def test_image_layout_3d_ambiguous_no_frames_cap():
    """Test 3D ambiguous with codec that lacks IC.FRAMES -> channels."""
    caps = IC.GRAY | IC.RGB | IC.UINT | IC.SZ1 | IC.ALPHA | IC.EXTRA
    r = image_layout((10, 48, 64), numpy.uint8, caps=int(caps))
    assert r.samples == 64
    assert r.frames == 1


def test_image_layout_3d_ambiguous_heuristic_small():
    """Test 3D ambiguous heuristic: trailing dim <= 4 -> channels."""
    r = image_layout((10, 48, 3), numpy.uint8)
    assert r.samples == 3
    assert r.photometric == Photometric.RGB
    assert r.frames == 1


def test_image_layout_3d_ambiguous_heuristic_large():
    """Test 3D ambiguous heuristic: trailing dim > 4 -> no channels."""
    r = image_layout((10, 48, 64), numpy.uint8)
    assert r.samples == 1
    assert r.frames == 10
    assert r.photometric == Photometric.GRAY


def test_image_layout_4d_frames_rgb():
    """Test (N, H, W, 3) -> N frames of RGB."""
    r = image_layout((5, 48, 64, 3), numpy.uint8)
    assert r.frames == 5
    assert r.height == 48
    assert r.width == 64
    assert r.samples == 3
    assert r.photometric == Photometric.RGB


def test_image_layout_4d_volumetric_rgb():
    """Test (D, H, W, 3) with volumetric=True."""
    r = image_layout((8, 48, 64, 3), numpy.uint8, volumetric=True)
    assert r.depth == 8
    assert r.height == 48
    assert r.width == 64
    assert r.samples == 3
    assert r.photometric == Photometric.RGB
    assert r.frames == 1


def test_image_layout_4d_volumetric_gray():
    """Test (D, H, W) with volumetric=True."""
    r = image_layout((8, 48, 64), numpy.uint8, volumetric=True)
    assert r.depth == 8
    assert r.height == 48
    assert r.width == 64
    assert r.samples == 1
    assert r.photometric == Photometric.GRAY
    assert r.frames == 1


def test_image_layout_5d_frames_volumetric_rgb():
    """Test (N, D, H, W, S) with volumetric=True."""
    r = image_layout((2, 4, 48, 64, 3), numpy.uint8, volumetric=True)
    assert r.frames == 2
    assert r.depth == 4
    assert r.samples == 3
    assert r.photometric == Photometric.RGB


def test_image_layout_planar_3d():
    """Test (S, H, W) with planar=True -> planar RGB."""
    r = image_layout((3, 48, 64), numpy.uint8, planar=True)
    assert r.planar is True
    assert r.samples == 3
    assert r.height == 48
    assert r.width == 64
    assert r.photometric == Photometric.RGB
    assert r.frames == 1


def test_image_layout_planar_4d_frames():
    """Test (N, S, H, W) with planar=True -> N frames of planar."""
    r = image_layout((5, 3, 48, 64), numpy.uint8, planar=True)
    assert r.planar is True
    assert r.frames == 5
    assert r.samples == 3


def test_image_layout_planar_volumetric():
    """Test (S, D, H, W) with planar=True and volumetric=True."""
    r = image_layout((3, 8, 48, 64), numpy.uint8, planar=True, volumetric=True)
    assert r.planar is True
    assert r.depth == 8
    assert r.samples == 3
    assert r.photometric == Photometric.RGB


def test_image_layout_planar_ndim_too_small():
    """Test planar with ndim < 3 raises."""
    with pytest.raises(ValueError, match='planar requires'):
        image_layout((48, 64), numpy.uint8, planar=True)


def test_image_layout_volumetric_ndim_too_small():
    """Test volumetric with ndim < 3 raises."""
    with pytest.raises(ValueError, match='volumetric requires'):
        image_layout((48, 64), numpy.uint8, volumetric=True)


@pytest.mark.parametrize(
    ('dtype', 'expected_sf'),
    [
        (numpy.uint8, SampleFormat.UINT),
        (numpy.uint16, SampleFormat.UINT),
        (numpy.uint32, SampleFormat.UINT),
        (numpy.int8, SampleFormat.SINT),
        (numpy.int16, SampleFormat.SINT),
        (numpy.int32, SampleFormat.SINT),
        (numpy.float32, SampleFormat.FLOAT),
        (numpy.float64, SampleFormat.FLOAT),
        (numpy.complex64, SampleFormat.COMPLEX),
        (numpy.complex128, SampleFormat.COMPLEX),
        (numpy.bool_, SampleFormat.BOOL),
    ],
)
def test_image_layout_sampleformat(dtype, expected_sf):
    """Test sampleformat is correctly inferred from dtype."""
    r = image_layout((64, 64), dtype=dtype)
    assert r.sampleformat == expected_sf


@pytest.mark.parametrize(
    ('dtype', 'expected_isz'),
    [
        (numpy.uint8, 1),
        (numpy.uint16, 2),
        (numpy.float32, 4),
        (numpy.float64, 8),
        (numpy.complex128, 16),
    ],
)
def test_image_layout_itemsize(dtype, expected_isz):
    """Test itemsize is correctly inferred from dtype."""
    r = image_layout((64, 64), dtype=dtype)
    assert r.itemsize == expected_isz
    assert r.bitspersample == expected_isz * 8


def test_image_layout_bool_dtype():
    """Test bool dtype produces SF_BOOL with itemsize 1."""
    r = image_layout((64, 64), dtype=numpy.bool_)
    assert r.sampleformat == SampleFormat.BOOL
    assert r.itemsize == 1
    assert r.bitspersample == 8


def test_image_layout_bps_default():
    """Test bitspersample defaults to itemsize * 8."""
    r = image_layout((64, 64), dtype=numpy.uint16)
    assert r.bitspersample == 16


def test_image_layout_bps_custom_with_cap():
    """Test custom bitspersample accepted when IC.BPS in caps."""
    caps = IC.GRAY | IC.UINT | IC.SZ2 | IC.BPS
    r = image_layout(
        (64, 64), dtype=numpy.uint16, bitspersample=12, caps=int(caps)
    )
    assert r.bitspersample == 12


def test_image_layout_bps_custom_no_cap():
    """Test custom bitspersample rejected when IC.BPS not in caps."""
    caps = IC.GRAY | IC.UINT | IC.SZ2
    with pytest.raises(ValueError, match=r'bitspersample.*custom'):
        image_layout(
            (64, 64), dtype=numpy.uint16, bitspersample=12, caps=int(caps)
        )


def test_image_layout_bps_same_as_native():
    """Test bitspersample hint matching native passes without IC.BPS."""
    caps = IC.GRAY | IC.UINT | IC.SZ2
    r = image_layout(
        (64, 64), dtype=numpy.uint16, bitspersample=16, caps=int(caps)
    )
    assert r.bitspersample == 16


def test_image_layout_bps_no_caps():
    """Test custom bitspersample with caps=0 (no validation)."""
    r = image_layout((64, 64), dtype=numpy.uint16, bitspersample=12)
    assert r.bitspersample == 12


@pytest.mark.parametrize(
    ('name', 'expected'),
    [
        ('gray', Photometric.GRAY),
        ('GRAY', Photometric.GRAY),
        ('GRAYSCALE', Photometric.GRAY),
        ('minisblack', Photometric.GRAY),
        ('rgb', Photometric.RGB),
        ('RGB', Photometric.RGB),
        ('rgba', Photometric.RGB),
        ('palette', Photometric.PALETTE),
    ],
)
def test_image_layout_photometric_string(name, expected):
    """Test photometric string normalization."""
    r = image_layout((64, 64, 3), numpy.uint8, photometric=name)
    assert r.photometric == expected


@pytest.mark.parametrize(
    ('name', 'expected'),
    [
        ('cmyk', Photometric.CMYK),
        ('YCBCR', Photometric.YCBCR),
        ('lab', Photometric.CIELAB),
        ('cielab', Photometric.CIELAB),
    ],
)
def test_image_layout_photometric_string_multi(name, expected):
    """Test photometric string normalization for multi-sample types."""
    r = image_layout((64, 64, 4), numpy.uint8, photometric=name)
    assert r.photometric == expected


def test_image_layout_photometric_int():
    """Test photometric as integer."""
    r = image_layout((64, 64, 4), numpy.uint8, photometric=Photometric.CMYK)
    assert r.photometric == Photometric.CMYK


def test_image_layout_photometric_invalid():
    """Test invalid photometric string raises."""
    with pytest.raises(ValueError, match='not recognised'):
        image_layout((64, 64, 3), numpy.uint8, photometric='INVALID')


@pytest.mark.parametrize('val', [True, 'SEPARATE', 'separate'])
def test_image_layout_planar_true_strings(val):
    """Test planar=True from boolean or 'SEPARATE' string."""
    r = image_layout((3, 48, 64), numpy.uint8, planar=val)
    assert r.planar is True


@pytest.mark.parametrize('val', [False, 'CONTIG', 'contig'])
def test_image_layout_planar_false_strings(val):
    """Test planar=False from boolean or 'CONTIG' string."""
    r = image_layout((48, 64, 3), numpy.uint8, planar=val)
    assert r.planar is False


def test_image_layout_planar_invalid():
    """Test invalid planar string raises."""
    with pytest.raises(ValueError, match=r'planar.*not recognised'):
        image_layout((3, 48, 64), numpy.uint8, planar='INVALID')


@pytest.mark.parametrize(
    ('val', 'expected'),
    [
        ('associated', ExtraSample.ASSOCALPHA),
        ('ASSOCIATED', ExtraSample.ASSOCALPHA),
        ('premultiplied', ExtraSample.ASSOCALPHA),
        ('unassociated', ExtraSample.UNASSALPHA),
        ('UNASSOCIATED', ExtraSample.UNASSALPHA),
        ('straight', ExtraSample.UNASSALPHA),
    ],
)
def test_image_layout_extrasample_string(val, expected):
    """Test extrasample string normalization."""
    r = image_layout((48, 64, 4), numpy.uint8, extrasample=val)
    assert r.extratype == expected


def test_image_layout_extrasample_int():
    """Test extrasample as integer."""
    r = image_layout(
        (48, 64, 4), numpy.uint8, extrasample=ExtraSample.ASSOCALPHA
    )
    assert r.extratype == ExtraSample.ASSOCALPHA


def test_image_layout_extrasample_invalid():
    """Test invalid extrasample raises."""
    with pytest.raises(ValueError, match='not recognised'):
        image_layout((48, 64, 4), numpy.uint8, extrasample='INVALID')


def test_image_layout_alpha_single_extra_auto():
    """Test single unspecified extra -> auto UNASSALPHA."""
    r = image_layout((48, 64, 4), numpy.uint8)  # RGB + 1 extra
    assert r.extratype == ExtraSample.UNASSALPHA
    assert r.extraindex == 3
    assert r.extracount == 1


def test_image_layout_alpha_multiple_extras_no_auto():
    """Test multiple extras with no hint -> no auto alpha."""
    r = image_layout(
        (48, 64, 5), numpy.uint8, photometric='rgb'
    )  # RGB + 2 extras
    assert r.extratype == ExtraSample.UNSPECIFIED
    assert r.extraindex == 0
    assert r.extracount == 2


def test_image_layout_alpha_with_hint():
    """Test alpha with explicit hint on multi-extra."""
    r = image_layout((48, 64, 5), numpy.uint8, extrasample='associated')
    assert r.extratype == ExtraSample.ASSOCALPHA
    assert r.extraindex == 3
    assert r.extracount == 2


def test_image_layout_no_extra_no_alpha():
    """Test exact-fit photometric -> no alpha, no extra."""
    r = image_layout((48, 64, 3), numpy.uint8)  # exactly RGB
    assert r.extratype == ExtraSample.UNSPECIFIED
    assert r.extraindex == 0
    assert r.extracount == 0


def test_image_layout_caps_zero_skips():
    """Test caps=0 skips all validation."""
    r = image_layout((48, 64, 3), dtype=numpy.float32)  # no caps -> passes
    assert r.sampleformat == SampleFormat.FLOAT


def test_image_layout_caps_reject_sf():
    """Test caps rejects unsupported sample format."""
    caps = IC.UINT | IC.SZ4 | IC.GRAY | IC.RGB
    with pytest.raises(ValueError, match='sample format'):
        image_layout((48, 64), dtype=numpy.float32, caps=int(caps))


def test_image_layout_caps_reject_sz():
    """Test caps rejects unsupported item size."""
    caps = IC.UINT | IC.SZ1 | IC.GRAY
    with pytest.raises(ValueError, match='item size'):
        image_layout((48, 64), dtype=numpy.uint16, caps=int(caps))


def test_image_layout_caps_reject_photo():
    """Test caps rejects unsupported photometric."""
    caps = IC.UINT | IC.SZ1 | IC.GRAY  # no IC.RGB
    with pytest.raises(ValueError, match='photometric'):
        image_layout((48, 64, 3), numpy.uint8, caps=int(caps))


def test_image_layout_caps_reject_frames():
    """Test caps rejects frames when IC.FRAMES not set."""
    caps = IC.UINT | IC.SZ1 | IC.GRAY
    with pytest.raises(ValueError, match='frames'):
        image_layout((5, 48, 64), numpy.uint8, frames=True, caps=int(caps))


def test_image_layout_caps_reject_alpha():
    """Test caps rejects alpha when IC.ALPHA not set."""
    caps = IC.UINT | IC.SZ1 | IC.GRAY | IC.RGB
    with pytest.raises(ValueError, match='alpha'):
        image_layout((48, 64, 4), numpy.uint8, caps=int(caps))


def test_image_layout_caps_reject_extra():
    """Test caps rejects multiple extras when IC.EXTRA not set."""
    caps = IC.UINT | IC.SZ1 | IC.GRAY | IC.RGB | IC.ALPHA
    with pytest.raises(ValueError, match='extra samples'):
        image_layout((48, 64, 6), numpy.uint8, caps=int(caps))


def test_image_layout_caps_reject_planar():
    """Test caps rejects planar when IC.PLANAR not set."""
    caps = IC.UINT | IC.SZ1 | IC.GRAY | IC.RGB
    with pytest.raises(ValueError, match='planar'):
        image_layout((3, 48, 64), numpy.uint8, planar=True, caps=int(caps))


def test_image_layout_caps_reject_volumetric():
    """Test caps rejects volumetric when IC.DEPTH not set."""
    caps = IC.UINT | IC.SZ1 | IC.GRAY
    with pytest.raises(ValueError, match='volumetric'):
        image_layout((8, 48, 64), numpy.uint8, volumetric=True, caps=int(caps))


def test_image_layout_caps_accept():
    """Test caps accept with all matching flags."""
    caps = (
        IC.UINT
        | IC.SZ1
        | IC.RGB
        | IC.ALPHA
        | IC.EXTRA
        | IC.FRAMES
        | IC.PLANAR
        | IC.DEPTH
    )
    r = image_layout(
        (2, 4, 8, 48, 64),
        numpy.uint8,
        planar=True,
        volumetric=True,
        extrasample='associated',
        caps=int(caps),
    )
    assert r.frames == 2
    assert r.depth == 8
    assert r.samples == 4
    assert r.planar is True
    assert r.extratype == ExtraSample.ASSOCALPHA


def test_image_layout_caps_bool():
    """Test IC.BOOL cap required for bool dtype."""
    caps = IC.UINT | IC.SZ1 | IC.GRAY
    with pytest.raises(ValueError, match='sample format'):
        image_layout((48, 64), dtype=numpy.bool_, caps=int(caps))

    caps_ok = IC.BOOL | IC.SZ1 | IC.GRAY
    r = image_layout((48, 64), dtype=numpy.bool_, caps=int(caps_ok))
    assert r.sampleformat == SampleFormat.BOOL


def test_image_layout_samples_less_than_photo():
    """Test samples < photo_samples raises."""
    with pytest.raises(ValueError, match=r'samples.*less than'):
        image_layout((48, 64, 2), numpy.uint8, photometric='rgb')


def test_image_layout_enum_types():
    """Test returned values are proper enum types."""
    r = image_layout((48, 64, 3), numpy.uint8)
    assert isinstance(r.photometric, Photometric)
    assert isinstance(r.sampleformat, SampleFormat)
    assert isinstance(r.extratype, ExtraSample)


def test_image_layout_intflag():
    """Test IC IntFlag composition."""
    flags = IC.GRAY | IC.RGB | IC.UINT | IC.SZ1
    assert isinstance(flags, IC)
    assert IC.GRAY in flags
    assert IC.CMYK not in flags


def test_image_layout_cmyk():
    """Test CMYK with 4 samples."""
    r = image_layout((48, 64, 4), numpy.uint8, photometric='cmyk')
    assert r.photometric == Photometric.CMYK
    assert r.samples == 4
    assert r.extracount == 0


def test_image_layout_cmyka():
    """Test CMYK+alpha with 5 samples."""
    r = image_layout((48, 64, 5), numpy.uint8, photometric='cmyk')
    assert r.photometric == Photometric.CMYK
    assert r.samples == 5
    assert r.extracount == 1
    assert r.extratype == ExtraSample.UNASSALPHA


def test_image_layout_multi_batch_frames():
    """Test (2, 3, 48, 64, 3) -> frames=6."""
    r = image_layout((2, 3, 48, 64, 3), numpy.uint8)
    assert r.frames == 6
    assert r.samples == 3
    assert r.photometric == Photometric.RGB


def test_image_layout_3d_photo_gray_no_channels():
    """Test (N, H, W) with photometric=gray (1 sample) -> frames."""
    r = image_layout((10, 48, 64), numpy.uint8, photometric='gray')
    assert r.frames == 10
    assert r.samples == 1
    assert r.photometric == Photometric.GRAY


def test_image_layout_4d_gray_photo_hint_frames():
    """Test (N, H, W, 1) with photometric=gray -> N frames, not H*N frames."""
    r = image_layout((10, 48, 64, 1), numpy.uint8, photometric='gray')
    assert r.frames == 10
    assert r.height == 48
    assert r.width == 64
    assert r.samples == 1
    assert r.photometric == Photometric.GRAY


def test_image_layout_5d_gray_photo_hint_frames():
    """Test 5D with single-sample photometric hint."""
    # (2, 3, 2, 4, 3) with photometric='gray' must produce 12 frames of 4x3
    # gray images, NOT 6 frames of 4x3 with 2 samples (channels) misinterpreted
    # from the trailing non-channel dim.

    for photo in ('gray', 'minisblack', 'miniswhite'):
        r = image_layout((2, 3, 2, 4, 3), numpy.uint8, photometric=photo)
        assert r.frames == 12, photo
        assert r.height == 4, photo
        assert r.width == 3, photo
        assert r.samples == 1, photo

    # float dtype: same rule applies
    r = image_layout((2, 3, 2, 4, 3), numpy.float32, photometric='gray')
    assert r.frames == 12
    assert r.height == 4
    assert r.width == 3
    assert r.samples == 1

    # trailing dim == 1 is still treated as a channel axis (1 sample),
    # so the remaining dims are frame dimensions
    r = image_layout((10, 48, 64, 1), numpy.uint8, photometric='gray')
    assert r.frames == 10
    assert r.height == 48
    assert r.width == 64
    assert r.samples == 1


def test_image_layout_3d_frames_false():
    """Test (H, W, S) with frames=False forces channel interpretation."""
    r = image_layout((48, 64, 64), numpy.uint8, frames=False)
    assert r.frames == 1
    assert r.samples == 64


# mypy: allow-untyped-defs
# mypy: check-untyped-defs=False
