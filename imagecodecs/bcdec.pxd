# imagecodecs/bcdec.pxd
# cython: language_level = 3

# Cython declarations for the `bcdec 963c5e5` library.
# https://github.com/iOrange/bcdec

cdef extern from 'bcdec.h' nogil:

    int BCDEC_VERSION_MAJOR
    int BCDEC_VERSION_MINOR

    int BCDEC_BC1_BLOCK_SIZE
    int BCDEC_BC2_BLOCK_SIZE
    int BCDEC_BC3_BLOCK_SIZE
    int BCDEC_BC4_BLOCK_SIZE
    int BCDEC_BC5_BLOCK_SIZE
    int BCDEC_BC6H_BLOCK_SIZE
    int BCDEC_BC7_BLOCK_SIZE

    int BCDEC_BC1_COMPRESSED_SIZE(int w, int h)
    int BCDEC_BC2_COMPRESSED_SIZE(int w, int h)
    int BCDEC_BC3_COMPRESSED_SIZE(int w, int h)
    int BCDEC_BC4_COMPRESSED_SIZE(int w, int h)
    int BCDEC_BC5_COMPRESSED_SIZE(int w, int h)
    int BCDEC_BC6H_COMPRESSED_SIZE(int w, int h)
    int BCDEC_BC7_COMPRESSED_SIZE(int w, int h)

    void bcdec_bc1(
        const void* compressedBlock,
        void* decompressedBlock,
        int destinationPitch
    )

    void bcdec_bc2(
        const void* compressedBlock,
        void* decompressedBlock,
        int destinationPitch
    )

    void bcdec_bc3(
        const void* compressedBlock,
        void* decompressedBlock,
        int destinationPitch
    )

    void bcdec_bc4(
        const void* compressedBlock,
        void* decompressedBlock,
        int destinationPitch
    )

    void bcdec_bc5(
        const void* compressedBlock,
        void* decompressedBlock,
        int destinationPitch
    )

    # BCDEC_BC4BC5_PRECISE
    #
    # void bcdec_bc4(
    #     const void* compressedBlock,
    #     void* decompressedBlock,
    #     int destinationPitch,
    #     int isSigned
    # )
    #
    # void bcdec_bc5(
    #     const void* compressedBlock,
    #     void* decompressedBlock,
    #     int destinationPitch,
    #     int isSigned
    # )
    #
    # void bcdec_bc4_float(
    #     const void* compressedBlock,
    #     void* decompressedBlock,
    #     int destinationPitch,
    #     int isSigned
    # )
    #
    # void bcdec_bc5_float(
    #     const void* compressedBlock,
    #     void* decompressedBlock,
    #     int destinationPitch,
    #     int isSigned
    # )

    void bcdec_bc6h_float(
        const void* compressedBlock,
        void* decompressedBlock,
        int destinationPitch,
        int isSigned
    )

    void bcdec_bc6h_half(
        const void* compressedBlock,
        void* decompressedBlock,
        int destinationPitch,
        int isSigned
    )

    void bcdec_bc7(
        const void* compressedBlock,
        void* decompressedBlock,
        int destinationPitch
    )


cdef extern from 'bcdec_dds.h' nogil:

    int DDS_FOURCC_DDS
    int DDS_FOURCC_DXT1
    int DDS_FOURCC_DXT3
    int DDS_FOURCC_DXT5
    int DDS_FOURCC_DX10

    int DXGI_FORMAT_BC4_UNORM
    int DXGI_FORMAT_BC5_UNORM
    int DXGI_FORMAT_BC6H_UF16
    int DXGI_FORMAT_BC6H_SF16
    int DXGI_FORMAT_BC7_UNORM

    # DDS_HEADER.flags
    int DDSD_CAPS
    int DDSD_HEIGHT
    int DDSD_WIDTH
    int DDSD_PITCH
    int DDSD_PIXELFORMAT
    int DDSD_MIPMAPCOUNT
    int DDSD_LINEARSIZE
    int DDSD_DEPTH

    int DDS_HEADER_FLAGS_TEXTURE
    int DDS_HEADER_FLAGS_MIPMAP
    int DDS_HEADER_FLAGS_VOLUME
    int DDS_HEADER_FLAGS_PITCH
    int DDS_HEADER_FLAGS_LINEARSIZE

    # DDS_HEADER.caps
    int DDSCAPS_COMPLEX
    int DDSCAPS_MIPMAP
    int DDSCAPS_TEXTURE

    int DDS_SURFACE_FLAGS_TEXTURE
    int DDS_SURFACE_FLAGS_MIPMAP
    int DDS_SURFACE_FLAGS_CUBEMAP

    # DDS_HEADER.caps2
    int DDSCAPS2_CUBEMAP
    int DDSCAPS2_CUBEMAP_POSITIVEX
    int DDSCAPS2_CUBEMAP_NEGATIVEX
    int DDSCAPS2_CUBEMAP_POSITIVEY
    int DDSCAPS2_CUBEMAP_NEGATIVEY
    int DDSCAPS2_CUBEMAP_POSITIVEZ
    int DDSCAPS2_CUBEMAP_NEGATIVEZ
    int DDSCAPS2_VOLUME

    int DDSCAPS2_CUBEMAP_ALLFACES

    # DDS_HEADER_DXT10.resourceDimension
    int DDS_DIMENSION_TEXTURE1D
    int DDS_DIMENSION_TEXTURE2D
    int DDS_DIMENSION_TEXTURE3D

    # DDS_HEADER_DXT10.miscFlag
    int DDS_RESOURCE_MISC_TEXTURECUBE

    # DDS_HEADER_DXT10.miscFlags2
    int DDS_ALPHA_MODE_UNKNOWN
    int DDS_ALPHA_MODE_STRAIGHT
    int DDS_ALPHA_MODE_PREMULTIPLIED
    int DDS_ALPHA_MODE_OPAQUE
    int DDS_ALPHA_MODE_CUSTOM

    ctypedef struct DDS_PIXELFORMAT_t:
        unsigned int size
        unsigned int flags
        unsigned int fourCC
        unsigned int RGBBitCount
        unsigned int RBitMask
        unsigned int GBitMask
        unsigned int BBitMask
        unsigned int ABitMask

    ctypedef struct DDS_HEADER_t:
        unsigned int size
        unsigned int flags
        unsigned int height
        unsigned int width
        # union
        int pitch
        unsigned int linearSize
        # end union
        unsigned int depth
        unsigned int mipMapCount
        unsigned int[11] reserved1
        DDS_PIXELFORMAT_t ddspf
        unsigned int caps
        unsigned int caps2
        unsigned int caps3
        unsigned int caps4
        unsigned int reserved2

    ctypedef struct DDS_HEADER_DXT10_t:
        unsigned int dxgiFormat
        unsigned int resourceDimension
        unsigned int miscFlag
        unsigned int arraySize
        unsigned int miscFlags2
