/* bcdec_dds.h */
/* Structures and defines for the DDS file format */
/* https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dx-graphics-dds */

#ifndef DDS_HEADER_INCLUDED
#define DDS_HEADER_INCLUDED

#define _FOURCC(a, b, c, d) ((a) | ((b) << 8) | ((c) << 16) | ((d) << 24))

#define DDS_FOURCC_DDS   _FOURCC('D', 'D', 'S', ' ')
#define DDS_FOURCC_DXT1  _FOURCC('D', 'X', 'T', '1')
#define DDS_FOURCC_DXT3  _FOURCC('D', 'X', 'T', '3')
#define DDS_FOURCC_DXT5  _FOURCC('D', 'X', 'T', '5')
#define DDS_FOURCC_DX10  _FOURCC('D', 'X', '1', '0')

#define DXGI_FORMAT_BC4_UNORM  80
#define DXGI_FORMAT_BC5_UNORM  83
#define DXGI_FORMAT_BC6H_UF16  95
#define DXGI_FORMAT_BC6H_SF16  96
#define DXGI_FORMAT_BC7_UNORM  98

/* DDS_HEADER.flags */
#define DDSD_CAPS         0x00000001
#define DDSD_HEIGHT       0x00000002
#define DDSD_WIDTH        0x00000004
#define DDSD_PITCH        0x00000008
#define DDSD_PIXELFORMAT  0x00001000
#define DDSD_MIPMAPCOUNT  0x00020000
#define DDSD_LINEARSIZE   0x00080000
#define DDSD_DEPTH 	      0x00800000

#define DDS_HEADER_FLAGS_TEXTURE     0x00001007  /* DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT */
#define DDS_HEADER_FLAGS_MIPMAP      0x00020000  /* DDSD_MIPMAPCOUNT */
#define DDS_HEADER_FLAGS_VOLUME      0x00800000  /* DDSD_DEPTH */
#define DDS_HEADER_FLAGS_PITCH       0x00000008  /* DDSD_PITCH */
#define DDS_HEADER_FLAGS_LINEARSIZE  0x00080000  /* DDSD_LINEARSIZE */

/* DDS_HEADER.caps */
#define DDSCAPS_COMPLEX  0x00000008
#define DDSCAPS_MIPMAP   0x00400000
#define DDSCAPS_TEXTURE  0x00001000

#define DDS_SURFACE_FLAGS_TEXTURE  0x00001000  /* DDSCAPS_TEXTURE */
#define DDS_SURFACE_FLAGS_MIPMAP   0x00400008  /* DDSCAPS_COMPLEX | DDSCAPS_MIPMAP */
#define DDS_SURFACE_FLAGS_CUBEMAP  0x00000008  /* DDSCAPS_COMPLEX */

/* DDS_HEADER.caps2 */
#define DDSCAPS2_CUBEMAP            0x00000200  /* DDSCAPS2_CUBEMAP */
#define DDSCAPS2_CUBEMAP_POSITIVEX  0x00000600  /* DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_POSITIVEX */
#define DDSCAPS2_CUBEMAP_NEGATIVEX  0x00000a00  /* DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_NEGATIVEX */
#define DDSCAPS2_CUBEMAP_POSITIVEY  0x00001200  /* DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_POSITIVEY */
#define DDSCAPS2_CUBEMAP_NEGATIVEY  0x00002200  /* DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_NEGATIVEY */
#define DDSCAPS2_CUBEMAP_POSITIVEZ  0x00004200  /* DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_POSITIVEZ */
#define DDSCAPS2_CUBEMAP_NEGATIVEZ  0x00008200  /* DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_NEGATIVEZ */
#define DDSCAPS2_VOLUME             0x00200000  /* DDSCAPS2_VOLUME */

#define DDSCAPS2_CUBEMAP_ALLFACES ( \
    DDSCAPS2_CUBEMAP_POSITIVEX | DDSCAPS2_CUBEMAP_NEGATIVEX | \
    DDSCAPS2_CUBEMAP_POSITIVEY | DDSCAPS2_CUBEMAP_NEGATIVEY | \
    DDSCAPS2_CUBEMAP_POSITIVEZ | DDSCAPS2_CUBEMAP_NEGATIVEZ )

/* DDS_HEADER_DXT10.resourceDimension */
#define DDS_DIMENSION_TEXTURE1D 0x2  /* D3D10_RESOURCE_DIMENSION_TEXTURE1D */
#define DDS_DIMENSION_TEXTURE2D 0x3  /* D3D10_RESOURCE_DIMENSION_TEXTURE2D */
#define DDS_DIMENSION_TEXTURE3D 0x4  /* D3D10_RESOURCE_DIMENSION_TEXTURE3D */

/* DDS_HEADER_DXT10.miscFlag */
#define DDS_RESOURCE_MISC_TEXTURECUBE  0x4  /* D3D10_RESOURCE_MISC_TEXTURECUBE */

/* DDS_HEADER_DXT10.miscFlags2 */
#define DDS_ALPHA_MODE_UNKNOWN 	      0x0
#define DDS_ALPHA_MODE_STRAIGHT       0x1
#define DDS_ALPHA_MODE_PREMULTIPLIED  0x2
#define DDS_ALPHA_MODE_OPAQUE         0x3
#define DDS_ALPHA_MODE_CUSTOM         0x4

typedef struct DDS_PIXELFORMAT {
    unsigned int size;
    unsigned int flags;
    unsigned int fourCC;
    unsigned int RGBBitCount;
    unsigned int RBitMask;
    unsigned int GBitMask;
    unsigned int BBitMask;
    unsigned int ABitMask;
} DDS_PIXELFORMAT_t;

typedef struct DDS_HEADER {
    unsigned int size;
    unsigned int flags;
    unsigned int height;
    unsigned int width;
    union {
        int pitch;
        unsigned int linearSize;
    };
    unsigned int depth;  /* if DDS_HEADER_FLAGS_VOLUME is in flags */
    unsigned int mipMapCount;
    unsigned int reserved1[11];
    DDS_PIXELFORMAT_t ddspf;
    unsigned int caps;
    unsigned int caps2;
    unsigned int caps3;
    unsigned int caps4;
    unsigned int reserved2;
} DDS_HEADER_t;

typedef struct DDS_HEADER_DXT10 {
    unsigned int dxgiFormat;
    unsigned int resourceDimension;
    unsigned int miscFlag;
    unsigned int arraySize;
    unsigned int miscFlags2;
} DDS_HEADER_DXT10_t;

#endif /* DDS_HEADER_INCLUDED */
