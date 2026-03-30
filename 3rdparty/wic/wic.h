/* wic.h */

/*
 * Windows Imaging Component (WIC) decode and encode header.
 *
 * C API wrapping COM-based WIC operations. Windows only.
 */

#ifndef WIC_H
#define WIC_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint8_t* data;  /* decoded pixel data (caller frees via wic_decode_free) */
    uint32_t width;
    uint32_t height;
    uint32_t stride;      /* bytes per row */
    uint32_t components;  /* 1 (gray), 3 (RGB), or 4 (RGBA) */
    uint32_t bpc;         /* bits per component: 8 or 16 */
    uint32_t frame_count; /* total frames in the image */
} wic_decode_result_t;

/* Initialize the module-level WIC factory singleton.
 * Must be called once before any other wic_* function.
 * Returns 0 on success, HRESULT on failure.
 */
int32_t
wic_factory_init(void);

/* Release the factory singleton and uninitialize COM.
 *
 * Once called, wic_factory_init cannot be called again in the same process.
 * Intended for use at process exit only.
 */
void
wic_factory_destroy(void);

/* Query image properties without decoding pixels.
 *
 * Uses frame 0 for format detection. frame_count is optional (may be NULL).
 * Returns 0 on success, HRESULT on failure.
 */
int32_t
wic_get_info(
    const uint8_t* src,
    size_t srcsize,
    uint32_t* width,
    uint32_t* height,
    uint32_t* components,
    uint32_t* bpc,
    uint32_t* frame_count
);

/* Decode frame_index directly into a caller-supplied buffer.
 *
 * dst_stride is the row stride in bytes.
 * dst_size must be >= dst_stride * height.
 * Returns 0 on success, HRESULT on failure.
 */
int32_t
wic_copy_pixels(
    const uint8_t* src,
    size_t srcsize,
    uint32_t frame_index,
    uint8_t* dst,
    uint32_t dst_stride,
    size_t dst_size
);

/* Decode a single frame from an in-memory image buffer.
 *
 * Returns 0 on success, negative on platform error, positive HRESULT on
 * WIC/COM failure.
 */
int32_t
wic_decode_(
    const uint8_t* src,
    size_t srcsize,
    uint32_t frame_index,
    wic_decode_result_t* result
);

/* Check whether WIC can decode the data.
 *
 * Returns 1 if decodable, 0 if not, -1 on COM error.
 */
int32_t
wic_check_(
    const uint8_t* src,
    size_t srcsize
);

/* Free pixel data returned in wic_decode_result_t. */
void
wic_decode_free(uint8_t* data);

/* Container format identifiers for encoding. */
#define WIC_FORMAT_BMP  0
#define WIC_FORMAT_PNG  1
#define WIC_FORMAT_JPEG 2
#define WIC_FORMAT_TIFF 3
#define WIC_FORMAT_GIF  4
#define WIC_FORMAT_WMP  5  /* JPEG XR / HD Photo */
#define WIC_FORMAT_HEIF 6  /* requires HEIF Image Extensions */
#define WIC_FORMAT_WEBP 7  /* requires WebP Image Extensions */

/* Encode pixel data to a WIC-supported format.
 *
 * src points to tightly-packed pixel rows (stride = width * components * bpc/8).
 * format is one of the WIC_FORMAT_* constants.
 * quality is 0-100 for lossy formats (JPEG, WMP) or -1 for default.
 *
 * On success, *dst and *dstsize receive a malloc'd buffer + size.
 * The caller must free *dst via wic_encode_free().
 * Returns 0 on success, HRESULT on failure.
 */
int32_t
wic_encode_(
    const uint8_t* src,
    uint32_t width,
    uint32_t height,
    uint32_t components,
    uint32_t bpc,
    int32_t format,
    int32_t quality,
    uint8_t** dst,
    size_t* dstsize
);

/* Free encoded data returned by wic_encode_. */
void
wic_encode_free(uint8_t* data);

/* Return a version identifier string, or NULL if WIC is unavailable. */
const char*
wic_version_string(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* WIC_H */
