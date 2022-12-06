# imagecodecs/rgbe.pxd
# cython: language_level = 3

# Cython declarations for the modified `rgbe` library.
# https://www.graphics.cornell.edu/~bjw/rgbe/rgbe.c

cdef extern from 'rgbe.h':

    char* RGBE_VERSION

    int RGBE_VALID_PROGRAMTYPE
    int RGBE_VALID_GAMMA
    int RGBE_VALID_EXPOSURE

    int RGBE_RETURN_SUCCESS
    int RGBE_RETURN_FAILURE
    int RGBE_READ_ERROR
    int RGBE_WRITE_ERROR
    int RGBE_FORMAT_ERROR
    int RGBE_MEMORY_ERROR

    ctypedef struct rgbe_header_info:
        int valid
        float gamma
        float exposure
        char programtype[16]

    ctypedef struct rgbe_stream_t:
        char *data
        size_t size
        size_t pos
        int owner

    int RGBE_WriteHeader(
        rgbe_stream_t *fp,
        int width,
        int height,
        rgbe_header_info *info
    ) nogil

    int RGBE_ReadHeader(
        rgbe_stream_t *fp,
        int *width,
        int *height,
        rgbe_header_info *info
    ) nogil

    int RGBE_WritePixels(
        rgbe_stream_t *fp,
        float *data,
        int numpixels
    ) nogil

    int RGBE_ReadPixels(
        rgbe_stream_t *fp,
        float *data,
        int numpixels
    ) nogil

    int RGBE_WritePixels_RLE(
        rgbe_stream_t *fp,
        float *data,
        int scanline_width,
        int num_scanlines
    ) nogil

    int RGBE_ReadPixels_RLE(
        rgbe_stream_t *fp,
        float *data,
        int scanline_width,
        int num_scanlines
    ) nogil

    rgbe_stream_t *rgbe_stream_new(
        size_t size,
        char *data
    ) nogil

    void rgbe_stream_del(
        rgbe_stream_t *stream
    ) nogil

    size_t rgbe_stream_read(
        void *ptr,
        size_t size,
        size_t nmemb,
        rgbe_stream_t *stream
    ) nogil

    size_t rgbe_stream_write(
        const void *ptr,
        size_t size,
        size_t nmemb,
        rgbe_stream_t *stream
    ) nogil

    int rgbe_stream_printf(
        rgbe_stream_t *stream,
        const char *format,
        ...
    ) nogil

    char* rgbe_stream_gets(
        char *str,
        size_t n,
        rgbe_stream_t *stream
    ) nogil
