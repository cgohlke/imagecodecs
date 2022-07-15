# imagecodecs/jetraw.pxd
# cython: language_level = 3

# Cython declarations for the `jetraw 22.02.16.1` library.
# https://github.com/Jetraw/Jetraw


from libc.stdint cimport uint16_t, uint32_t, int32_t

cdef extern from 'jetraw/jetraw.h':

    ctypedef enum dp_status:
        dp_success
        dp_memory_error
        dp_not_initialized
        dp_unknown_error
        dp_license_error
        dp_file_read_error
        dp_file_write_error
        dp_file_corrupt
        dp_unknown_identifier
        dp_parameter_error
        dp_image_too_small
        dp_out_of_range
        dp_tiff_file_cannot_open
        dp_tiff_not_initialized
        dp_tiff_handle_in_use
        dp_tiff_file_update_error
        dp_tiff_wrong_file_mode
        dp_bad_image

    const char* dp_status_description(
        dp_status status
    ) nogil

    dp_status jetraw_encode(
        const uint16_t* pImgBuffer,
        uint32_t imgWidth,
        uint32_t imgHeight,
        char* pDstBuffer,
        int32_t* pDstLen
    ) nogil

    dp_status jetraw_decode(
        const char* pSrcBuffer,
        int32_t srcLen,
        uint16_t* pImgBuffer,
        int32_t imgPixels
    ) nogil

    const char* jetraw_version() nogil


cdef extern from 'dpcore/dpcore.h':

    ctypedef char CHARTYPE

    int dpcore_init() nogil

    void dpcore_set_loglevel(
        int level
    ) nogil

    dp_status dpcore_set_logfile(
        const CHARTYPE* file_path
    ) nogil

    dp_status dpcore_load_parameters(
        const CHARTYPE* file_path
    ) nogil

    dp_status dpcore_prepare_image(
        uint16_t* imgbuf,
        int32_t imgsize,
        const char* identifier,
        float error_bound
    ) nogil

    dp_status dpcore_embed_meta(
        uint16_t* imgbuf,
        int32_t imgsize,
        const char* identifier,
        float error_bound
    ) nogil

    int dpcore_identifier_count() nogil

    dp_status dpcore_get_identifiers(
        char* buf,
        int* bufsize
    ) nogil
