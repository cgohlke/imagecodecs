# imagecodecs/libspng.pxd
# cython: language_level = 3

# Cython declarations for the `libspng 0.7.3` library.
# https://github.com/randy408/libspng/

from libc.stdio cimport FILE
from libc.stdint cimport uint8_t, uint16_t, uint32_t, int32_t

cdef extern from 'spng.h' nogil:

    int SPNG_VERSION_MAJOR
    int SPNG_VERSION_MINOR
    int SPNG_VERSION_PATCH

    enum spng_errno:
        SPNG_IO_ERROR
        SPNG_IO_EOF
        SPNG_OK
        SPNG_EINVAL
        SPNG_EMEM
        SPNG_EOVERFLOW
        SPNG_ESIGNATURE
        SPNG_EWIDTH
        SPNG_EHEIGHT
        SPNG_EUSER_WIDTH
        SPNG_EUSER_HEIGHT
        SPNG_EBIT_DEPTH
        SPNG_ECOLOR_TYPE
        SPNG_ECOMPRESSION_METHOD
        SPNG_EFILTER_METHOD
        SPNG_EINTERLACE_METHOD
        SPNG_EIHDR_SIZE
        SPNG_ENOIHDR
        SPNG_ECHUNK_POS
        SPNG_ECHUNK_SIZE
        SPNG_ECHUNK_CRC
        SPNG_ECHUNK_TYPE
        SPNG_ECHUNK_UNKNOWN_CRITICAL
        SPNG_EDUP_PLTE
        SPNG_EDUP_CHRM
        SPNG_EDUP_GAMA
        SPNG_EDUP_ICCP
        SPNG_EDUP_SBIT
        SPNG_EDUP_SRGB
        SPNG_EDUP_BKGD
        SPNG_EDUP_HIST
        SPNG_EDUP_TRNS
        SPNG_EDUP_PHYS
        SPNG_EDUP_TIME
        SPNG_EDUP_OFFS
        SPNG_EDUP_EXIF
        SPNG_ECHRM
        SPNG_EPLTE_IDX
        SPNG_ETRNS_COLOR_TYPE
        SPNG_ETRNS_NO_PLTE
        SPNG_EGAMA
        SPNG_EICCP_NAME
        SPNG_EICCP_COMPRESSION_METHOD
        SPNG_ESBIT
        SPNG_ESRGB
        SPNG_ETEXT
        SPNG_ETEXT_KEYWORD
        SPNG_EZTXT
        SPNG_EZTXT_COMPRESSION_METHOD
        SPNG_EITXT
        SPNG_EITXT_COMPRESSION_FLAG
        SPNG_EITXT_COMPRESSION_METHOD
        SPNG_EITXT_LANG_TAG
        SPNG_EITXT_TRANSLATED_KEY
        SPNG_EBKGD_NO_PLTE
        SPNG_EBKGD_PLTE_IDX
        SPNG_EHIST_NO_PLTE
        SPNG_EPHYS
        SPNG_ESPLT_NAME
        SPNG_ESPLT_DUP_NAME
        SPNG_ESPLT_DEPTH
        SPNG_ETIME
        SPNG_EOFFS
        SPNG_EEXIF
        SPNG_EIDAT_TOO_SHORT
        SPNG_EIDAT_STREAM
        SPNG_EZLIB
        SPNG_EFILTER
        SPNG_EBUFSIZ
        SPNG_EIO
        SPNG_EOF
        SPNG_EBUF_SET
        SPNG_EBADSTATE
        SPNG_EFMT
        SPNG_EFLAGS
        SPNG_ECHUNKAVAIL
        SPNG_ENCODE_ONLY
        SPNG_EOI
        SPNG_ENOPLTE
        SPNG_ECHUNK_LIMITS
        SPNG_EZLIB_INIT
        SPNG_ECHUNK_STDLEN
        SPNG_EINTERNAL
        SPNG_ECTXTYPE
        SPNG_ENOSRC
        SPNG_ENODST
        SPNG_EOPSTATE
        SPNG_ENOTFINAL

    enum spng_text_type:
        SPNG_TEXT
        SPNG_ZTXT
        SPNG_ITXT

    enum spng_color_type:
        SPNG_COLOR_TYPE_GRAYSCALE
        SPNG_COLOR_TYPE_TRUECOLOR
        SPNG_COLOR_TYPE_INDEXED
        SPNG_COLOR_TYPE_GRAYSCALE_ALPHA
        SPNG_COLOR_TYPE_TRUECOLOR_ALPHA

    enum spng_filter:
        SPNG_FILTER_NONE
        SPNG_FILTER_SUB
        SPNG_FILTER_UP
        SPNG_FILTER_AVERAGE
        SPNG_FILTER_PAETH

    enum spng_filter_choice:
        SPNG_DISABLE_FILTERING
        SPNG_FILTER_CHOICE_NONE
        SPNG_FILTER_CHOICE_SUB
        SPNG_FILTER_CHOICE_UP
        SPNG_FILTER_CHOICE_AVG
        SPNG_FILTER_CHOICE_PAETH
        SPNG_FILTER_CHOICE_ALL

    enum spng_interlace_method:
        SPNG_INTERLACE_NONE
        SPNG_INTERLACE_ADAM7

    enum spng_format:
        SPNG_FMT_RGBA8
        SPNG_FMT_RGBA16
        SPNG_FMT_RGB8
        SPNG_FMT_GA8
        SPNG_FMT_GA16
        SPNG_FMT_G8
        SPNG_FMT_PNG
        SPNG_FMT_RAW

    enum spng_ctx_flags:
        SPNG_CTX_IGNORE_ADLER32
        SPNG_CTX_ENCODER

    enum spng_decode_flags:
        SPNG_DECODE_USE_TRNS
        SPNG_DECODE_USE_GAMA
        SPNG_DECODE_USE_SBIT
        SPNG_DECODE_TRNS
        SPNG_DECODE_GAMMA
        SPNG_DECODE_PROGRESSIVE

    enum spng_crc_action:
        SPNG_CRC_ERROR
        SPNG_CRC_DISCARD
        SPNG_CRC_USE

    enum spng_encode_flags:
        SPNG_ENCODE_PROGRESSIVE
        SPNG_ENCODE_FINALIZE

    struct spng_ihdr:
        uint32_t width
        uint32_t height
        uint8_t bit_depth
        uint8_t color_type
        uint8_t compression_method
        uint8_t filter_method
        uint8_t interlace_method

    struct spng_plte_entry:
        uint8_t red
        uint8_t green
        uint8_t blue
        uint8_t alpha

    struct spng_plte:
        uint32_t n_entries
        spng_plte_entry[256] entries

    struct spng_trns:
        uint16_t gray
        uint16_t red
        uint16_t green
        uint16_t blue
        uint32_t n_type3_entries
        uint8_t[256] type3_alpha

    struct spng_chrm_int:
        uint32_t white_point_x
        uint32_t white_point_y
        uint32_t red_x
        uint32_t red_y
        uint32_t green_x
        uint32_t green_y
        uint32_t blue_x
        uint32_t blue_y

    struct spng_chrm:
        double white_point_x
        double white_point_y
        double red_x
        double red_y
        double green_x
        double green_y
        double blue_x
        double blue_y

    struct spng_iccp:
        char[80] profile_name
        size_t profile_len
        char* profile

    struct spng_sbit:
        uint8_t grayscale_bits
        uint8_t red_bits
        uint8_t green_bits
        uint8_t blue_bits
        uint8_t alpha_bits

    struct spng_text:
        char[80] keyword
        int type_ 'type'
        size_t length
        char* text
        uint8_t compression_flag
        uint8_t compression_method
        char* language_tag
        char* translated_keyword

    struct spng_bkgd:
        uint16_t gray
        uint16_t red
        uint16_t green
        uint16_t blue
        uint16_t plte_index

    struct spng_hist:
        uint16_t[256] frequency

    struct spng_phys:
        uint32_t ppu_x
        uint32_t ppu_y
        uint8_t unit_specifier

    struct spng_splt_entry:
        uint16_t red
        uint16_t green
        uint16_t blue
        uint16_t alpha
        uint16_t frequency

    struct spng_splt:
        char[80] name
        uint8_t sample_depth
        uint32_t n_entries
        spng_splt_entry* entries

    struct spng_time:
        uint16_t year
        uint8_t month
        uint8_t day
        uint8_t hour
        uint8_t minute
        uint8_t second

    struct spng_offs:
        int32_t x
        int32_t y
        uint8_t unit_specifier

    struct spng_exif:
        size_t length
        char* data

    struct spng_chunk:
        size_t offset
        uint32_t length
        uint8_t[4] type
        uint32_t crc

    enum spng_location:
        SPNG_AFTER_IHDR
        SPNG_AFTER_PLTE
        SPNG_AFTER_IDAT

    struct spng_unknown_chunk:
        uint8_t[4] type
        size_t length
        void* data
        spng_location location

    enum spng_option:
        SPNG_KEEP_UNKNOWN_CHUNKS
        SPNG_IMG_COMPRESSION_LEVEL
        SPNG_IMG_WINDOW_BITS
        SPNG_IMG_MEM_LEVEL
        SPNG_IMG_COMPRESSION_STRATEGY
        SPNG_TEXT_COMPRESSION_LEVEL
        SPNG_TEXT_WINDOW_BITS
        SPNG_TEXT_MEM_LEVEL
        SPNG_TEXT_COMPRESSION_STRATEGY
        SPNG_FILTER_CHOICE
        SPNG_CHUNK_COUNT_LIMIT
        SPNG_ENCODE_TO_BUFFER

    ctypedef void* spng_malloc_fn(
        size_t size
    )

    ctypedef void* spng_realloc_fn(
        void* ptr,
        size_t size
    )

    ctypedef void* spng_calloc_fn(
        size_t count,
        size_t size
    )

    ctypedef void spng_free_fn(
        void* ptr
    )

    struct spng_alloc:
        spng_malloc_fn* malloc_fn
        spng_realloc_fn* realloc_fn
        spng_calloc_fn* calloc_fn
        spng_free_fn* free_fn

    struct spng_row_info:
        uint32_t scanline_idx
        uint32_t row_num
        int pass_
        uint8_t filter

    ctypedef struct spng_ctx:
        pass

    ctypedef int spng_read_fn(
        spng_ctx* ctx,
        void* user,
        void* dest,
        size_t length
    )

    ctypedef int spng_write_fn(
        spng_ctx* ctx,
        void* user,
        void* src,
        size_t length
    )

    ctypedef int spng_rw_fn(
        spng_ctx* ctx,
        void* user,
        void* dst_src,
        size_t length
    )

    spng_ctx* spng_ctx_new(
        int flags
    )

    spng_ctx* spng_ctx_new2(
        spng_alloc* alloc,
        int flags
    )

    void spng_ctx_free(
        spng_ctx* ctx
    )

    int spng_set_png_buffer(
        spng_ctx* ctx,
        const void* buf,
        size_t size
    )

    int spng_set_png_stream(
        spng_ctx* ctx,
        spng_rw_fn* rw_func,
        void* user
    )

    int spng_set_png_file(
        spng_ctx* ctx,
        FILE* file
    )

    void* spng_get_png_buffer(
        spng_ctx* ctx,
        size_t* len,
        int* error
    )

    int spng_set_image_limits(
        spng_ctx* ctx,
        uint32_t width,
        uint32_t height
    )

    int spng_get_image_limits(
        spng_ctx* ctx,
        uint32_t* width,
        uint32_t* height
    )

    int spng_set_chunk_limits(
        spng_ctx* ctx,
        size_t chunk_size,
        size_t cache_size
    )

    int spng_get_chunk_limits(
        spng_ctx* ctx,
        size_t* chunk_size,
        size_t* cache_size
    )

    int spng_set_crc_action(
        spng_ctx* ctx,
        int critical,
        int ancillary
    )

    int spng_set_option(
        spng_ctx* ctx,
        spng_option option,
        int value
    )

    int spng_get_option(
        spng_ctx* ctx,
        spng_option option,
        int* value
    )

    int spng_decoded_image_size(
        spng_ctx* ctx,
        int fmt,
        size_t* len
    )

    int spng_decode_image(
        spng_ctx* ctx,
        void* out,
        size_t len,
        int fmt,
        int flags
    )

    int spng_decode_scanline(
        spng_ctx* ctx,
        void* out,
        size_t len
    )

    int spng_decode_row(
        spng_ctx* ctx,
        void* out,
        size_t len
    )

    int spng_decode_chunks(
        spng_ctx *ctx
    )

    int spng_get_row_info(
        spng_ctx* ctx,
        spng_row_info* row_info
    )

    int spng_encode_image(
        spng_ctx* ctx,
        const void* img,
        size_t len,
        int fmt,
        int flags
    )

    int spng_encode_scanline(
        spng_ctx* ctx,
        const void* scanline,
        size_t len
    )

    int spng_encode_row(
        spng_ctx* ctx,
        const void* row,
        size_t len
    )

    int spng_encode_chunks(
        spng_ctx* ctx
    )

    int spng_get_ihdr(
        spng_ctx* ctx,
        spng_ihdr* ihdr
    )

    int spng_get_plte(
        spng_ctx* ctx,
        spng_plte* plte
    )

    int spng_get_trns(
        spng_ctx* ctx,
        spng_trns* trns
    )

    int spng_get_chrm(
        spng_ctx* ctx,
        spng_chrm* chrm
    )

    int spng_get_chrm_int(
        spng_ctx* ctx,
        spng_chrm_int* chrm_int
    )

    int spng_get_gama(
        spng_ctx* ctx,
        double* gamma
    )

    int spng_get_gama_int(
        spng_ctx* ctx,
        uint32_t* gama_int
    )

    int spng_get_iccp(
        spng_ctx* ctx,
        spng_iccp* iccp
    )

    int spng_get_sbit(
        spng_ctx* ctx,
        spng_sbit* sbit
    )

    int spng_get_srgb(
        spng_ctx* ctx,
        uint8_t* rendering_intent
    )

    int spng_get_text(
        spng_ctx* ctx,
        spng_text* text,
        uint32_t* n_text
    )

    int spng_get_bkgd(
        spng_ctx* ctx,
        spng_bkgd* bkgd
    )

    int spng_get_hist(
        spng_ctx* ctx,
        spng_hist* hist
    )

    int spng_get_phys(
        spng_ctx* ctx,
        spng_phys* phys
    )

    int spng_get_splt(
        spng_ctx* ctx,
        spng_splt* splt,
        uint32_t* n_splt
    )

    int spng_get_time(
        spng_ctx* ctx,
        spng_time* time
    )

    int spng_get_unknown_chunks(
        spng_ctx* ctx,
        spng_unknown_chunk* chunks,
        uint32_t* n_chunks
    )

    int spng_get_offs(
        spng_ctx* ctx,
        spng_offs* offs
    )

    int spng_get_exif(
        spng_ctx* ctx,
        spng_exif* exif
    )

    int spng_set_ihdr(
        spng_ctx* ctx,
        spng_ihdr* ihdr
    )

    int spng_set_plte(
        spng_ctx* ctx,
        spng_plte* plte
    )

    int spng_set_trns(
        spng_ctx* ctx,
        spng_trns* trns
    )

    int spng_set_chrm(
        spng_ctx* ctx,
        spng_chrm* chrm
    )

    int spng_set_chrm_int(
        spng_ctx* ctx,
        spng_chrm_int* chrm_int
    )

    int spng_set_gama(
        spng_ctx* ctx,
        double gamma
    )

    int spng_set_gama_int(
        spng_ctx* ctx,
        uint32_t gamma
    )

    int spng_set_iccp(
        spng_ctx* ctx,
        spng_iccp* iccp
    )

    int spng_set_sbit(
        spng_ctx* ctx,
        spng_sbit* sbit
    )

    int spng_set_srgb(
        spng_ctx* ctx,
        uint8_t rendering_intent
    )

    int spng_set_text(
        spng_ctx* ctx,
        spng_text* text,
        uint32_t n_text
    )

    int spng_set_bkgd(
        spng_ctx* ctx,
        spng_bkgd* bkgd
    )

    int spng_set_hist(
        spng_ctx* ctx,
        spng_hist* hist
    )

    int spng_set_phys(
        spng_ctx* ctx,
        spng_phys* phys
    )

    int spng_set_splt(
        spng_ctx* ctx,
        spng_splt* splt,
        uint32_t n_splt
    )

    int spng_set_time(
        spng_ctx* ctx,
        spng_time* time
    )

    int spng_set_unknown_chunks(
        spng_ctx* ctx,
        spng_unknown_chunk* chunks,
        uint32_t n_chunks
    )

    int spng_set_offs(
        spng_ctx* ctx,
        spng_offs* offs
    )

    int spng_set_exif(
        spng_ctx* ctx,
        spng_exif* exif
    )

    const char* spng_strerror(
        int err
    )

    const char* spng_version_string()
