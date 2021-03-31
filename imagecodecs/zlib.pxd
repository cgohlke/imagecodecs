# imagecodecs/zlib.pxd
# cython: language_level = 3

# Cython declarations for the `zlib 1.2.11` library.
# https://github.com/madler/zlib

cdef extern from 'zlib.h':

    char* ZLIB_VERSION
    int ZLIB_VERNUM
    int ZLIB_VER_MAJOR
    int ZLIB_VER_MINOR
    int ZLIB_VER_REVISION
    int ZLIB_VER_SUBREVISION

    const char* zlibVersion() nogil

    #  flush values
    int Z_NO_FLUSH
    int Z_PARTIAL_FLUSH
    int Z_SYNC_FLUSH
    int Z_FULL_FLUSH
    int Z_FINISH
    int Z_BLOCK
    int Z_TREES

    # return codes
    int Z_OK
    int Z_STREAM_END
    int Z_NEED_DICT
    int Z_ERRNO
    int Z_STREAM_ERROR
    int Z_DATA_ERROR
    int Z_MEM_ERROR
    int Z_BUF_ERROR
    int Z_VERSION_ERROR

    # compression levels
    int Z_NO_COMPRESSION
    int Z_BEST_SPEED
    int Z_BEST_COMPRESSION
    int Z_DEFAULT_COMPRESSION

    # compression strategy
    int Z_FILTERED
    int Z_HUFFMAN_ONLY
    int Z_RLE
    int Z_FIXED
    int Z_DEFAULT_STRATEGY

    # data_type
    int Z_BINARY
    int Z_TEXT
    int Z_ASCII
    int Z_UNKNOWN

    int Z_DEFLATED
    int Z_NULL

    ctypedef unsigned char Bytef
    ctypedef unsigned long uLong
    ctypedef unsigned long uLongf
    ctypedef unsigned int uInt
    ctypedef void* voidpf
    ctypedef void* voidp
    ctypedef const void* voidpc
    ctypedef size_t z_size_t
    ctypedef long z_off_t
    ctypedef long long z_off64_t

    ctypedef voidpf (*alloc_func)(
        voidpf opaque,
        uInt items,
        uInt size
    ) nogil

    ctypedef void (*free_func)(
        voidpf opaque,
        voidpf address
    ) nogil

    struct internal_state:
        pass

    ctypedef struct z_stream:
        const Bytef* next_in
        uInt avail_in
        uLong total_in
        Bytef* next_out
        uInt avail_out
        uLong total_out
        const char* msg
        internal_state* state
        alloc_func zalloc
        free_func zfree
        voidpf opaque
        int data_type
        uLong adler
        uLong reserved

    ctypedef z_stream* z_streamp

    ctypedef struct gz_header:
        int text
        uLong time
        int xflags
        int os
        Bytef* extra
        uInt extra_len
        uInt extra_max
        Bytef* name
        uInt name_max
        Bytef* comment
        uInt comm_max
        int hcrc
        int done

    ctypedef gz_header* gz_headerp

    int deflate(
        z_streamp strm,
        int flush
    ) nogil

    int deflateEnd(
        z_streamp strm
    ) nogil

    int inflate(
        z_streamp strm,
        int flush
    ) nogil

    int inflateEnd(
        z_streamp strm
    ) nogil

    int deflateSetDictionary(
        z_streamp strm,
        const Bytef* dictionary,
        uInt dictLength
    ) nogil

    int deflateGetDictionary(
        z_streamp strm,
        Bytef* dictionary,
        uInt* dictLength
    ) nogil

    int deflateCopy(
        z_streamp dest,
        z_streamp source
    ) nogil

    int deflateReset(
        z_streamp strm
    ) nogil

    int deflateParams(
        z_streamp strm,
        int level,
        int strategy
    ) nogil

    int deflateTune(
        z_streamp strm,
        int good_length,
        int max_lazy,
        int nice_length,
        int max_chain
    ) nogil

    uLong deflateBound(
        z_streamp strm,
        uLong sourceLen
    ) nogil

    int deflatePending(
        z_streamp strm,
        unsigned* pending,
        int* bits
    ) nogil

    int deflatePrime(
        z_streamp strm,
        int bits,
        int value
    ) nogil

    int deflateSetHeader(
        z_streamp strm,
        gz_headerp head
    ) nogil

    int inflateSetDictionary(
        z_streamp strm,
        const Bytef* dictionary,
        uInt dictLength
    ) nogil

    int inflateGetDictionary(
        z_streamp strm,
        Bytef* dictionary,
        uInt* dictLength
    ) nogil

    int inflateSync(
        z_streamp strm
    ) nogil

    int inflateCopy(
        z_streamp dest,
        z_streamp source
    ) nogil

    int inflateReset(
        z_streamp strm
    ) nogil

    int inflateReset2(
        z_streamp strm,
        int windowBits
    ) nogil

    int inflatePrime(
        z_streamp strm,
        int bits,
        int value
    ) nogil

    long inflateMark(
        z_streamp strm
    ) nogil

    int inflateGetHeader(
        z_streamp strm,
        gz_headerp head
    ) nogil

    ctypedef unsigned(*in_func)(
        void*,
        const unsigned char**
    ) nogil

    ctypedef int(*out_func)(
        void*,
        unsigned char*,
        unsigned
    ) nogil

    int inflateBack(
        z_streamp strm,
        in_func inf,
        void* in_desc,
        out_func out,
        void* out_desc
    ) nogil

    int inflateBackEnd(
        z_streamp strm
    ) nogil

    uLong zlibCompileFlags() nogil

    int compress(
        Bytef* dest,
        uLongf* destLen,
        const Bytef* source,
        uLong sourceLen
    ) nogil

    int compress2(
        Bytef* dest,
        uLongf* destLen,
        const Bytef* source,
        uLong sourceLen,
        int level
    ) nogil

    uLong compressBound(
        uLong sourceLen
    ) nogil

    int uncompress(
        Bytef* dest,
        uLongf* destLen,
        const Bytef* source,
        uLong sourceLen
    ) nogil

    int uncompress2(
        Bytef* dest,
        uLongf* destLen,
        const Bytef* source,
        uLong* sourceLen
    ) nogil

    # GZIP

    ctypedef struct gzFile_s:
        unsigned have
        unsigned char* next
        z_off64_t pos

    ctypedef gzFile_s* gzFile

    gzFile gzdopen(
        int fd,
        const char* mode
    ) nogil

    int gzbuffer(
        gzFile file,
        unsigned size
    ) nogil

    int gzsetparams(
        gzFile file,
        int level,
        int strategy
    ) nogil

    int gzread(
        gzFile file,
        voidp buf,
        unsigned len
    ) nogil

    z_size_t gzfread(
        voidp buf,
        z_size_t size,
        z_size_t nitems,
        gzFile file
    ) nogil

    int gzwrite(
        gzFile file,
        voidpc buf,
        unsigned len
    ) nogil

    z_size_t gzfwrite(
        voidpc buf,
        z_size_t size,
        z_size_t nitems,
        gzFile file
    ) nogil

    int gzputs(
        gzFile file,
        const char* s
    ) nogil

    char* gzgets(
        gzFile file,
        char* buf,
        int len
    ) nogil

    int gzputc(
        gzFile file,
        int c
    ) nogil

    int gzgetc(
        gzFile file
    ) nogil

    int gzungetc(
        int c,
        gzFile file
    ) nogil

    int gzflush(
        gzFile file,
        int flush
    ) nogil

    int gzrewind(
        gzFile file
    ) nogil

    int gzeof(
        gzFile file
    ) nogil

    int gzdirect(
        gzFile file
    ) nogil

    int gzclose(
        gzFile file
    ) nogil

    int gzclose_r(
        gzFile file
    ) nogil

    int gzclose_w(
        gzFile file
    ) nogil

    const char* gzerror(
        gzFile file,
        int* errnum
    ) nogil

    void gzclearerr(
        gzFile file
    ) nogil

    # CRC

    uLong adler32(
        uLong adler,
        const Bytef* buf,
        uInt len
    ) nogil

    uLong adler32_z(
        uLong adler,
        const Bytef* buf,
        z_size_t len
    ) nogil

    uLong adler32_combine(
        uLong adler1,
        uLong adler2,
        z_off_t len2
    ) nogil

    uLong crc32(
        uLong crc,
        const Bytef* buf,
        uInt len
    ) nogil

    uLong crc32_z(
        uLong adler,
        const Bytef* buf,
        z_size_t len
    ) nogil
