# imagecodecs/wavpack.pxd

# Cython declarations for the `WavPack 5.9` library.
# https://github.com/dbry/WavPack

from libc.stdint cimport int16_t, int32_t, int64_t, uint16_t, uint32_t


cdef extern from 'wavpack/wavpack.h' nogil:

    ctypedef struct RiffChunkHeader:
        char[4] ckID
        uint32_t ckSize
        char[4] formType

    ctypedef struct ChunkHeader:
        char[4] ckID
        uint32_t ckSize

    ctypedef struct WaveHeader:
        uint16_t FormatTag
        uint16_t NumChannels
        uint32_t SampleRate
        uint32_t BytesPerSecond
        uint16_t BlockAlign
        uint16_t BitsPerSample
        uint16_t cbSize
        uint16_t ValidBitsPerSample
        int32_t ChannelMask
        uint16_t SubFormat
        char[14] GUID

    ctypedef struct WavpackHeader:
        char[4] ckID
        uint32_t ckSize
        int16_t version
        unsigned char block_index_u8
        unsigned char total_samples_u8
        uint32_t total_samples
        uint32_t block_index
        uint32_t block_samples
        uint32_t flags
        uint32_t crc

    int BYTES_STORED
    int MONO_FLAG
    int HYBRID_FLAG
    int JOINT_STEREO
    int CROSS_DECORR
    int HYBRID_SHAPE
    int FLOAT_DATA
    int INT32_DATA
    int HYBRID_BITRATE
    int HYBRID_BALANCE
    int INITIAL_BLOCK
    int FINAL_BLOCK
    int FALSE_STEREO
    int NEW_SHAPING
    int MONO_DATA
    int HAS_CHECKSUM
    int DSD_FLAG

    int SHIFT_LSB
    int SHIFT_MASK
    int MAG_LSB
    int MAG_MASK
    int SRATE_LSB
    int SRATE_MASK

    int MIN_STREAM_VERS
    int MAX_STREAM_VERS

    int WAVPACK_MAX_CHANS
    int WAVPACK_MAX_CLI_CHANS

    long long MAX_WAVPACK_SAMPLES

    int ID_UNIQUE
    int ID_OPTIONAL_DATA
    int ID_ODD_SIZE
    int ID_LARGE
    int ID_DUMMY
    int ID_ENCODER_INFO
    int ID_DECORR_TERMS
    int ID_DECORR_WEIGHTS
    int ID_DECORR_SAMPLES
    int ID_ENTROPY_VARS
    int ID_HYBRID_PROFILE
    int ID_SHAPING_WEIGHTS
    int ID_FLOAT_INFO
    int ID_INT32_INFO
    int ID_WV_BITSTREAM
    int ID_WVC_BITSTREAM
    int ID_WVX_BITSTREAM
    int ID_CHANNEL_INFO
    int ID_DSD_BLOCK
    int ID_RIFF_HEADER
    int ID_RIFF_TRAILER
    int ID_ALT_HEADER
    int ID_ALT_TRAILER
    int ID_CONFIG_BLOCK
    int ID_MD5_CHECKSUM
    int ID_SAMPLE_RATE
    int ID_ALT_EXTENSION
    int ID_ALT_MD5_CHECKSUM
    int ID_NEW_CONFIG_BLOCK
    int ID_CHANNEL_IDENTITIES
    int ID_WVX_NEW_BITSTREAM
    int ID_BLOCK_CHECKSUM

    ctypedef struct WavpackConfig:
        float bitrate
        float shaping_weight
        int bits_per_sample
        int bytes_per_sample
        int qmode
        int flags
        int xmode
        int num_channels
        int float_norm_exp
        int32_t block_samples
        int32_t worker_threads
        int32_t sample_rate
        int32_t channel_mask
        unsigned char[16] md5_checksum
        unsigned char md5_read
        int num_tag_strings
        char** tag_strings

    int CONFIG_HYBRID_FLAG
    int CONFIG_JOINT_STEREO
    int CONFIG_CROSS_DECORR
    int CONFIG_HYBRID_SHAPE
    int CONFIG_FAST_FLAG
    int CONFIG_HIGH_FLAG
    int CONFIG_VERY_HIGH_FLAG
    int CONFIG_BITRATE_KBPS
    int CONFIG_SHAPE_OVERRIDE
    int CONFIG_JOINT_OVERRIDE
    int CONFIG_DYNAMIC_SHAPING
    int CONFIG_CREATE_EXE
    int CONFIG_CREATE_WVC
    int CONFIG_OPTIMIZE_WVC
    int CONFIG_COMPATIBLE_WRITE
    int CONFIG_CALC_NOISE
    int CONFIG_EXTRA_MODE
    int CONFIG_SKIP_WVX
    int CONFIG_MD5_CHECKSUM
    int CONFIG_MERGE_BLOCKS
    int CONFIG_PAIR_UNDEF_CHANS
    int CONFIG_OPTIMIZE_32BIT
    int CONFIG_OPTIMIZE_MONO

    int QMODE_BIG_ENDIAN
    int QMODE_SIGNED_BYTES
    int QMODE_UNSIGNED_WORDS
    int QMODE_REORDERED_CHANS
    int QMODE_DSD_LSB_FIRST
    int QMODE_DSD_MSB_FIRST
    int QMODE_DSD_IN_BLOCKS
    int QMODE_DSD_AUDIO
    int QMODE_ADOBE_MODE
    int QMODE_NO_STORE_WRAPPER
    int QMODE_CHANS_UNASSIGNED
    int QMODE_IGNORE_LENGTH
    int QMODE_RAW_PCM
    int QMODE_EVEN_BYTE_DEPTH

    ctypedef struct WavpackStreamReader:
        int32_t (*read_bytes)(void* id, void* data, int32_t bcount)
        uint32_t (*get_pos)(void* id)
        int (*set_pos_abs)(void* id, uint32_t pos)
        int (*set_pos_rel)(void* id, int32_t delta, int mode)
        int (*push_back_byte)(void* id, int c)
        uint32_t (*get_length)(void* id)
        int (*can_seek)(void* id)
        int32_t (*write_bytes)(void* id, void* data, int32_t bcount)

    ctypedef struct WavpackStreamReader64:
        int32_t (*read_bytes)(void* id, void* data, int32_t bcount)
        int32_t (*write_bytes)(void* id, void* data, int32_t bcount)
        int64_t (*get_pos)(void* id)
        int (*set_pos_abs)(void* id, int64_t pos)
        int (*set_pos_rel)(void* id, int64_t delta, int mode)
        int (*push_back_byte)(void* id, int c)
        int64_t (*get_length)(void* id)
        int (*can_seek)(void* id)
        int (*truncate_here)(void* id)
        int (*close)(void* id)

    ctypedef int (*WavpackBlockOutput)(void* id, void* data, int32_t bcount)

    ctypedef struct WavpackContext:
        pass

    int OPEN_WVC
    int OPEN_TAGS
    int OPEN_WRAPPER
    int OPEN_2CH_MAX
    int OPEN_NORMALIZE
    int OPEN_STREAMING
    int OPEN_EDIT_TAGS
    int OPEN_FILE_UTF8
    int OPEN_DSD_NATIVE
    int OPEN_DSD_AS_PCM
    int OPEN_ALT_TYPES
    int OPEN_NO_CHECKSUM
    int OPEN_THREADS_SHFT
    int OPEN_THREADS_MASK

    int MODE_WVC
    int MODE_LOSSLESS
    int MODE_HYBRID
    int MODE_FLOAT
    int MODE_VALID_TAG
    int MODE_HIGH
    int MODE_FAST
    int MODE_EXTRA
    int MODE_APETAG
    int MODE_SFX
    int MODE_VERY_HIGH
    int MODE_MD5
    int MODE_XMODE
    int MODE_DNS

    int WP_FORMAT_WAV
    int WP_FORMAT_W64
    int WP_FORMAT_CAF
    int WP_FORMAT_DFF
    int WP_FORMAT_DSF
    int WP_FORMAT_AIF

    WavpackContext* WavpackOpenRawDecoder(
        void* main_data,
        int32_t main_size,
        void* corr_data,
        int32_t corr_size,
        int16_t version,
        char* error,
        int flags,
        int norm_offset
    )

    WavpackContext* WavpackOpenFileInputEx64(
        WavpackStreamReader64* reader,
        void* wv_id,
        void* wvc_id,
        char* error,
        int flags,
        int norm_offset
    )

    WavpackContext* WavpackOpenFileInputEx(
        WavpackStreamReader* reader,
        void* wv_id,
        void* wvc_id,
        char* error,
        int flags,
        int norm_offset
    )

    WavpackContext* WavpackOpenFileInput(
        const char* infilename,
        char* error,
        int flags,
        int norm_offset
    )

    int WavpackGetMode(
        WavpackContext* wpc
    )

    int WavpackVerifySingleBlock(
        unsigned char* buffer,
        int verify_checksum
    )

    int WavpackGetQualifyMode(
        WavpackContext* wpc
    )

    char* WavpackGetErrorMessage(
        WavpackContext* wpc
    )

    int WavpackGetVersion(
        WavpackContext* wpc
    )

    char* WavpackGetFileExtension(
        WavpackContext* wpc
    )

    unsigned char WavpackGetFileFormat(
        WavpackContext* wpc
    )

    uint32_t WavpackGetNumSamples(
        WavpackContext* wpc
    )

    int64_t WavpackGetNumSamples64(
        WavpackContext* wpc
    )

    uint32_t WavpackGetNumSamplesInFrame(
        WavpackContext* wpc
    )

    uint32_t WavpackGetSampleIndex(
        WavpackContext* wpc
    )

    int64_t WavpackGetSampleIndex64(
        WavpackContext* wpc
    )

    int WavpackGetNumErrors(
        WavpackContext* wpc
    )

    int WavpackLossyBlocks(
        WavpackContext* wpc
    )

    uint32_t WavpackGetSampleRate(
        WavpackContext* wpc
    )

    uint32_t WavpackGetNativeSampleRate(
        WavpackContext* wpc
    )

    int WavpackGetBitsPerSample(
        WavpackContext* wpc
    )

    int WavpackGetBytesPerSample(
        WavpackContext* wpc
    )

    int WavpackGetNumChannels(
        WavpackContext* wpc
    )

    int WavpackGetChannelMask(
        WavpackContext* wpc
    )

    int WavpackGetReducedChannels(
        WavpackContext* wpc
    )

    int WavpackGetFloatNormExp(
        WavpackContext* wpc
    )

    int WavpackGetMD5Sum(
        WavpackContext* wpc,
        unsigned char[16] data
    )

    void WavpackGetChannelIdentities(
        WavpackContext* wpc,
        unsigned char* identities
    )

    uint32_t WavpackGetChannelLayout(
        WavpackContext* wpc,
        unsigned char* reorder
    )

    uint32_t WavpackGetWrapperBytes(
        WavpackContext* wpc
    )

    unsigned char* WavpackGetWrapperData(
        WavpackContext* wpc
    )

    void WavpackFreeWrapper(
        WavpackContext* wpc
    )

    void WavpackSeekTrailingWrapper(
        WavpackContext* wpc
    )

    double WavpackGetProgress(
        WavpackContext* wpc
    )

    uint32_t WavpackGetFileSize(
        WavpackContext* wpc
    )

    int64_t WavpackGetFileSize64(
        WavpackContext* wpc
    )

    double WavpackGetRatio(
        WavpackContext* wpc
    )

    double WavpackGetAverageBitrate(
        WavpackContext* wpc,
        int count_wvc
    )

    double WavpackGetInstantBitrate(
        WavpackContext* wpc
    )

    int WavpackSeekSample(
        WavpackContext* wpc,
        uint32_t sample
    )

    int WavpackSeekSample64(
        WavpackContext* wpc,
        int64_t sample
    )

    uint32_t WavpackUnpackSamples(
        WavpackContext* wpc,
        int32_t* buffer,
        uint32_t samples
    )

    WavpackContext* WavpackCloseFile(
        WavpackContext* wpc
    )

    int WavpackGetNumTagItems(
        WavpackContext* wpc
    )

    int WavpackGetTagItem(
        WavpackContext* wpc,
        const char* item,
        char* value,
        int size
    )

    int WavpackGetTagItemIndexed(
        WavpackContext* wpc,
        int index,
        char* item,
        int size
    )

    int WavpackGetNumBinaryTagItems(
        WavpackContext* wpc
    )

    int WavpackGetBinaryTagItem(
        WavpackContext* wpc,
        const char* item,
        char* value,
        int size
    )

    int WavpackGetBinaryTagItemIndexed(
        WavpackContext* wpc,
        int index,
        char* item,
        int size
    )

    WavpackContext* WavpackOpenFileOutput(
        WavpackBlockOutput blockout,
        void* wv_id,
        void* wvc_id
    )

    void WavpackSetFileInformation(
        WavpackContext* wpc,
        char* file_extension,
        unsigned char file_format
    )

    int WavpackSetConfiguration(
        WavpackContext* wpc,
        WavpackConfig* config,
        uint32_t total_samples
    )

    int WavpackSetConfiguration64(
        WavpackContext* wpc,
        WavpackConfig* config,
        int64_t total_samples,
        const unsigned char* chan_ids
    )

    int WavpackSetChannelLayout(
        WavpackContext* wpc,
        uint32_t layout_tag,
        const unsigned char* reorder
    )

    int WavpackAddWrapper(
        WavpackContext* wpc,
        void* data,
        uint32_t bcount
    )

    int WavpackStoreMD5Sum(
        WavpackContext* wpc,
        unsigned char[16] data
    )

    int WavpackPackInit(
        WavpackContext* wpc
    )

    int WavpackPackSamples(
        WavpackContext* wpc,
        int32_t* sample_buffer,
        uint32_t sample_count
    )

    int WavpackFlushSamples(
        WavpackContext* wpc
    )

    void WavpackUpdateNumSamples(
        WavpackContext* wpc,
        void* first_block
    )

    void* WavpackGetWrapperLocation(
        void* first_block,
        uint32_t* size
    )

    double WavpackGetEncodedNoise(
        WavpackContext* wpc,
        double* peak
    )

    int WavpackAppendTagItem(
        WavpackContext* wpc,
        const char* item,
        const char* value,
        int vsize
    )

    int WavpackAppendBinaryTagItem(
        WavpackContext* wpc,
        const char* item,
        const char* value,
        int vsize
    )

    int WavpackDeleteTagItem(
        WavpackContext* wpc,
        const char* item
    )

    int WavpackWriteTag(
        WavpackContext* wpc
    )

    void WavpackFloatNormalize(
        int32_t* values,
        int32_t num_values,
        int delta_exp
    )

    void WavpackLittleEndianToNative(
        void* data,
        char* format
    )

    void WavpackNativeToLittleEndian(
        void* data,
        char* format
    )

    void WavpackBigEndianToNative(
        void* data,
        char* format
    )

    void WavpackNativeToBigEndian(
        void* data,
        char* format
    )

    uint32_t WavpackGetLibraryVersion()

    const char* WavpackGetLibraryVersionString()
