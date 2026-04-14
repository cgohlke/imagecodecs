# imagecodecs/openjph.pxd
# distutils: language = c++

# Cython declarations for the `openjph 0.27.0` library.
# https://github.com/aous72/OpenJPH

from libc.stdint cimport (
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)
from libc.stdio cimport FILE
from libcpp cimport bool


cdef extern from 'openjph/ojph_version.h' nogil:

    int OPENJPH_VERSION_MAJOR
    int OPENJPH_VERSION_MINOR
    int OPENJPH_VERSION_PATCH

cdef extern from 'openjph/ojph_defs.h' namespace 'ojph' nogil:

    ctypedef uint8_t ui8
    ctypedef int8_t si8
    ctypedef uint16_t ui16
    ctypedef int16_t si16
    ctypedef uint32_t ui32
    ctypedef int32_t si32
    ctypedef uint64_t ui64
    ctypedef int64_t si64

    int NUM_FRAC_BITS

cdef extern from 'openjph/ojph_base.h' namespace 'ojph' nogil:

    cdef cppclass size:
        size(ui32 w, ui32 h)
        ui32 w
        ui32 h
        ui64 area()

    cdef cppclass point:
        point(ui32 x, ui32 y)
        ui32 x
        ui32 y

    struct rect:
        point org
        size siz

cdef extern from 'openjph/ojph_message.h' namespace 'ojph' nogil:

    cdef enum OJPH_MSG_LEVEL:
        OJPH_MSG_ALL_MSG
        OJPH_MSG_INFO
        OJPH_MSG_WARN
        OJPH_MSG_ERROR
        OJPH_MSG_NO_MSG

    void set_message_level(OJPH_MSG_LEVEL level) except +

    cdef cppclass message_base:
        pass
        # virtual void operator() (
        #     int warn_code,
        #     const char* file_name,
        #     int line_num,
        #     const char* fmt, ...
        # ) except +

    cdef cppclass message_info(message_base):
        pass

    void set_info_stream(FILE* s) except +
    void configure_info(message_info* info) except +
    message_info* get_info() except +

    cdef cppclass message_warning(message_base):
        pass

    void set_warning_stream(FILE* s) except +
    void configure_warning(message_warning* warn) except +
    message_warning* get_warning() except +

    cdef cppclass message_error(message_base):
        pass

    void set_error_stream(FILE* s) except +
    void configure_error(message_error* error) except +
    message_error* get_error() except +

cdef extern from 'openjph/ojph_file.h' namespace 'ojph' nogil:

    cdef enum:
        OJPH_SEEK_SET 'ojph::outfile_base::OJPH_SEEK_SET'
        OJPH_SEEK_CUR 'ojph::outfile_base::OJPH_SEEK_CUR'
        OJPH_SEEK_END 'ojph::outfile_base::OJPH_SEEK_END'

    cdef cppclass outfile_base:
        pass

    cdef cppclass mem_outfile(outfile_base):
        mem_outfile() except +
        void open() except +
        void open(size_t initial_size, bool clear_mem) except +
        size_t write(const void* ptr, size_t size) except +
        si64 tell() except +
        int seek(si64 offset, int) except +
        void close() except +
        const ui8* get_data() except +
        # const ui8* get_data() const
        size_t get_used_size() const
        size_t get_buf_size() const
        void expand_storage(size_t new_size, bool clear_all) except +

    cdef cppclass infile_base:
        pass

    cdef cppclass mem_infile(infile_base):
        mem_infile() except +
        void open(const ui8* data, size_t size) except +
        size_t read(void* ptr, size_t size) except +
        int seek(si64 offset, int) except +
        si64 tell() except +
        bool eof() except +
        void close() except +

cdef extern from 'openjph/ojph_mem.h' namespace 'ojph' nogil:

    void* ojph_aligned_malloc(size_t alignment, size_t size)
    void ojph_aligned_free(void* pointer)

    # TODO: is there a better way to define unnanmed enum in class?
    cdef enum:
        LFT_UNDEFINED 'ojph::line_buf::LFT_UNDEFINED'
        LFT_BYTE 'ojph::line_buf::LFT_BYTE'
        LFT_16BIT 'ojph::line_buf::LFT_16BIT'
        LFT_32BIT 'ojph::line_buf::LFT_32BIT'
        LFT_64BIT 'ojph::line_buf::LFT_64BIT'
        LFT_INTEGER 'ojph::line_buf::LFT_INTEGER'
        LFT_SIZE_MASK 'ojph::line_buf::LFT_SIZE_MASK'

    cdef cppclass line_buf:
        line_buf() except +
        size_t size
        ui32 pre_size
        ui32 flags
        # ctypedef struct union:
        si32* i32
        si64* i64
        float* f32
        void* p

cdef extern from 'openjph/ojph_params.h' nogil:

    cdef cppclass param_siz:
        param_siz(param_siz* p) except +

        void set_image_extent(point extent) except +
        void set_tile_size(size s) except +
        void set_image_offset(point offset) except +
        void set_tile_offset(point offset) except +
        void set_num_components(ui32 num_comps) except +
        void set_component(
            ui32 comp_num,
            const point& downsampling,
            ui32 bit_depth,
            bool is_signed
        ) except +

        point get_image_extent() const
        point get_image_offset() const
        size get_tile_size() const
        point get_tile_offset() const
        ui32 get_num_components() const
        ui32 get_bit_depth(ui32 comp_num) const
        bool is_signed(ui32 comp_num) const
        point get_downsampling(ui32 comp_num) const
        ui32 get_recon_width(ui32 comp_num) const
        ui32 get_recon_height(ui32 comp_num) const

    cdef cppclass param_coc:
        param_coc(param_cod* p) except +

        void set_num_decomposition(ui32 num_decompositions) except +
        void set_block_dims(ui32 width, ui32 height) except +
        void set_precinct_size(int num_levels, size* precinct_size) except +
        void set_reversible(bool reversible) except +

        ui32 get_num_decompositions() const
        size get_block_dims() const
        size get_log_block_dims() const
        bool is_reversible() const
        size get_precinct_size(ui32 level_num) const
        size get_log_precinct_size(ui32 level_num) const
        bool get_block_vertical_causality() const

    cdef cppclass param_cod:
        param_cod(param_cod* p) except +

        void set_num_decomposition(ui32 num_decompositions) except +
        void set_block_dims(ui32 width, ui32 height) except +
        void set_precinct_size(int num_levels, size* precinct_size) except +
        void set_progression_order(const char* name) except +
        void set_color_transform(bool color_transform) except +
        void set_reversible(bool reversible) except +

        param_coc get_coc(ui32 component_idx) const
        ui32 get_num_decompositions() const
        size get_block_dims() const
        size get_log_block_dims() const
        bool is_reversible() const
        size get_precinct_size(ui32 level_num) const
        size get_log_precinct_size(ui32 level_num) const
        int get_progression_order() const
        const char* get_progression_order_as_string() const
        int get_num_layers() const
        bool is_using_color_transform() const
        bool packets_may_use_sop() const
        bool packets_use_eph() const
        bool get_block_vertical_causality() const

    cdef cppclass param_qcd:
        param_qcd(param_qcd* p) except +

        void set_irrev_quant(float delta) except +
        void set_irrev_quant(ui32 comp_idx, float delta) except +

    cdef cppclass param_nlt:
        enum class special_comp_num:
            ALL_COMPS

        enum class nonlinearity:
            OJPH_NLT_NO_NLT
            OJPH_NLT_GAMMA_STYLE_NLT
            OJPH_NLT_LUT_STYLE_NLT
            OJPH_NLT_BINARY_COMPLEMENT_NLT
            OJPH_NLT_UNDEFINED

        # ui16 special_comp_num
        # ui8 nonlinearity
        param_nlt(param_nlt* p) except +
        void set_nonlinear_transform(
            ui32 comp_num,
            ui8 nl_type
        ) except +
        bool get_nonlinear_transform(
            ui32 comp_num,
            ui8& bit_depth,
            bool& is_signed,
            ui8& nl_type
        ) except +

    cdef cppclass comment_exchange:
        comment_exchange() except +
        void set_string(const char* str) except +
        void set_data(const char* data, ui16 len) except +

cdef extern from 'openjph/ojph_codestream.h' namespace 'ojph' nogil:

    cdef cppclass codestream:
        codestream() except +
        void restart() except +
        void set_planar(bool planar) except +
        void set_profile(const char* s) except +
        void set_tilepart_divisions(
            bool at_resolutions,
            bool at_components
        ) except +
        bool is_tilepart_division_at_resolutions() except +
        bool is_tilepart_division_at_components() except +
        void request_tlm_marker(bool needed)except +
        bool is_tlm_requested() except +
        void write_headers(
            outfile_base* file,
            const comment_exchange* comments,
            ui32 num_comments
        ) except +
        line_buf* exchange(
            line_buf* line,
            ui32& next_component
        ) except +
        void flush() except +

        void enable_resilience() except +
        void read_headers(infile_base* file) except +
        void restrict_input_resolution(
            ui32 skipped_res_for_data,
            ui32 skipped_res_for_recon
        ) except +
        void create() except +
        line_buf* pull(ui32 &comp_num) except +
        void close() except +

        param_siz access_siz()
        param_cod access_cod()
        param_qcd access_qcd()
        param_nlt access_nlt()
        bool is_planar() const
