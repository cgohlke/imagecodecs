# imagecodecs/meshoptimizer.pxd

# Cython declarations for the `meshoptimizer 1.1` library.
# https://github.com/zeux/meshoptimizer

cdef extern from 'meshoptimizer.h' nogil:

    int MESHOPTIMIZER_VERSION

    cdef struct meshopt_Stream:
        const void* data
        size_t size
        size_t stride

    size_t meshopt_generateVertexRemap(
        unsigned int* destination,
        const unsigned int* indices,
        size_t index_count,
        const void* vertices,
        size_t vertex_count,
        size_t vertex_size
    )

    size_t meshopt_generateVertexRemapMulti(
        unsigned int* destination,
        const unsigned int* indices,
        size_t index_count,
        size_t vertex_count,
        const meshopt_Stream* streams,
        size_t stream_count
    )

    size_t meshopt_generateVertexRemapCustom(
        unsigned int* destination,
        const unsigned int* indices,
        size_t index_count,
        const float* vertex_positions,
        size_t vertex_count,
        size_t vertex_positions_stride,
        int (*callback)(
            void*,
            unsigned int,
            unsigned int
        ),
        void* context
    )

    void meshopt_remapVertexBuffer(
        void* destination,
        const void* vertices,
        size_t vertex_count,
        size_t vertex_size,
        const unsigned int* remap
    )

    void meshopt_remapIndexBuffer(
        unsigned int* destination,
        const unsigned int* indices,
        size_t index_count,
        const unsigned int* remap
    )

    void meshopt_generateShadowIndexBuffer(
        unsigned int* destination,
        const unsigned int* indices,
        size_t index_count,
        const void* vertices,
        size_t vertex_count,
        size_t vertex_size,
        size_t vertex_stride
    )

    void meshopt_generateShadowIndexBufferMulti(
        unsigned int* destination,
        const unsigned int* indices,
        size_t index_count,
        size_t vertex_count,
        const meshopt_Stream* streams,
        size_t stream_count
    )

    void meshopt_generatePositionRemap(
        unsigned int* destination,
        const float* vertex_positions,
        size_t vertex_count,
        size_t vertex_positions_stride
    )

    void meshopt_generateAdjacencyIndexBuffer(
        unsigned int* destination,
        const unsigned int* indices,
        size_t index_count,
        const float* vertex_positions,
        size_t vertex_count,
        size_t vertex_positions_stride
    )

    void meshopt_generateTessellationIndexBuffer(
        unsigned int* destination,
        const unsigned int* indices,
        size_t index_count,
        const float* vertex_positions,
        size_t vertex_count,
        size_t vertex_positions_stride
    )

    size_t meshopt_generateProvokingIndexBuffer(
        unsigned int* destination,
        unsigned int* reorder,
        const unsigned int* indices,
        size_t index_count,
        size_t vertex_count
    )

    void meshopt_optimizeVertexCache(
        unsigned int* destination,
        const unsigned int* indices,
        size_t index_count,
        size_t vertex_count
    )

    void meshopt_optimizeVertexCacheStrip(
        unsigned int* destination,
        const unsigned int* indices,
        size_t index_count,
        size_t vertex_count
    )

    void meshopt_optimizeVertexCacheFifo(
        unsigned int* destination,
        const unsigned int* indices,
        size_t index_count,
        size_t vertex_count,
        unsigned int cache_size
    )

    void meshopt_optimizeOverdraw(
        unsigned int* destination,
        const unsigned int* indices,
        size_t index_count,
        const float* vertex_positions,
        size_t vertex_count,
        size_t vertex_positions_stride,
        float threshold
    )

    size_t meshopt_optimizeVertexFetch(
        void* destination,
        unsigned int* indices,
        size_t index_count,
        const void* vertices,
        size_t vertex_count,
        size_t vertex_size
    )

    size_t meshopt_optimizeVertexFetchRemap(
        unsigned int* destination,
        const unsigned int* indices,
        size_t index_count,
        size_t vertex_count
    )

    size_t meshopt_encodeIndexBuffer(
        unsigned char* buffer,
        size_t buffer_size,
        const unsigned int* indices,
        size_t index_count
    )

    size_t meshopt_encodeIndexBufferBound(
        size_t index_count,
        size_t vertex_count
    )

    void meshopt_encodeIndexVersion(
        int version
    )

    int meshopt_decodeIndexBuffer(
        void* destination,
        size_t index_count,
        size_t index_size,
        const unsigned char* buffer,
        size_t buffer_size
    )

    int meshopt_decodeIndexVersion(
        const unsigned char* buffer,
        size_t buffer_size
    )

    size_t meshopt_encodeIndexSequence(
        unsigned char* buffer,
        size_t buffer_size,
        const unsigned int* indices,
        size_t index_count
    )

    size_t meshopt_encodeIndexSequenceBound(
        size_t index_count,
        size_t vertex_count
    )

    int meshopt_decodeIndexSequence(
        void* destination,
        size_t index_count,
        size_t index_size,
        const unsigned char* buffer,
        size_t buffer_size
    )

    size_t meshopt_encodeMeshlet(
        unsigned char* buffer,
        size_t buffer_size,
        const unsigned int* vertices,
        size_t vertex_count,
        const unsigned char* triangles,
        size_t triangle_count
    )

    size_t meshopt_encodeMeshletBound(
        size_t max_vertices,
        size_t max_triangles
    )

    int meshopt_decodeMeshlet(
        void* vertices,
        size_t vertex_count,
        size_t vertex_size,
        void* triangles,
        size_t triangle_count,
        size_t triangle_size,
        const unsigned char* buffer,
        size_t buffer_size
    )

    int meshopt_decodeMeshletRaw(
        unsigned int* vertices,
        size_t vertex_count,
        unsigned int* triangles,
        size_t triangle_count,
        const unsigned char* buffer,
        size_t buffer_size
    )

    size_t meshopt_encodeVertexBuffer(
        unsigned char* buffer,
        size_t buffer_size,
        const void* vertices,
        size_t vertex_count,
        size_t vertex_size
    )

    size_t meshopt_encodeVertexBufferBound(
        size_t vertex_count,
        size_t vertex_size
    )

    size_t meshopt_encodeVertexBufferLevel(
        unsigned char* buffer,
        size_t buffer_size,
        const void* vertices,
        size_t vertex_count,
        size_t vertex_size,
        int level,
        int version
    )

    void meshopt_encodeVertexVersion(
        int version
    )

    int meshopt_decodeVertexBuffer(
        void* destination,
        size_t vertex_count,
        size_t vertex_size,
        const unsigned char* buffer,
        size_t buffer_size
    )

    int meshopt_decodeVertexVersion(
        const unsigned char* buffer,
        size_t buffer_size
    )

    void meshopt_decodeFilterOct(
        void* buffer,
        size_t count,
        size_t stride
    )

    void meshopt_decodeFilterQuat(
        void* buffer,
        size_t count,
        size_t stride
    )

    void meshopt_decodeFilterExp(
        void* buffer,
        size_t count,
        size_t stride
    )

    void meshopt_decodeFilterColor(
        void* buffer,
        size_t count,
        size_t stride
    )

    cdef enum meshopt_EncodeExpMode:
        meshopt_EncodeExpSeparate
        meshopt_EncodeExpSharedVector
        meshopt_EncodeExpSharedComponent
        meshopt_EncodeExpClamped

    void meshopt_encodeFilterOct(
        void* destination,
        size_t count,
        size_t stride,
        int bits,
        const float* data
    )
    void meshopt_encodeFilterQuat(
        void* destination,
        size_t count,
        size_t stride,
        int bits,
        const float* data
    )
    void meshopt_encodeFilterExp(
        void* destination,
        size_t count,
        size_t stride,
        int bits,
        const float* data,
        meshopt_EncodeExpMode mode
    )
    void meshopt_encodeFilterColor(
        void* destination,
        size_t count,
        size_t stride,
        int bits,
        const float* data
    )

    cdef enum:
        meshopt_SimplifyLockBorder
        meshopt_SimplifySparse
        meshopt_SimplifyErrorAbsolute
        meshopt_SimplifyPrune
        meshopt_SimplifyRegularize
        meshopt_SimplifyPermissive
        meshopt_SimplifyRegularizeLight

    cdef enum:
        meshopt_SimplifyVertex_Lock
        meshopt_SimplifyVertex_Protect
        meshopt_SimplifyVertex_Priority

    size_t meshopt_simplify(
        unsigned int* destination,
        const unsigned int* indices,
        size_t index_count,
        const float* vertex_positions,
        size_t vertex_count,
        size_t vertex_positions_stride,
        size_t target_index_count,
        float target_error,
        unsigned int options,
        float* result_error
    )

    size_t meshopt_simplifyWithAttributes(
        unsigned int* destination,
        const unsigned int* indices,
        size_t index_count,
        const float* vertex_positions,
        size_t vertex_count,
        size_t vertex_positions_stride,
        const float* vertex_attributes,
        size_t vertex_attributes_stride,
        const float* attribute_weights,
        size_t attribute_count,
        const unsigned char* vertex_lock,
        size_t target_index_count,
        float target_error,
        unsigned int options,
        float* result_error
    )

    size_t meshopt_simplifyWithUpdate(
        unsigned int* indices,
        size_t index_count,
        float* vertex_positions,
        size_t vertex_count,
        size_t vertex_positions_stride,
        float* vertex_attributes,
        size_t vertex_attributes_stride,
        const float* attribute_weights,
        size_t attribute_count,
        const unsigned char* vertex_lock,
        size_t target_index_count,
        float target_error,
        unsigned int options,
        float* result_error
    )

    size_t meshopt_simplifySloppy(
        unsigned int* destination,
        const unsigned int* indices,
        size_t index_count,
        const float* vertex_positions,
        size_t vertex_count,
        size_t vertex_positions_stride,
        const unsigned char* vertex_lock,
        size_t target_index_count,
        float target_error,
        float* result_error
    )

    size_t meshopt_simplifyPrune(
        unsigned int* destination,
        const unsigned int* indices,
        size_t index_count,
        const float* vertex_positions,
        size_t vertex_count,
        size_t vertex_positions_stride,
        float target_error
    )

    size_t meshopt_simplifyPoints(
        unsigned int* destination,
        const float* vertex_positions,
        size_t vertex_count,
        size_t vertex_positions_stride,
        const float* vertex_colors,
        size_t vertex_colors_stride,
        float color_weight,
        size_t target_vertex_count
    )

    float meshopt_simplifyScale(
        const float* vertex_positions,
        size_t vertex_count,
        size_t vertex_positions_stride
    )

    size_t meshopt_stripify(
        unsigned int* destination,
        const unsigned int* indices,
        size_t index_count,
        size_t vertex_count,
        unsigned int restart_index
    )

    size_t meshopt_stripifyBound(
        size_t index_count
    )

    size_t meshopt_unstripify(
        unsigned int* destination,
        const unsigned int* indices,
        size_t index_count,
        unsigned int restart_index
    )

    size_t meshopt_unstripifyBound(
        size_t index_count
    )

    cdef struct meshopt_VertexCacheStatistics:
        unsigned int vertices_transformed
        unsigned int warps_executed
        float acmr
        float atvr

    cdef meshopt_VertexCacheStatistics meshopt_analyzeVertexCache(
        const unsigned int* indices,
        size_t index_count,
        size_t vertex_count,
        unsigned int cache_size,
        unsigned int warp_size,
        unsigned int primgroup_size
    )

    cdef struct meshopt_VertexFetchStatistics:
        unsigned int bytes_fetched
        float overfetch

    cdef meshopt_VertexFetchStatistics meshopt_analyzeVertexFetch(
        const unsigned int* indices,
        size_t index_count,
        size_t vertex_count,
        size_t vertex_size
    )

    cdef struct meshopt_OverdrawStatistics:
        unsigned int pixels_covered
        unsigned int pixels_shaded
        float overdraw

    cdef meshopt_OverdrawStatistics meshopt_analyzeOverdraw(
        const unsigned int* indices,
        size_t index_count,
        const float* vertex_positions,
        size_t vertex_count,
        size_t vertex_positions_stride
    )

    cdef struct meshopt_CoverageStatistics:
        float[3] coverage
        float extent

    cdef meshopt_CoverageStatistics meshopt_analyzeCoverage(
        const unsigned int* indices,
        size_t index_count,
        const float* vertex_positions,
        size_t vertex_count,
        size_t vertex_positions_stride
    )

    cdef struct meshopt_Meshlet:
        unsigned int vertex_offset
        unsigned int triangle_offset
        unsigned int vertex_count
        unsigned int triangle_count

    size_t meshopt_buildMeshlets(
        meshopt_Meshlet* meshlets,
        unsigned int* meshlet_vertices,
        unsigned char* meshlet_triangles,
        const unsigned int* indices,
        size_t index_count,
        const float* vertex_positions,
        size_t vertex_count,
        size_t vertex_positions_stride,
        size_t max_vertices,
        size_t max_triangles,
        float cone_weight
    )

    size_t meshopt_buildMeshletsScan(
        meshopt_Meshlet* meshlets,
        unsigned int* meshlet_vertices,
        unsigned char* meshlet_triangles,
        const unsigned int* indices,
        size_t index_count,
        size_t vertex_count,
        size_t max_vertices,
        size_t max_triangles
    )

    size_t meshopt_buildMeshletsBound(
        size_t index_count,
        size_t max_vertices,
        size_t max_triangles
    )

    size_t meshopt_buildMeshletsFlex(
        meshopt_Meshlet* meshlets,
        unsigned int* meshlet_vertices,
        unsigned char* meshlet_triangles,
        const unsigned int* indices,
        size_t index_count,
        const float* vertex_positions,
        size_t vertex_count,
        size_t vertex_positions_stride,
        size_t max_vertices,
        size_t min_triangles,
        size_t max_triangles,
        float cone_weight,
        float split_factor
    )

    size_t meshopt_buildMeshletsSpatial(
        meshopt_Meshlet* meshlets,
        unsigned int* meshlet_vertices,
        unsigned char* meshlet_triangles,
        const unsigned int* indices,
        size_t index_count,
        const float* vertex_positions,
        size_t vertex_count,
        size_t vertex_positions_stride,
        size_t max_vertices,
        size_t min_triangles,
        size_t max_triangles,
        float fill_weight
    )

    void meshopt_optimizeMeshlet(
        unsigned int* meshlet_vertices,
        unsigned char* meshlet_triangles,
        size_t triangle_count,
        size_t vertex_count
    )

    void meshopt_optimizeMeshletLevel(
        unsigned int* meshlet_vertices,
        size_t vertex_count,
        unsigned char* meshlet_triangles,
        size_t triangle_count,
        int level
    )

    cdef struct meshopt_Bounds:
        float[3] center
        float radius
        float[3] cone_apex
        float[3] cone_axis
        float cone_cutoff
        signed char[3] cone_axis_s8
        signed char cone_cutoff_s8

    cdef meshopt_Bounds meshopt_computeClusterBounds(
        const unsigned int* indices,
        size_t index_count,
        const float* vertex_positions,
        size_t vertex_count,
        size_t vertex_positions_stride
    )

    cdef meshopt_Bounds meshopt_computeMeshletBounds(
        const unsigned int* meshlet_vertices,
        const unsigned char* meshlet_triangles,
        size_t triangle_count,
        const float* vertex_positions,
        size_t vertex_count,
        size_t vertex_positions_stride
    )

    cdef meshopt_Bounds meshopt_computeSphereBounds(
        const float* positions,
        size_t count,
        size_t positions_stride,
        const float* radii,
        size_t radii_stride
    )

    size_t meshopt_extractMeshletIndices(
        unsigned int* vertices,
        unsigned char* triangles,
        const unsigned int* indices,
        size_t index_count
    )

    size_t meshopt_partitionClusters(
        unsigned int* destination,
        const unsigned int* cluster_indices,
        size_t total_index_count,
        const unsigned int* cluster_index_counts,
        size_t cluster_count,
        const float* vertex_positions,
        size_t vertex_count,
        size_t vertex_positions_stride,
        size_t target_partition_size
    )

    void meshopt_spatialSortRemap(
        unsigned int* destination,
        const float* vertex_positions,
        size_t vertex_count,
        size_t vertex_positions_stride
    )

    void meshopt_spatialSortTriangles(
        unsigned int* destination,
        const unsigned int* indices,
        size_t index_count,
        const float* vertex_positions,
        size_t vertex_count,
        size_t vertex_positions_stride
    )

    void meshopt_spatialClusterPoints(
        unsigned int* destination,
        const float* vertex_positions,
        size_t vertex_count,
        size_t vertex_positions_stride,
        size_t cluster_size
    )

    size_t meshopt_opacityMapMeasure(
        unsigned char* levels,
        unsigned int* sources,
        int* omm_indices,
        const unsigned int* indices,
        size_t index_count,
        const float* vertex_uvs,
        size_t vertex_count,
        size_t vertex_uvs_stride,
        unsigned int texture_width,
        unsigned int texture_height,
        int max_level,
        float target_edge
    )

    void meshopt_opacityMapRasterize(
        unsigned char* result,
        int level,
        int states,
        const float* uv0,
        const float* uv1,
        const float* uv2,
        const unsigned char* texture_data,
        size_t texture_stride,
        size_t texture_pitch,
        unsigned int texture_width,
        unsigned int texture_height
    )

    size_t meshopt_opacityMapEntrySize(
        int level,
        int states
    )

    size_t meshopt_opacityMapCompact(
        unsigned char* data,
        size_t data_size,
        unsigned char* levels,
        unsigned int* offsets,
        size_t omm_count,
        int* omm_indices,
        size_t triangle_count,
        int states
    )

    unsigned short meshopt_quantizeHalf(
        float v
    )

    float meshopt_quantizeFloat(
        float v,
        int N
    )

    float meshopt_dequantizeHalf(
        unsigned short h
    )

    void meshopt_setAllocator(
        void* (*allocate)(size_t),
        void (*deallocate)(void*)
    )
