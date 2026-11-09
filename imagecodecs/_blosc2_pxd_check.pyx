from libc.stdint cimport int64_t, uint8_t
from blosc2 cimport *

cdef int64_t check(blosc2_schunk* schunk, uint8_t* (*alloc)(int64_t, void*)):
    return blosc2_schunk_to_buffer_alloc(schunk, alloc, NULL)
