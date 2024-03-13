# imagecodecs/_shared.pxd
# cython: language_level = 3

# Shared function definitions for imagecodecs extensions.

from libc.stdint cimport uint8_t

cdef const uint8_t[::1] _readable_input(object data)

cdef const uint8_t[::1] _writable_input(object data)

cdef const uint8_t[::1] _inplace_input(object data)

cdef _parse_output(out, ssize_t outsize=*, outgiven=*, outtype=*)

cdef _create_output(out, ssize_t size, const char* string=*)

cdef _return_output(out, ssize_t size, ssize_t used, outgiven)

cdef _create_array(out, shape, dtype, strides=*, zero=*, contig=*)

cdef tuple _squeeze_shape(tuple shape, ssize_t ndim)

cdef _default_value(value, default, smallest, largest)

cdef _default_threads(numthreads)
