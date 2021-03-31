# imagecodecs/_shared.pxd
# cython: language_level = 3

# Shared function definitions for imagecodecs extensions.

from libc.stdint cimport uint8_t

cdef const uint8_t[::1] _readable_input(data)

cdef const uint8_t[::1] _writable_input(data)

cdef const uint8_t[::1] _inplace_input(data)

cdef _parse_output(out, ssize_t outsize=*, outgiven=*, outtype=*)

cdef object _create_output(object out, ssize_t size, const char* string=*)

cdef object _return_output(object out, ssize_t size, ssize_t used, outgiven)

cdef _create_array(out, shape, dtype, strides=*, zero=*)

cdef _default_value(value, default, smallest, largest)
