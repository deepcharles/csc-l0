cimport cython
import numpy as np
cimport numpy as np

np.import_array()

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef np.int32_t[::] get_path_matrix_cy(np.float32_t[::] cost_vec, Py_ssize_t atom_length):
    cdef Py_ssize_t n_samples = cost_vec.shape[0] + atom_length - 1
    cdef Py_ssize_t end
    cdef np.float32_t sum_of_costs

    cdef np.int32_t[::] path_vec = np.zeros(n_samples+1, dtype=np.int32)
    cdef np.float32_t[::] sum_of_costs_vec = np.zeros(n_samples+1, dtype=np.float32)
    
    path_vec[0] = -1

    for end in range(atom_length, n_samples+1):
        sum_of_costs = sum_of_costs_vec[end-atom_length] + cost_vec[end-atom_length]
        if sum_of_costs < sum_of_costs_vec[end-1]:
            path_vec[end] = end-atom_length
            sum_of_costs_vec[end] = sum_of_costs
        else:
            sum_of_costs_vec[end] = sum_of_costs_vec[end-1]
            path_vec[end] = path_vec[end-1]

    return path_vec