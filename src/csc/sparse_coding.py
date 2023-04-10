from typing import Optional

import einops as ei
import numexpr as ne
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from scipy.fft import dct

from .get_path_matrix import get_path_matrix_cy
from .utils import from_path_matrix_to_event_indexes

NO_CONSTRAINT = "NoConstraint"
AT_MOST_ONE_ACTIVATION = "AtMostOneActivation"
AT_MOST_R_ACTIVATIONS = "AtMostRActivations"
LIST_OF_CONSTRAINTS = [NO_CONSTRAINT, AT_MOST_ONE_ACTIVATION, AT_MOST_R_ACTIVATIONS]


def update_z(
    signal: NDArray,
    dictionary: NDArray,
    penalty: float = 1.0,
    constraint_str: str = "NoConstraint",
    n_activations: Optional[int] = None,
) -> NDArray:
    err_msg = f"Constraint '{constraint_str}' not in {LIST_OF_CONSTRAINTS}."
    assert constraint_str in LIST_OF_CONSTRAINTS, err_msg

    if constraint_str == AT_MOST_R_ACTIVATIONS:
        err_msg = "Please provide a number of activations (`n_activations`)."
        assert n_activations is not None, err_msg

    err_msg = f"Signal must 1D, not {signal.shape}."
    assert signal.ndim == 1 or signal.shape[1] == 1, err_msg

    n_samples = signal.shape[0]
    (n_atoms, atom_length) = dictionary.shape

    if constraint_str == AT_MOST_ONE_ACTIVATION:
        # Comput cost vector from the correlations
        correlation_vec = ei.rearrange(
            [np.correlate(signal.flatten(), atom) for atom in dictionary],
            "n_atoms n_corrs -> n_atoms n_corrs",
        )
        cost_vec = penalty - ei.reduce(
            ne.evaluate("correlation_vec**2"), "n_atoms n_corrs->n_corrs", "max"
        ).astype(np.float32)

        # find the best temporal support for the activations
        path_vec = get_path_matrix_cy(cost_vec=cost_vec, atom_length=atom_length)
        path_vec = np.asarray(path_vec)
        activated_time_indexes = np.array(
            from_path_matrix_to_event_indexes(path_vec), dtype=int
        )

        # from the temporal support, retrieve the activation values
        atom_axis = 0
        activated_atom_indexes = np.argmax(
            abs(correlation_vec)[:, activated_time_indexes], axis=atom_axis
        )
        activations = np.zeros((n_samples - atom_length + 1, n_atoms))
        activated_indexes = np.s_[activated_time_indexes, activated_atom_indexes]
        activations[activated_indexes] = correlation_vec.T[activated_indexes]

    elif constraint_str == NO_CONSTRAINT:
        signal_windowed = sliding_window_view(signal, window_shape=atom_length)
        lstsq_solution, residuals, *_ = np.linalg.lstsq(a=dictionary.T, b=signal_windowed.T, rcond=None)
        norm_vec = np.linalg.norm(signal_windowed, axis=1)
        cost_vec = ne.evaluate("penalty + residuals-norm_vec**2").astype(np.float32)

        # find the best temporal support for the activations
        path_vec = get_path_matrix_cy(cost_vec=cost_vec, atom_length=atom_length)
        path_vec = np.asarray(path_vec)
        activated_time_indexes = np.array(
            from_path_matrix_to_event_indexes(path_vec), dtype=int
        )

        # from the temporal support, retrieve the activation values
        activations = np.zeros((n_samples - atom_length + 1, n_atoms))
        activations[activated_time_indexes] = lstsq_solution.T[activated_time_indexes]

    elif constraint_str == AT_MOST_R_ACTIVATIONS:
        raise NotImplementedError

    return activations


def update_z_dct(
    signal: NDArray,
    atom_length: int,
    penalty: float = 1,
    n_activations: int = 10,
) -> NDArray:
    err_msg = f"Signal must 1D, not {signal.shape}."
    assert signal.ndim == 1 or signal.shape[1] == 1, err_msg

    n_samples = signal.shape[0]

    # compute cost vector from the DCT transform
    windowed = sliding_window_view(signal, window_shape=atom_length)
    dct_transform = dct(x=windowed, axis=1, type=2, norm="ortho")
    dct_transform_squared = ne.evaluate("dct_transform**2")
    dct_power_spectra_partitionned = np.partition(
        dct_transform_squared, kth=-n_activations, axis=1
    )
    cost_vec = penalty - dct_power_spectra_partitionned[:, :-n_activations].sum(axis=1)

    # find the best temporal support for the activations
    path_vec = get_path_matrix_cy(cost_vec, atom_length=atom_length)
    path_vec = np.asarray(path_vec)
    activated_time_indexes = np.array(
        from_path_matrix_to_event_indexes(path_vec), dtype=int
    )

    # from the temporal support, retrieve the activation values
    # get minimun coeff value for each timestamp
    threshold_arr = dct_power_spectra_partitionned[
        activated_time_indexes, -n_activations
    ][:, None]
    # set everything below to 0
    activations = np.zeros_like(dct_transform)
    keep_mask = dct_transform_squared[activated_time_indexes] >= threshold_arr
    activations[activated_time_indexes] = np.where(
        keep_mask, dct_transform[activated_time_indexes], 0
    )
    return activations
