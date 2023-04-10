import numpy as np
from numpy.typing import NDArray
import einops as ei


def from_path_matrix_to_event_indexes(path_matrix: NDArray) -> list:
    """Convert path matrix to list of event indexes."""
    n_samples = path_matrix.shape[0] - 1
    event_indexes = list()
    ind = n_samples
    while ind > 0:
        event_indexes.append(ind)
        ind = path_matrix[ind]
    event_indexes.sort()
    return event_indexes[:-1]


def get_temporal_support(activations: NDArray) -> NDArray:
    """Return the temporal support of the activations, as an array of indexes."""
    temporal_support, _ = np.nonzero(activations)
    return np.unique(temporal_support)


def get_reconstruction(activations: NDArray, dictionary: NDArray) -> NDArray:
    """Return the reconstructed signal (convolution of activations and atoms)."""
    reconstruction = ei.rearrange(
        [
            np.convolve(activations_1d, atom, mode="full")
            for (activations_1d, atom) in zip(activations.T, dictionary)
        ],
        "n_atoms n_corrs -> n_atoms n_corrs",
    ).sum(axis=0)
    return reconstruction
