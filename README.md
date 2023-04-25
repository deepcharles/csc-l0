# Sparse convolutional sparse coding with $\ell_0$ constraint

## Install

```bash
python -m pip install .
```

## Run on a step detection task

```python
from dicodile.data.gait import get_gait_data
from alphacsc.init_dict import init_dictionary
import csc
import einops as ei
import matplotlib.pyplot as plt


def fig_ax(figsize=(20, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xmargin(0)
    return fig, ax


# load data
trial = get_gait_data(subject=6, trial=1)
signal = trial["data"][["RAX", "RAY", "RAZ"]].pow(2).sum(axis=1).to_numpy().flatten()  # shape (n_samples,)


# set dictionary size
n_atoms = 5

# set individual atom (patch) size.
n_times_atom = 70

dictionary = init_dictionary(
    signal[None, None, :],
    n_atoms=n_atoms,
    n_times_atom=n_times_atom,
    rank1=False,
    window=True,
    D_init="chunk",
    random_state=60,
).squeeze()  # shape (n_atoms, n_times_atom)


# Alternate between sparse coding and dictionary learning
n_iter = 10
penalty = 1e-3
for _ in range(n_iter):
    activations = csc.update_z(
        signal=signal,
        dictionary=dictionary,
        penalty=penalty,
        constraint_str=csc.NO_CONSTRAINT,
    )
    dictionary = alphacsc.update_d.update_d(
        X=signal[None, :],
        Z=ei.rearrange(activations, "n_times n_atoms -> n_atoms 1 n_times"),
        n_times_atom=n_times_atom,
    )[0]


# Show activated time indexes
fig, ax = fig_ax()
ax.plot(signal)
activations = csc.update_z(signal=signal, dictionary=dictionary, penalty=1)
activated_time_indexes = csc.get_temporal_support(activations)
for b in activated_time_indexes:
    ax.axvline(b, color="k", ls="--")

# Show reconstruction
from alphacsc.utils import construct_X_multi

Z_hat = ei.rearrange(activations, "n_times n_atoms -> 1 n_atoms n_times")
reconstruction = construct_X_multi(Z_hat, dictionary[:, None, :])[0][0]

fig, ax = fig_ax()
ax.plot(signal[1_000:1_500])
ax.plot(reconstruction[1_000:1_500])

```


## Test the CDL

```python

import numpy as np
from dicodile.data.gait import get_gait_data
from numpy.typing import NDArray

import csc
import alphacsc
from alphacsc.update_d import _embed, solve_unit_norm_dual


def update_d(X, Z, n_times_atom, lambd0=None, ds_init=None, debug=False,
             solver_kwargs=dict(), verbose=0):
    """Learn d's in time domain.

    Parameters
    ----------
    X : array, shape (n_trials, n_times)
        The data for sparse coding
    Z : array, shape (n_trials, n_atoms, n_times - n_times_atom + 1)
        The code for which to learn the atoms
    n_times_atom : int
        The shape of atoms.
    lambd0 : array, shape (n_atoms,) | None
        The init for lambda.
    debug : bool
        If True, check grad.
    solver_kwargs : dict
        Parameters for the solver
    verbose : int
        Verbosity level.

    Returns
    -------
    d_hat : array, shape (k, n_times_atom)
        The atom to learn from the data.
    lambd_hats : float
        The dual variables
    """
    n_trials = len(Z)
    n_atoms = Z[0].shape[1]

    if lambd0 is None:
        lambd0 = 10. * np.ones(n_atoms)

    lhs = np.zeros((n_times_atom * n_atoms, ) * 2)
    rhs = np.zeros(n_times_atom * n_atoms)
    for i in range(n_trials):

        ZZi = []
        n_times_valid = Z[i].shape[0]
        Zki = np.zeros(n_times_valid + 2*(n_times_atom - 1))
        for k in range(n_atoms):
            Zki[n_times_atom - 1:-(n_times_atom - 1)] = Z[i][:, k]
            ZZik = _embed(Zki, n_times_atom)
            # n_times_atom, n_times = ZZik.shape
            ZZi.append(ZZik)

        ZZi = np.concatenate(ZZi, axis=0)
        lhs += np.dot(ZZi, ZZi.T)
        rhs += np.dot(ZZi, X[i])

    factr = solver_kwargs.get('factr', 1e7)  # default value
    d_hat, lambd_hats = solve_unit_norm_dual(lhs, rhs, lambd0=lambd0,
                                             factr=factr, debug=debug,
                                             lhs_is_toeplitz=False)
    d_hat = d_hat.reshape(n_atoms, n_times_atom)[:, ::-1]
    return d_hat, lambd_hats


# load data
trial = get_gait_data(subject=6, trial=1)
signal = trial["data"][["RAX", "RAY", "RAZ"]].pow(2).sum(axis=1).to_numpy().flatten()  # shape (n_samples,)
X = [signal]

# init dictionary
n_atoms, n_times_atoms = 5, 70
dictionary = np.random.randn(n_atoms, n_times_atoms)
dictionary /= np.linalg.norm(dictionary, axis=1).reshape(-1, 1)

# alternate between csc and cdl
n_iter = 10
penalty = 1
lambd_hats = None
for k_iter in range(n_iter):
    print(k_iter, end="\t")
    print("csc", end="...")
    activations = csc.update_z(
        X=[signal],
        dictionary=dictionary,
        penalty=penalty,
        constraint_str=csc.NO_CONSTRAINT,
    )
    print("ok", end=" ")
    print("cdl", end="...")
    dictionary, lambd_hats = update_d(
        X=[signal],
        Z=activations,
        n_times_atom=n_times_atoms,
        lambd0=lambd_hats
    )
    print("ok")


```
