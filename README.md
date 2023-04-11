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
