import numpy as np

from lisaorbits import Orbits
from segwo.response import compute_strain2link
from segwo.cov import construct_mixing_from_pytdi, compose_mixings, construct_covariance_from_psds, project_covariance
from lisaconstants import c
from lisaconstants.indexing import LINKS

from pytdi.core import LISATDICombination
from pytdi.michelson import X2_ETA, Y2_ETA, Z2_ETA

X2_ETA: LISATDICombination
Y2_ETA: LISATDICombination
Z2_ETA: LISATDICombination

# Define a dictionary of TDI combinations mapping our noises into each of the 6
# single links (aka, the 6 eta variables)
ETA_COMBS: dict[int, LISATDICombination] = {}

# We start by defining it for eta 12:
#
# A PyTDI combination is defined as a dictionary, where the keys are the names
# of the input variables (N_tm_12, N_oms_12, etc.), and the values are lists of
# tuples. Each tuple contains a scaling coefficient and a list of delays to be
# applied on the corresponding variable.
ETA_COMBS[12] = LISATDICombination(
    {
        "N_tm_21": [(1, ["D_12"])],
        "N_tm_12": [(1, [])],
        "N_oms_12": [(1, [])],
    }
)

# Use PyTDI symmetry operations to define the other links

# We can do cyclic permuations, rotating indices 1->2->3->1
ETA_COMBS[23] = ETA_COMBS[12].rotated()
ETA_COMBS[31] = ETA_COMBS[23].rotated()

# We can do reflections along the axis going through the respective spacecraft,
# ie., exchanging indices 2<->3, 3<->1, and 1<->2
ETA_COMBS[13] = ETA_COMBS[12].reflected(1)
ETA_COMBS[21] = ETA_COMBS[23].reflected(2)
ETA_COMBS[32] = ETA_COMBS[31].reflected(3)

# Define a dictionary containing the TDI combinations mapping the noises into
# the eta variables, making sure that the keys map the measurement labels used
# in `X2_ETA`, `Y2_ETA`, and `Z2_ETA` (to allow composing them)
ETA_COMBS_SET = {f"eta_{mosa}": ETA_COMBS[mosa] for mosa in LINKS}

# Define list of noise labels (our input variables)
noise_list = [f"N_oms_{mosa}" for mosa in LINKS] + [f"N_tm_{mosa}" for mosa in LINKS]

# Define TDI combinations transforming XYZ into AET
A = (Z2_ETA - X2_ETA) / np.sqrt(2)
E = (X2_ETA - 2 * Y2_ETA + Z2_ETA) / np.sqrt(6)
T = (X2_ETA + Y2_ETA + Z2_ETA) / np.sqrt(3)
# ---------------------------------------------------------------------------
# TDI combinations
# ---------------------------------------------------------------------------
# Form the dictionary mapping the TDI combinations into the eta variables
xyz_eta_dict = {"X": X2_ETA, "Y": Y2_ETA, "Z": Z2_ETA}

# Compute and simplify the AET combinations for single links
A_ETA = (A @ xyz_eta_dict).simplified()
E_ETA = (E @ xyz_eta_dict).simplified()
T_ETA = (T @ xyz_eta_dict).simplified()

# Compute and simplify the AET combinations for noises
A_NOISE = (A_ETA @ ETA_COMBS_SET).simplified()
E_NOISE = (E_ETA @ ETA_COMBS_SET).simplified()
T_NOISE = (T_ETA @ ETA_COMBS_SET).simplified()

def compute_covariance(f, ltts, oms_ref = 15e-12, tm_ref = 3e-15):
    """
    Compute the noise covariance in the TDI variables for given frequencies and mixing matrix.
    Parameters
    ----------
    f : np.array
        Array of frequency values.
    ltts : np.array
        Light travel times for the constellation.
    Returns
    -------
    noise_cov_tdi_pytdi : np.array
        The noise covariance in the TDI variables, projected using the provided mixing matrix.
    """
    

    # The OMS noise is defined in terms of displacement (meters), which we convert
    # to fractional frequency shifts
    displ_2_ffd = 2 * np.pi * f / c
    oms = (oms_ref) ** 2 * displ_2_ffd**2 * (1 + ((2e-3) / f) ** 4)

    # The TM noise is defined in terms of acceleration (m/s^2), which we convert to
    # fractional frequency shifts
    acc_2_ffd = 1 / (2 * np.pi * f * c)
    tm = (tm_ref) ** 2 * acc_2_ffd**2 * (1 + (0.4e-3 / f) ** 2) * (1 + (f / 8e-3) ** 4)
    
    # Construct the overall noise covariance
    noise_cov = construct_covariance_from_psds([oms] * 6 + [tm] * 6)

    # Construct the mixing matrix for the noise covariance
    # Compute the corresponding mixing matrix
    noise2aet = construct_mixing_from_pytdi(f, noise_list, [A_NOISE, E_NOISE, T_NOISE], ltts)
    
    noise_cov_tdi_pytdi = project_covariance(noise_cov, noise2aet)

    return noise_cov_tdi_pytdi

def compute_strain2x(frequencies, betas, lambs, ltts, positions):
    """
    Compute the strain2x matrix for given frequencies, sky locations, and orbits.

    Parameters
    ----------
    frequencies : np.array
        Array of frequency values.
    betas : np.array
        Array of beta (latitude) values for sky locations.
    lambs : np.array
        Array of lambda (longitude) values for sky locations.
    ltts : np.array
        Light travel times for the constellation.
    positions : np.array
        Spacecraft positions for the constellation.
    Returns
    -------
    strain2x : np.array
        The composed mixing matrix to go directly from strain to TDI variables.
    """
    # Compute the complex signal response for the given frequencies and sky locations
    strain2link = compute_strain2link(frequencies, betas, lambs, ltts, positions, method="baghi+23")
    # Shape: times, frequencies, pixels, links, polarizations
    # Ie., it's the mixing matrix to go from h+/hx in terms of strain to single link
    # response in terms of fractional frequency deviation. See segwo documentation for details.
    strain2link.shape

    # Define eta variables for the links
    eta_list = [f"eta_{mosa}" for mosa in LINKS]

    # Construct the mixing matrix from pytdi
    link2x = construct_mixing_from_pytdi(frequencies, eta_list, [A, E, T], ltts)

    # Add a new axis for each pixel
    link2x = link2x[:, :, np.newaxis, :, :]

    # Compose the mixing matrices to go directly from strain to TDI variables
    strain2x = compose_mixings([strain2link, link2x])

    return strain2x


from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from lisaorbits import Orbits  # noqa: F401

# Adjust to your project layout, e.g. from lisagwresponse.jaxgb import JaxGB
from jaxgb.jaxgb import JaxGB
from jaxgb.tdi import to_tdi_combination, to_tdi_generation
from functools import partial

# This is a helper function to subtract the GB templates from the data
def _add_to(tdi_one, tdi_sum, kmin_arr, kmax, n):
    def _update_fpoint(idx, val):
        _mask = ((idx + kmin_arr) >= 0) & ((idx + kmin_arr) < kmax)
        return jax.lax.dynamic_update_index_in_dim(
            val,
            val[:, idx + kmin_arr] + _mask * tdi_sum[:, idx],
            idx + kmin_arr,
            axis=1,
        )


    return jax.lax.fori_loop(0, n, _update_fpoint, tdi_one)

vadd = jax.vmap(_add_to, in_axes=(0,0,0,None,None))

class MyJaxGB(JaxGB):
    """Galactic binary response with the velocity-corrected two-exponential kernel.

    Drop-in replacement for :class:`JaxGB`.  Only
    :meth:`_construct_slow_part` is overridden; all public methods
    (:meth:`get_link_responses`, :meth:`get_tdi`, …) are inherited
    unchanged.

    Parameters
    ----------
    orbits
        Orbits instance (lisaorbits).
    t_obs
        Observation duration (s).
    t0
        Start time (s).
    n
        Number of slow-response evaluation points (even integer).
    """

    def __init__(
        self,
        orbits: "Orbits",
        *,
        t_obs: float = 6.2914560e7,
        t0: float = 0.0,
        n: int = 128,
        tdi_generation=2.0,
        tdi_combination="AET"
    ) -> None:
        self.kwarg_get_tdi = dict(tdi_generation=tdi_generation, tdi_combination=tdi_combination)
        
        super().__init__(orbits, t_obs=t_obs, t0=t0, n=n)

    def get_data_minus_template(
        self,
        params: jax.Array,
        data: jax.Array,
        kmin: int | None = None,
        kmax: int | None = None,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Compute data minus template
        
        Parameters
        ----------
        params : jax.Array
            Physical parameters (8,) or (n_sources, 8)
        kmin : int
            Minimum frequency index
        kmax : int
            Maximum frequency index
        t_init : float, optional
            Initialization time
            
        Returns
        -------
        loglike : jnp.ndarray
            Log-likelihood value
        """
        kmins = self.get_kmin(f0=params[:, 0])

        # Compute residual: data - sum(templates)
        _x, _y, _z = self.get_tdi(params, **self.kwarg_get_tdi)
        templates = -jnp.stack([_x, _y, _z],axis=1)
        
        return vadd(data, templates, kmins-kmin, kmax, self.n)

    
import matplotlib.pyplot as plt
import corner

COL1 = 5
COL2 = 5

def get_normalisation_weight(len_current_samples, len_of_longest_samples):
    return np.ones(len_current_samples) * (len_of_longest_samples / len_current_samples)

def overlaid_corner(samples_list, sample_labels, name_save=None, corn_kw=None, title=None, weights=None):
    import matplotlib.lines as mlines

    n = len(samples_list)
    _, ndim = samples_list[0].shape
    max_len = max(len(s) for s in samples_list)
    colors = ["C0", "C1"]

    plot_range = []
    for d in range(ndim):
        lo = min(np.min(samples_list[i][:, d]) for i in range(n))
        hi = max(np.max(samples_list[i][:, d]) for i in range(n))
        plot_range.append([lo, hi])

    corn_kw = {} if corn_kw is None else dict(corn_kw)
    corn_kw.update(range=plot_range)

    if weights is None:
        weights = [get_normalisation_weight(len(samples_list[i]), max_len) for i in range(n)]
    else:
        weights = [
            get_normalisation_weight(len(samples_list[i]), max_len) * weights[i]
            for i in range(n)
        ]
    fig = plt.figure(figsize=(COL1*5, COL1*5))
    fig = corner.corner(
        samples_list[0],
        fig=fig,
        color=colors[0],
        weights=weights[0],
        **corn_kw,
    )
    axes = np.array(fig.axes).reshape((ndim, ndim))
    maxy_all = [[axes[i, i].get_ybound()[-1] for i in range(ndim)]]

    for i in range(1, n):
        fig = corner.corner(samples_list[i], fig=fig, color=colors[i], weights=weights[i], **corn_kw)
        axes = np.array(fig.axes).reshape((ndim, ndim))
        maxy_all.append([axes[j, j].get_ybound()[-1] for j in range(ndim)])

    maxy_all = np.asarray(maxy_all)
    axes = np.array(fig.axes).reshape((ndim, ndim))
    for i in range(ndim):
        axes[i, i].set_ylim(0.0, np.max(maxy_all[:, i]))

    axes[0,-1].legend(
        handles=[mlines.Line2D([], [], color=colors[i], label=sample_labels[i]) for i in range(n)],
        frameon=False,
        loc="upper right",
        title=title,
        fontsize=8,
        title_fontsize=8,
    )

    plt.subplots_adjust(hspace=0.15, wspace=0.15)

    if name_save is not None:
        plt.savefig(name_save + ".png", pad_inches=0.2, bbox_inches="tight", dpi=300)
    else:
        plt.show()
