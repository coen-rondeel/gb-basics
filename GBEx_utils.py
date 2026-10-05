from pathlib import Path

import h5py
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


# ===========================================================================
# Tutorial helpers (used by GBExploration.ipynb)
# ===========================================================================
import healpy as hp
from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.state import State
from lisaconstants import GRAVITATIONAL_CONSTANT as G_NEWTON
from lisaconstants import PARSEC_METER, SIDEREALYEAR_J2000DAY, SUN_MASS
from lisaorbits import EqualArmlengthOrbits

# GW phase needs float64; must be set before any JAX array is created.
jax.config.update("jax_enable_x64", True)

RNG_SEED = 20261006
PARAM_NAMES = ["f0", "fdot", "A", "beta", "lambda", "psi", "iota", "phi0"]

CORNER_KWARGS = dict(
    bins=40,
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)),
    plot_density=True,
    plot_datapoints=False,
    fill_contours=True,
    show_titles=True,
    title_kwargs=dict(fontsize=10),
    max_n_ticks=4,
    truth_color="crimson",
    labelpad=0.2,
    label_kwargs=dict(fontsize=11),
)

# ---------------------------------------------------------------------------
# Astrophysical <-> GW observables
# ---------------------------------------------------------------------------
def frequency_GW(Porb):
    """GW frequency [Hz] from orbital period [s]."""
    return 2 / Porb


def chirp_mass(m1, m2):
    """Chirp mass in the same unit as m1, m2."""
    return (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)


def frequency_derivative_GW(f_GW, Mc):
    """GW-driven fdot [Hz/s]; Mc in kg."""
    return 96 / 5 * np.pi ** (8 / 3) * (G_NEWTON * Mc / c**3) ** (5 / 3) * f_GW ** (11 / 3)


def amplitude_GW(f_GW, Mc, dL):
    """Strain amplitude; Mc in kg, dL in m."""
    return 2 * (G_NEWTON * Mc) ** (5 / 3) * (np.pi * f_GW) ** (2 / 3) / (c**4 * dL)


def distance_from_GW(fdot_GW, f_GW, A_GW):
    """Luminosity distance [m] from a measured fdot (GW-driven), f0 and A."""
    return 5 * c / (48 * np.pi**2) * fdot_GW / (f_GW**3 * A_GW)


# ---------------------------------------------------------------------------
# Verification binaries (read from VGB.hdf5)
# ---------------------------------------------------------------------------
VGB_FILE = Path(__file__).with_name("VGB.hdf5")

_VGB_SOURCE_NAMES = {
    "HMCnc": "HM Cnc",
    "ZTFJ1539": "ZTF J1539",
    "SDSSJ0651": "SDSS J0651",
    "CDm3011223": "CD-30 11223",
    "ESCet": "ES Cet",
}


def load_vgb_catalog(path=None):
    """Structured array with the raw VGB.hdf5 records (field names = HDF5 columns)."""
    with h5py.File(VGB_FILE if path is None else path, "r") as f:
        return f["data"][:]


def _load_vgb_sources(records=None):
    """Convert VGB.hdf5 records to the source-table conventions used in this module."""
    records = load_vgb_catalog() if records is None else records
    sources = {}
    for row in records:
        raw_name = row["Name"].decode("utf-8")
        name = _VGB_SOURCE_NAMES.get(raw_name, raw_name)
        sources[name] = dict(
            kind="VGB",
            f0=float(row["Frequency"]),
            fdot=float(row["FrequencyDerivative"]),
            A=float(row["Amplitude"]),
            cosiota=np.cos(float(row["Inclination"])),
            m1=float(row["Mass1"]),
            m2=float(row["Mass2"]),
            dL_kpc=float(row["Distance"]) / 1e3,
            lam=float(row["EclipticLongitude"]),
            beta=float(row["EclipticLatitude"]),
            psi=float(row["Polarization"]) + np.pi/4,
            phi0=float(row["InitialPhase"]) + np.pi/3,
            raw_name=raw_name,
        )
        # check psi and phi0
        # print(float(row["Polarization"]), float(row["InitialPhase"]), sources[name]["psi"], sources[name]["phi0"])
    return sources


VGB_RECORDS = load_vgb_catalog()
SOURCES = _load_vgb_sources(VGB_RECORDS)
SOURCES.update({
    "Synthetic 1": dict(kind="Benchmark", f0=3.500e-3, fdot=2.00e-17, A=8.77e-23, cosiota=1.00, m1=0.35, m2=0.35, dL_kpc=2.10, lam=1.00, beta=1.40, psi=0.50, phi0=1.20),
    "Synthetic 2": dict(kind="Benchmark", f0=3.500e-3, fdot=2.00e-17, A=8.77e-23, cosiota=0.50, m1=0.35, m2=0.35, dL_kpc=2.10, lam=2.50, beta=0.05, psi=0.80, phi0=0.50),
    "ES Cet": dict(kind="VGB", f0=3.22470771e-03, fdot=-1.66e-17, A=9.51718037e-23, cosiota=1.04719755e+00, m1=0.80, m2=0.16, dL_kpc=1.78, lam=4.29491420e-01, beta=-3.54892793e-01, psi=7.85398163e-01, phi0=1.04719755e+00)
})


def get_source(name):
    """Parameter vector [f0, fdot, A, beta, lambda, psi, iota, phi0] of a reference source."""
    s = SOURCES[name]
    return np.array([s["f0"], s["fdot"], s["A"], s["beta"], s["lam"], s["psi"] % np.pi, np.arccos(s["cosiota"]), s["phi0"]])  # psi has period pi


def print_sources_table():
    print(f"{'name':<13}{'type':<10}{'f0 [mHz]':>9}{'fdot [1e-17]':>13}{'A [1e-23]':>10}{'cos(i)':>8}{'dL [kpc]':>9}{'lam':>6}{'beta':>7}")
    for k, s in SOURCES.items():
        print(f"{k:<13}{s['kind']:<10}{s['f0']*1e3:>9.3f}{s['fdot']*1e17:>13.3f}{s['A']*1e23:>10.2f}{s['cosiota']:>8.2f}{s['dL_kpc']:>9.2f}{s['lam']:>6.2f}{s['beta']:>7.2f}")


def physical_summary(name):
    """Recompute chirp mass, GW-driven fdot and distance from the tabulated astrophysical inputs."""
    s = SOURCES[name]
    Mc = chirp_mass(s["m1"], s["m2"])
    fdot_gw = frequency_derivative_GW(s["f0"], Mc * SUN_MASS)
    dL_from_Mc = s["dL_kpc"]
    A_from_Mc = amplitude_GW(s["f0"], Mc * SUN_MASS, s["dL_kpc"] * 1e3 * PARSEC_METER)
    print(f"{name}: m1={s['m1']} Msun, m2={s['m2']} Msun, dL={dL_from_Mc} kpc, f0={s['f0']*1e3:.3f} mHz")
    print(f"  chirp mass            : {Mc:.3f} Msun")
    print(f"  fdot (GW-driven)      : {fdot_gw:.3e} Hz/s   (table: {s['fdot']:.3e})")
    print(f"  amplitude from Mc, dL : {A_from_Mc:.3e}      (table: {s['A']:.3e})")
    if s["fdot"] > 0:
        d = distance_from_GW(s["fdot"], s["f0"], s["A"]) / (1e3 * PARSEC_METER)
        print(f"  dL from (fdot, f0, A) : {d:.2f} kpc        (table: {s['dL_kpc']})")
    else:
        print("  fdot < 0: not GW-driven (mass transfer), so dL cannot be obtained from fdot.")


def strain_polarizations(t, f0, fdot, A, iota, phi0):
    """Source-frame h_plus, h_cross of a circular binary (illustrative phase convention)."""
    phase = phi0 + 2 * np.pi * (f0 * t + 0.5 * fdot * t**2)
    hp_ = A * (1 + np.cos(iota) ** 2) * np.cos(phase)
    hx_ = 2 * A * np.cos(iota) * np.sin(phase)
    return hp_, hx_


# ---------------------------------------------------------------------------
# Inner products and parameter transforms
# ---------------------------------------------------------------------------
def inner_product(d1, d2, inv_psd, df):
    """Noise-weighted inner product <d1|d2> = 4 df Re sum_k d1* d2 / S_n.

    d1, d2: (..., 3, N_freq) complex; inv_psd: (3, N_freq) one-sided 1/S_n.
    """
    return 4.0 * df * jnp.sum((jnp.conj(d1) * d2 * inv_psd).real, axis=(-2, -1))


def mismatch(h1, h2, inv_psd, df):
    """1 - <h1|h2> / sqrt(<h1|h1><h2|h2>)."""
    h1h2 = inner_product(h1, h2, inv_psd, df)
    h1h1 = inner_product(h1, h1, inv_psd, df)
    h2h2 = inner_product(h2, h2, inv_psd, df)
    return float(1.0 - h1h2 / jnp.sqrt(h1h1 * h2h2))


def get_parameter_labels(include_noise=False):
    labels = [r"$f_0$ [Hz]", r"$\dot f_0$ [Hz/s]", r"$A$", r"$\beta$", r"$\lambda$", r"$\psi$", r"$\iota$", r"$\phi_0$"]
    if include_noise:
        labels += [r"$\sigma_A$", r"$\sigma_E$", r"$\sigma_T$"]
    return labels


# ---------------------------------------------------------------------------
# Simulated LISA observation of one galactic binary
# ---------------------------------------------------------------------------
class GBAnalysis:
    """Equal-armlength LISA data (AET, TDI 2) = one GB injection + one coloured-noise draw.

    A wide frequency window (n_bins + 2*pad bins) is used for scans over f0;
    the analysis window is the central `n_bins` bins around the source.
    All arrays are (3, N_freq) for the A, E, T channels.
    """

    def __init__(self, source_params, tobs=None, n_bins=128, pad=64, seed=RNG_SEED, t0=0.0):
        self.source_params = np.asarray(source_params, dtype=float)
        self.tobs = SIDEREALYEAR_J2000DAY * 24 * 3600 if tobs is None else tobs
        self.df = 1.0 / self.tobs
        self.n_bins, self.pad = n_bins, pad
        self.orbits = EqualArmlengthOrbits()
        self.jaxgb = MyJaxGB(self.orbits, t_obs=self.tobs, t0=t0, n=n_bins, tdi_generation=2.0, tdi_combination="AET")

        self.kmin = int(np.array(self.jaxgb.get_kmin(self.source_params[0])))
        self.kmax = self.kmin + n_bins
        self.kmin_w, self.n_w = self.kmin - pad, n_bins + 2 * pad
        self.window = slice(pad, pad + n_bins)
        self.freqs_w = self.df * (np.arange(self.n_w) + self.kmin_w)
        self.freqs = self.freqs_w[self.window]

        # Noise CSD averaged over time (orbits are only approximately stationary)
        self.ltts = self.orbits.compute_ltt(jnp.linspace(t0, t0 + self.tobs, 10))
        self.cov_w = np.asarray(compute_covariance(self.freqs_w, self.ltts).mean(axis=0))  # (n_w, 3, 3)
        self.psd_w = np.stack([self.cov_w[:, i, i].real for i in range(3)], axis=0)  # (3, n_w)
        self.inv_psd_w = 1.0 / self.psd_w
        self.psd, self.inv_psd = self.psd_w[:, self.window], self.inv_psd_w[:, self.window]
        # E|n_k|^2 = S_n / (2 df): makes ln L = -1/2 <r|r> the exact Gaussian likelihood
        self.sigma0 = np.sqrt(self.psd.mean(axis=1) / (2 * self.df))
        self._chol_w = np.linalg.cholesky(self.cov_w / (2 * self.df))

        self._template_wide = jax.jit(self._template_wide_impl)
        self.rng = np.random.default_rng(seed)
        self.noise_w = self.draw_noise(self.rng)
        self.signal_w = np.asarray(self._template_wide(jnp.asarray(self.source_params[None])))[0]
        self.data_w = self.signal_w + self.noise_w
        self.noise, self.signal, self.data = (x[:, self.window] for x in (self.noise_w, self.signal_w, self.data_w))
        self.sigma_empirical = np.sqrt(np.mean(np.abs(self.noise) ** 2, axis=1))

    # -- building blocks ---------------------------------------------------
    def _template_wide_impl(self, params):
        zeros = jnp.zeros((params.shape[0], 3, self.n_w), dtype=jnp.complex128)
        return -self.jaxgb.get_data_minus_template(params, zeros, self.kmin_w, self.n_w)

    def draw_noise(self, rng, size=()):
        """Coloured noise n_k = L_k z_k with L L^H = Cov/(2 df); returns (*size, 3, n_w)."""
        shape = (*size, self.n_w, 3)
        z = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)
        return np.einsum("kij,...kj->...ik", self._chol_w, z)

    def template(self, params):
        """Template (3, n_bins) in the analysis window for one parameter vector."""
        return np.asarray(self._template_wide(jnp.asarray(np.atleast_2d(params))))[0][:, self.window]

    def inner(self, d1, d2):
        return inner_product(d1, d2, self.inv_psd, self.df)

    def optimal_snr(self, h):
        return float(jnp.sqrt(self.inner(h, h)))

    def matched_filter_snr(self, d, h):
        return self.inner(d, h) / jnp.sqrt(self.inner(h, h))

    # -- SNR studies ---------------------------------------------------------
    def snr_realizations(self, n_real=500, seed=1):
        """Matched-filter SNR of the injected template over independent noise draws."""
        noise = self.draw_noise(np.random.default_rng(seed), size=(n_real,))[..., self.window]
        return np.asarray(self.matched_filter_snr(self.signal[None] + noise, self.signal))

    def snr_vs_frequency(self, offsets_bins):
        """Matched-filter and intrinsic SNR for templates with f0 shifted by `offsets_bins` * df."""
        offsets_bins = np.asarray(offsets_bins)
        if np.abs(offsets_bins).max() > self.pad:
            raise ValueError("offsets exceed the padded window")
        p = np.repeat(self.source_params[None], len(offsets_bins), axis=0)
        p[:, 0] += offsets_bins * self.df
        h = self._template_wide(jnp.asarray(p))
        mf = inner_product(self.data_w, h, self.inv_psd_w, self.df) / jnp.sqrt(inner_product(h, h, self.inv_psd_w, self.df))
        opt = jnp.sqrt(inner_product(h, h, self.inv_psd_w, self.df))
        return np.asarray(mf), np.asarray(opt)

    def snr_vs_sky(self, nside=8, chunk=128):
        """Matched-filter and intrinsic SNR maps: only (beta, lambda) change, other parameters fixed."""
        npix = hp.nside2npix(nside)
        theta, phi = hp.pix2ang(nside, np.arange(npix))
        mf, opt = np.empty(npix), np.empty(npix)
        for s in range(0, npix, chunk):
            sl = slice(s, min(s + chunk, npix))
            p = np.repeat(self.source_params[None], sl.stop - sl.start, axis=0)
            p[:, 3], p[:, 4] = np.pi / 2 - theta[sl], phi[sl]
            h = self._template_wide(jnp.asarray(p))[:, :, self.window]
            hh = jnp.sqrt(self.inner(h, h))
            mf[sl] = np.asarray(self.inner(self.data[None], h) / hh)
            opt[sl] = np.asarray(hh)
        return mf, opt

    # -- likelihood ----------------------------------------------------------
    def make_log_likelihood(self, fit_noise=False, batch_size=256):
        """Batched ln L for Eryn. params: (N, 8) or (N, 11) with sigma_A, sigma_E, sigma_T last.

        fit_noise=False: ln L = -1/2 <r|r> with the known PSD (constant normalisation dropped).
        fit_noise=True : circular complex Gaussian with Cov = diag(sigma^2), constant in frequency,
                         including the -N ln(pi sigma^2) term.
        """
        data = jnp.asarray(self.data)
        inv_psd, df, sl = jnp.asarray(self.inv_psd), self.df, self.window

        @jax.jit
        def _ll(params):
            gb = params[:, :8]
            resid = data[None] - self._template_wide_impl(gb)[:, :, sl]
            if not fit_noise:
                return -0.5 * inner_product(resid, resid, inv_psd, df)
            sigma2 = params[:, 8:11] ** 2
            power = jnp.sum(jnp.abs(resid) ** 2, axis=-1)  # (N, 3)
            return -jnp.sum(resid.shape[-1] * jnp.log(jnp.pi * sigma2) + power / sigma2, axis=-1)

        def log_likelihood(params):
            params = np.atleast_2d(np.asarray(params, dtype=float))
            out = np.empty(len(params))
            for s in range(0, len(params), batch_size):  # fixed batch shape avoids recompiling
                chunk = params[s : s + batch_size]
                m = len(chunk)
                if m < batch_size:
                    chunk = np.vstack([chunk, np.repeat(chunk[-1:], batch_size - m, axis=0)])
                out[s : s + m] = np.asarray(_ll(jnp.asarray(chunk)))[:m]
            return out

        return log_likelihood

    # -- plots ---------------------------------------------------------------
    def plot_data(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for i, ch in enumerate("AET"):
            axes[i].semilogy(self.freqs, np.abs(self.data[i]) ** 2, alpha=0.6, label="data = signal + noise")
            axes[i].semilogy(self.freqs, np.abs(self.signal[i]) ** 2, label="signal only")
            axes[i].semilogy(self.freqs, self.psd[i] / (2 * self.df), "k--", label=r"noise variance $S_n/2\Delta f$")
            axes[i].set_title(f"{ch} channel")
            axes[i].set_xlabel("Frequency [Hz]")
            axes[i].tick_params(axis="x", rotation=30)
        axes[0].set_ylabel("Power per frequency bin [strain$^2$]")
        axes[0].legend(fontsize=8)
        plt.tight_layout()
        plt.show()

    def plot_covariance_structure(self):
        """Magnitude of the A,E,T cross-spectral matrix and of its inverse at the first bin."""
        cov = self.cov_w[0]
        inv = np.linalg.inv(self.cov_w)[0]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
        for ax, mat, title in zip(axes, (cov, inv), ("CSD", "inverse CSD")):
            im = ax.matshow(np.log10(np.abs(mat) + 1e-300), cmap="viridis")
            ax.set_title(r"$\log_{10}|$" + title + r"$|$", pad=15)
            ax.set_xticks(range(3), list("AET"))
            ax.set_yticks(range(3), list("AET"))
            fig.colorbar(im, ax=ax, shrink=0.8)
        plt.show()


def plot_noise_curve(analysis, source_names=None, fmin=1e-4, fmax=1e-1, n=400):
    """Wide-band noise amplitude spectral density (A channel) with reference-source frequencies marked."""
    f = np.logspace(np.log10(fmin), np.log10(fmax), n)
    cov = np.asarray(compute_covariance(f, analysis.ltts)).mean(axis=0)
    plt.figure(figsize=(9, 5))
    plt.loglog(f, np.sqrt(np.abs(cov[:, 0, 0])), label="A channel noise (TDI 2)")
    plt.loglog(f, np.sqrt(np.abs(cov[:, 2, 2])), label="T channel noise (TDI 2)")
    for name in source_names or []:
        # plt.scatter(SOURCES[name]["f0"], SOURCES[name]["A"], marker="*", s=100, label=name)
        # plt.text(SOURCES[name]["f0"], SOURCES[name]["A"], name, rotation=90, va="bottom", ha="right", fontsize=8)
        plt.axvline(SOURCES[name]["f0"], ls=":", color="gray")
        plt.text(SOURCES[name]["f0"], plt.ylim()[1], name, rotation=90, va="top", ha="right", fontsize=8)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"$\sqrt{S_n}$ [strain/$\sqrt{\rm Hz}$]")
    plt.legend()
    plt.show()


def plot_reference_sources(names, tobs=None, n_bins=128):
    """Signal power in the A channel (frequency axis in bins around f0) for several sources."""
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.5), sharey=False)
    for ax, name in zip(np.atleast_1d(axes), names):
        a = GBAnalysis(get_source(name), tobs=tobs, n_bins=n_bins, pad=1)
        ax.semilogy(a.freqs/1e-3, np.abs(a.signal[0]) ** 2, label="A signal")
        ax.semilogy(a.freqs/1e-3, a.psd[0] / (2 * a.df), "k--", label="noise variance")
        ax.set_title(f"{name}\nf0={SOURCES[name]['f0']*1e3:.2f} mHz", fontsize=10)
        ax.set_xlabel("Frequency [mHz]")
        ax.tick_params(axis="x", rotation=30)
    np.atleast_1d(axes)[0].set_ylabel("Power per bin")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def interactive_waveform_explorer(analysis):
    """Sliders for inclination, polarisation and sky position; shows the A, E, T signal power."""
    import ipywidgets as widgets

    base = analysis.source_params.copy()

    def _plot(cos_iota=float(np.cos(base[6])), psi=float(base[5]), beta=float(base[3]), lam=float(base[4])):
        p = base.copy()
        p[3], p[4], p[5], p[6] = beta, lam, psi, np.arccos(cos_iota)
        h = analysis.template(p)
        plt.figure(figsize=(8, 4))
        for i, ch in enumerate("AET"):
            plt.semilogy(analysis.freqs/1e-3, np.abs(h[i]) ** 2 + 1e-60, label=f"{ch}")
        plt.ylim(1e-50, None)
        plt.xlabel("Frequency [mHz]")
        plt.ylabel("Signal power per bin")
        plt.legend()
        plt.xticks(rotation=30)
        plt.show()

    widgets.interact(
        _plot,
        cos_iota=widgets.FloatSlider(min=-1, max=1, step=0.05, value=float(np.cos(base[6])), description="cos(iota)"),
        psi=widgets.FloatSlider(min=0, max=2 * np.pi, step=0.1, value=float(base[5]), description="psi"),
        beta=widgets.FloatSlider(min=-np.pi / 2, max=np.pi / 2, step=0.05, value=float(base[3]), description="beta"),
        lam=widgets.FloatSlider(min=0, max=2 * np.pi, step=0.1, value=float(base[4]), description="lambda"),
    )


def animate_binary_gw(cos_iota=0.79, psi=0.0, arm_angle=0.0, n_frames=36, strain_scale=0.8):
    """Animate one orbit of a binary and the induced stretching of a ring of free masses and one arm.

    Panels: 3D orbit with the line of sight, the orbit as seen from LISA, the ring + arm
    (strain exaggerated by `strain_scale`) and the arm strain over time.
    Same convention as `strain_polarizations`: h+ ~ (1+cos^2 iota)/2 cos(2 phi), hx ~ cos(iota) sin(2 phi),
    with phi the orbital phase. `psi` rotates the polarization axes, `arm_angle` is the arm direction on the sky plane.
    """
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML

    plt.rcParams["animation.embed_limit"] = 100
    c = float(cos_iota)
    s = np.sqrt(1 - c**2)
    iota = np.arccos(c)
    phi = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    hplus, hcross = (1 + c**2) / 2 * np.cos(2 * phi), c * np.sin(2 * phi)

    p = np.array([np.cos(psi), np.sin(psi)])
    q = np.array([-np.sin(psi), np.cos(psi)])
    u = np.array([np.cos(arm_angle), np.sin(arm_angle)])
    a = arm_angle - psi
    h_arm = (1 + c**2) / 2 * np.cos(2 * phi) * np.cos(2 * a) + c * np.sin(2 * phi) * np.sin(2 * a)  # DeltaL/L in units of A

    def displace(r, hp_, hx_):
        """Free-mass displacement 1/2 h.r for points r (2, N), in the (p, q) basis."""
        xp, xq = p @ r, q @ r
        dp = 0.5 * strain_scale * (hp_ * xp + hx_ * xq)
        dq = 0.5 * strain_scale * (hx_ * xp - hp_ * xq)
        return r + dp * p[:, None] + dq * q[:, None]

    def orbit_xyz(ph):
        x, y, z = np.cos(ph), np.sin(ph) * c, np.sin(ph) * s  # orbital plane tilted by iota about the x axis
        return np.cos(psi) * x - np.sin(psi) * y, np.sin(psi) * x + np.cos(psi) * y, z

    ring = np.array([np.cos(np.linspace(0, 2 * np.pi, 97)), np.sin(np.linspace(0, 2 * np.pi, 97))])
    masses = ring[:, ::8]
    th = np.linspace(0, 2 * np.pi, 200)
    ox, oy, oz = orbit_xyz(th)

    fig = plt.figure(figsize=(11, 9), dpi=60)  # low dpi keeps the embedded animation (and the notebook file) small
    ax3 = fig.add_subplot(2, 2, 1, projection="3d")
    axv = fig.add_subplot(2, 2, 2)
    axr = fig.add_subplot(2, 2, 3)
    axt = fig.add_subplot(2, 2, 4)

    ax3.plot(ox, oy, oz, color="gray", lw=1)
    Lx, Ly, Lz = s * np.sin(psi), -s * np.cos(psi), c  # orbital angular momentum
    ax3.quiver(0, 0, 0, Lx, Ly, Lz, color="green", lw=2, arrow_length_ratio=0.15)
    ax3.text(1.15 * Lx, 1.15 * Ly, 1.15 * Lz, "orbital axis", color="green")
    ax3.quiver(0, 0, 0, 0, 0, 1.6, color="k", lw=2, arrow_length_ratio=0.1)
    ax3.text(0, 0, 1.75, "to LISA (line of sight)")
    ax3.set_xlim(-1.2, 1.2); ax3.set_ylim(-1.2, 1.2); ax3.set_zlim(-1.2, 1.9)
    ax3.set_box_aspect((1, 1, 1.3)); ax3.set_axis_off()
    ax3.set_title(f"Binary orbit, $\\iota$ = {np.degrees(iota):.0f}$^\\circ$")
    s1_3d, = ax3.plot([], [], [], "o", color="C1", ms=12)
    s2_3d, = ax3.plot([], [], [], "o", color="C0", ms=12)

    axv.plot(ox, oy, color="gray", lw=1)
    axv.plot([-1.2 * p[0], 1.2 * p[0]], [-1.2 * p[1], 1.2 * p[1]], "k:", lw=0.8)
    axv.plot([-1.2 * q[0], 1.2 * q[0]], [-1.2 * q[1], 1.2 * q[1]], "k:", lw=0.8)
    axv.set_xlim(-1.4, 1.4); axv.set_ylim(-1.4, 1.4); axv.set_aspect("equal"); axv.axis("off")
    axv.set_title("The same orbit seen from LISA\n(dotted: $+$ polarization axes)")
    s1_v, = axv.plot([], [], "o", color="C1", ms=12)
    s2_v, = axv.plot([], [], "o", color="C0", ms=12)

    axr.plot(ring[0], ring[1], color="lightgray", lw=1)
    axr.plot([-u[0], u[0]], [-u[1], u[1]], color="lightcoral", lw=1, ls="--")
    # ring_line, = axr.plot([], [], color="C0", lw=2)
    mass_pts, = axr.plot([], [], "o", color="C0", ms=6)
    arm_line, = axr.plot([], [], color="crimson", lw=4)
    arm_pts, = axr.plot([], [], "o", color="crimson", ms=9)
    axr.set_xlim(-1.6, 1.6); axr.set_ylim(-1.6, 1.6); axr.set_aspect("equal"); axr.axis("off")
    axr.set_title("Free masses and one arm\n(strain exaggerated)")
    txt = axr.text(0, -1.5, "", ha="center")

    t = phi / (2 * np.pi)
    axt.plot(t, h_arm, label=r"arm: $\Delta L/L$ [units of $\mathscr{A}$]", color="crimson", lw=2.5)
    axt.plot(t, hplus, label=r"$h_+$", color="C0", ls="--", lw=2)
    axt.plot(t, hcross, label=r"$h_\times$", color="C2", ls=":", lw=2.5)
    
    marker = axt.axvline(0, color="k")
    axt.set_xlabel("orbital phase [orbits]"); axt.set_ylabel("relative strain")
    axt.set_title("Two GW cycles per orbit"); axt.legend(loc="upper right", fontsize=8)
    axt.set_ylim(-1.15, 1.15)
    fig.tight_layout()

    def update(k):
        ph = phi[k]
        for pt3, pv, sign in ((s1_3d, s1_v, 1), (s2_3d, s2_v, -1)):
            x, y, z = orbit_xyz(ph + (0 if sign == 1 else np.pi))
            pt3.set_data_3d([x], [y], [z])
            pv.set_data([x], [y])
        d = displace(ring, hplus[k], hcross[k])
        # ring_line.set_data(d[0], d[1])
        m = displace(masses, hplus[k], hcross[k])
        mass_pts.set_data(m[0], m[1])
        e = displace(np.stack([-u, u], axis=1), hplus[k], hcross[k])
        arm_line.set_data(e[0], e[1])
        arm_pts.set_data(e[0], e[1])
        marker.set_xdata([t[k], t[k]])
        txt.set_text(f"$h_+$ = {hplus[k]:+.2f}   $h_\\times$ = {hcross[k]:+.2f}")
        return ()

    anim = FuncAnimation(fig, update, frames=n_frames, interval=70, blit=False)
    html = HTML(anim.to_jshtml())
    plt.close(fig)
    return html


def interactive_binary_animation():
    """Sliders for inclination, polarization angle and arm direction; press 'Run Interact' to render."""
    import ipywidgets as widgets
    from IPython.display import display

    def _run(cos_iota, psi, arm_angle):
        display(animate_binary_gw(cos_iota, psi, arm_angle))

    widgets.interact_manual(
        _run,
        cos_iota=widgets.FloatSlider(min=-1, max=1, step=0.05, value=0.79, description="cos(iota)"),
        psi=widgets.FloatSlider(min=0, max=np.pi, step=np.pi / 16, value=0.0, description="psi", readout_format=".2f"),
        arm_angle=widgets.FloatSlider(min=0, max=np.pi, step=np.pi / 8, value=0.0, description="arm angle", readout_format=".2f"),
    )


def galactic_plane_ecliptic(n=720):
    """Galactic plane (b=0) and Galactic centre in ecliptic coordinates.

    Returns (lon, lat) of the plane sorted by lon in [-pi, pi), and (lon_gc, lat_gc); all in radians.
    """
    rot = hp.Rotator(coord=["G", "E"])
    theta, phi = rot(np.full(n, np.pi / 2), np.linspace(0, 2 * np.pi, n, endpoint=False))
    lon = (phi + np.pi) % (2 * np.pi) - np.pi
    order = np.argsort(lon)
    theta_gc, phi_gc = rot(np.pi / 2, 0.0)
    return lon[order], np.pi / 2 - theta[order], (phi_gc + np.pi) % (2 * np.pi) - np.pi, np.pi / 2 - theta_gc


def ecliptic_to_equatorial(lon, lat):
    """Convert ecliptic longitude/latitude to equatorial RA/Dec; inputs and outputs are radians."""
    rot = hp.Rotator(coord=["E", "C"])
    theta, ra = rot(np.pi / 2 - np.asarray(lat), np.asarray(lon))
    return ra, np.pi / 2 - theta


def plot_sky_map(values, title, unit="", cmap="viridis", truth=None, galactic_plane=True):
    """Mollweide view of a HEALPix map in ecliptic coordinates.

    truth = (beta, lambda) is marked with a red star; the Galactic plane (magenta dashed) and centre (white star)
    are drawn if `galactic_plane` is True.
    """
    hp.mollview(values, title=title, unit=unit, norm=None, cmap=cmap)
    hp.graticule()
    if galactic_plane:
        lon, lat, lon_gc, lat_gc = galactic_plane_ecliptic()
        hp.projplot(np.degrees(lon), np.degrees(lat), lonlat=True, color="magenta", ls="--", lw=1.5)
        hp.projscatter(np.degrees(lon_gc), np.degrees(lat_gc), lonlat=True, marker="*", color="white", edgecolors="k", s=150)
        hp.projtext(np.degrees(lon_gc), np.degrees(lat_gc) - 8, "Galactic centre", lonlat=True, color="white", fontsize=8, ha="center")
    if truth is not None:
        hp.projscatter(np.pi / 2 - truth[0], truth[1], marker="*", color="red", s=120)
    plt.show()


def interactive_sky_snr(analysis, nside=8, quantity="intrinsic"):
    """Explore how the SNR depends on the sky position of the source.

    Shows the SNR map (ecliptic coordinates, longitude increasing to the left like `plot_sky_map`),
    the Galactic plane, the true source and a selected position, plus the A, E, T signal power at the
    selected position (dashed: at the true position). With an interactive backend (`%matplotlib widget`)
    click on the map to move the selected position; otherwise use the sliders.

    quantity: "intrinsic" (SNR if the source were at that position) or "matched" (matched filter on our data).
    """
    import matplotlib
    import ipywidgets as widgets

    mf_map, opt_map = analysis.snr_vs_sky(nside=nside)
    values, label = (opt_map, r"intrinsic SNR $\rho_{\rm opt}$") if quantity == "intrinsic" else (mf_map, r"matched-filter SNR $\rho_{\rm MF}$")

    lon = np.linspace(-np.pi, np.pi, 360)
    lat = np.linspace(-np.pi / 2, np.pi / 2, 180)
    LON, LAT = np.meshgrid(lon, lat)
    grid = hp.get_interp_val(values, np.pi / 2 - LAT, (-LON) % (2 * np.pi))  # plot x = -lambda (astro convention)
    gal_lon, gal_lat, gc_lon, gc_lat = galactic_plane_ecliptic()
    true_beta, true_lam = analysis.source_params[3], analysis.source_params[4]
    h_true = analysis.template(analysis.source_params)
    wrap = lambda lam: -((lam + np.pi) % (2 * np.pi) - np.pi)  # lambda -> plot x

    def _draw(beta, lam, fig):
        p = analysis.source_params.copy()
        p[3], p[4] = beta, lam
        h = analysis.template(p)
        opt, mf = analysis.optimal_snr(h), float(analysis.matched_filter_snr(analysis.data, h))

        fig.clear()
        ax = fig.add_subplot(1, 2, 1, projection="mollweide")
        mesh = ax.pcolormesh(LON, LAT, grid, shading="auto", cmap="viridis")
        fig.colorbar(mesh, ax=ax, orientation="horizontal", shrink=0.8, pad=0.08, label=label)
        ax.plot(-gal_lon, gal_lat, color="magenta", ls="--", lw=1.5, label="Galactic plane")
        ax.plot(wrap(gc_lon), gc_lat, "w*", ms=13, mec="k", ls="", label="Galactic centre")
        ax.plot(wrap(true_lam), true_beta, "r*", ms=14, mec="k", ls="", label="true source")
        ax.plot(wrap(lam), beta, "o", color="orange", mec="k", ms=9, ls="", label="selected")
        ticks = np.arange(-150, 151, 30)
        ax.set_xticks(np.radians(ticks))
        ax.set_xticklabels([f"{int(-d) % 360}°" for d in ticks], fontsize=7)
        ax.grid(True, alpha=0.4)
        ax.set_title(f"$\\beta$={beta:.2f}, $\\lambda$={lam:.2f}\n"
                     f"$\\rho_{{\\rm opt}}$={opt:.1f}, $\\rho_{{\\rm MF}}$={mf:.1f}", fontsize=10)

        ax2 = fig.add_subplot(1, 2, 2)
        x = (analysis.freqs - analysis.source_params[0]) / analysis.df
        for i, ch in enumerate("AET"):
            ax2.semilogy(x, np.abs(h[i]) ** 2 + 1e-60, color=f"C{i}", label=f"{ch} (selected)")
            ax2.semilogy(x, np.abs(h_true[i]) ** 2 + 1e-60, color=f"C{i}", ls="--", alpha=0.5)
        ax2.set_ylim(1e-50, None)
        ax2.set_xlabel(r"Frequency offset from true $f_0$ [bins]")
        ax2.set_ylabel("Signal power per bin")
        ax2.set_title("Signal at the selected position\n(dashed: true position)", fontsize=10)
        handles, labels = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        fig.legend(handles + h2, labels + l2, loc="lower center", ncol=7, fontsize=8)
        fig.subplots_adjust(bottom=0.2, wspace=0.3)
        return ax

    if any(k in matplotlib.get_backend().lower() for k in ("ipympl", "widget")):
        fig = plt.figure(figsize=(13, 5.5))
        ax = _draw(true_beta, true_lam, fig)

        def _on_click(event):
            if event.inaxes is None or event.inaxes.name != "mollweide" or event.xdata is None:
                return
            _draw(event.ydata, (-event.xdata) % (2 * np.pi), fig)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("button_press_event", _on_click)
        plt.show()
    else:
        print("Sliders mode. For click-to-select run `%matplotlib widget` first (needs ipympl).")

        def _render(beta, lam):
            fig = plt.figure(figsize=(13, 5.5))
            _draw(beta, lam, fig)
            plt.show()

        widgets.interact(
            _render,
            beta=widgets.FloatSlider(min=-np.pi / 2, max=np.pi / 2, step=0.05, value=float(true_beta), description="beta"),
            lam=widgets.FloatSlider(min=0, max=2 * np.pi, step=0.05, value=float(true_lam), description="lambda"),
        )


def samples_to_sky_counts(samples, nside=64):
    """HEALPix counts of posterior samples; columns 3, 4 are (beta, lambda)."""
    pix = hp.ang2pix(nside, np.pi / 2 - samples[:, 3], samples[:, 4])
    return np.bincount(pix, minlength=hp.nside2npix(nside))


# ---------------------------------------------------------------------------
# Priors and MCMC
# ---------------------------------------------------------------------------
def make_priors(f0_center, f0_halfwidth, A_center, sigma0=None):
    """Uniform priors for the GB parameters (+ 3 noise sigmas if `sigma0` is given).

    Returns (priors, periodic, ndims, bounds) in the format expected by Eryn.
    """
    A_lo, A_hi = A_center / 4, A_center * 4
    fdot_lo, fdot_hi = -1e-15, 3e-15
    bounds = [(f0_center - f0_halfwidth, f0_center + f0_halfwidth), (fdot_lo, fdot_hi), (A_lo, A_hi),
              (-np.pi / 2, np.pi / 2), (0.0, 2 * np.pi), (0.0, np.pi), (0.0, np.pi), (0.0, 2 * np.pi)]
    if sigma0 is not None:
        bounds += [(s / 100, 5.0 * s) for s in sigma0]  # multiplicative range around the reference sigma
    priors = {"gb": ProbDistContainer({i: uniform_dist(lo, hi) for i, (lo, hi) in enumerate(bounds)})}
    periodic = {"gb": {4: 2 * np.pi, 5: np.pi, 7: 2 * np.pi}}  # psi has period pi
    return priors, periodic, len(bounds), bounds

def initialize_gb_coords(
    priors,
    rng,
    start_params,
    bounds,
    ntemps,
    nwalkers,
    rel_scatter=1e-4,
):
    """
    Symmetry-aware initialization for the GB parameters.

    The phase/polarization variables are initialized in terms of
    the combinations

        alpha = phi0 + 2 psi
        beta  = phi0 - 2 psi

    and the inclination-polarization symmetry

        (iota, psi) -> (pi-iota, -psi)

    is also included when the inclination prior spans [0, pi].
    """
    psi_idx = 5
    iota_idx = 6
    phi0_idx = 7
    
    ndims = len(start_params)

    coords = np.zeros((ntemps, nwalkers, 1, ndims))

    # ------------------------------------------------------------
    # 1. Local cloud around the starting point
    # ------------------------------------------------------------
    coords = priors["gb"].rvs(size=(ntemps, nwalkers)).reshape(ntemps, nwalkers, 1, ndims)
    
    for i in [0,1,2,3,4]: # f0, fdot, A, beta, lambda
        truth = start_params[i]

        if i == 0:
            scale = min(rel_scatter, 1e-5) * abs(truth)
        else:
            scale = (
                rel_scatter * abs(truth)
                if truth != 0
                else rel_scatter
            )

        lo, hi = bounds[i]
        margin = 1e-5 * (hi - lo)

        coords[:, :, 0, i] = np.clip(
            rng.normal(
                truth,
                scale,
                size=(ntemps, nwalkers),
            ),
            lo + margin,
            hi - margin,
        )

    # ------------------------------------------------------------
    # 2. Draw one common local cloud for phi0 and psi
    # ------------------------------------------------------------

    psi_lo, psi_hi = bounds[psi_idx]
    phi0_lo, phi0_hi = bounds[phi0_idx]

    psi_period = psi_hi - psi_lo
    phi0_period = phi0_hi - phi0_lo

    psi_truth = start_params[psi_idx]
    phi0_truth = start_params[phi0_idx]

    psi_scale = (
        rel_scatter * abs(psi_truth)
        if psi_truth != 0
        else rel_scatter
    )

    phi0_scale = (
        rel_scatter * abs(phi0_truth)
        if phi0_truth != 0
        else rel_scatter
    )

    psi_base = rng.normal(
        psi_truth,
        psi_scale,
        size=(ntemps, nwalkers),
    )

    phi0_base = rng.normal(
        phi0_truth,
        phi0_scale,
        size=(ntemps, nwalkers),
    )

    # ------------------------------------------------------------
    # 3. Four phase/polarization symmetry images
    # ------------------------------------------------------------

    modes = np.array([
        # dphi0       dpsi
        [0.0,         0.0],
        [np.pi,       np.pi / 2],
        [2*np.pi,     0.0],
        [np.pi,      -np.pi / 2],
    ])

    nmodes = len(modes)

    # ------------------------------------------------------------
    # 4. Add inclination symmetry as a second binary choice
    # ------------------------------------------------------------

    iota_lo, iota_hi = bounds[iota_idx]

    inclination_symmetry = (
        iota_lo <= 0.0
        and iota_hi >= np.pi
    )

    if inclination_symmetry:
        incl_modes = np.array([False, True])
    else:
        incl_modes = np.array([False])

    # Total number of modes
    all_modes = [
        (phase_mode, flip_iota)
        for phase_mode in range(nmodes)
        for flip_iota in incl_modes
    ]

    nmodes_total = len(all_modes)

    # ------------------------------------------------------------
    # 5. Populate walkers
    # ------------------------------------------------------------

    for temp in range(ntemps):

        permutation = rng.permutation(nwalkers)

        mode_idx = np.empty(nwalkers, dtype=int)
        mode_idx[permutation] = (
            np.arange(nwalkers) % nmodes_total
        )

        for k, (phase_mode, flip_iota) in enumerate(all_modes):

            mask = mode_idx == k

            if not np.any(mask):
                continue

            dphi0, dpsi = modes[phase_mode]

            # phase/polarization transformation
            coords[temp, mask, 0, phi0_idx] = (
                phi0_lo
                + np.mod(
                    phi0_base[temp, mask]
                    - phi0_lo
                    + dphi0,
                    phi0_period,
                )
            )

            coords[temp, mask, 0, psi_idx] = (
                psi_lo
                + np.mod(
                    psi_base[temp, mask]
                    - psi_lo
                    + dpsi,
                    psi_period,
                )
            )

            # inclination-polarization symmetry
            if flip_iota:

                coords[temp, mask, 0, iota_idx] = (
                    iota_lo
                    + iota_hi
                    - coords[
                        temp, mask, 0, iota_idx
                    ]
                )

                coords[temp, mask, 0, psi_idx] = (
                    psi_lo
                    + np.mod(
                        -coords[
                            temp, mask, 0, psi_idx
                        ]
                        - psi_lo,
                        psi_period,
                    )
                )

    return coords

def run_gb_mcmc(log_likelihood, priors, periodic, ndims, bounds, start_params,
                nwalkers=32, ntemps=8, n_iterations=2000, burn=1000,
                rel_scatter=1e-4, seed=RNG_SEED, progress=True):
    """Run parallel-tempered ensemble MCMC with a tight, symmetry-aware
    initial ensemble.

    The GB waveform has the discrete symmetries

        (phi0, psi)
            -> (phi0 + pi, psi)
            -> (phi0,       psi + pi/2)
            -> (phi0 + pi, psi + pi/2)

    modulo

        phi0 -> phi0 + 2*pi
        psi  -> psi  + pi.

    These are the four distinct points in the symmetry orbit within the
    current parameter ranges

        phi0 in [0, 2*pi)
        psi  in [0, pi).

    Walkers are distributed as evenly as possible among the four modes,
    independently for every temperature.

    Returns
    -------
    sampler : Eryn EnsembleSampler
    """

    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    sampler = EnsembleSampler(
        nwalkers,
        ndims,
        log_likelihood,
        priors,
        branch_names=["gb"],
        tempering_kwargs=dict(ntemps=ntemps),
        periodic=periodic,
        vectorize=True,
    )

    # ------------------------------------------------------------------
    # Initial coordinates
    # ------------------------------------------------------------------
    coords = initialize_gb_coords(
    priors=priors,
    rng=rng,
    start_params=start_params,
    bounds=bounds,
    ntemps=ntemps,
    nwalkers=nwalkers,
    rel_scatter=rel_scatter,
    )
    
    # ------------------------------------------------------------------
    # Initialize Eryn state
    # ------------------------------------------------------------------

    state = State({"gb": coords})

    inds = {
        "gb": np.ones(
            (ntemps, nwalkers, 1),
            dtype=bool,
        )
    }

    state.log_prior = sampler.compute_log_prior(
        state.branches_coords
    )

    state.log_like = sampler.compute_log_like(
        state.branches_coords,
        logp=state.log_prior,
        inds=inds,
    )[0]

    sampler.run_mcmc(
        state,
        n_iterations,
        progress=progress,
        burn=burn,
    )

    return sampler


def get_cold_samples(sampler, discard_fraction=0.5):
    """Flattened T=1 chain: samples (N, ndim) and log-likelihoods (N, 1)."""
    n_steps = sampler.get_chain(discard=0)["gb"].shape[0]
    discard = int(discard_fraction * n_steps)
    chain = sampler.get_chain(discard=discard)["gb"]
    logl = sampler.get_log_like(discard=discard)
    return chain[:, 0].reshape(-1, chain.shape[-1]), logl[:, 0].reshape(-1, 1)


def summarize_posterior(samples, truth, labels):
    """Print median, std, relative precision and bias of each parameter."""
    med, std = np.median(samples, axis=0), np.std(samples, axis=0)
    print(f"{'Parameter':<18} | {'Median':<18} | {'Std':<12} | {'Precision':<10} | {'Bias':<12} | Bias (sigma)")
    print("-" * 100)
    for i, lbl in enumerate(labels):
        prec = np.nan if truth[i] == 0 else std[i] / abs(truth[i])
        bias = med[i] - truth[i]
        sig = bias / std[i] if std[i] > 0 else np.nan
        print(f"{lbl:<18} | {med[i]:<18.10e} | {std[i]:<12.4e} | {prec:<10.2e} | {bias:<12.4e} | {sig:+.2f}")


def plot_posterior(samples, logl, truth, labels, title, indices=None):
    """Corner plot of the cold chain (with log L as last column when `indices` is None)."""
    data = np.hstack([samples, logl])
    labs = list(labels) + [r"$\log L$"]
    tr = np.append(truth, np.nan)
    if indices is not None:
        data, labs, tr = data[:, indices], [labs[i] for i in indices], tr[indices]
    fig = corner.corner(data, labels=labs, truths=tr, **CORNER_KWARGS)
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.show()
