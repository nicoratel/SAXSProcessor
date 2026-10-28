import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
import fabio
import os
import re
import math
import glob
from pathlib import Path
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import pandas as pd

# simps was removed in SciPy >= 1.14
try:
    from scipy.integrate import simpson
except ImportError:  # very old SciPy
    from scipy.integrate import simps as simpson

# PyFAI imports
try:
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
except ModuleNotFoundError:
    from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import pyFAI.detectors

# SasModels imports (optional)
try:
    from sasmodels.core import load_model
    from sasmodels.data import Data2D
    from sasmodels.direct_model import DirectModel
    SASMODELS_AVAILABLE = True
except ImportError:
    SASMODELS_AVAILABLE = False
    print("Warning: sasmodels not available. Form factor calculations will be disabled.")
try:
    from .waxsprocessor import WAXSProcessor
    WAXS = True
except ImportError:
    WAXS = False
try:
    from .saxsprocessor import SAXSProcessor
    SAXS = True
except ImportError:
    SAXS = False

class NematicOrderCalculator:
    """Calculate nematic order parameter S using Maier-Saupe distribution."""

    def __init__(self, form_factor=None):
        """Initialize with optional form factor object."""
        self.form_factor = form_factor
        # FIX: always defined, so convolve_with_form_factor can test them
        self.chi_ff = None
        self.I_ff = None

        if form_factor is not None:
            print(f" NematicOrderCalculator initialized with CylinderFormFactor")
            print(f"  Form factor profiles will be extracted on-the-fly for each q")
        else:
            print(f" NematicOrderCalculator initialized without form factor")
            print(f"  Pure Maier-Saupe fitting will be used")

    @staticmethod
    def Z_MS(m, x0):
        """Normalization constant for Maier-Saupe distribution."""
        beta0 = np.radians(x0)
        integrand = lambda beta: np.exp(m * np.cos(beta - beta0)**2) * np.sin(beta)
        Z, _ = quad(integrand, 0, np.pi, epsabs=1e-12, epsrel=1e-12)
        return Z

    @staticmethod
    def ms_distribution(beta, m, x0):
        """Normalized Maier-Saupe distribution."""
        beta0 = np.radians(x0)
        D = np.exp(m * np.cos(beta - beta0)**2)
        return D / NematicOrderCalculator.Z_MS(m, x0)

    @staticmethod
    def compute_S_old(m, x0):
        """Compute nematic order parameter S from Maier-Saupe parameters."""
        num_integrand = lambda theta: (0.5 * (3*np.cos(theta - np.radians(x0))**2 - 1)) \
                                    * np.exp(m*np.cos(theta - np.radians(x0))**2) * np.sin(theta)
        den_integrand = lambda theta: np.exp(m*np.cos(theta - np.radians(x0))**2) * np.sin(theta)

        num, _ = quad(num_integrand, 0, np.pi, epsabs=1e-10, epsrel=1e-10)
        den, _ = quad(den_integrand, 0, np.pi, epsabs=1e-10, epsrel=1e-10)

        return num / den

    @staticmethod
    def compute_S(m):
        P2 = lambda x: 0.5*(3*x**2 - 1)
        num = quad(lambda x: P2(x) * np.exp(m * x**2), -1, 1, epsabs=1e-10, epsrel=1e-10)[0]
        den = quad(lambda x: np.exp(m * x**2), -1, 1, epsabs=1e-10, epsrel=1e-10)[0]
        return num / den

    @staticmethod
    def compute_area_fractions(theta, I_oriented, I_iso):
        """Compute oriented and isotropic fractions from integrated areas.

        Both component profiles must be on the same intensity scale and
        sampled over the same angular domain.
        """
        theta = np.asarray(theta, dtype=float)
        I_oriented = np.asarray(I_oriented, dtype=float)
        I_iso = np.asarray(I_iso, dtype=float)

        valid = (np.isfinite(theta) & np.isfinite(I_oriented)
                 & np.isfinite(I_iso))
        theta = theta[valid]
        I_oriented = I_oriented[valid]
        I_iso = I_iso[valid]

        if len(theta) < 2:
            return {
                "A_oriented": np.nan, "A_iso": np.nan, "A_total": np.nan,
                "f_oriented": np.nan, "f_iso": np.nan,
            }

        # Simpson integration requires monotonically increasing x.
        order = np.argsort(theta)
        theta = theta[order]
        I_oriented = I_oriented[order]
        I_iso = I_iso[order]

        A_oriented = float(simpson(I_oriented, x=theta))
        A_iso = float(simpson(I_iso, x=theta))
        A_total = A_oriented + A_iso

        if not np.isfinite(A_total) or A_total <= 0:
            f_oriented = np.nan
            f_iso = np.nan
        else:
            f_oriented = A_oriented / A_total
            f_iso = A_iso / A_total

        return {
            "A_oriented": A_oriented,
            "A_iso": A_iso,
            "A_total": A_total,
            "f_oriented": f_oriented,

            "f_iso": f_iso,
        }

    @staticmethod
    def compute_mean_S(S, f_iso):
        """Return the mean order parameter <S> = (1 - f_iso) * S."""
        if not np.isfinite(S) or not np.isfinite(f_iso):
            return np.nan
        if not 0.0 <= f_iso <= 1.0:
            raise ValueError("f_iso must be between 0 and 1.")
        return (1.0 - f_iso) * S

    def convolve_with_form_factor(self, chi_array, m, x0, n_phi=360):
        """
        Compute azimuthal profile by convolving form factor with Maier-Saupe distribution.

        Parameters:
        -----------
        chi_array : array
            Azimuthal angles for calculation (°)
        m, x0 : float
            Maier-Saupe parameters
        n_phi : int
            Number of integration points

        Returns:
        --------
        intensities : array
            Computed intensities
        """
        if self.chi_ff is None or self.I_ff is None:
            raise ValueError("Form factor profile not provided")

        phi_ref = self.chi_ff[np.argmax(self.I_ff)]

        chi_extended = np.concatenate([self.chi_ff - 360, self.chi_ff, self.chi_ff + 360])
        I_extended = np.concatenate([self.I_ff, self.I_ff, self.I_ff])
        ff_interp = interp1d(chi_extended, I_extended, kind='cubic',
                             bounds_error=False, fill_value='extrapolate')

        phi = np.linspace(0, 360, n_phi, endpoint=False)
        d_phi = phi[1] - phi[0]

        phi_rad = np.radians(phi)
        x0_rad = np.radians(x0)
        f_ms_unnormalized = np.exp(m * np.cos(phi_rad - x0_rad)**2)
        integral_f_ms = np.sum(f_ms_unnormalized) * d_phi
        f_ms = f_ms_unnormalized / integral_f_ms

        # Vectorised version of the former Python loop over chi
        chi_array = np.atleast_1d(np.asarray(chi_array, dtype=float))
        angles_for_ff = chi_array[:, None] - phi[None, :] + phi_ref
        F2 = ff_interp(angles_for_ff)
        return np.sum(F2 * f_ms[None, :], axis=1) * d_phi

    def fit_azimuthal_profile(self, theta_exp, I_exp,
                              qvalue_ff=0.034,
                              threshold_ff=0.001,
                              target=90.0,
                              smooth=False,
                              window_length=11,
                              polyorder=2,
                              plot=False,
                              apply_mirror=False,
                              apply_fitmask=False,
                              verbose=True,
                              processor=None):
        """
        Fit experimental azimuthal profile with Maier-Saupe model.

        Parameters:
        -----------
        theta_exp, I_exp : arrays
            Experimental data
        qvalue_ff : float
            Q value for form factor extraction (if applicable)
        threshold_ff : float
            Threshold for form factor extraction
        target : float
            Center of mask (°)
        smooth : bool
            Apply Savitzky-Golay smoothing
        window_length, polyorder : int
            Smoothing parameters
        plot : bool
            Display result
        apply_mirror : bool
            Apply mirror symmetry to the data
        apply_fitmask : bool
            Restrict the fit to target ± 35°
        processor :
            If given, the figure is saved next to the data.

        Returns:
        --------
        results : dict
            Dictionary with fit results including S parameter
        """
        theta_exp = np.asarray(theta_exp, dtype=float)
        I_exp = np.asarray(I_exp, dtype=float)

        if apply_mirror:
            theta_exp, I_exp = self.mirror_profile(theta_exp, I_exp)
        if smooth:
            I_fit_data = savgol_filter(I_exp, window_length, polyorder)
        else:
            I_fit_data = I_exp.copy()

        if self.form_factor is not None:
            self.chi_ff, self.I_ff = self.form_factor.extract_azim_profile_formfactor(
                self.form_factor.I_ff2D, q0=qvalue_ff, threshold=threshold_ff, plot=False)
        else:
            self.chi_ff, self.I_ff = None, None

        if self.chi_ff is not None and self.I_ff is not None:
            def model_func(theta, I0, m, x0, b):
                I_ms = self.convolve_with_form_factor(theta, m, x0)
                return I0 * I_ms + b
        else:
            def model_func(theta, I0, m, x0, b):
                theta_rad = np.radians(theta)
                I_ms = self.ms_distribution(theta_rad, m, x0)
                return I0 * I_ms + b

        # Apply the fit mask if requested
        if apply_fitmask:
            mask_width = 70.0  # total width
            half_width = mask_width / 2
            lower = target - half_width
            upper = target + half_width

            # handle wrap-around at ±180°
            if lower < -180:
                mask = (theta_exp >= (lower + 360)) | (theta_exp <= upper)
            elif upper > 180:
                mask = (theta_exp >= lower) | (theta_exp <= (upper - 360))
            else:
                mask = (theta_exp >= lower) & (theta_exp <= upper)

            theta_fit = theta_exp[mask]
            I_fit = I_fit_data[mask]
        else:
            theta_fit = theta_exp
            I_fit = I_fit_data

        # FIX: keep the normalisation factor, so that the model and the data
        # are compared on the same scale when computing R² and plotting.
        norm = np.max(I_fit)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("Cannot fit profile: maximum fitted intensity is zero or invalid.")
        I_fit = I_fit / norm

        p0 = (1.0, 5.0, target, np.median(I_fit))
        bounds = (
            [0,      0,   target - 2, 0],
            [np.inf, 500, target + 2, np.inf]
        )

        try:
            popt, pcov = curve_fit(model_func, theta_fit, I_fit,
                                   p0=p0, bounds=bounds, maxfev=20000)
        except (RuntimeError, ValueError) as e:
            print(f"Fitting error: {e}")
            return None

        I0_opt, m_opt, x0_opt, b_opt = popt
        S_opt = self.compute_S(m_opt)
        I_model = model_func(theta_exp, *popt)           # normalised scale

        # FIX: R² on the points actually fitted, both on normalised scale
        residuals = I_fit - model_func(theta_fit, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((I_fit - np.mean(I_fit))**2)
        r_squared = 1 - (ss_res / ss_tot)

        if plot:
            fig = plt.figure(figsize=(10, 6))
            plt.plot(theta_exp, I_exp / norm, 'o',
                     label='Experimental', alpha=0.6)
            plt.plot(theta_exp, I_model, '-',
                     label=f'Fit (m={m_opt:.2f}, S={S_opt:.3f}, R²={r_squared:.3f})', linewidth=2)
            if self.chi_ff is not None:
                plt.plot(self.chi_ff, self.I_ff / np.max(self.I_ff), '--',
                         label='Form factor', alpha=0.5)
            plt.xlabel('θ (°)')
            plt.ylabel('Normalized intensity')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.title(f'Maier-Saupe fit (x0={x0_opt:.1f}°)')
            plt.tight_layout()
            # FIX: save BEFORE show (otherwise the saved figure is empty),
            # and include q in the name (otherwise each q overwrites the previous)
            if processor is not None:
                figname = (f'{processor.path}/{processor.samplename}_B={processor.B}mT_'
                           f'Img{processor.file_number}_q={qvalue_ff:.4f}_azimprofile_mixturefit.png')
                plt.savefig(figname)
                print(f' /!\\ Figure saved in {figname}')
            plt.show()
            plt.close(fig)

        x0_opt = (x0_opt + 180) % 360 - 180

                # ----------------------------
        # Component reconstruction
        # ----------------------------
        if self.chi_ff is not None and self.I_ff is not None:
            C90 = self.convolve_with_form_factor(theta_exp, m_opt, x0_opt)
        else:
            theta_rad = np.radians(theta_exp)
            C90 = self.ms_distribution(theta_rad, m_opt, x0_opt)

        # curve_fit was performed on I_fit / norm. Restore the experimental
        # intensity scale before integrating component areas.
        I_oriented = norm * I0_opt * C90
        I_iso = norm * b_opt * np.ones_like(theta_exp)

        # ----------------------------
        # Integrated areas and phase-intensity fractions
        # ----------------------------
        fractions = self.compute_area_fractions(
            theta_exp, I_oriented, I_iso
        )
        wparr = fractions["f_oriented"]
        wiso = fractions["f_iso"]
        S_mean = self.compute_mean_S(S_opt, wiso)

        results = {
            'I0': I0_opt,
            'm': m_opt,
            'x0': x0_opt,
            'b': b_opt,
            'S': S_opt,
            'S_mean': S_mean,
            'R2': r_squared,
            'I_model': I_model,
            'norm': norm,
            'wparr': wparr,
            'wiso': wiso,
            'A_oriented': fractions['A_oriented'],
            'A_iso': fractions['A_iso'],
            'A_total': fractions['A_total'],
            'popt': popt,
            'pcov': pcov
        }
        if verbose:
            print("\nFitting results:")
            relevant_items = ['S', 'S_mean', 'R2', 'wparr', 'wiso']
            for key, value in results.items():
                if key in relevant_items:
                    print(f"{key}: {value}")
        return results

    

    @staticmethod
    def mirror_profile(theta_exp, I_exp):
        """
        Fill the gaps of the azimuthal profile using the 180° rotation symmetry.
        A peak seen at θ generates a peak at θ + 180° (e.g. -89° -> 91°).
        Only missing data is duplicated, not the whole profile.
        """
        theta_exp = np.asarray(theta_exp, dtype=float)
        I_exp = np.asarray(I_exp, dtype=float)

        # Map angles to [-180, 180[
        theta_norm = ((theta_exp + 180) % 360) - 180

        # 0.1° grid (3600 points, no -180/180 duplicate)
        n = 3600
        theta_grid = np.linspace(-180, 180, n, endpoint=False)
        I_grid = np.full(n, np.nan)

        # Place experimental data (keep the highest value)
        for t, I in zip(theta_norm, I_exp):
            idx = int(np.argmin(np.abs(theta_grid - t)))
            if np.isnan(I_grid[idx]) or I > I_grid[idx]:
                I_grid[idx] = I

        # 180° rotation = shift by n/2 indices
        I_rot = np.roll(I_grid, n // 2)

        # Fill only the holes with the rotated value
        holes = np.isnan(I_grid)
        I_grid[holes] = I_rot[holes]

        mask_valid = ~np.isnan(I_grid)
        return theta_grid[mask_valid], I_grid[mask_valid]


# ============================================================================
# CYLINDER FORM FACTOR CALCULATION USING SASMODELS
# ============================================================================

if SASMODELS_AVAILABLE:

    class CylinderFormFactor:
        """
        Cylinder form factor (sasmodels) and its azimuthal profile at a given q.

        Conventions
        -----------
        - Detector image shape is (npix_z, npix_x) = (rows, columns).
        - Qx varies along columns, Qz along rows.
        - chi = atan2(Qz, Qx), in degrees, in [-180, 180[.
          With theta=90, phi=0 the cylinder axis is along x, so the
          intensity maximum is along Qz (chi = +/-90 deg).
          Use `check_orientation()` to verify on your sasmodels version.
        - Model angles (theta, phi, theta_pd, phi_pd) are in degrees;
          radius_pd and L_pd are relative widths (sigma/mean).
        """

        def __init__(self,
                     processor=None,
                     npix_x=None,
                     npix_z=None,
                     pix_size=None,
                     distance=None,
                     wl=None,
                     # Model parameters
                     radius=78,
                     L=840,
                     theta=90,
                     phi=0,
                     radius_pd=0.3,
                     L_pd=0.75,
                     theta_pd=0,
                     phi_pd=0,
                     background=0.00001,
                     scale=1,
                     chi_sign=1,
                     plot=False,
                     verbose=False):
            """
            Parameters
            ----------
            processor : SAXSProcessor, optional
                Detector geometry is inherited from it if given.
            npix_x, npix_z : int
                Number of pixels along x (columns) and z (rows).
                Required if processor is None.
            pix_size : float
                Pixel size (m), square pixels assumed.
            distance : float
                Sample-detector distance (m).
            wl : float
                Wavelength (Angstrom).
            radius, L : float
                Cylinder radius and length (Angstrom).
            theta, phi : float
                Orientation angles (deg). phi is NOT a volume fraction.
            radius_pd, L_pd : float
                Relative polydispersity (sigma/mean, gaussian).
            theta_pd, phi_pd : float
                Angular polydispersity (deg), 0 = perfectly oriented.
            chi_sign : +1 or -1
                Set to -1 if your experimental chi has the opposite
                handedness (only matters if the profile is not symmetric).
            """
            if processor is not None:
                self.npix_x = int(processor.num_pixel_x)
                self.npix_z = int(processor.num_pixel_z)
                self.wl = processor.wl * 1e10          # m -> Angstrom
                self.D = processor.D
                self.pixel_size = processor.pixel_size_x
                if verbose:
                    print(f"Detector geometry inherited from {processor.samplename}")
            else:
                if None in [npix_x, npix_z, pix_size, distance, wl]:
                    raise ValueError("Provide either a processor or all of "
                                     "npix_x, npix_z, pix_size, distance, wl")
                self.npix_x = int(npix_x)
                self.npix_z = int(npix_z)
                self.wl = wl
                self.D = distance
                self.pixel_size = pix_size
            self.distance = self.D

            if verbose:
                print(f"  - Detector: {self.npix_x} (x) x {self.npix_z} (z) pixels")
                print(f"  - Pixel size: {self.pixel_size * 1e3:.3f} mm")
                print(f"  - Distance: {self.D:.3f} m")
                print(f"  - Wavelength: {self.wl:.5f} A")

            self.radius = radius
            self.L = L
            self.theta = theta
            self.phi = phi
            self.radius_pd = radius_pd
            self.L_pd = L_pd
            self.theta_pd = theta_pd
            self.phi_pd = phi_pd
            self.background = background
            self.scale = scale
            self.chi_sign = -1 if chi_sign < 0 else 1

            if L_pd > 0.5 or radius_pd > 0.5:
                warnings.warn(f"Very large polydispersity (radius_pd={radius_pd}, "
                              f"L_pd={L_pd}): gaussian tails are truncated at 0 "
                              "by sasmodels. Is this intended?")

            self.kernel = load_model("cylinder")
            self._I_ff2D = None
            self._build_q_grid()

            if plot:
                self.compute_2d_map(plot=True)

        # ------------------------------------------------------------------
        # Geometry
        # ------------------------------------------------------------------
        def _build_q_grid(self):
            """Q grids of shape (npix_z, npix_x), at pixel centres."""
            ix = (np.arange(self.npix_x) + 0.5 - self.npix_x / 2) * self.pixel_size
            iz = (np.arange(self.npix_z) + 0.5 - self.npix_z / 2) * self.pixel_size
            X, Z = np.meshgrid(ix, iz, indexing="xy")      # (nz, nx)

            k = 2.0 * np.pi / self.wl
            two_theta = np.arctan2(np.hypot(X, Z), self.D)
            q = np.maximum(2.0 * k * np.sin(two_theta / 2.0), 1e-10)   # exact |q|
            psi = np.arctan2(Z, X)                                      # in-plane angle

            self.Qx = q * np.cos(psi)
            self.Qz = q * np.sin(psi)
            # Note: sasmodels 2D only takes the in-plane (Qx, Qz) components;
            # the small out-of-plane (Ewald sphere) component is neglected.

        # ------------------------------------------------------------------
        # sasmodels evaluation
        # ------------------------------------------------------------------
        def _params(self):
            params = {
                "radius": self.radius,
                "radius_pd": self.radius_pd,
                "radius_pd_type": "gaussian",
                "radius_pd_n": 50,
                "radius_pd_nsigma": 6,
                "length": self.L,
                "length_pd": self.L_pd,
                "length_pd_type": "gaussian",
                "length_pd_n": 50,
                "length_pd_nsigma": 6,
                "theta": self.theta,
                "phi": self.phi,
                "background": self.background,
                "scale": self.scale,
            }
            if self.theta_pd:
                params.update(theta_pd=self.theta_pd, theta_pd_type="gaussian",
                              theta_pd_n=35, theta_pd_nsigma=3)
            if self.phi_pd:
                params.update(phi_pd=self.phi_pd, phi_pd_type="gaussian",
                              phi_pd_n=35, phi_pd_nsigma=3)
            return params

        def _evaluate(self, qx, qz):
            """Evaluate I(qx, qz) for arrays of any shape (returns same shape)."""
            shape = np.shape(qx)
            qx = np.ascontiguousarray(qx, dtype=float).ravel()
            qz = np.ascontiguousarray(qz, dtype=float).ravel()
            z = np.zeros_like(qx)
            data = Data2D(x=qx, y=qz, z=z, dx=None, dy=None, dz=z)
            I = DirectModel(data, self.kernel)(**self._params())
            return np.asarray(I).reshape(shape)

        # ------------------------------------------------------------------
        # 2D map (for display / cross-checks)
        # ------------------------------------------------------------------
        @property
        def I_ff2D(self):
            """2D form factor, shape (npix_z, npix_x). Computed on first access."""
            if self._I_ff2D is None:
                self._I_ff2D = self.compute_2d_map()
            return self._I_ff2D

        def compute_2d_map(self, plot=False, vmin=-6):
            """Compute the 2D intensity on the detector grid (npix_z, npix_x)."""
            I2D = self._evaluate(self.Qx, self.Qz)
            if plot:
                extent = [self.Qx[0, 0], self.Qx[0, -1], self.Qz[0, 0], self.Qz[-1, 0]]
                plt.figure(figsize=(8, 6))
                plt.imshow(I2D / I2D.max(), origin="lower", extent=extent,
                           norm=LogNorm(vmin=10.0 ** vmin, vmax=1.0), cmap="jet")
                plt.colorbar(label="Normalized intensity")
                plt.xlabel("Qx (A$^{-1}$)")
                plt.ylabel("Qz (A$^{-1}$)")
                plt.title(f"Cylinder form factor, R={self.radius} A, L={self.L} A")
                plt.tight_layout()
                plt.show()
            return I2D

        # ------------------------------------------------------------------
        # Azimuthal profile (direct, analytical evaluation)
        # ------------------------------------------------------------------
        def azimuthal_profile(self, q0=0.034, n_chi=540, threshold=0.0, n_q=1,
                              plot=False):
            """
            Azimuthal profile I(chi) at |q| = q0, evaluated directly on a
            circle (no detector pixelisation, no pyFAI).

            Parameters
            ----------
            q0 : float
                |q| (A^-1).
            n_chi : int
                Number of chi points (bin centres over [-180, 180[).
            threshold, n_q : float, int
                Optional average over an annulus q0*(1 +/- threshold) sampled
                with n_q values. Default (n_q=1): exact circle at q0.
            Returns
            -------
            chi : array (deg, ascending), I : array
            """
            chi = -180.0 + (np.arange(n_chi) + 0.5) * 360.0 / n_chi
            c = np.radians(chi)

            if n_q > 1 and threshold > 0:
                qs = np.linspace(q0 * (1 - threshold), q0 * (1 + threshold), n_q)
            else:
                qs = np.array([q0])

            qx = qs[:, None] * np.cos(c)[None, :]
            qz = qs[:, None] * np.sin(c)[None, :]
            I = self._evaluate(qx, qz).mean(axis=0)

            chi = self.chi_sign * chi
            order = np.argsort(chi)
            chi, I = chi[order], I[order]

            if plot:
                plt.figure(figsize=(8, 5))
                plt.plot(chi, I, linewidth=2)
                plt.xlabel(r"$\chi$ (deg)")
                plt.ylabel("Intensity")
                plt.title(f"Form factor azimuthal profile, q = {q0:.4f} A$^{{-1}}$")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
            return chi, I

        def extract_azim_profile_formfactor(self, Iflat=None, q0=0.034,
                                            threshold=0.001, plot=False,
                                            n_chi=540, method="direct"):
            """
            Same signature as before, so NematicOrderCalculator keeps working.

            method="direct"   : analytical evaluation on the circle |q|=q0
                                (Iflat and threshold are ignored).
            method="detector" : integrate the 2D map (Iflat or self.I_ff2D)
                                with pyFAI, as in the old implementation.
                                Only a cross-check: the annulus must be at
                                least ~1 pixel wide in q, i.e. use a
                                threshold of a few 1e-3, not 1e-4.
            """
            if method == "direct":
                return self.azimuthal_profile(q0, n_chi=n_chi, plot=plot)
            if method == "detector":
                I_2D = self.I_ff2D if Iflat is None else np.asarray(Iflat).reshape(
                    self.npix_z, self.npix_x)
                return self._profile_from_detector(I_2D, q0, threshold, n_chi, plot)
            raise ValueError("method must be 'direct' or 'detector'")

        def _profile_from_detector(self, I_2D, q0, threshold, n_chi, plot=False):
            pix = self.pixel_size
            det = pyFAI.detectors.Detector(pixel1=pix, pixel2=pix,
                                           max_shape=(self.npix_z, self.npix_x))
            ai = AzimuthalIntegrator(dist=self.D,
                                     poni1=self.npix_z / 2 * pix,   # axis 1 = rows (z)
                                     poni2=self.npix_x / 2 * pix,   # axis 2 = columns (x)
                                     detector=det,
                                     wavelength=self.wl * 1e-10)
            chi, I = ai.integrate_radial(
                I_2D, n_chi,
                radial_range=(q0 * (1 - threshold), q0 * (1 + threshold)),
                radial_unit="q_A^-1")
            # pyFAI's chi handedness may differ from ours: compare with the
            # direct profile and adjust chi_sign if needed.
            chi = self.chi_sign * np.asarray(chi)
            order = np.argsort(chi)
            chi, I = chi[order], np.asarray(I)[order]
            if plot:
                plt.figure(figsize=(8, 5))
                plt.plot(chi, I, linewidth=2)
                plt.xlabel(r"$\chi$ (deg)")
                plt.ylabel("Intensity")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
            return chi, I

        # ------------------------------------------------------------------
        # Sanity check
        # ------------------------------------------------------------------
        def check_orientation(self, q0=0.034):
            """Print the chi of the profile maximum (expected +/-90 for theta=90, phi=0)."""
            chi, I = self.azimuthal_profile(q0)
            chi_max = chi[np.argmax(I)]
            print(f"Maximum of the form factor at q={q0:.4f}: chi = {chi_max:.1f} deg")
            return chi_max


class BatchNematic:
    """
    Process multiple SAXS files and extract nematic order parameters.
    Simplified: uses a single NematicOrderCalculator with CylinderFormFactor.
    """

    def __init__(self,
                 path,
                 instrument='ID02',
                 qvalues=None,
                 file_filter='*_eiger2*_raw.h5',
                 threshold=0.0001,
                 binning=1,
                 mask=None,
                 reference_file: str = None,
                 k=1,
                 autosubstract: bool = True,
                 form_factor=None,
                 output_dir=None,
                 structure=None,
                 case='SAXS',
                 target=90,
                 apply_fitmask=False
                 ):
        """
        Initialize batch processor.

        Parameters:
        -----------
        path : str
            Directory containing data files
        instrument : str
            Beamline name
        qvalues : list
            Q values for azimuthal profiles (Å⁻¹)
        file_filter : str
            Wildcard pattern for files
        threshold : float
            Q tolerance for azimuthal profile extraction
        binning : int
            Binning factor
        mask : str
            Path to mask file
        reference_file : str
            Path to reference file for background subtraction
        k : float
            Scaling factor for background subtraction
        autosubstract : bool
            Enable automatic background subtraction
        form_factor : CylinderFormFactor or None
            Form factor object (profiles extracted on-the-fly for each q)
        output_dir : str
            Directory to save output files
        structure:
            CrystalStructure instance for WAXS data analysis
        case:
            Indicates whether treating SAXS/WAXS
        target: int
            Target value for azimuthal peak (90 or 0°)
        apply_fitmask:
            Indicates whether apply mask on fit data. Mask is centered around target.
        """
        self.target = target
        self.apply_fitmask = apply_fitmask
        self.case = case
        if structure is not None:
            self.structure = structure
        self.path = path
        self.instrument = instrument
        self.qvalues = qvalues if qvalues is not None else None
        self.reflections = structure.reflections if structure is not None else None
        self.threshold = threshold
        self.binning = binning
        self.mask = mask
        self.reference_file = reference_file
        self.k = k
        self.autosubstract = autosubstract
        self.h5_filelist = sorted(glob.glob(os.path.join(path, file_filter)),
                                  key=self._extract_number)
        self.output_dir = output_dir if output_dir is not None else self.path

        # Create a single NematicOrderCalculator with the form factor
        self.nematic_calc = NematicOrderCalculator(form_factor=form_factor)

        print(f"\n{'=' * 60}")
        print(f"BatchNematic initialized")
        print(f"{'=' * 60}")
        print(f"Path: {self.path}")
        if self.case == 'SAXS':
            print(f"Q values: {self.qvalues}")
        if self.case == 'WAXS':
            print(f'Reflections: {self.structure.reflections}')
        print(f"Found {len(self.h5_filelist)} files to process")
        print(f"Output directory: {self.output_dir}")
        print(f"{'=' * 60}\n")

    def _extract_number(self, file_path):
        """Extract file number for sorting."""
        filename = os.path.basename(file_path)
        extension = filename.split('.')[-1]

        if self.instrument == 'ID02':
            number = filename.split('_')[2]
        elif self.instrument == 'SWING':
            number = filename.split('_')[1]
        elif extension == 'edf':
            basename = filename.split('.')[0]
            number = basename.split('_')[-1].split('-')[0]
        else:
            number = '0'

        return int(number)

    def process_all(self, save_profiles=True, plot=False, apply_mirror=False):
        """
        Process all files and compute nematic order parameters.

        Parameters:
        -----------
        save_profiles : bool
            Save azimuthal profiles
        plot : bool
            Display fitting results
        apply_mirror : bool
            Apply mirror symmetry to experimental profiles

        Returns:
        --------
        df : DataFrame
            Results table
        """
        results = []
        logfile = os.path.join(self.output_dir, f'batch_processing_target={self.target}.log')
        log_lines = []

        for file in self.h5_filelist:
            try:
                print(f"\n{'=' * 60}")
                print(f"Processing: {os.path.basename(file)}")
                print(f"{'=' * 60}")
                if self.case == 'SAXS':
                    processor = SAXSProcessor(
                        file,
                        instrument=self.instrument,
                        binning=self.binning,
                        mask=self.mask,
                        reference_file=self.reference_file,
                        k=self.k,
                        autosubstract=self.autosubstract
                    )
                elif self.case == 'WAXS':
                    processor = WAXSProcessor(
                        file,
                        structure=self.structure,
                        mask=self.mask,
                        mapping=True,
                        instrument=self.instrument,
                        output_dir=self.output_dir
                    )
                if self.case == 'SAXS':
                    for qvalue in self.qvalues:
                        print(f"\nProcessing q = {qvalue:.4f} Å⁻¹...")

                        # Extract experimental azimuthal profile
                        chi_exp, I_exp = processor.extract_azimuthal_profile(
                            qvalue,
                            threshold=self.threshold,
                            save=save_profiles,
                            output_dir=self.output_dir
                        )

                        # Fit with NematicOrderCalculator (form factor extracted on-the-fly)
                        fit_results = self.nematic_calc.fit_azimuthal_profile(
                            chi_exp, I_exp,
                            qvalue_ff=qvalue,
                            threshold_ff=self.threshold,
                            target=self.target,
                            smooth=False,
                            plot=plot,
                            apply_mirror=apply_mirror,
                            apply_fitmask=self.apply_fitmask,
                            processor=processor
                        )

                        if fit_results is not None:
                            results.append({
                                'File': os.path.basename(file),
                                'Sample': processor.samplename,
                                'B (mT)': processor.B,
                                'File_Number': processor.file_number,
                                'q (Å⁻¹)': qvalue,
                                'S': fit_results['S'],
                                'wparr': fit_results['wparr'],
                                'wiso': fit_results['wiso'],
                                'm': fit_results['m'],
                                'x0 (°)': fit_results['x0'],
                                'I0': fit_results['I0'],
                                'b (offset)': fit_results['b'],
                                'R²': fit_results['R2']
                            })

                            success_msg = (f"  ✓ Success | q={qvalue:.4f} | S={fit_results['S']:.4f} | "
                                           f"m={fit_results['m']:.2f} | R²={fit_results['R2']:.4f}")
                            print(success_msg)
                            log_lines.append(success_msg)
                        else:
                            fail_msg = f"  ✗ Fit failed for q = {qvalue:.4f} Å⁻¹"
                            print(fail_msg)
                            log_lines.append(fail_msg)
                if self.case == 'WAXS':
                    for reflection in self.reflections:
                        ref_str = "".join(str(a) for a in reflection)
                        print(f"\nProcessing reflection = {ref_str}...")

                        # Extract experimental azimuthal profile
                        chi_exp, I_exp = processor.extract_azimuthal_profile(
                            reflection,
                            save=save_profiles)

                        fit_results = self.nematic_calc.fit_azimuthal_profile(
                            chi_exp, I_exp,
                            smooth=False,
                            plot=plot,
                            apply_mirror=apply_mirror,
                            target=self.target,
                            apply_fitmask=self.apply_fitmask
                        )

                        if fit_results is not None:
                            results.append({
                                'File': os.path.basename(file),
                                'Sample': processor.samplename,
                                'B (mT)': processor.B,
                                'File_Number': processor.file_number,
                                'reflection': ref_str,
                                'S': fit_results['S'],
                                'wparr': fit_results['wparr'],
                                'wiso': fit_results['wiso'],
                                'm': fit_results['m'],
                                'x0 (°)': fit_results['x0'],
                                'I0': fit_results['I0'],
                                'b (offset)': fit_results['b'],
                                'R²': fit_results['R2']
                            })

                            success_msg = (f"  ✓ Success | reflection={ref_str} | S={fit_results['S']:.4f} | "
                                           f"m={fit_results['m']:.2f} | R²={fit_results['R2']:.4f}")
                            print(success_msg)
                            log_lines.append(success_msg)
                        else:
                            fail_msg = f"  ✗ Fit failed for reflection = {ref_str}"
                            print(fail_msg)
                            log_lines.append(fail_msg)

            except Exception as e:
                error_msg = f"ERROR processing {os.path.basename(file)}: {str(e)}"
                print(f"\n{error_msg}")
                log_lines.append(error_msg)
                import traceback
                traceback.print_exc()

        # Save log
        print(f"\n{'=' * 60}")
        print("Saving results...")
        print(f"{'=' * 60}")

        with open(logfile, 'w') as f:
            f.write('\n'.join(log_lines))
        print(f"✓ Log file saved: {logfile}")

        # Create DataFrame
        if len(results) > 0:
            df = pd.DataFrame(results)

            # Sort by file number and q value
            if self.case == 'SAXS':
                df = df.sort_values(['File_Number', 'q (Å⁻¹)'])
            if self.case == 'WAXS':
                df = df.sort_values(['File_Number', 'reflection'])

            output_csv = os.path.join(self.output_dir, f'nematic_order_results_target={self.target}.csv')
            df.to_csv(output_csv, index=False)
            print(f"✓ Results CSV saved: {output_csv}")

            return df
        else:
            print("⚠ No successful fits to save")
            return pd.DataFrame()
