from __future__ import annotations
from typing import *
import copy
import os
import glob
import numpy as np
from numpy.typing import *
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit, minimize
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d


class SAXSTools:
    @staticmethod
    def gaussian(x: np.array, mu: float = 0, sigma: float = 1 ) -> np.array:
        """ Gaussian function """
        a = 1 / (sigma * np.sqrt(2 * np.pi))
        return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    @staticmethod
    def lorentzian(x : np.array, mu : float=0, sigma : float=1 ) -> np.array:
        """ Lorentzian function """
        a = 1/np.pi
        return a * (sigma/((x - mu)**2 + sigma**2))

    @staticmethod
    def pseudo_voigt(x: np.array, amp: float, mu: float, sigma: float, eta: float, ) -> np.array:
        """
        Pseudo-Voigt profile, a linear combination of Gaussian and Lorentzian profiles.

        Parameters
        ----------
        x : np.array
            The input array where the profile is evaluated.
        amp : float
            Amplitude of the profile.
        mu : float
            Center of the profile.
        sigma : float
            Standard deviation of the Gaussian component.
        eta : float
            Mixing parameter between Gaussian and Lorentzian components, must be between 0 and 1
        
        Returns
        -------
        np.array
            The evaluated pseudo-Voigt profile at the input x values.
        """
        if not (0 <= eta <= 1):
            raise ValueError(f"eta must be between 0 and 1, got {eta}")
        gauss_norm = (sigma * np.sqrt(2 * np.pi)) * SAXSTools.gaussian(x, mu, sigma)
        lorentz_norm = (sigma * np.pi) * SAXSTools.lorentzian(x, mu, sigma)
        return amp * (eta * lorentz_norm + (1 - eta) * gauss_norm)

    @staticmethod
    def splitted_pseudo_voigt(
        x: np.array,
        first_params: Annotated[tuple[float, float, float, float, float], 'Parameters of the first pseudo-Voigt function : (amp, cen, sig, eta, off)'],
        second_params: Annotated[tuple[float, float, float, float, float], 'Parameters of the second pseudo-Voigt function : (amp, cen, sig, eta, off)'],
    ) -> np.array:
        """
        Splitted pseudo-Voigt profile, a linear combination of two pseudo-Voigt profiles with a sigmoid transition.

        Parameters
        ----------
        x : np.array
            The input array where the profile is evaluated.
        first_params : tuple
            Parameters of the first pseudo-Voigt function: (amp, cen, sig, eta, off).
        second_params : tuple
            Parameters of the second pseudo-Voigt function: (amp, cen, sig, eta, off).
        
        Returns
        -------
        np.array
            The evaluated splitted pseudo-Voigt profile at the input x values.
        """
        if first_params[1] != second_params[1]:
            raise ValueError(f'The centers must be same got : ({first_params[1]}, {second_params[1]})')
        else:
            mu = first_params[1]
            
        off1 = first_params[4]
        off2 = second_params[4]
        
        t = 1/(1 + np.exp(- (x - mu)/1e-3)) # Sigmoid transition function
        
        return (1 - t) * (SAXSTools.pseudo_voigt(x, *first_params[:-1]) + off1) + t * (SAXSTools.pseudo_voigt(x, *second_params[:-1]) + off2)

    @staticmethod
    def sum_pseudo_voigts(
        x: np.array,
        *params: Annotated[tuple, "Parameters of n pseudo-Voigt profiles, now 4 params each: (amp, cen, sigma, eta) and 1 offset param"]
    ) -> np.array:
        """
        Sum of n pseudo-Voigt profiles, each with an individual offset.
        len(params) must be 4*n + 1.
        """
        n = (len(params) - 1) // 4
        y = np.zeros_like(x)
        for i in range(n):
            amp = params[4*i]
            mu = params[4*i + 1]
            sigma = params[4*i + 2]
            eta = params[4*i + 3]
            y += SAXSTools.pseudo_voigt(x, amp, mu, sigma, eta)
        
        offset = params[-1]
        return y + offset
    
    @staticmethod
    def asym2sig( x: np.array, A: float, mu: float, w1: float, w2: float, w3: float, off:float ) -> np.array:
        """
        Asymetric double sigmoid function
        ref: https://www.originlab.com/doc/Origin-Help/Asym2Sig-FitFunc

        Parameters
        ----------
        x : np.array
            The input array where the profile is evaluated.
        A : float
            Amplitude of the profile. (Must be > 0)
            NB: This don't represent the height of the peak, but the height of the sigmoids function.
        mu : float
            Center of the profile.
        w1 : float
            Distance beetween the two inflection points of the sigmoids function.
        w2 : float
            Width of the first sigmoid (left side).
        w3 : float
            Width of the second sigmoid (right side).
        off : float
            Offset of the profile.
        
        Returns
        -------
        np.array
            The evaluated asymetric double sigmoid profile at the input x values.

        Raises
        ------
        ValueError
            If A <= 0, w1 < 0, w2 <= 0, or w3 <= 0.
        
        Notes
        -----
        Estimation of the FWHM (Full Width at Half Maximum) is not straightforward for this function.
        The FWHM can be approximated by the distance between the two inflection points plus the widths:
        FWHM ≈ w1 + (w2 + w3).
        """
        if A <= 0:
            raise ValueError(f'Amplitude must me > 0, got : A={A}')
        if w1 < 0 or w2 <= 0 or w3 <= 0:
            raise ValueError(f'Width must be > 0, got : w1={w1} ; w2={w2} ; w3={w3}')
        low_sig = 1/(1 + np.exp(-(x - mu + w1/2)/w2))
        high_sig = 1/(1 + np.exp(-(x - mu - w1/2)/w3)) 
        return off + A * low_sig * (1 - high_sig)

    @staticmethod
    def tuple_to_dict(
        entries: Annotated[tuple, "Key names used in the dictionary"],
        values: Annotated[tuple, "Values associated with keys in periodic ways"]
        ) -> dict:
        """
        Factory method to convert a tuple of entries and a tuple of values into a dictionary.
        
        Parameters
        ----------
        entries : tuple
            A tuple of property names (keys) for the dictionary.
        values : tuple
            A tuple of values corresponding to the properties. The length of this tuple must be a multiple
            of the number of entries.

        Returns
        -------
        dict
            A dictionary where each key is an entry from `entries` and the corresponding value is a list of values
            from `values`, grouped by the number of entries.
        
        Raises
        ------
        ValueError
            If the number of values is not a multiple of the number of entries, or if any key is missing from the dictionary.
        """

        n_props = len(entries)
        if len(values) % n_props != 0:
            raise ValueError("The number of values is not a multiple of the number of properties.")
        result = {entry: values[i::n_props] for i, entry in enumerate(entries)}
        return result
    
    @staticmethod
    def find_peaks_dico(y : np.array,
                        n_expected_peaks : int = 3,
                        max_iter : int = 20,
                        min_prominence : int | None= None,
                        max_prominence : int | None = None,
                        verbose : bool = False):
        
        # Initialize search bounds for prominence
        low_prominence = min_prominence if not None else 1e-6
        high_prominence =(
            max_prominence if max_prominence is not None
            else max(
                abs(np.max(y) + np.min(y)),
                abs(np.max(y) - np.min(y))
            )
        )
        
        best_peak_indices = np.array([])
        best_peak_properties = {}

        for i in range(max_iter):
            mid_prominence = (low_prominence + high_prominence) / 2

            # Detect peaks using current prominence threshold
            peak_indices, peak_properties = find_peaks(
                y,
                height=(None, None),
                threshold=(None, None),
                distance=None,
                prominence=mid_prominence,
                width=(None, None),
                wlen=None,
                rel_height=0.5,
                plateau_size=(None, None)
            )

            diff = len(peak_indices) - n_expected_peaks
            if verbose:
                print(f"[{i+1}/{max_iter}] : {len(peak_indices)} found | ub : {high_prominence}; umb : {mid_prominence}; lb : {low_prominence}")

            # Keep best match so far
            if abs(diff ) < abs(len(best_peak_indices) - n_expected_peaks) or (abs(diff) == abs(len(best_peak_indices) - n_expected_peaks) and len(peak_indices) > len(best_peak_indices)) :
                best_peak_indices = peak_indices
                best_peak_properties = peak_properties

            # Early exit if target number of peaks found
            if diff == 0:
                break

            # Binary search logic
            if len(peak_indices) < n_expected_peaks:
                high_prominence = mid_prominence
            else:
                low_prominence = mid_prominence
        
        return best_peak_indices, best_peak_properties
            


class SAXSData:
    def __init__(
        self,
        x: Annotated[np.array, "x values"],
        y: Annotated[np.array, "y values"],
        dy: Annotated[np.array | None, "y uncertainty"] = None,
        name: Annotated[str | None, "Measurement name"] = None,
        infos: Annotated[dict[str, Any] | None, "Dictionnary about additionals infos"] = None
    ):
        """
        Create a SAXSData object with x, y, and optional dy values.

        Parameters
        ----------
        x : np.array
            The x values (e.g., scattering vector).
        y : np.array
            The y values (e.g., intensity).
        dy : np.array, optional
            The uncertainty in y values. If None, defaults to 5% of y.
        name : str, optional
            The name of the measurement. Defaults to None.
        infos : dict, optional
            A dictionary containing additional information about the measurement. Defaults to None.
        
        Raises
        ------
        ValueError
            If the shapes of x, y, dy do not match.            
        """
        self.x = np.asarray(x)
        self.y = np.asarray(y)

        if dy is None:
            self.dy = 0.05 * self.y
        else:
            self.dy = np.asarray(dy)
        
        if self.x.shape != self.y.shape or self.y.shape != self.dy.shape:
            raise ValueError(
                f"Shape mismatch: x={self.x.shape}, y={self.y.shape}, dy={self.dy.shape}"
            )

        self.name = name
        self.infos = infos
        self.mask = np.zeros_like(self.y, dtype=bool)  # ensure mask matches y

    def _validate_mask(self):
        """Ensure the mask matches the shape of y."""
        if self.mask.shape != self.y.shape:
            print("Warning: mask shape mismatch. Auto-resetting mask.")
            self.mask = np.zeros_like(self.y, dtype=bool)

    def apply_mask(self, mask=None, xmin=None, xmax=None) -> SAXSData:
        """
        Apply a mask to the data based on given conditions.

        Parameters
        ----------
        mask : array-like, optional
            A boolean mask to apply. If None, a mask is created based on xmin and xmax.
        xmin : float, optional
            Minimum x value for the mask. If None, defaults to -np.inf.
        xmax : float, optional
            Maximum x value for the mask. If None, defaults to np.inf.
        
        Returns
        -------
        SAXSData
            The SAXSData object with the applied mask.
        
        Raises
        ------
        ValueError
            If the length of the mask does not match the length of x.
        """
        if mask is not None:
            if len(mask) != len(self.x):
                raise ValueError("Mask length must match data length")
            self.mask = mask
        else:
            if xmin is None:
                xmin = -np.inf
            if xmax is None:
                xmax = np.inf
            self.mask = (self.x < xmin) | (self.x > xmax)
        return self

    def remove_mask(self) -> SAXSData:
        self.mask = np.full_like(self.x, False, dtype=bool)
        return self

    def invert_mask(self) -> SAXSData:
        self.mask = ~self.mask
        return self
    
    def get_mask(self) -> SAXSData:
        self._validate_mask()
        return self.mask
    
    def get_mask_bounds(self) -> tuple[Optional[int], Optional[int]]:
        self._validate_mask()
        indices = np.where(~self.mask)[0]
        if len(indices) == 0:
            return None, None
        return self.x[indices].min(), self.x[indices].max()

    def get_filtered_data(self) -> tuple[Optional[np.array], Optional[np.array], Optional[np.array]]:
        self._validate_mask()
        return self.x[~self.mask], self.y[~self.mask], self.dy[~self.mask]

    def get_raw_data(self) -> tuple[Optional[np.array], Optional[np.array], Optional[np.array]]:
        return self.x, self.y, self.dy
    
    def get_infos(self) -> dict:
        return self.infos

    def set_data(self, y_new, dy_new=None):
        """
        Update the y and dy data of the SAXSData object.

        Parameters
        ----------
        y_new : np.array
            The new y values to set.
        dy_new : np.array, optional
            The new dy values to set. If None, defaults to 5% of y_new.
        
        Raises
        ------
        ValueError
            If the shape of y_new does not match the shape of x, or if dy_new is provided and its shape does not match the shape of x.
        """
        y_new = np.array(y_new)
        if y_new.shape != self.x.shape:
            raise ValueError("New y data must match the shape of x.")
        
        self.y = y_new

        if dy_new is not None:
            dy_new = np.array(dy_new)
            if dy_new.shape != self.x.shape:
                raise ValueError("New dy data must match the shape of x.")
            self.dy = dy_new
        else:
            self.dy = y_new * 0.05
    
    def copy(self) -> SAXSData:
        """ Create a copy of the SAXSData object. """
        self._validate_mask()
        
        return SAXSData(
            name=self.name,
            x=np.copy(self.x),
            y=np.copy(self.y),
            dy=np.copy(self.dy) if self.dy is not None else None,
            infos=self.infos if self.dy is not None else None
        ).with_mask(np.copy(self.mask))

    def with_mask(self, mask_array) -> SAXSData:
        self.mask = mask_array
        return self
    
    def __len__(self) -> int:
        return len(self.x)


class SAXSPeakInfo:
    def __init__(self):
        '''
        SAXSPeakInfo is a class to store and manage sets of peaks with their properties.
        '''
        self.peak_sets: dict[str, dict[str, Any]] = {}

    def set(self, name: str, q_values: np.array, q_values_std: np.array = None, FWHM: np.array = None, Imin : float = None,  properties: dict[str, NDArray] = None):
        """
        Store a set of peaks with their properties.

        Parameters
        ----------
        name : str
            Identifier for the peak set (e.g., 'standard', 'pseudo_voigt').
        q_values : np.array
            Array of peak q values.
        q_values_std : np.array, optional
            Array of standard deviations for the q values. If None, defaults to 5% of q_values.
        FWHM : np.array, optional
            Array of Full Width at Half Maximum (FWHM) values for the peaks. If None, it is not stored.
        I_min : float, optional
            Value of the minimum of intensity between the peaks
        properties : dict[str, NDArray], optional
            Dictionary of additional properties associated with the peaks. Each key should have an array of values.
        
        Raises
        ------
        ValueError
            If the length of any property array does not match the length of q_values.
        """
        properties = properties or {}
        q_values_std = q_values_std if q_values_std is not None else q_values
        if q_values is not None :
            n = len(q_values)
        for key, val in properties.items():
            if len(val) != n:
                raise ValueError(f"Length mismatch in property '{key}' (expected {n}, got {len(val)})")
        self.peak_sets[name] = {'q_values': q_values, 'q_values_std': q_values_std, 'FWHM': FWHM, 'Imin' : Imin, **properties}
    
    """
    Different getters for peak sets.
    Each getter retrieves a specific property or set of properties from the stored peak sets.

    - `get`: Retrieves the entire peak set by name.
    - `get_q_values`: Retrieves the q values of the peaks for a given name.
    - `get_FWHM`: Retrieves the FWHM values of the peaks for a given name.

    Parameters
    ----------
    name : str
        The name of the peak set to retrieve.
    """
    def get(self, name: str) -> dict:
        return self.peak_sets[name]

    def get_q_values(self, name: str) -> NDArray:
        return self.peak_sets.get(name, {}).get("q_values", np.array([]))
    
    def get_FWHM(self, name: str) -> NDArray:
        return self.peak_sets.get(name, {}).get("FWHM", np.array([]))
    
    def get_Imin(self, name : str ):
        return self.peak_sets.get(name, {}).get("Imin", float)

    def get_indices(self, name: str, q_reference: np.array) -> NDArray:
        """
        Return the indices of the closest q values in the peak set to the reference q values.

        Parameters
        ----------
        name : str
            The name of the peak set to retrieve.
        q_reference : np.array
            The reference q values to which the indices will be matched.
        
        Returns
        -------
        Indices : NDArray
            An array of indices corresponding to the closest q values in the peak set to the reference q values.
        """
        q_values = self.get_q_values(name)
        if q_values.size == 0:
            return np.array([])

        indices = np.array([int(np.argmin(np.abs(q_reference - q))) for q in q_values])

        return indices

    def get_property(self, name: str, prop: str, default=None) -> Any:
        """
        Return a property array (e.g. 'prominence') from a named peak set.

        Parameters
        ----------
        name : str
            The name of the peak set to retrieve.
        prop : str
            The property to retrieve from the peak set (e.g., 'prominence').
        default : Any, optional
            Default value to return if the property is not found. Defaults to None.
        
        Returns
        -------
        Any
            The property array if found, otherwise the default value.
        """
        return self.peak_sets.get(name, {}).get(prop, default)

    def names(self) -> list[str]:
        """ List the names of all stored peak sets."""
        return list(self.peak_sets.keys())

    def remove(self, name: str):
        """
        Delete a peak set by name.
        """
        if name in self.peak_sets:
            del self.peak_sets[name]


    def clear(self):
        """ Clear all stored peak sets. """
        self.peak_sets.clear()

    def __contains__(self, name: str) -> bool:
        return name in self.peak_sets

    def __getitem__(self, name: str) -> dict[str, Any]:
        return self.get(name)

    def __repr__(self) -> str:
        return f"<SAXSPeakInfo with {len(self.peak_sets)} sets: {', '.join(self.names())}>"


class SAXSExperiment:
    def __init__(
        self,
        data_dict: Annotated[
            dict[str, tuple[np.array, np.array, np.array | None]],
            "Data dictionnary (name: (q, Iq, [Iq_std]))"
        ]=None,
        peaks: Annotated[SAXSPeakInfo | None, "Information on the peaks"]=None,
        name: Annotated[str |None, "Name of the experiment"]=None,
    ):  
        """
        Create a SAXSExperiment object with a dictionary of SAXSData objects.

        Parameters
        ----------
        data_dict : dict[str, tuple[np.array, np.array, np.array | None]], optional
            A dictionary where keys are names of SAXSData attributes and values are tuples containing:
            - q values (np.array)
            - y values (np.array)
            - dy values (np.array, optional). If dy is not provided, it defaults to 5% of y.
        peaks : SAXSPeakInfo, optional
            An instance of SAXSPeakInfo containing information about peaks. If None, no peaks are stored.
        name : str, optional
            The name of the experiment. If None, no name is assigned.
        
        Raises
        ------
        ValueError
            If any entry in `data_dict` is not a tuple of length 2 or 3, or if the q and y arrays do not match in shape.
        """
        for key, data_tuple in data_dict.items():
            if not isinstance(data_tuple, tuple) or not (2 <= len(data_tuple) <= 3):
                raise ValueError(f"Each entrie must be a tuple as (q, y, [dy]) : got '{data_tuple}'")
            
            q, y = data_tuple[:2]
            dy = data_tuple[2] if len(data_tuple) == 3 else y * 0.05
            self.add_data(key, q, y, dy)

        self.name = name
        self.peaks = peaks if isinstance(peaks, SAXSPeakInfo) else SAXSPeakInfo()

        self.metadata = {}     

    def add_data(
        self,
        name: Annotated[str, "Name of attribute to add"],
        x: Annotated[np.array, "q"],
        y: Annotated[np.array, "y(x)"],
        dy: Annotated[np.array | None, "Error on y"] = None,
        infos: Annotated[dict[str, Any] | None, "Dictionnary about additionals infos"] = None
    ):
        """
        Dynamically adds a new SAXSData curve to the experiment.

        Parameters
        ----------
        name : str
            The name of the SAXSData attribute to create.
        x : np.array
            The x values (e.g., scattering vector).
        y : np.array
            The y values (e.g., intensity).
        dy : np.array, optional
            The uncertainty in y values. If None, defaults to 5% of y.
        infos : dict[str, Any], optional
            A dictionary containing additional information about the measurement. Defaults to None.

        Raises
        ------
        ValueError
            If the name is not a valid Python identifier or if the shapes of x, y, dy do not match.
        """

        if not name.isidentifier():
            raise ValueError(f"'{name}' is not a valid attribute name.")
        
        new_data = SAXSData(x, y, dy, name=name, infos=infos)
        setattr(self, name, new_data)

    def apply_masks(self, qmin=None, qmax=None):
        """
        Apply the same mask to each SAXSData object in the experiment.

        Parameters
        ----------
        qmin : float, optional
            Minimum q value for the mask. If None, defaults to -np.inf.
        qmax : float, optional
            Maximum q value for the mask. If None, defaults to np.inf.

        Raises
        ------
        ValueError
            If qmin is greater than qmax.
        """
        if qmin is None:
            qmin = - np.inf
        if qmax is None:
            qmax = np.inf

        if qmin > qmax:
            raise ValueError(f"qmin must be inferior to qmax got ({qmin}, {qmax})")
        
        for _, attr_value in self.__dict__.items():
            if isinstance(attr_value, SAXSData):
                attr_value.apply_mask(xmin=qmin, xmax=qmax)

    def copy(self) -> SAXSExperiment:
        """
        Returns a deep copy of the SAXS experiment by copying each SAXSData object and other attributes.

        Returns
        -------
        SAXSExperiment
            A new instance of SAXSExperiment with copied SAXSData objects and attributes.
        """
        # Create a new instance without parameters
        new_exp = self.__class__.__new__(self.__class__)
        
        for attr_name, attr_value in self.__dict__.items():
            # For each attribute, if it's a SAXSData, we make a copy via its copy() method
            if isinstance(attr_value, SAXSData):
                setattr(new_exp, attr_name, attr_value.copy())
            else:
                # For the other attributes, we make a classic deep copy
                try:
                    setattr(new_exp, attr_name, copy.deepcopy(attr_value))
                except Exception as e:
                    print(f"Warning: couldn't deepcopy {attr_name} ({type(attr_value)}): {e}")

        return new_exp

    def plot(
        self,
        curves=None,
        show_uncertainty=False,
        peaks=False,
        show_peak_errorbars=False,
        plot_type='plot',
        ax=None,
        legend=True,
        **kwargs
    ):
        """
        Wrapper for matplotlib plotting, allowing flexible display of SAXSData curves and peaks.

        Parameters
        ----------
        curves : list of str, optional
            Names of SAXSData attributes to plot. If None, plot all SAXSData in self.
        show_uncertainty : bool, optional
            If True, plot uncertainty (error bars or shaded area) for each curve.
        peaks : bool or list of str, optional
            If True, plot all peaks in self.peaks. If list of str, plot only those peaks. If False or None, do not plot peaks.
        show_peak_errorbars : bool, optional
            If True, show error bars for peak positions.
        plot_type : str, optional
            One of 'plot', 'semilogx', 'semilogy', 'loglog'.
        ax : matplotlib.axes.Axes, optional
            Axis to plot on. If None, creates a new figure.
        legend : bool, optional
            Whether to show legend.
        **kwargs : dict
            Additional keyword arguments passed to the plot function.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axis with the plotted data.
        
        Raises
        ------
        ValueError
            If `plot_type` is not one of the recognized types ('plot', 'semilogx', 'semilogy', 'loglog').
        Warning
            If a specified curve or peak is not found in the experiment.
        """

        if ax is None:
            _, ax = plt.subplots()

        # Sélection des courbes à tracer
        if curves is None:
            curves = [k for k, v in self.__dict__.items() if isinstance(v, SAXSData)]

        plot_func = getattr(ax, plot_type, None)
        if plot_func is None:
            raise ValueError(f"Unknown plot_type: {plot_type}")

        for curve_name in curves:
            data = getattr(self, curve_name, None)
            if data is None:
                print(f"Warning: Curve '{curve_name}' not found in experiment.")
                continue
            x, y, dy = data.get_filtered_data()
            label = curve_name
            # 🔍 Ajout des pics détectés dans la légende
            peak_text = ""
            if peaks and hasattr(self, 'peaks') and len(curves) > 0:
                # Gestion des méthodes de détection
                if "Standard" in self.peaks:
                    '''print("Imin de la courbe est")
                    Imin = self.peaks["Standard"].get("Imin")
                    print(Imin)'''
                    q_std = self.peaks["Standard"].get("q_values", [])
                    if len(q_std) > 0:
                        peak_text += " | Std=" + ", ".join(f"{q:.3f}" for q in q_std)
                #if "Asym2Sig" in self.peaks:
                    #q_asym = self.peaks["Asym2Sig"].get("q_values", [])
                    #if len(q_asym) > 0:
                    #    peak_text += " | Asym=" + ", ".join(f"{q:.3f}" for q in q_asym)

            label = f"{label}{peak_text}"

            # Tracé principal
            plot_func(x, y, label=label, **kwargs)

            # --- Tracé de Imin * q^-m pour la courbe 'Iq' ---
            if peaks and curve_name == "Iq":
                if hasattr(self, "peaks") and "Standard" in self.peaks:
                    Imin = self.peaks["Standard"].get("Imin", None)
                else:
                    Imin = None

                m = self.metadata.get("power_law_order", None)

                if Imin is not None and m is not None:
                    y_powerlaw = Imin * x**(-m)
                    ax.plot(x, y_powerlaw, "--", color="grey", alpha=0.7)

            # --- Ajouter la droite Imin sur la courbe preprocess ---
            if peaks and curve_name == "Iq_preprocess":
               if hasattr(self, 'peaks') and "Standard" in self.peaks:
                    Imin = self.peaks["Standard"].get("Imin", None)
                    if Imin is not None:
                        ax.axhline(Imin, linestyle="--", color="grey", alpha=0.7, label="Imin")

            # Incertitudes (si demandées)
            if show_uncertainty and dy is not None:
                if plot_type in ['plot', 'semilogx', 'semilogy', 'loglog']:
                    ax.fill_between(x, y - dy, y + dy, alpha=0.2)


        # --- Affichage des pics sous forme de marqueurs (x) ---
        if peaks and hasattr(self, 'peaks') and len(curves) > 0:
            first_curve = getattr(self, curves[0], None)
            if first_curve is not None:
                x_first, y_first, _ = first_curve.get_filtered_data()
            if peaks is True:
                peak_names = list(self.peaks.names())
            elif isinstance(peaks, list):
                peak_names = peaks
            else:
                peak_names = []

            for peak_name in peak_names:
                if peak_name not in self.peaks:
                    print(f"Warning: Peak '{peak_name}' not found in experiment.peaks.")
                    continue
                peak_info = self.peaks[peak_name]
                q_vals = peak_info.get('q_values', None)
                q_err = peak_info.get('q_values_std', None)
                if q_vals is not None:
                    y_peaks = np.interp(q_vals, x_first, y_first)
                    if show_peak_errorbars and q_err is not None:
                        ax.errorbar(q_vals, y_peaks, xerr=q_err, fmt='x', label=f"{peak_name} peaks")
                    else:
                        ax.plot(q_vals, y_peaks, 'x', label=f"{peak_name} peaks")

        if legend:
            ax.legend()
        ax.set_xlabel("q (Å⁻¹)")
        ax.set_ylabel("Intensity")
        ax.set_title(self.name if self.name else "SAXS Experiment")
        plt.tight_layout()
        if ax is None:
            plt.show()
        return ax

    def gaussian(self, sigma=1) -> SAXSExperiment:
        """
        Apply Gaussian smoothing to the y data of the Iq_preprocess attribute.
        
        Parameters
        ----------
        sigma : float, optional
            Standard deviation for Gaussian kernel. Default is 1.
        """
        data = getattr(self.Iq_preprocess, 'y')
        smoothed = gaussian_filter1d(data, sigma=sigma)
        setattr(self.Iq_preprocess, 'y', smoothed)
        return self

    def moving_average(self, window=5) -> SAXSExperiment:
        """
        Apply a moving average filter to the y data of the Iq_preprocess attribute.
        
        Parameters
        ----------
        window : int
            Size of the moving average window. Must be at least 1.
        
        Raises
        ------
        ValueError
            If window size is less than 1, or if it is larger than the length of the data, or if it is even.
        """
        if window < 1:
            raise ValueError("Window size must be at least 1")
        elif window == 1:
            return self
        elif window > len(getattr(self.Iq_preprocess, 'y')):
            raise ValueError("Window size is larger than data length")
        elif window % 2 == 0:
            window += 1
        
        data = getattr(self.Iq_preprocess, 'y')
        kernel = np.ones(window) / window
        smoothed = np.convolve(data, kernel, mode='same')
        setattr(self.Iq_preprocess, 'y', smoothed)
        return self

    def savgol(self, window=11, polyorder=3) -> SAXSExperiment:
        """
        Apply a Savitzky-Golay filter to the y data of the Iq_preprocess attribute.
        
        Parameters
        ----------
        window : int
            Size of the filter window. Must be odd and at least 3.
        polyorder : int
            Order of the polynomial used to fit the samples. Must be less than window size.
        
        Raises
        ------
        ValueError
            If window size is less than 3, or if polyorder is greater than or equal to window size, or if window size is even.
        """
        if window % 2 == 0:
            window += 1
        data = getattr(self.Iq_preprocess, 'y')
        if window > len(data):
            raise ValueError("window_length too large for data size")
        smoothed = savgol_filter(data, window, polyorder)
        setattr(self.Iq_preprocess, 'y', smoothed)
        return self
    
    def normalize(self) -> SAXSExperiment:
        """ Normalize the y data of the Iq_preprocess attribute to the range [0, 1]."""
        data = getattr(self.Iq_preprocess, 'y')
        norm = (data - np.min(data))/(np.max(data)-np.min(data))
        setattr(self.Iq_preprocess, 'y', norm)
        return self
    
    def up_scale(self, res=None, n_pt=1e4) -> SAXSExperiment:
        """
        Upscale the Iq_preprocess data to a finer resolution.

        Parameters
        ----------
        res : float, optional
            Resolution for the upscaling. If None, uses n_pt to determine the number of points.
        n_pt : int, optional
            Number of points for the upscaled data. Default is 10,000.
        """
        x_raw, y_raw, dy_raw = self.Iq_preprocess.get_raw_data()
        xmin, xmax = self.Iq_preprocess.get_mask_bounds()

        if res is not None:
            N = int((x_raw[-1] - x_raw[0])/res)
        else:
            N = int(n_pt)

        x_fine = np.linspace(x_raw[0], x_raw[-1], N)

        interp_y = interp1d(x_raw, y_raw, kind='cubic', bounds_error=False, fill_value='extrapolate')
        interp_dy = interp1d(x_raw, dy_raw, kind='cubic', bounds_error=False, fill_value='extrapolate')

        y_fine = interp_y(x_fine)
        dy_fine = interp_dy(x_fine)

        self.add_data('Iq_preprocess', x_fine, y_fine, dy_fine, self.Iq_preprocess.infos)
        self.Iq_preprocess.apply_mask(xmin=xmin, xmax=xmax)

        return self
    
    def feat_power_law(
        self,
        init_order: int | None=None,
        order_range=None,
        qpredetec = None,
        input_attr='Iq_preprocess',
        output_attr='Iq_preprocess',
        verbose=True
    ) -> SAXSExperiment:
        
 
        if hasattr(self, input_attr):
            attr_data = getattr(self, input_attr)
            q, y, dy = attr_data.get_filtered_data()
        else:
            raise AttributeError(f'{self.__class__.__name__} as no attribute {input_attr}')
        
        if init_order is not None:
            if order_range is None:
                y *= q**init_order
                dy *= q**init_order

                infos = {
                    'order': init_order
                }
                self.add_data(output_attr, q, y, dy, infos)
                return self

        y_flatten = y * q ** 3.0

        try:
            #print(qpredetec)
            if qpredetec is None :
                best_peaks_indices, _ = SAXSTools.find_peaks_dico(
                    y_flatten,
                    3,
                    20,
                    1e-6,
                    None,
                    False
                )
            else :
                best_peaks_indices = qpredetec
                
            '''print("1er methode", best_peaks_indices, best_peaks_indices[1], len(best_peaks_indices))
            #best_peaks_indices2, _ = SAXSTools.find_peaks_dico(
                    y_flatten,
                    3,
                    20,
                    1e-4,
                    None,
                    False
                )
            print("2eme methode",best_peaks_indices2)'''

            if best_peaks_indices is None or len(best_peaks_indices) < 2:
                raise ValueError("Not enough peaks for feat_power_law")
            

            def local_min_between(y, q, i1, i2):
                rel_idx = np.argmin(y[i1:i2])
                idx = i1 + rel_idx
                return idx, q[idx], y[idx]
            

            
            f_idx = best_peaks_indices[0]
            s_idx = best_peaks_indices[1]

            if len(best_peaks_indices) >= 3:
                t_idx = best_peaks_indices[2]
                
            else:
                t_idx = len(q) - 1
            
            # ===== DEBUG PLOT =====
            idx12, q12, I12 = local_min_between(y_flatten, q, f_idx, s_idx)
            idx23, q23, I23 = local_min_between(y_flatten, q, s_idx, t_idx)

            if verbose : 
                plt.figure(figsize=(8, 6))
                plt.loglog(q, y_flatten, label=r"$y \cdot q^3$", color="black")

                # Pics détectés
                plt.scatter(
                    q[best_peaks_indices],
                    y_flatten[best_peaks_indices],
                    color="red",
                    s=80,
                    zorder=5,
                    label="Detected peaks"
                )

                # Minima entre pics (pour m=3)
                plt.scatter(
                    [q12, q23],
                    [I12, I23],
                    color="blue",
                    s=80,
                    zorder=6,
                    label="Local minima (m=3)"
                )

                plt.xlabel("q")
                plt.ylabel(r"$I(q)\,q^3$")
                plt.legend()
                plt.title("DEBUG: y_flatten, peaks and minima")
                plt.show()


            def objective(m, q, y):
                m = m[0]
                y_scaled = y * q**m

                mean_val = np.mean(y_scaled)
                eps = 1e-12

                num = np.abs(y_scaled - mean_val)
                den = np.abs(y_scaled + mean_val) + eps

                return np.sum(num / den)
            

            def constraint_equal_minima(m, q, y, f_idx, s_idx, t_idx):
                m = m[0]
                y_scaled = y * q**m

                min1 = np.min(y_scaled[f_idx:s_idx])
                min2 = np.min(y_scaled[s_idx:t_idx])

                return min1 - min2

            from scipy.optimize import NonlinearConstraint

            constraint = NonlinearConstraint(
                fun=lambda m: constraint_equal_minima(m, q, y, f_idx, s_idx, t_idx),
                lb=0.0,
                ub=0.0)
            
            result = minimize(
                fun=objective,
                x0=[3.0],
                args=(q, y),
                method="trust-constr",
                bounds=[(2.0, 6)],
                constraints=[constraint],
                options={
                    "verbose": 3 if verbose else 0,
                    "gtol": 1e-8,
                    "xtol": 1e-8,
                    "maxiter": 500
                }
            )


            if not result.success:
                raise ValueError("Optimization failed")

            opt_order = result.x[0]

            y_opt = y * q**opt_order

            idx12_opt, q12_opt, I12_opt = local_min_between(y_opt, q, f_idx, s_idx)
            idx23_opt, q23_opt, I23_opt = local_min_between(y_opt, q, s_idx, t_idx)

            if verbose: 
                print('ok')
                plt.figure(figsize=(8, 6))
                plt.loglog(q, y_opt, label=fr"$y \cdot q^{opt_order:.2f}$", color="black")

                plt.scatter(
                    q[best_peaks_indices],
                    y_opt[best_peaks_indices],
                    color="red",
                    s=80,
                    label="Peaks (same indices)"
                )

                plt.scatter(
                    [q12_opt, q23_opt],
                    [I12_opt, I23_opt],
                    color="green",
                    s=80,
                    label="Minima after optimization"
                )

                plt.legend()
                plt.title("After optimization")
                plt.show()

        except Exception as e:
            print(f"[feat_power_law] fallback → cancel_power_law ({e})")

            return self.cancel_power_law(
                init_order=init_order,
                order_range=order_range,
                input_attr=input_attr,
                output_attr=output_attr,
                verbose=verbose
            )

        y *= q**opt_order
        dy *= q**opt_order

        infos = {
            'order': opt_order,
            'Intmin': I12_opt if 'I12_opt' in locals() else None
        }
        self.add_data(output_attr, q, y, dy, infos)

        return self


    def cancel_power_law(
        self,
        init_order: int | None=None,
        order_range=None,
        input_attr='Iq_preprocess',
        output_attr='Iq_preprocess',
        verbose=False
    ) -> SAXSExperiment:
        """
        Cancels a power law trend in scattering data by multiplying the intensity
        by q raised to an estimated or given exponent. This is typically used to flatten curves
        in log-log scale by compensating for power-law behavior in SAXS data.

        Parameters
        ----------
        init_order : int or None, optional
            Initial guess or fixed power law exponent. If provided without `order_range`,
            the data is multiplied by q^init_order directly.
        order_range : tuple or None, optional
            Range of values (min, max) for optimizing the power law exponent. If provided,
            a fit is performed to find the optimal exponent within this range.
        input_attr : str, optional
            Name of the attribute from which to retrieve the input data (default is 'Iq_preprocess').
        output_attr : str, optional
            Name of the attribute to store the result after cancelling the power law (default is 'Iq_preprocess').
        verbose : bool, optional
            If True, prints detailed optimization output during the minimization process.

        Returns
        -------
        self : SAXSExperiment
            Returns the instance of the class with the updated attribute containing corrected data
            and associated information, including the used or optimized power law order.

        Raises
        -------
        AttributeError :
            If the specified input attribute does not exist in the SAXSExperiment object.
        RuntimeError :
            If the optimization process for determining the power law exponent fails.
        """
        
        if hasattr(self, input_attr):
            attr_data = getattr(self, input_attr)
            q, y, dy = attr_data.get_filtered_data()
        else:
            raise AttributeError(f'{self.__class__.__name__} as no attribute {input_attr}')
        
        if init_order is not None:
            if order_range is None:
                y *= q**init_order
                dy *= q**init_order

                infos = {
                    'order': init_order
                }
                self.add_data(output_attr, q, y, dy, infos)
                return self
            else:
                p0 = [init_order]
                bounds = [order_range]
        else:
            if order_range is not None:
                p0 = [sum(order_range)/len(order_range)]
                bounds = [order_range]
            else:
                p0 = 3.5
                bounds = [(2.5, 4.0)]
        
        def model(x, m) -> np.array:
            """ Power law model """
            return x ** m
        
        
        def loss(m, x, y) -> float:
            
            """ Compute a custom loss value for optimizing the power law exponent.

            This loss function evaluates the deviation of a scaled dataset from its mean,
            using a symmetric normalized difference. It is designed to be minimized
            during optimization of the power-law exponent in scattering data analysis.

            Parameters
            ----------
            m : float
                The exponent value used in the power law model.
            x : array-like
                The independent variable (typically the q-vector).
            y : array-like
                The dependent variable (typically intensity values).

            Returns
            -------
            int
                The computed loss as a scalar value. (Note: the actual return is a float,
                so the return type hint should be corrected to float.)"""

            
            return sum(y * abs(y * model(x, m) - np.mean(y * model(x, m)))/abs(y * model(x, m) + np.mean(y * model(x, m))))
        
        result = minimize(
            loss,
            p0,
            method='Powell',
            args=(q, y),
            bounds=bounds,
            options={'disp': verbose}
        )

        if not result.success:
            raise RuntimeError("Power law cancel failed: " + result.message)
        
        opt_order = result.x[0]

        y *= q**opt_order
        dy *= q**opt_order

        infos = {
            'order': opt_order,
            'Intmin': None
        }
        self.add_data(output_attr, q, y, dy, infos)

        return self

    def find_peaks_standard(
        self,
        input_attr: Annotated[str, "Name of the attribute"]='Iq_peakfinder',
        output_attr: Annotated[str, "Name of the property to add"]='Standard',
        n_expected_peaks: Annotated[int, "Number of peaks to find"] = 3,
        min_prominence: Annotated[float, "Minimal peak's prominence"] = 1e-6,
        max_prominence: Annotated[float  , "Maximal peak's prominence"] = None,
        max_iter: Annotated[int, "Maximum number of iterations"] = 20,
        verbose: Annotated[bool, "Activate verbose"] = False
    ) -> SAXSExperiment:
        """
        Detects a predefined number of peaks in scattering data using a prominence-based
        binary search strategy for optimal thresholding.

        This method analyzes residual scattering data and uses iterative binary search
        over the prominence parameter to detect approximately `n_expected_peaks` peaks.

        Parameters
        ----------
        input_attr : str, optional
            Name of the attribute containing the data to analyze.
        output_attr : str, optional
            Name under which to store the detected peaks in the `peaks` container.
        n_expected_peaks : int, optional
            Target number of peaks to detect in the data.
        min_prominence : float, optional
            Minimum prominence to start the binary search.
        max_prominence : float or None, optional
            Maximum prominence for binary search; estimated from data if None.
        max_iter : int, optional
            Maximum number of iterations for the prominence binary search.
        verbose : bool, optional
            If True, prints progress and diagnostic information.

        Returns
        -------
        SAXSExperiment
            The current SAXSExperiment instance with updated peak information.

        Raises
        ------
        AttributeError
            If the specified `input_attr` is not found in the instance.
        """
        # Validate the input attribute exists and retrieve filtered data
        if hasattr(self, input_attr):
            attr_data = getattr(self, input_attr)
            q, y, dy = attr_data.get_filtered_data()
        else:
            raise AttributeError(f'{self.__class__.__name__} as no attribute {input_attr}')

        best_peak_indices, best_peak_properties = SAXSTools.find_peaks_dico(
                        y,
                        n_expected_peaks,
                        max_iter,
                        min_prominence,
                        max_prominence,
                        verbose)
        
        # Ensure indices are valid integers

        # --- CAS : aucun pic détecté ---
        if best_peak_indices is None or len(best_peak_indices) == 0:
            
            print(f"[File:{self.name}] ⚠ La recherche de pic a échoué.")

            # On stocke des valeurs None propres
            self.peaks.set(
                name=output_attr,
                q_values=None,
                q_values_std=None,
                FWHM=None,
                Imin=None,
                properties=None
            )

            return self
        
        else : 

            best_peak_indices = np.asarray(best_peak_indices, dtype=int)
            
            # Calculate FWHM for each detected peak
            FWHM = np.zeros_like(best_peak_indices, dtype=float)
            for i, _ in enumerate(best_peak_indices):
                left = int(np.round(best_peak_properties['left_ips'][i]))
                right = int(np.round(best_peak_properties['right_ips'][i]))
                peak_l_base = q[left]
                peak_r_base = q[right]
                FWHM[i] = peak_r_base - peak_l_base

            # Calcule la constante Imin de la courbe 
            if len(best_peak_indices) >= 2 :
                # Trouver I(q_peak)
                ind1 = best_peak_indices[0]
                ind2 = best_peak_indices[1]
                I_btw2peaks1 = y[ind1:ind2]  # float() pour sérialiser proprement
                #I_btw2peaks2 = float(y[ind1:ind2])
                I_min = np.min(I_btw2peaks1)
                # print(ind1, ind2, I_min)
            else :
                if len(best_peak_indices) < 2:
                    ind1 = best_peak_indices[0]
                    indd = len(y) - 1
                    I_btw2peaks = y[ind1:indd]
                    I_min = np.min(I_btw2peaks)

                else:
                    I_min = 0; 



            # Store the results in the SAXSPeakInfo object
            self.peaks.set(
                name=output_attr,
                q_values=q[best_peak_indices],
                q_values_std=dy[best_peak_indices],
                FWHM=FWHM,
                Imin = I_min,
                properties=best_peak_properties
            )

            return self
    
    def find_peaks_spv(
        self,
        input_attr: Annotated[str, "Name of the attribute"]='Iq_peakfinder',
        output_attr: Annotated[str, "Name of the property to add"]='SPV',
        verbose: Annotated[bool, "Activate verbose"] = False,
        plot: Annotated[bool, "Activate plotting iterations"] = False
    ) -> np.array:
        """
        Fit a single asymmetric peak in scattering data using a split pseudo-Voigt model.

        This method attempts to locate the main peak using a standard peak finder,
        estimates an initial model based on the detected peak, and fits the region
        using a custom split pseudo-Voigt function. Optionally, it can visualize both
        the initial guess and the final fitted model.

        Parameters
        ----------
        input_attr : str, optional
            Name of the data attribute containing the scattering curve (default is 'Iq_peakfinder').
        output_attr : str, optional
            Name of the peak property key under which results will be stored (default is 'SPV').
        verbose : bool, optional
            If True, prints information about detected peaks and fitting status.
        plot : bool, optional
            If True, displays plots showing initial guesses and final fitting results.

        Returns
        -------
        popt : np.array
            Optimized parameters of the split pseudo-Voigt model:
            [mu, amp1, sigma1, eta1, amp2, sigma2, eta2, offset].

        Raises
        ------
        AttributeError
            If the specified `input_attr` does not exist in the current instance.
        ValueError
            If no peak is found and the standard peak detection fails.
        """
        # Validate the input attribute exists and retrieve filtered data
        if hasattr(self, input_attr):
            attr_data = getattr(self, input_attr)
            q, y, dy = attr_data.copy().get_filtered_data()
        else:
            raise AttributeError(f'{self.__class__.__name__} as no attribute {input_attr}')
        
        # Create a temporary SAXSExperiment for peak analysis
        working_experiment = SAXSExperiment(
            {input_attr : (q, y, dy)},
            name=self.name
        )

        # === Use standard peak detection to find the most prominent peak ===
        working_experiment.find_peaks_standard(
            input_attr=input_attr,
            output_attr=output_attr,
            n_expected_peaks=1,
            verbose=verbose
        )

        if output_attr in working_experiment.peaks:
            peak_dict = working_experiment.peaks[output_attr]
        else:
            raise ValueError(f'{working_experiment.__class__.__name__}.peaks as no key {output_attr}')
        
        # === Extract initial peak information ===
        peak_q_value = peak_dict['q_values'][0]
        peak_q_value_std = peak_dict['q_values_std'][0]
        peak_prominence = peak_dict['prominences'][0]
        peak_height = peak_dict['peak_heights'][0]
        
        peak_l_base = q[int(peak_dict['left_ips'][0])]
        peak_r_base = q[int(peak_dict['right_ips'][0])]
        FWHM = (peak_r_base - peak_l_base)

        if verbose:
            print(f"[File:{self.name}] Peak found at q={peak_q_value:.4f} with prominence {peak_prominence:.4f} | width : {FWHM:.2f}")

        # === Define the splitted pseudo-Voigt model ===
        def model(x, mu, amp1, sig1, eta1, amp2, sig2, eta2, off) -> np.array:
            first_params = (amp1, mu, sig1, eta1, off)
            second_params = (amp2, mu, sig2, eta2, off)
            return SAXSTools.splitted_pseudo_voigt(x, first_params, second_params)

        # === Construct initial guess for the fit ===
        p0 = [
            peak_q_value,       # Center
            # Parameters of the first pseudo-voigt
            peak_height,        # Scale
            FWHM/2,             # Sigma
            0.5,                # Eta
            # Parameters of the second pseudo-voigt
            peak_height,        # Scale
            FWHM/2,             # Sigma
            0.5,                # Eta
            np.min(y)           # Offset
        ]

        # === Construct fitting bounds ===
        bounds = (
            (
                peak_q_value - peak_q_value_std,    # Center
                # === Bounds of the first pseudo-voigt ===
                0,          # Scale
                1e-10,      # Sigma
                0,          # Eta
                # === Bounds of the second pseudo-voigt ===
                0,          # Scale
                1e-10,      # Sigma
                0,          # Eta
                np.min(y) - 1e-3     # Offset
            ),
            (
                peak_q_value + peak_q_value_std,    # Center
                # === Bounds of the first pseudo-voigt ===
                np.inf,                             # Scale
                FWHM,       # Sigma
                1,          # Eta
                # === Bounds of the second pseudo-voigt ===
                np.inf,                             # Scale
                FWHM,       # Sigma
                1,          # Eta
                np.min(y) + 1e-3      # Offset
            )
        )
        
        # === Plot initial guess ===
        if plot:

            plt.figure()
            plt.loglog(q, y, label=input_attr)
            plt.loglog(q, model(q, *p0), label='SPV model')
            plt.plot(peak_q_value, peak_height, 'rx', label='Peak')
            plt.axvline(peak_l_base, color='g', label='left_base')
            plt.axvline(peak_r_base, color='g', label='right_base')
            plt.title('Initial guess')
            plt.ylabel('Intensity')
            plt.xlabel("q (Å⁻¹)")
            plt.tight_layout()
            plt.legend()
            plt.show()

        # Build a mask around the peak based on initial sigmas
        mu, _, sig1, _, _, sig2, _, _ = p0
        sector_mask = (q <= mu)
        k = 1  # Number of standard deviations around the center

        peak_mask = np.zeros_like(q, dtype=bool)
        peak_mask[sector_mask] = (np.abs(q[sector_mask] - mu) <= k * sig1)
        peak_mask[~sector_mask] = (np.abs(q[~sector_mask] - mu) <= k * sig2)

        # === Fit the model to the masked region ===
        try:
            popt, _ = curve_fit(
                model,
                q[peak_mask],
                y[peak_mask],
                p0=p0,
                bounds=bounds,
            )
        except Exception as e:
            print(f"[File:{self.name}] SPV peak finding failed: {e}")
            popt = p0

        # === Extract fitted parameters ===
        mu_opt, amp1_opt, sig1_opt, eta1_opt, amp2_opt, sig2_opt, eta2_opt, off_opt = popt
        y_model = model(q, *popt)

        # Estimate peak properties from model
        q_value = q[np.argmin(np.max(y_model)- y_model)]
        q_value_std = q_value * 0.05 # Assuming a 5% uncertainty on q_value
        FWHM_opt = sig1_opt + sig2_opt
        
        infos = SAXSTools.tuple_to_dict(
            entries=('amplitude', 'center', 'sigma', 'eta', 'offset'),
            values=(
                (amp1_opt, amp2_opt),
                (mu_opt, mu_opt),
                (sig1_opt, sig2_opt),
                (eta1_opt, eta2_opt),
                (off_opt, off_opt)
            )
        )

        # === Store peak properties === 
        self.peaks.set(
            name=output_attr,
            q_values=np.array((q_value,)),
            q_values_std=np.array((q_value_std,)),
            FWHM=np.array((FWHM_opt,)),
            properties=infos
        )
        
        # Store fitted model and residuals
        self.add_data('SPV_theory', q, y_model, infos=infos)
        self.add_data('SPV_residuals', q, y - y_model, infos=infos)

        # === Plot final result ===
        if plot:
            # Calculate peak's bounds (mu +/- 2 * sigma)
            q_lb = mu_opt - 2 * sig1_opt
            q_rb = mu_opt + 2 * sig2_opt

            plt.figure()
            plt.loglog(q, y, label=input_attr)
            plt.loglog(q, y_model, label=f'SPV model')
            plt.plot(q_value, model(q_value, *popt), 'rx', label='Peak')
            plt.axvline(q_lb, color='g', label='left_base')
            plt.axvline(q_rb, color='g', label='right_base')
            plt.title(f'Final optimization')
            plt.ylabel('Intensity')
            plt.xlabel("q (Å⁻¹)")
            plt.tight_layout()
            plt.legend()
            plt.show()

        return popt
    
    def find_peaks_spv_batch(
        self,
        n_expected_peaks: Annotated[int, "Number of peaks to find"]=3,
        input_attr: Annotated[str, "Name of the attribute"]='Iq_peakfinder',
        output_attr: Annotated[str, "Name of the property to add"]='SPV',
        verbose: Annotated[bool, "Activate verbose"] = False,
        plot: Annotated[bool, "Activate plotting iterations"] = False
    ) -> SAXSExperiment:
        """
        Find multiple peaks by fitting a batch of split pseudo-Voigt profiles with a sharing offset.

        This method fits multiple asymmetric peaks in the given dataset using a sequential 
        approach followed by a global optimization. The asymmetric peak profile is based on 
        the `splitted_pseudo_voigt` function from `SAXSTools` with a single offset.

        Parameters
        ----------
        n_expected_peaks : int, optional
            Number of peaks to find and fit (default is 3).
        input_attr : str, optional
            Name of the attribute containing the input data to analyze
            (default is 'Iq_peakfinder').
        output_attr : str, optional
            Name of the property where the peak fitting results will be stored
            (default is 'SPV').
        verbose : bool, optional
            If True, print progress and diagnostic messages during fitting
            (default is False).
        plot : bool, optional
            If True, display plots of initial and final fits for each peak and
            the combined global fit (default is False).

        Returns
        -------
        SAXSExperiment
            The current SAXSExperiment instance with peak and model data stored.

        Raises
        ------
        AttributeError
            If the specified `input_attr` is not found.
        ValueError
            If an invalid number of parameters is passed to the global model.

        Notes
        -----
        - Each peak is fitted sequentially using `find_peaks_spv`, and the data is masked 
        after each fit to avoid refitting the same peak.
        - A final global optimization is performed on all found peaks simultaneously.
        - Each peak is defined by eight parameters:
            center, amplitude1, sigma1, eta1, amplitude2, sigma2, eta2, offset.
        """

        # Validate the input attribute exists and retrieve filtered data
        if hasattr(self, input_attr):
            attr_data = getattr(self, input_attr)
            q, y, dy = attr_data.copy().get_filtered_data()
        else:
            raise AttributeError(f'{self.__class__.__name__} as no attribute {input_attr}')
        
        infos = {}
        all_popt = np.array(())
        n_found_peak = 0

        # Preserve initial data for global fit
        q_init = q.copy()
        y_init = y.copy()

        # === Sequentially detect and fit expected number of peaks ===
        for i in range(n_expected_peaks):
            
            # Create a temporary SAXSExperiment for peak analysis
            working_experiment = SAXSExperiment(
                {'working_curve' : (q, y, dy)},
                name=f"{self.name}_working_experiment_{i+1}"
            )

            try:
                # Detect and fit one peak
                popt = working_experiment.find_peaks_spv(
                    input_attr='working_curve',
                    output_attr='working_result',
                    verbose=verbose,
                    plot=False
                    )
            except Exception as e:
                print(f'[File:{self.name}][{i+1}/{n_expected_peaks}] SPV peak finding failed: {e}')
                break
            
            working_peak_dict = working_experiment.peaks.get('working_result')
            
            # Merge peak parameters
            if infos == {}:
                infos = working_peak_dict
            else:
                for key, value in working_peak_dict.items():
                    infos[key] = np.concatenate(
                        [np.atleast_1d(infos[key]), np.atleast_1d(value)]
                    )
            
            # Mask region around the peak to detect next
            working_peak_r_base = infos['center'][i][-1] + infos['sigma'][i][-1]
            working_experiment.apply_masks(qmin=working_peak_r_base)
            q, y, dy = working_experiment.SPV_residuals.get_filtered_data()

            all_popt = np.append(all_popt, popt)
            n_found_peak += 1


        def model(x, *all_popt) -> np.array:
            """
            Sum of multiple split pseudo-Voigt (SPV) profiles.

            Each peak is modeled by two pseudo-Voigt components sharing the same center (`mu`) and offset (`off`),
            but with different amplitudes (`amp1`, `amp2`), widths (`sig1`, `sig2`), and mixing parameters (`eta1`, `eta2`).

            Parameters
            ----------
            x : np.array
                The input array over which the combined SPV model is evaluated.
            *all_popt : float
                Flattened sequence of parameters for all peaks. Each peak requires
                8 parameters in the following order:
                    - mu (center position)
                    - amp1 (amplitude of the first pseudo-Voigt)
                    - sig1 (sigma of the first pseudo-Voigt)
                    - eta1 (mixing parameter of the first pseudo-Voigt)
                    - amp2 (amplitude of the second pseudo-Voigt)
                    - sig2 (sigma of the second pseudo-Voigt)
                    - eta2 (mixing parameter of the second pseudo-Voigt)
                    - off (offset shared by both components)

            Returns
            -------
            fun : np.array
                The summed intensity values of all SPV peaks evaluated at `x`.

            Raises
            ------
            ValueError
                If the total number of parameters is not a multiple of 8.
            """
            if len(all_popt) % 8 != 0:
                raise ValueError(f'all_popt must be a multiple of 8, got: {all_popt}')
            else:
                n = len(all_popt) // 8

            fun = np.zeros_like(x)

            # Sum contributions of each SPV peak
            for i in range(n):
                mu, amp1, sig1, eta1, amp2, sig2, eta2, off = all_popt[8*i: 8*(i+1)]
                first_p = (amp1, mu, sig1, eta1, off)
                second_p = (amp2, mu, sig2, eta2, off)
                fun += SAXSTools.splitted_pseudo_voigt(x, first_p, second_p)
            return fun
        
        # === Prepare initial parameters (p0) and bounds for global fit ===
        p0 = all_popt.copy()

        lower_bounds = []
        upper_bounds = []

        for i in range(n_found_peak):
            # Retrieve each set of fitted parameters
            mu, amp1, sig1, _, amp2, sig2, _, off = all_popt[8*i: 8*(i+1)]
            fw = infos['FWHM'][i]

            # Lower bounds
            lower_bounds.extend([
                mu - sig1,          # mu
                0,                  # amp1
                1e-10,              # sig1
                0.0,                # eta1
                0,                  # amp2
                1e-10,              # sig2
                0.0,                # eta2
                off - 1e-2          # off
            ])

            # Upper bounds
            upper_bounds.extend([
                mu + sig2,          # mu
                amp1 * 2,           # amp1
                fw,                 # sig1
                1.0,                # eta1
                amp2 * 2,           # amp2
                fw,                 # sig2
                1.0,                # eta2
                off + 1e-2          # off
            ])

            # Verify that all initial parameters are within the bounds
            for j, (p, lb, ub) in enumerate(zip(p0, lower_bounds, upper_bounds)):
                if not (lb <= p <= ub):
                    raise ValueError(f'[File:{self.name}] SPV parameter {j+1} of the peak n°{i+1} initial guess is out of bounds: lb={lb}; p0={p}; ub={ub}')

        bounds = (lower_bounds, upper_bounds)

        # === Perform final curve_fit on full model with all peaks ===
        try:
            popt_global, _ = curve_fit(
                model,
                q_init,
                y_init,
                p0=p0,
                bounds=bounds
            )
        except Exception as e:
            print(f"[File:{self.name}][Global SPV Optimization] Failed during curve_fit: {e}")
            popt_global = p0

        # === Store results and plot if required ===
        y_model_global = model(q_init, *popt_global)

        q_values_global = np.zeros_like(infos['q_values'])
        FWHM_global = np.zeros_like(infos['FWHM'])

        keys = ('amplitude', 'center', 'sigma', 'eta', 'offset')
        # Prepare lists to collect parameters for each peak
        param_lists = {k: [] for k in keys}

        for i in range(n_found_peak):
            popt = all_popt[8 * i : 8 * (i + 1)]
            mu, amp1, sig1, eta1, amp2, sig2, eta2, off = popt
            peak_model = model(q_init, *popt)
            q_values_global[i] = q_init[np.argmin(np.abs(peak_model - np.max(peak_model)))]
            FWHM_global[i] = sig1 + sig2

            # Collect parameters as tuples for each peak
            param_lists['amplitude'].append((amp1, amp2))
            param_lists['center'].append((mu, mu))
            param_lists['sigma'].append((sig1, sig2))
            param_lists['eta'].append((eta1, eta2))
            param_lists['offset'].append((off, off))

        # Convert lists to dict
        infos_global = {k: np.array(v) for k, v in param_lists.items()}

        q_values_std_global = q_values_global * 0.05  # Assuming a 5% uncertainty for all peaks (WIP)

        # === Save final fit as data in experiment ===
        self.add_data('SPV_theory', q_init, y_model_global, infos=infos_global)
        self.add_data('SPV_residuals', q_init, y_init - y_model_global, infos=infos_global)

        # === Store results in peaks object ===
        self.peaks.set(
            name=output_attr,
            q_values=q_values_global,
            q_values_std=q_values_std_global,
            FWHM=FWHM_global,
            properties=infos_global
        )

        # === Plot final result ===
        if plot:
            plt.figure()
            plt.loglog(q_init, y_init, label=input_attr)
            plt.loglog(q_init, y_model_global, label='SPV model')
            plt.plot(q_values_global, model(q_values_global, *popt_global), 'rx', label='Peaks')
            for i in range(n_found_peak):
                popt = popt_global[8 * i : 8 * (i + 1)]
                plt.loglog(q_init, model(q_init, *popt), '--', label=f'SPV n°{i+1}')
            plt.title(f'[File:{self.name}] Global SPV Optimization')
            plt.ylabel('Intensity')
            plt.xlabel("q (Å⁻¹)")
            plt.tight_layout()
            plt.legend()
            plt.show()
    
        return self

    def find_peaks_spv_off(
        self,
        input_attr: Annotated[str, "Name of the attribute"]='Iq_peakfinder',
        output_attr: Annotated[str, "Name of the property to add"]='SPV_off',
        verbose: Annotated[bool, "Activate verbose"] = False,
        plot: Annotated[bool, "Activate plotting iterations"] = False
    ) -> np.array:
        """
        Fit a single asymmetric peak using a split pseudo-Voigt model with independent offsets.

        This method detects a single peak in the given scattering curve, estimates an initial guess 
        for the peak position and shape, and then fits it using a split pseudo-Voigt (SPV) function 
        that allows independent background offsets on each side of the peak. The fitted parameters 
        and modeled curve are saved in the experiment for further analysis.

        Parameters
        ----------
        input_attr : str, optional
            Name of the data attribute containing the scattering curve (default is 'Iq_peakfinder').
        output_attr : str, optional
            Name of the peak property key under which results will be stored (default is 'SPV_off').
        verbose : bool, optional
            If True, prints information about detected peaks and fitting status.
        plot : bool, optional
            If True, displays plots showing initial guesses and final fitting results.

        Returns
        -------
        popt : np.array
            Optimized parameters of the split pseudo-Voigt model:
            [mu, scale, sigma1, eta1, offset1, sigma2, eta2, offset2].

        Raises
        ------
        AttributeError
            If the input attribute does not exist in the experiment.
        ValueError
            If the peak detection fails to produce a valid result.
        """

        # Validate the input attribute exists and retrieve filtered data
        if hasattr(self, input_attr):
            attr_data = getattr(self, input_attr)
            q, y, dy = attr_data.copy().get_filtered_data()
        else:
            raise AttributeError(f'{self.__class__.__name__} as no attribute {input_attr}')
        
        # Create a temporary SAXSExperiment for peak analysis
        working_experiment = SAXSExperiment(
            {input_attr : (q, y, dy)}
        )

        # === Use standard peak detection to find the most prominent peak ===
        working_experiment.find_peaks_standard(
            input_attr=input_attr,
            output_attr=output_attr,
            n_expected_peaks=1,
            verbose=verbose
        )

        if output_attr in working_experiment.peaks:
            peak_dict = working_experiment.peaks[output_attr]
        else:
            raise ValueError(f'{working_experiment.__class__.__name__}.peaks as no key {output_attr}')
        
        # === Extract initial peak information ===
        peak_q_value = peak_dict['q_values'][0]
        peak_q_value_std = peak_dict['q_values_std'][0]
        peak_prominence = peak_dict['prominences'][0]
        peak_height = peak_dict['peak_heights'][0]
        
        peak_l_base = q[int(peak_dict['left_ips'][0])]
        peak_r_base = q[int(peak_dict['right_ips'][0])]
        FWHM = (peak_r_base - peak_l_base)

        if verbose:
            print(f"[File:{self.name}] Peak found at q={peak_q_value:.4f} with prominence {peak_prominence:.4f} | width : {FWHM:.2f}")

        # === Define the splitted pseudo-Voigt model ===
        def model(x, mu, scale, sig1, eta1, off1, sig2, eta2, off2) -> np.array:
            """
            Evaluate a split pseudo-Voigt (SPV) profile with asymmetric offsets.

            This function models an asymmetric peak using two pseudo-Voigt profiles—
            one on each side of the central position `mu`—with shared center but separate 
            parameters for width (sigma), shape (eta), and offset.

            The amplitudes are computed from a shared `scale` and each offset to ensure
            continuity at the peak apex.

            Parameters
            ----------
            x : np.array
                The input array over which the model is evaluated.
            mu : float
                Center position of the peak.
            scale : float
                Total intensity at the peak apex. Used to compute amplitudes with offsets.
            sig1 : float
                Standard deviation (width) of the left-side pseudo-Voigt profile.
            eta1 : float
                Lorentzian fraction (0 to 1) of the left-side profile.
            off1 : float
                Baseline offset of the left-side profile.
            sig2 : float
                Standard deviation (width) of the right-side pseudo-Voigt profile.
            eta2 : float
                Lorentzian fraction (0 to 1) of the right-side profile.
            off2 : float
                Baseline offset of the right-side profile.

            Returns
            -------
            np.array
                The evaluated SPV model over the input `x`.

            Notes
            -----
            The amplitudes are derived internally as:
                amp1 = scale - off1
                amp2 = scale - off2
            This enforces continuity at `mu` and avoids directly optimizing the amplitudes.
            """
            amp1 = scale - off1
            amp2 = scale - off2
            first_params = (amp1, mu, sig1, eta1, off1)
            second_params = (amp2, mu, sig2, eta2, off2)
            return SAXSTools.splitted_pseudo_voigt(x, first_params, second_params)

        # === Construct initial guess for the fit ===
        p0 = [
            peak_q_value,       # Center
            peak_height,        # Scale
            # Parameters of the first pseudo-voigt
            FWHM/2,             # Sigma
            0.5,                # Eta
            np.min(y),          # Offset
            # Parameters of the second pseudo-voigt
            FWHM/2,             # Sigma
            0.5,                # Eta
            np.min(y)           # Offset
        ]

        # === Construct fitting bounds ===
        bounds = (
            (
                peak_q_value - peak_q_value_std,    # Center
                0,                                  # Scale
                # === Bounds of the first pseudo-voigt ===
                1e-10,      # Sigma
                0,          # Eta
                np.min(y) - 1e-3,    # Offset
                # === Bounds of the second pseudo-voigt ===
                1e-10,      # Sigma
                0,          # Eta
                np.min(y) - 1e-3     # Offset
            ),
            (
                peak_q_value + peak_q_value_std,    # Center
                np.inf,                             # Scale
                # === Bounds of the first pseudo-voigt ===
                FWHM,       # Sigma
                1,          # Eta
                np.min(y) + 1e-3,     # Offset
                # === Bounds of the second pseudo-voigt ===
                FWHM,       # Sigma
                1,          # Eta
                np.min(y) + 1e-3      # Offset
            )
        )
        
        # === Plot initial guess ===
        if plot:

            plt.figure()
            plt.loglog(q, y, label=input_attr)
            plt.loglog(q, model(q, *p0), label='SPV_off')
            plt.plot(peak_q_value, peak_height, 'rx', label='Peak')
            plt.axvline(peak_l_base, color='g', label='left_base')
            plt.axvline(peak_r_base, color='g', label='right_base')
            plt.title('Initial guess')
            plt.ylabel('Intensity')
            plt.xlabel("q (Å⁻¹)")
            plt.tight_layout()
            plt.legend()
            plt.show()

        # Build a mask around the peak based on initial sigmas
        mu, _, sig1, _, _, sig2, _, _ = p0
        sector_mask = (q <= mu)
        k = 1  # Standard deviation number

        peak_mask = np.zeros_like(q, dtype=bool)
        peak_mask[sector_mask] = (np.abs(q[sector_mask] - mu) <= k * sig1)
        peak_mask[~sector_mask] = (np.abs(q[~sector_mask] - mu) <= k * sig2)

        # === Fit the model to the masked region ===
        try:
            popt, _ = curve_fit(
                model,
                q[peak_mask],
                y[peak_mask],
                p0=p0,
                bounds=bounds,
            )
        except Exception as e:
            print(f"[File:{self.name}] SPV_off peak finding failed: {e}")
            popt = p0

        # === Extract fitted parameters ===
        mu_opt, scale_opt, sig1_opt, eta1_opt, off1_opt, sig2_opt, eta2_opt, off2_opt = popt
        amp1_opt = scale_opt - off1_opt
        amp2_opt = scale_opt - off2_opt
        y_model = model(q, *popt)

        # Estimate peak properties from model
        q_value = q[np.argmin(np.max(y_model)- y_model)]
        q_value_std = q_value * 0.05  # Assuming a 5% uncertainty for the peak position
        FWHM_opt = sig1_opt + sig2_opt
 
        infos = SAXSTools.tuple_to_dict(
            entries=('amplitude', 'center', 'sigma', 'eta', 'offset'),
            values=(
                ((amp1_opt, amp2_opt),
                (mu_opt, mu_opt),
                (sig1_opt, sig2_opt),
                (eta1_opt, eta2_opt),
                (off1_opt, off2_opt))
            )
        )

        # === Store peak properties ===
        self.peaks.set(
            name=output_attr,
            q_values=np.array((q_value,)),
            q_values_std=np.array((q_value_std,)),
            FWHM=np.array((FWHM_opt,)),
            properties=infos
        )

        # Store fitted model and residuals
        self.add_data('SPV_off_theory', q, y_model, infos=infos)
        self.add_data('SPV_off_residuals', q, y - y_model, infos=infos)

        # === Plot final result ===
        if plot:
            # Calculate peak's bounds (mu +/- 2 * sigma)
            q_lb = mu_opt - 2 * sig1_opt
            q_ub = mu_opt + 2 * sig2_opt

            plt.figure()
            plt.loglog(q, y, label=input_attr)
            plt.loglog(q, y_model, label=f'SPV_off')
            plt.plot(q_value, model(q_value, *popt), 'rx', label='Peak')
            plt.axvline(q_lb, color='g', label='left_base')
            plt.axvline(q_ub, color='g', label='right_base')
            plt.title(f'Final optimization')
            plt.ylabel('Intensity')
            plt.xlabel("q (Å⁻¹)")
            plt.tight_layout()
            plt.legend()
            plt.show()

        return popt
    
    def find_peaks_spv_off_batch(
        self,
        n_expected_peaks: Annotated[int, "Number of peaks to find"]=3,
        input_attr: Annotated[str, "Name of the attribute"]='Iq_peakfinder',
        output_attr: Annotated[str, "Name of the property to add"]='SPV_off',
        verbose: Annotated[bool, "Activate verbose"] = False,
        plot: Annotated[bool, "Activate plotting iterations"] = False
    ) -> SAXSExperiment:
        """
        Find multiple peaks by fitting a batch of split pseudo-Voigt profiles with different offset.

        This method fits multiple asymmetric peaks in the given dataset using a sequential 
        approach followed by a global optimization. The asymmetric peak profile is based on 
        the `splitted_pseudo_voigt` function from `SAXSTools` with different offset.

        Parameters
        ----------
        n_expected_peaks : int, optional
            Number of peaks to find and fit (default is 3).
        input_attr : str, optional
            Name of the attribute containing the input data to analyze
            (default is 'Iq_peakfinder').
        output_attr : str, optional
            Name of the property where the peak fitting results will be stored
            (default is 'SPV_off').
        verbose : bool, optional
            If True, print progress and diagnostic messages during fitting
            (default is False).
        plot : bool, optional
            If True, display plots of initial and final fits for each peak and
            the combined global fit (default is False).

        Returns
        -------
        SAXSExperiment
            The current SAXSExperiment instance with peak and model data stored.

        Raises
        ------
        AttributeError
            If the input attribute specified by `input_attr` does not exist.
        ValueError
            If global curve fitting fails or parameter arrays have inconsistent lengths.

        Notes
        -----
        - Each peak is fitted sequentially using `find_peaks_spv`, and the data is masked 
        after each fit to avoid refitting the same peak.
        - A final global optimization is performed on all found peaks simultaneously.
        - Each peak is defined by eight parameters:
            center, scale, sigma1, eta1, offset1, sigma2, eta2, offset2.
        """
        # Validate the input attribute exists and retrieve filtered data
        if hasattr(self, input_attr):
            attr_data = getattr(self, input_attr)
            q, y, dy = attr_data.copy().get_filtered_data()
        else:
            raise AttributeError(f'{self.__class__.__name__} as no attribute {input_attr}')
        
        infos = {}
        all_popt = np.array(())
        n_found_peak = 0

        # Preserve initial data for global fit
        q_init = q.copy()
        y_init = y.copy()

        # === Sequentially detect and fit expected number of peaks ===
        for i in range(n_expected_peaks):
            
            # Create a temporary SAXSExperiment for peak analysis
            working_experiment = SAXSExperiment(
                {'working_curve' : (q, y, dy)}
            )

            try:
                # Detect and fit one peak
                popt = working_experiment.find_peaks_spv_off(
                    input_attr='working_curve',
                    output_attr='working_result',
                    verbose=verbose,
                    plot=False
                    )
            except Exception as e:
                print(f'[File:{self.name}][{i+1}/{n_expected_peaks}] SPV_off peak finding failed: {e}')
                break
            
            working_peak_dict = working_experiment.peaks.get('working_result')
            
            # Merge peak parameters
            if infos == {}:
                infos = working_peak_dict
            else:
                for key, value in working_peak_dict.items():
                    infos[key] = np.concatenate(
                        [np.atleast_1d(infos[key]), np.atleast_1d(value)]
                    )
            
            # Mask region around the peak to detect next
            working_peak_r_base = infos['center'][i][-1] + infos['sigma'][i][-1]
            working_experiment.apply_masks(qmin=working_peak_r_base)
            q, y, dy = working_experiment.SPV_off_residuals.get_filtered_data()

            all_popt = np.append(all_popt, popt)
            n_found_peak += 1

        def model(x, *all_popt) -> np.array:
            """
            Sum of multiple split pseudo-Voigt (SPV) profiles.

            This model represents a superposition of `n` SPV peaks, where each peak
            is defined by 8 parameters describing two pseudo-Voigt components with
            asymmetric offsets. The total function is the sum of all individual peaks.

            Parameters
            ----------
            x : np.array
                The input array over which the combined SPV model is evaluated.
            *all_popt : float
                Flattened sequence of parameters for all peaks. Each peak requires
                8 parameters in the following order:
                    - mu (center position)
                    - scale (overall amplitude scale)
                    - sig1 (sigma of the first pseudo-Voigt component)
                    - eta1 (mixing parameter of the first pseudo-Voigt)
                    - off1 (offset of the first pseudo-Voigt)
                    - sig2 (sigma of the second pseudo-Voigt component)
                    - eta2 (mixing parameter of the second pseudo-Voigt)
                    - off2 (offset of the second pseudo-Voigt)

            Returns
            -------
            fun : np.array
                The summed intensity values of all SPV peaks evaluated at `x`.

            Raises
            ------
            ValueError
                If the total number of parameters is not a multiple of 8.
            """
            if len(all_popt) % 8 != 0:
                raise ValueError(f'all_popt must be a multiple of 8, got: {all_popt}')
            else:
                n = len(all_popt) // 8
            
            fun = np.zeros_like(x)

            # Sum contributions of each SPV peak
            for i in range(n):
                mu, scale, sig1, eta1, off1, sig2, eta2, off2 = all_popt[8*i: 8*(i+1)]
                amp1 = scale - off1
                amp2 = scale - off2
                first_p = (amp1, mu, sig1, eta1, off1)
                second_p = (amp2, mu, sig2, eta2, off2)
                fun += SAXSTools.splitted_pseudo_voigt(x, first_p, second_p)
            return fun
        
        # === Prepare initial parameters (p0) and bounds for global fit ===
        p0 = all_popt.copy()

        lower_bounds = []
        upper_bounds = []

        for i in range(n_found_peak):
            # Retrieve each set of fitted parameters
            mu, scale, sig1, _, off1, sig2, _, off2 = all_popt[8*i: 8*(i+1)]
            fw = infos['FWHM'][i]

            # === Lower bounds ===
            lower_bounds.extend([
                mu - sig1,            # mu
                0,                    # scale
                1e-10,                # sig1
                0.0,                  # eta1
                off1 - 1e-2,          # off1
                1e-10,                # sig2
                0.0,                  # eta2
                off2 - 1e-2           # off2
            ])

            # === Upper bounds ===
            upper_bounds.extend([
                mu + sig2,           # mu
                scale * 2,           # scale
                fw,                  # sig1
                1.0,                 # eta1
                off1 + 1e-2,         # off1
                fw,                  # sig2
                1.0,                 # eta2
                off2 + 1e-2          # off2
            ])

            # Verify that all initial parameters are within the bounds
            for j, (p, lb, ub) in enumerate(zip(p0, lower_bounds, upper_bounds)):
                if not (lb <= p <= ub):
                    raise ValueError(f'[File:{self.name}] SPV_off parameter {j+1} of the peak n°{i+1} initial guess is out of bounds: lb={lb}; p0={p}; ub={ub}')


        bounds = (lower_bounds, upper_bounds)

        # === Perform final curve_fit on full model with all peaks ===
        try:
            popt_global, _ = curve_fit(
                model,
                q_init,
                y_init,
                p0=p0,
                bounds=bounds
            )
        except Exception as e:
            print(f"[File:{self.name}][Global SPV_off Optimization] Failed during curve_fit: {e}")
            popt_global = p0

        # === Store results and plot if required ===
        y_model_global = model(q_init, *popt_global)

        q_values_global = np.zeros_like(infos['q_values'])
        FWHM_global = np.zeros_like(infos['FWHM'])

        keys = ('amplitude', 'center', 'sigma', 'eta', 'offset')
        # Prepare lists to collect parameters for each peak
        param_lists = {k: [] for k in keys}

        for i in range(n_found_peak):
            popt = all_popt[8 * i : 8 * (i + 1)]
            mu, scale, sig1, eta1, off1, sig2, eta2, off2 = popt
            amp1 = scale - off1
            amp2 = scale - off2

            peak_model = model(q_init, *popt)
            q_values_global[i] = q_init[np.argmin(np.abs(peak_model - np.max(peak_model)))]
            FWHM_global[i] = sig1 + sig2

            # Collect parameters as tuples for each peak
            param_lists['amplitude'].append((amp1, amp2))
            param_lists['center'].append((mu, mu))
            param_lists['sigma'].append((sig1, sig2))
            param_lists['eta'].append((eta1, eta2))
            param_lists['offset'].append((off1, off2))

        # Convert lists to dict
        infos_global = {k: np.array(v) for k, v in param_lists.items()}

        q_values_std_global = q_values_global * 0.05  # Assuming a 5% uncertainty for all peaks (WIP)     

        # === Save final fit as data in experiment ===
        self.add_data('SPV_off_theory', q_init, y_model_global, infos=infos_global)
        self.add_data('SPV_off_residuals', q_init, y_init - y_model_global, infos=infos_global)

        # === Store results in peaks object ===
        self.peaks.set(
            name=output_attr,
            q_values=q_values_global,
            q_values_std=q_values_std_global,
            FWHM=FWHM_global,
            properties=infos_global
        )

        # === Plot final result ===
        if plot:
            plt.figure()
            plt.loglog(q_init, y_init, label=input_attr)
            plt.loglog(q_init, y_model_global, label='SPV_off model')
            plt.plot(q_values_global, model(q_values_global, *popt_global), 'rx', label='Peaks')
            for i in range(n_found_peak):
                mu, scale, sig1, eta1, _, sig2, eta2, _ = popt_global[8 * i : 8 * (i + 1)]
                plt.loglog(q_init, model(q_init, mu, scale, sig1, eta1, 0, sig2, eta2, 0), '--', label=f'SPV_off n°{i+1}')
            plt.title(f'[File:{self.name}] Global SPV_off Optimization')
            plt.ylabel('Intensity')
            plt.xlabel("q (Å⁻¹)")
            plt.tight_layout()
            plt.legend()
            plt.show()
        
        return self

    def find_peaks_asym2sig(
        self,
        input_attr: Annotated[str, "Name of the attribute"]='Iq_peakfinder',
        output_attr: Annotated[str, "Name of the property to add"]='Asym2Sig',
        verbose: Annotated[bool, "Activate verbose"] = False,
        plot: Annotated[bool, "Activate plotting iterations"] = False
    ) -> tuple:
        """
        Detect and fit a single peak in the input data using an asymmetric two-sigmoid model.

        This method extracts data from a specified attribute, performs initial peak finding 
        using a standard method, then fits the peak with the asymmetric two-sigmoid function 
        `SAXSTools.asym2sig`. The fit accounts for peak asymmetry by allowing different widths 
        on either side of the peak center.

        Parameters
        ----------
        input_attr : str, optional
            Name of the attribute containing the data to analyze (default is 'Iq_peakfinder').
        output_attr : str, optional
            Name of the property to store the fit results in the peaks dictionary (default is 'Asym2Sig').
        verbose : bool, optional
            If True, prints detailed information during processing (default is False).
        plot : bool, optional
            If True, plots initial guess and final fit results (default is False).

        Returns
        -------
        tuple
            - popt (np.array): Optimized parameters of the asymmetric two-sigmoid fit.
            The parameters are (amplitude, center, w1, w2, w3, offset) where
            w1 is the distance between the two sigmoid centers,
            w2 and w3 describe the width variations on the low and high q sides, respectively.
            - (float, float): The calculated left and right base positions (q_lb, q_ub) defining the peak boundaries.

        Raises
        ------
        AttributeError
            If the input attribute is not found in the object.
        ValueError
            If the peak finding does not return expected results.
        """
        # Validate the input attribute exists and retrieve filtered data
        if hasattr(self, input_attr):
            attr_data = getattr(self, input_attr)
            q, y, dy = attr_data.copy().get_filtered_data()
        else:
            raise AttributeError(f'{self.__class__.__name__} as no attribute {input_attr}')

        # Create a temporary SAXSExperiment for peak analysis
        working_experiment = SAXSExperiment(
            {input_attr : (q, y, dy)},
            name=self.name
        )

        # === Use standard peak detection to find the most prominent peak ===
        working_experiment.find_peaks_standard(
            input_attr=input_attr,
            output_attr=output_attr,
            n_expected_peaks=1,
            verbose=verbose
        )
        
        if output_attr in working_experiment.peaks:
            peak_dict = working_experiment.peaks[output_attr]
        else:
            raise ValueError(f'[File:{self.name}]{working_experiment.__class__.__name__}.peaks as no key {output_attr}')
        
        peak_q_value = peak_dict['q_values'][0]
        peak_prominence = peak_dict['prominences'][0]
        FWHM = peak_dict['FWHM'][0]

        peak_height = peak_dict['peak_heights'][0]
        
        peak_l_base = q[int(peak_dict['left_ips'][0])]
        peak_r_base = q[int(peak_dict['right_ips'][0])]
        
        
        if verbose:
            print(f"[File:{self.name}] Peak found at q={peak_q_value:.4f} with prominence {peak_prominence:.4f} | width : {FWHM:.2f}")
        
        # === Construct initial guess for the fit ===
        p0 = [
            2*(peak_height),                        # Amplitude
            peak_q_value,                           # Center
            FWHM/4,                                 # w1 (distance between centers of the two sigmoids)
            abs(peak_q_value - peak_l_base)/2,      # w2 (low q var)
            abs(peak_q_value - peak_r_base)/2,      # w3 (high q var)
            0                                       # Offset
        ]

        # === Construct fitting bounds ===
        bounds = (
            (
                0,                                  # Amplitude
                peak_q_value - FWHM/8,              # Center
                0,                                  # w1 (distance between centers of the two sigmoids)
                0,                                  # w2 (low q var)
                0,                                  # w3 (high q var)
                -np.inf                             # Offset
            ),
            (
                np.inf,                             # Amplitude
                peak_q_value + FWHM/8,              # Center
                np.inf,                             # w1 (distance between centers of the two sigmoids)
                np.inf,                             # w2 (low q var)
                np.inf,                             # w3 (high q var)
                np.inf                              # Offset
            ),
        )
        
        # === Plot initial guess ===
        if plot:

            plt.figure()
            plt.loglog(q, y, label=input_attr)
            plt.loglog(q, SAXSTools.asym2sig(q, *p0), label='Asym2Sig')
            plt.plot(peak_q_value, peak_height, 'rx', label='Peak')
            plt.axvline(peak_l_base, color='g', label='left_base')
            plt.axvline(peak_r_base, color='g', label='right_base')
            plt.title('Initial guess')
            plt.ylabel('Intensity')
            plt.xlabel("q (Å⁻¹)")
            plt.tight_layout()
            plt.legend()
            plt.show()

        # Build a mask around the peak based on the  initial center and widths
        _, mu, w1, w2, w3, _ = p0

        sector_mask = (q <= mu)
        k = 2  # Number of standard deviations around each sigmoid center
        qmin = mu - w1/2 - k * w2
        qmax = mu + w1/2 + k * w3
        
        peak_mask = np.zeros_like(q, dtype=bool)
        peak_mask[sector_mask] = (q[sector_mask] >= qmin)
        peak_mask[~sector_mask] = (q[~sector_mask] <= qmax)

        # === Fit the model to the masked region ===
        try:
            popt, _ = curve_fit(
                SAXSTools.asym2sig,
                q[peak_mask],
                y[peak_mask],
                p0=p0,
                bounds=bounds,
            )
        except Exception as e:
            print(f"[File:{self.name}] Asym2Sig peak finding failed: {e}")
            popt = p0

        # === Extract fitted parameters ===
        _, mu_opt, w1_opt, w2_opt, w3_opt, _ = popt
        y_model = SAXSTools.asym2sig(q, *popt)

        # Estimate peak properties from SAXSTools.asym2sig
        q_value = q[np.argmin(np.max(y_model)- y_model)]
        q_value_std = q_value * 0.05  # Assuming a 5% uncertainty for the peak position

        q_lb = mu_opt - w1_opt/2 - k * w2_opt
        q_ub = mu_opt + w1_opt/2 + k * w3_opt
        FWHM_opt = w1_opt + w2_opt + w3_opt

        infos = SAXSTools.tuple_to_dict(
            entries=('amplitude', 'center', 'w1', 'w2', 'w3', 'offset'),
            values=popt
        )
        
        # === Store peak properties === 
        self.peaks.set(
            name=output_attr,
            q_values=np.array((q_value,)),
            q_values_std=np.array((q_value_std,)),
            FWHM=np.array((FWHM_opt,)),
            properties=infos
        )

        # Store fitted model and residuals
        self.add_data('asym2sig_theory', q, y_model, infos=infos)
        self.add_data('asym2sig_residuals', q, y - y_model, infos=infos)

        # === Plot final result ===
        if plot:
            plt.figure()
            plt.loglog(q, y, label=input_attr)
            plt.loglog(q, y_model, label=f'Asym2Sig')
            plt.plot(q_value, SAXSTools.asym2sig(q_value, *popt), 'rx', label='Peak')
            plt.axvline(q_lb, color='g', label='left_base')
            plt.axvline(q_ub, color='g', label='right_base')
            plt.title(f'Final optimization')
            plt.ylabel('Intensity')
            plt.xlabel("q (Å⁻¹)")
            plt.tight_layout()
            plt.legend()
            plt.show()

        return popt, (q_lb, q_ub)

    def find_peaks_asym2sig_batch(
        self,
        n_expected_peaks: Annotated[int, "Number of peaks to find"]=3,
        input_attr: Annotated[str, "Name of the attribute"]='Iq_peakfinder',
        output_attr: Annotated[str, "Name of the property to add"]='Asym2Sig',
        verbose: Annotated[bool, "Activate verbose"] = False,
        plot: Annotated[bool, "Activate plotting iterations"] = False
    ) -> SAXSExperiment:
        """
        Find and fit multiple peaks using the asymmetric two-sigmoid model.

        This method fits multiple asymmetric peaks in the given dataset using a sequential 
        approach followed by a global optimization. The asymmetric peak profile is based on 
        the `asym2sig` function from `SAXSTools`, which models skewed peaks using two sigmoids.

        Parameters
        ----------
        n_expected_peaks : int, optional
            The number of peaks expected in the data (default is 3).
        input_attr : str, optional
            Name of the attribute containing the SAXS data to be analyzed (default is 'Iq_peakfinder').
        output_attr : str, optional
            Name of the peak property to store results under (default is 'Asym2Sig').
        verbose : bool, optional
            If True, enables detailed print statements for debugging (default is False).
        plot : bool, optional
            If True, plots both initial guesses and final model results (default is False).

        Returns
        -------
        self : SAXSExperiment
            The current instance with peak fitting results stored in the `peaks` and `data` containers.

        Raises
        ------
        AttributeError
            If the specified input attribute is not found.
        ValueError
            If any initial parameter guesses are out of the defined fitting bounds.
        
        Notes
        -----
        - Each peak is fitted sequentially using `find_peaks_asym2sig`, and the data is masked 
        after each fit to avoid refitting the same peak.
        - A final global optimization is performed on all found peaks simultaneously.
        - Each peak is defined by six parameters: amplitude, center, w1, w2, w3, and offset.
        """

        # Validate the input attribute exists and retrieve filtered data
        if hasattr(self, input_attr):
            attr_data = getattr(self, input_attr)
            q, y, dy = attr_data.copy().get_filtered_data()
        else:
            raise AttributeError(f'{self.name} as no attribute {input_attr}')
        
        infos = {}
        all_popt = np.array(())
        global_mask_bounds = [None, None]
        n_found_peak = 0

        # Preserve initial data for global fit
        q_init = q.copy()
        y_init = y.copy()

        # === Sequentially detect and fit expected number of peaks ===
        for i in range(n_expected_peaks):
            
            # Create a temporary SAXSExperiment for peak analysis
            working_experiment = SAXSExperiment(
                {'working_curve' : (q, y, dy)},
                name=f'{self.name}_working_experiment_{i+1}'
            )

            try:
                # Detect and fit one peak
                popt, fit_mask_bounds = working_experiment.find_peaks_asym2sig(
                    input_attr='working_curve',
                    output_attr='working_result',
                    verbose=verbose,
                    plot=False
                    )
            except Exception as e:
                print(f'[File:{self.name}][{i+1}/{n_expected_peaks}] Asym2Sig peak finding failed: {e}')
                break
            
            working_peak_dict = working_experiment.peaks.get('working_result')
            
            # Merge peak parameters
            if infos == {}:
                infos = working_peak_dict
            else:
                for key, value in working_peak_dict.items():
                    infos[key] = np.concatenate(
                        [np.atleast_1d(infos[key]), np.atleast_1d(value)]
                    )
            # Mask region around the peak to detect next
            if global_mask_bounds[0] is None: # Query the left bounds of the first peak
                global_mask_bounds[0] = fit_mask_bounds[0]
            global_mask_bounds[-1] = fit_mask_bounds[-1] # Query the right bounds of the last peak
            working_experiment.apply_masks(qmin=fit_mask_bounds[-1])
            q, y, dy = working_experiment.asym2sig_residuals.get_filtered_data()

            all_popt = np.append(all_popt, popt)
            n_found_peak += 1

        def model(x, *all_popt) -> np.array:
            """
            Sum of multiple asymetric double sigmoids (Asym2Sig) profiles.

            Parameters
            ----------
            x : np.array
                The input array over which the combined Asym2Sig model is evaluated.

            *all_popt : float
                Flattened sequence of parameters for all peaks. Each peak requires
                6 parameters in the following order:
                    - A (amplitude)
                    - mu (center of the profile)
                    - w1 (distance beetween the two inflection points of the sigmoids function)
                    - w2 (width of the first sigmoid)
                    - w3 (width of the second sigmoid)
                    - off (offset of the profile)

            Returns
            -------
            fun : np.array
                The summed intensity values of all Asym2Sig peaks evaluated at `x`.

            Raises
            ------
            ValueError
                If the total number of parameters is not a multiple of 6.
            """
            if len(all_popt) % 6 != 0:
                raise ValueError(f'all_popt must be a multiple of 6, got: {all_popt}')
            else:
                n = len(all_popt) // 6

            fun = np.zeros_like(x)

            # Sum contributions of each SPV peak
            for i in range(n):
                fun += SAXSTools.asym2sig(x, *all_popt[6*i: 6*(1+i)])
            return fun

        # === Prepare initial parameters (p0) and bounds for global fit ===
        p0 = all_popt.copy()

        lower_bounds = []
        upper_bounds = []

        for i in range(n_found_peak):
            # === Lower bounds ===
            lower_bounds.extend([
                0,          # scale
                0,          # mu
                0,          # w1
                0,          # w2
                0,          # w3
                -np.inf     # off
            ])

            # === Upper bounds ===
            upper_bounds.extend([
                np.inf,     # scale
                np.inf,     # mu
                np.inf,     # w1
                np.inf,     # w2
                np.inf,     # w3
                np.inf      # off
            ])

            # Verify that all initial parameters are within the bounds
            for j, (p, lb, ub) in enumerate(zip(p0, lower_bounds, upper_bounds)):
                if not (lb <= p <= ub):
                    raise ValueError(f'[File:{self.name}] Asym2Sig parameter {j+1} of the peak n°{i+1} initial guess is out of bounds: lb={lb}; p0={p}; ub={ub}')

        bounds = (lower_bounds, upper_bounds)

        # === Perform final curve_fit on full model with all peaks ===
        try:
            # Create a mask to fit only the region where the peaks were initialy detected
            global_mask = (global_mask_bounds[0] <= q_init) & (q_init <= global_mask_bounds[-1])

            # Fit the masked curve
            popt_global, _ = curve_fit(
                model,
                q_init[global_mask],
                y_init[global_mask],
                p0=p0,
                bounds=bounds
            )

        except Exception as e:
            print(f"[File:{self.name}][Global Asym2Sig Optimization] Failed during curve_fit: {e}")
            popt_global = p0

        # === Store results and plot if required ===
        y_model_global = model(q_init, *popt_global)
        
        q_values_global = np.zeros_like(infos.get('q_values', np.array([])))
        FWHM_global = np.zeros_like(infos['FWHM'])

        keys = ('amplitude', 'center', 'w1', 'w2', 'w3', 'offset')
        
        # Prepare lists to collect parameters for each peak
        param_lists = {k: [] for k in keys}

        for i in range(n_found_peak):
            popt = all_popt[6 * i : 6 * (i + 1)]
            amp, mu, w1, w2, w3, off = popt
            peak_model = model(q_init, *popt)
            q_values_global[i] = q_init[np.argmin(np.abs(peak_model - np.max(peak_model)))]
            FWHM_global[i] = w1 + w2 + w3

            # Collect parameters as tuples for each peak
            param_lists['amplitude'].append(amp)
            param_lists['center'].append(mu)
            param_lists['w1'].append(w1)
            param_lists['w2'].append(w2)
            param_lists['w3'].append(w3)
            param_lists['offset'].append(off)


        # Convert lists to dict
        infos_global = {k: np.array(v) for k, v in param_lists.items()}

        q_values_std_global = q_values_global * 0.05  # Assuming a 5% uncertainty for all peaks (WIP)

        # === Save final fit as data in experiment ===
        self.add_data('Asym2Sig_theory', q_init, y_model_global, infos=infos_global)
        self.add_data('Asym2Sig_residuals', q_init, y_init - y_model_global, infos=infos_global)

        # === Store results in peaks object ===
        self.peaks.set(
            name=output_attr,
            q_values=q_values_global,
            q_values_std=q_values_std_global,
            FWHM=FWHM_global,
            properties=infos_global
        )

        if plot:
            plt.figure()
            plt.loglog(q_init, y_init, label=input_attr)
            plt.loglog(q_init, y_model_global, label='Asym2Sig model')
            plt.plot(q_values_global, model(q_values_global, *popt_global), 'rx', label='Peaks')
            for i in range(n_found_peak):
                A, mu, w1, w2, w3, _ = popt_global[6 * i : 6 * (i + 1)]
                plt.loglog(q_init, SAXSTools.asym2sig(q_init, A, mu, w1, w2, w3, np.min(q_init)), '--', label=f'Asym2Sig n°{i+1}')
            plt.axvspan(global_mask_bounds[0], global_mask_bounds[-1], color='g', alpha=0.1)
            plt.title('Global Asym2Sig Optimization')
            plt.ylabel('Intensity')
            plt.xlabel("q (Å⁻¹)")
            plt.tight_layout()
            plt.legend()
            plt.show()

        return self

class PreprocessPipeline:
    def __init__(self, batch):
        """
        Initialize a PreprocessPipeline instance.

        Parameters
        ----------
        batch : SAXSBatch
            The SAXSBatch instance containing experiments to preprocess.
        """
        self.batch = batch
        
    def savgol(self, **kwargs) -> PreprocessPipeline:
        """
        Apply Savitzky-Golay smoothing to all experiments in the batch.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed to each experiment's savgol method.

        Returns
        -------
        PreprocessPipeline
            Returns self to allow method chaining.
        """
        for exp in self.batch.experiments.values():
            exp.savgol(**kwargs)
        return self

    def up_scale(self, **kwargs) -> PreprocessPipeline:
        """
        Better the resolution of all experiments in the batch.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed to each experiment's up_scale method.

        Returns
        -------
        PreprocessPipeline
            Returns self to allow method chaining.
        """
        for exp in self.batch.experiments.values():
            exp.up_scale(**kwargs)
        return self
    
    def feat_power_law(self, average=False, bounds=(None, None), verbose=False) -> PreprocessPipeline:
        """
        Remove the power-law background from all experiments in the batch.

        This method optionally calculates an average power-law order from all experiments and applies
        it to each experiment for a consistent background subtraction.

        Parameters
        ----------
        average : bool, optional
            If True, compute the average power-law order across experiments and apply it; 
            if False, process each experiment independently (default is False).
        bounds : tuple, optional
            Tuple of (qmin, qmax) bounds for masking the data before power-law cancellation 
            (default is (None, None), i.e., no mask).
        verbose : bool, optional
            If True, print the average power-law order after processing (default is False).

        Returns
        -------
        PreprocessPipeline
            Returns self to allow method chaining.
        """
        
        avg_order = 0 # Initialize accumulator for averaging power-law order
        n = len(self.batch) # Number of experiments in the batch

        if average:
            # First pass: estimate power-law order for each experiment and accumulate average
            for exp in self.batch.experiments.values():
                exp.apply_masks(*bounds)        # Apply data mask within given bounds (qmin, qmax)
                exp.feat_power_law(
                    order_range=(2.5, 4.0),     # Restrict search to order values between 3 and 5
                    input_attr='Iq_preprocess',
                    output_attr='Iq_wo_pl'

                )
                order = exp.Iq_wo_pl.infos['order'] # Extract the fitted power-law order
                exp.metadata['power_law_order'] = order
                avg_order += order/n            # Accumulate for averaging
                exp.metadata['power_law_order'] = avg_order

            # Second pass: apply the averaged order as fixed parameter for all experiments
            for exp in self.batch.experiments.values():
                exp.apply_masks()               # Clear or re-apply default masks
                exp.feat_power_law(
                    init_order=avg_order,       # Use averaged order for consistent background removal
                    input_attr='Iq_preprocess',
                    output_attr='Iq_preprocess',
                    verbose = verbose
                )
        else:

            valid_orders = []

            for exp in self.batch.experiments.values():

                if not hasattr(exp, "metadata"):
                    exp.metadata = {}

                exp.apply_masks(*bounds)

                try:
                    exp.feat_power_law(
                        order_range=(2.5, 4.5),
                        input_attr='Iq_preprocess',
                        output_attr='Iq_preprocess',
                        verbose = verbose
                    )
                except Exception as e:
                    print(f"[WARNING] {exp.name} feat_power_law failed → {e}")
                    exp.metadata['power_law_order'] = None
                    exp.metadata['Intensity_min'] = None
                    continue

                infos = exp.Iq_preprocess.infos or {}

                order = infos.get('order', None)
                IntensityMin = infos.get('Intmin', None)

                exp.metadata['power_law_order'] = order
                exp.metadata['Intensity_min'] = IntensityMin

                if order is not None:
                    valid_orders.append(order)

                print(
                    f"{exp.name} | order={order} | Imin={IntensityMin}"
                )

            if verbose and len(valid_orders) > 0:
                avg_order = sum(valid_orders) / len(valid_orders)
                print(
                    f"Average order = {avg_order:.3f} "
                    f"({len(valid_orders)}/{len(self.batch)} valid samples)"
                )

            return self

    def cancel_power_law(self, average=False, bounds=(None, None), verbose=False) -> PreprocessPipeline:
        """
        Remove the power-law background from all experiments in the batch.

        This method optionally calculates an average power-law order from all experiments and applies
        it to each experiment for a consistent background subtraction.

        Parameters
        ----------
        average : bool, optional
            If True, compute the average power-law order across experiments and apply it; 
            if False, process each experiment independently (default is False).
        bounds : tuple, optional
            Tuple of (qmin, qmax) bounds for masking the data before power-law cancellation 
            (default is (None, None), i.e., no mask).
        verbose : bool, optional
            If True, print the average power-law order after processing (default is False).

        Returns
        -------
        PreprocessPipeline
            Returns self to allow method chaining.
        """
        
        avg_order = 0 # Initialize accumulator for averaging power-law order
        n = len(self.batch) # Number of experiments in the batch

        if average:
            # First pass: estimate power-law order for each experiment and accumulate average
            for exp in self.batch.experiments.values():
                exp.apply_masks(*bounds)        # Apply data mask within given bounds (qmin, qmax)
                exp.cancel_power_law(
                    order_range=(2.5, 5.0),     # Restrict search to order values between 3 and 5
                    input_attr='Iq_preprocess',
                    output_attr='Iq_wo_pl'

                )
                order = exp.Iq_wo_pl.infos['order'] # Extract the fitted power-law order
                exp.metadata['power_law_order'] = order
                avg_order += order/n            # Accumulate for averaging
                exp.metadata['power_law_order'] = avg_order

            # Second pass: apply the averaged order as fixed parameter for all experiments
            for exp in self.batch.experiments.values():
                exp.apply_masks()               # Clear or re-apply default masks
                exp.cancel_power_law(
                    init_order=avg_order,       # Use averaged order for consistent background removal
                    input_attr='Iq_preprocess',
                    output_attr='Iq_preprocess',
                )
        else:
            # === Process each experiment independently (no averaging) ===
            print("On calcule un ordre pour chaque exp")
            for exp in self.batch.experiments.values():
                exp.apply_masks(*bounds) # Apply mask for power-law fitting
                exp.cancel_power_law(
                    order_range=(2.5, 5.0),
                    input_attr='Iq_preprocess',
                    output_attr='Iq_preprocess'
                )
                order = exp.Iq_preprocess.infos['order'] # Extract fitted order
                exp.metadata['power_law_order'] = order
                order = exp.Iq_preprocess.infos['order']
                IntensityMin = exp.Iq_preprocess.infos['Intmin']


                exp.metadata['power_law_order'] = order
                exp.metadata['Intensity_min'] = IntensityMin
                avg_order += order/n # Accumulate average order, but applied separately here

        if verbose:
                print(f'Average order: n={avg_order} [{n} samples]') 

        return self

class PeakFinderPipeline:
    def __init__(self, batch):
        """
        Initialize a PeakFinderPipeline instance.

        This class manages peak finding on all experiments within a SAXSBatch instance
        using various methods.

        Parameters
        ----------
        batch : SAXSBatch
            The SAXSBatch instance containing experiments to analyze.
        """
        self.batch = batch

    def standard(self, **kwargs) -> PeakFinderPipeline:
        """
        Apply the standard peak finding method to all experiments in the batch.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed to each experiment's find_peaks_standard method.

        Returns
        -------
        self : PeakFinderPipeline
            Returns self to allow method chaining.
        """
        for exp in self.batch.experiments.values():
            exp.find_peaks_standard(**kwargs)
        return self

    def pv(self, **kwargs) -> PeakFinderPipeline:
        """
        Apply the Pseudo-Voigt (PV) peak finding batch method to all experiments.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed to each experiment's find_peaks_pv_batch method.

        Returns
        -------
        self : PeakFinderPipeline
            Returns self to allow method chaining.
        """
        for exp in self.batch.experiments.values():
            exp.find_peaks_pv_batch(**kwargs)
        return self

    def spv(self, **kwargs) -> PeakFinderPipeline:
        """
        Apply the Split Pseudo-Voigt (SPV) peak finding batch method to all experiments.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed to each experiment's find_peaks_spv_batch method.

        Returns
        -------
        self : PeakFinderPipeline
            Returns self to allow method chaining.
        """
        for exp in self.batch.experiments.values():
            exp.find_peaks_spv_batch(**kwargs)
        return self

    def spv_off(self, **kwargs) -> PeakFinderPipeline:
        """
        Apply the Split Pseudo-Voigt with offset (SPV Off) peak finding batch method to all experiments.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed to each experiment's find_peaks_spv_off_batch method.

        Returns
        -------
        self : PeakFinderPipeline
            Returns self to allow method chaining.
        """
        for exp in self.batch.experiments.values():
            exp.find_peaks_spv_off_batch(**kwargs)
        return self

    def asym2sig(self, **kwargs) -> PeakFinderPipeline:
        """
        Apply the Asymmetric Double Sigmoid (asym2sig) peak finding batch method to all experiments.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed to each experiment's find_peaks_asym2sig_batch method.

        Returns
        -------
        self : PeakFinderPipeline
            Returns self to allow method chaining.
        """
        for exp in self.batch.experiments.values():
            exp.find_peaks_asym2sig_batch(**kwargs)
        return self
    
    def full(self, **kwargs) -> PeakFinderPipeline:
        """
        Run all peak finding methods in sequence on all experiments in the batch.

        The sequence includes:
        - standard peak finding,
        - split pseudo-voigt (spv),
        - split pseudo-voigt with offset (spv_off),
        - asymmetric double sigmoid (asym2sig).

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed to each peak finding method.

        Returns
        -------
        self : PeakFinderPipeline
            Returns self to allow method chaining.
        """
        self.standard(min_prominence=1e-2, **kwargs)
        #self.pv(**kwargs) # not implemented yet
        self.spv(**kwargs)
        self.spv_off(**kwargs)
        self.asym2sig(**kwargs)

        return self

class AnalysisPipeline:
    def __init__(self, batch):
        """
        Initialize an AnalysisPipeline instance.

        This class performs analysis on peaks found after using the PeakFinderPipeline.

        Parameters
        ----------
        batch : SAXSBatch
            The SAXSBatch instance on which the analysis will be performed.
        """
        self.batch = batch
    
    def positions(self) -> dict:
        """
        Retrieve the q_values of detected peaks for all experiments and methods.

        Returns
        -------
        dict
            A nested dictionary where the first-level keys are experiment identifiers,
            and the second-level keys are peak detection method names. The values are
            arrays of q_values corresponding to detected peaks for each method in each experiment.

        Example
        -------
        {
            'experiment1': {
                'methodA': array([...]),
                'methodB': array([...]),
            },
            'experiment2': {
                'methodA': array([...]),
                ...
            },
            ...
        }
        """
        return {
            key: {method: exp.peaks.get_q_values(method)
                  for method in exp.peaks.names()}
            for key, exp in self.batch.experiments.items()
        }
    
    def IMax(self) -> dict:
        """
        Retrieve the Intensity of all the q_values of detected peaks for all experiments and method standard.

        Returns
        -------
        

        
        """
        result = {}
        for key, exp in self.batch.experiments.items():

            # Récupération des données filtrées
            x, y, dy = exp.Iq_preprocess.get_filtered_data()

           

            
            q_peaks = exp.peaks['Standard'].get('q_values', None)
            if q_peaks is None:
                continue

            # Trouver I(q_peak)
            indices = [np.argmin(np.abs(x - qpk)) for qpk in q_peaks]
            #print( "les indices calculés sont :")
            #print(indices)
            I_peaks = [float(y[i]) for i in indices]   # float() pour sérialiser proprement

            result[key] = I_peaks

            """# Pour chaque méthode : Standard, Asym2Sig, SPV, SPV_off…
            for method in exp.peaks.names():

                q_peaks = exp.peaks[method].get('q_values', None)
                if q_peaks is None:
                    continue

                # Trouver I(q_peak)
                indices = [np.argmin(np.abs(x - qpk)) for qpk in q_peaks]
                print(method)
                print( "les indices calculés sont :")
                print(indices)
                I_peaks = [float(y[i]) for i in indices]   # float() pour sérialiser proprement

                result[key][method] = I_peaks"""

        return result
    
    def IMin(self) -> dict: 

        Result = {}   # dictionnaire final : {exp_name: {method: [I_peaks] } }

        for key, exp in self.batch.experiments.items():

            # Récupération des données filtrées
            x, y, dy = exp.Iq_preprocess.get_filtered_data()

            

            q_peaks = exp.peaks['Standard']['q_values']
            # print(q_peaks)

            if q_peaks is None:
                continue

            if len(q_peaks) >= 2 :
                # Trouver I(q_peak)
                ind1 = np.argmin(np.abs(x - q_peaks[0]))
                ind2 = np.argmin(np.abs(x - q_peaks[1]))
                I_btw2peaks1 = y[ind1:ind2]  # float() pour sérialiser proprement
                #I_btw2peaks2 = float(y[ind1:ind2])
                I_min = np.min(I_btw2peaks1)
                # print(ind1, ind2, I_min)

            else :
                if len(q_peaks) < 2:
                    ind1 = np.argmin(np.abs(x - q_peaks[0]))
                    indd = np.argmin(np.abs(x - (len(y) - 1)))
                    I_btw2peaks = y[ind1:indd]
                    I_min = np.min(I_btw2peaks)

                else:
                    I_min = 0; 

            Result[key] = I_min


        return Result
                
            

    def ratios(self, ref: int = 0):
        """
        Calculate the ratio of each q_value to a reference q_value for all experiments and methods.

        Parameters
        ----------
        ref : int, optional
            Index of the reference peak in the q_values array to normalize against (default is 0).

        Returns
        -------
        result : dict
            A nested dictionary where the first-level keys are experiment identifiers,
            the second-level keys are peak detection method names, and the values are
            arrays of ratios of q_values normalized by the reference q_value.

        Notes
        -----
        If the reference index `ref` is out of bounds for a given q_values array,
        that method will be skipped for that experiment.

        Example
        -------
        {
            'experiment1': {
                'methodA': array([1.0, 1.5, 2.0]),
                'methodB': array([1.0, 1.2]),
            },
            'experiment2': {
                'methodA': array([1.0, 0.8, 1.3]),
                ...
            },
            ...
        }
        """
        result = {}
        for key, exp in self.batch.experiments.items():
            result[key] = {}
            for method in exp.peaks.names():
                q_values = exp.peaks.get_q_values(method)
                if len(q_values) > ref:
                    result[key][method] = q_values / q_values[ref]
        return result
    
    def fwhm(self):
        """
        Retrieve the Full Width at Half Maximum (FWHM) values for each peak detection method
        in each experiment within the batch.

        Returns
        -------
        dict
            Nested dictionary where keys are experiment identifiers and values are dictionaries
            with method names as keys and corresponding FWHM arrays as values.

        Notes
        -----
        If FWHM values are not available for a given method, that method is skipped.
        """
        result = {}
        for key, exp in self.batch.experiments.items():
            result[key] = {}
            for method in exp.peaks.names():
                fwhm_values = exp.peaks.get_FWHM(method)
                if fwhm_values is not None:
                    result[key][method] = fwhm_values
        return result

    def chi2(self, target=None, ref: int = 0, ddof: int = 0):
        """
        Calculate the Chi-squared statistic for each method in each experiment based on the peak ratios.
        The Chi-squared statistic is calculated as the sum of the squared differences between the ratios and the target,
        normalized by the target values.
        Parameters
        ----------
        target : list, optional
            Target values to compare against. If None, uses a linear target from 1 to n
            (where n is the number of peaks).
        ref : int, optional
            Reference index to normalize the ratios. Default is 0, meaning the first peak is used
            for normalization.
        ddof : int, optional
            Delta degrees of freedom for the Chi-squared calculation. Default is 0.
        
        Returns
        -------
        dict
            A nested dictionary with experiments as keys and methods as sub-keys.
            Each value is the Chi-squared statistic for that method in that experiment.
        """
        result = {}
        for key, exp in self.batch.experiments.items():
            result[key] = {}
            for method in exp.peaks.names():
                q_values = exp.peaks.get_q_values(method)
                if len(q_values) > ref:
                    ratios = q_values / q_values[ref]
                    if target is None:
                        target_arr = np.arange(1, len(ratios) + 1)
                    else:
                        target_arr = np.asarray(target)
                    if ratios.size == target_arr.size:
                        chi2_stat = np.sum((ratios - target_arr) ** 2 / target_arr)
                        result[key][method] = chi2_stat
        return result

    def z_score(self, target=None, ref: int = 0) -> dict:
        """
        Calculate the Z-score for each method in each experiment based on the peak ratios.
        The Z-score is calculated as the sum of the absolute differences between the ratios and the target,
        normalized by the square root of the sum of squares of the ratios and the target.

        Parameters
        ----------
        target : list, optional
            Target values to compare against. If None, uses a linear target from 1 to n
            (where n is the number of peaks).
        ref : int, optional
            Reference index to normalize the ratios. Default is 0, meaning the first peak is used
            for normalization.

        Returns
        -------
        dict
            A nested dictionary with experiments as keys and methods as sub-keys.
            Each value is the Z-score for that method in that experiment.
        """
        result = {}
        for key, exp in self.batch.experiments.items():
            result[key] = {}
            for method in exp.peaks.names():
                q_values = exp.peaks.get_q_values(method)
                if len(q_values) > ref:
                    ratios = q_values / q_values[ref]
                    if target is None:
                        target_arr = np.arange(1, len(ratios) + 1)
                    else:
                        target_arr = np.asarray(target)
                    z = np.sum(np.abs(ratios - target_arr) / np.sqrt(ratios ** 2 + target_arr ** 2)) * 100
                    result[key][method] = z
        return result
    
    def mean_peak_positions(self) -> dict:
        """
        Return a dictionary with the mean and standard deviation of each peak position
        (across all experiments and methods).

        Returns
        -------
        dict
            A dictionary where keys are method names and values are lists of tuples
            (mean, std) for each peak index across all experiments.
        """
        positions = self.positions()
        all_methods = set()
        for exp in self.batch.experiments.values():
            all_methods.update(exp.peaks.names())
        result = {}
        for method in all_methods:
            # Collect all peak positions for this method across experiments
            peaks_list = []
            for key in self.batch.experiments.keys():
                pos = positions.get(key, {}).get(method, None)
                if pos is not None:
                    peaks_list.append(pos)
            if not peaks_list:
                continue
            # Pad shorter lists with np.nan to align peak indices
            max_len = max(len(p) for p in peaks_list)
            arr = np.full((len(peaks_list), max_len), np.nan)
            for i, p in enumerate(peaks_list):
                arr[i, :len(p)] = p
            # Compute mean and std, ignoring nan
            means = np.nanmean(arr, axis=0)
            stds = np.nanstd(arr, axis=0, ddof=1)
            result[method] = list(zip(means, stds))
        return result
    
    def all(self):
        """
        Run all analysis methods and return a dictionary with the results.
        
        Returns
        -------
        dict
            A dictionary containing the results of all analysis methods:
            - 'positions': Peak positions for each method
            - 'ratios': Peak ratios for each method
            - 'fwhm': FWHM values for each method
            - 'chi2': Chi2 scores for each method
            - 'z_score': Z-scores for each method
            - 'mean_peak_positions': Mean and std of peak positions across experiments
        """
        return {
            'positions': self.positions(),
            'ratios': self.ratios(ref=0),
            'fwhm': self.fwhm(),
            'chi2': self.chi2(),
            'z_score': self.z_score(),
            'mean_peak_positions': self.mean_peak_positions(),
            'IMax': self.IMax(),
            'IMin' : self.IMin()
        }

    def overview(self, methods: list = None):
        """
        Plot peak positions and peak ratios for each peak index, for each method, over the experiments.
        Each peak is represented by 2 graphs: 
        - Left: peak position for each method over the experiments
        - Middle: peak ratio for each method over the experiments
        - Right: FWHM for each method over the experiments.

        Additionally, two bar plots are generated:
        - Left: chi2 metric as bar plot over the experiments (per method)
        - Right: z-score metric as bar plot over the experiments (per method)

        Parameters
        ----------
        methods : list, optional
            List of method names to include in the plots. If None, all methods are included.
        """

        positions = self.positions()
        ratios = self.ratios()
        chi2 = self.chi2()
        z_score = self.z_score()
        fwhm = self.fwhm()
        IMax = self.IMax()
        IMin = self.IMin()

        experiments = list(self.batch.experiments.keys())
        all_methods = set()
        # Collect all methods from all experiments
        for exp in self.batch.experiments.values():
            all_methods.update(exp.peaks.names())
        if methods is not None:
            methods = [m for m in methods if m in all_methods]
        else:
            methods = sorted(all_methods)
            
        x = np.arange(len(experiments))
        width = 0.8 / len(methods) if methods else 0.8

        # Determine the maximum number of peaks across all experiments and methods
        max_peaks = 0
        for key in experiments:
            for method in methods:
                pos = positions.get(key, {}).get(method, None)
                if pos is not None:
                    max_peaks = max(max_peaks, len(pos))

        fig, axs = plt.subplots(max_peaks +1, 4, figsize=(14, 4 * (max_peaks + 1)))
        if max_peaks == 1:
            axs = np.array([axs])

        #print("le nombre max de pic est ")
        #print(max_peaks)

        for peak_idx in range(max_peaks):
            # Left: Peak positions for each method over experiments
            ax_pos = axs[peak_idx, 0]
            for method in methods:
                y = []
                for key in experiments:
                    pos = positions.get(key, {}).get(method, None)
                    if pos is not None and len(pos) > peak_idx:
                        y.append(pos[peak_idx])
                    else:
                        y.append(np.nan)
                ax_pos.plot(x, y, marker='o', label=method)
                
            ax_pos.set_xticks(x, experiments, rotation=45)
            ax_pos.set_title(f"Peak n°{peak_idx+1} Position")
            ax_pos.set_xlabel("Experiment")
            ax_pos.set_ylabel("q (Å⁻¹)")
            ax_pos.legend(fontsize='small')

            # Left: Peak positions for each method over experiments
            
            if peak_idx == 0: 
                ax_posnm = axs[0, 1]
                for method in methods:
                    y = []
                    for key in experiments:
                        pos = positions.get(key, {}).get(method, None)
                        newpos = 2 * np.pi / (pos * 10)
                        if newpos is not None and len(newpos) > 0:
                            y.append(newpos[peak_idx])
                        else:
                            y.append(np.nan)
                    ax_posnm.plot(x, y, marker='o', label=method)
                    
                ax_posnm.set_xticks(x, experiments, rotation=45)
                ax_posnm.set_title(f"Peak n°{1} Position")
                ax_posnm.set_xlabel("Experiment")
                ax_posnm.set_ylabel("d en nm")
                ax_posnm.legend(fontsize='small')

            # Middle: Peak ratios for each method over experiments
            if peak_idx != 0:  # Skip the first peak ratio plot
                ax_ratio = axs[peak_idx, 1]
                for method in methods:
                    y = []
                    for key in experiments:
                        ratio = ratios.get(key, {}).get(method, None)
                        if ratio is not None and len(ratio) > peak_idx:
                            y.append(ratio[peak_idx])
                        else:
                            y.append(np.nan)
                    ax_ratio.plot(x, y, marker='o', label=method)
                
                ax_ratio.set_xticks(x, experiments, rotation=45)
                ax_ratio.set_title(f"Peak n°{peak_idx+1} Ratio")
                ax_ratio.set_xlabel("Experiment")
                ax_ratio.set_ylabel("q/q0")
                ax_ratio.legend(fontsize='small')
            
            
            # Right: FWHM for each method over experiments
            ax_fwhm = axs[peak_idx, 2]
            for method in methods:
                y = []
                for key in experiments:
                    f = fwhm.get(key, {}).get(method, None)
                    if f is not None and len(f) > peak_idx:
                        y.append(f[peak_idx])
                    else:
                        y.append(np.nan)
                ax_fwhm.plot(x, y, marker='o', label=method)
            
            ax_fwhm.set_xticks(x, experiments, rotation=45)
            ax_fwhm.set_title(f"Peak n°{peak_idx+1} FWHM")
            ax_fwhm.set_xlabel("Experiment")
            ax_fwhm.set_ylabel("FWHM (Å⁻¹)")
            ax_fwhm.legend(fontsize='small')

            # Right: Peak Intensity for each method over experiments
            ax_I = axs[peak_idx, 3]
            y = []
            for key in experiments:
                Ima = IMax.get(key, {})
                Imi = IMin.get(key, {})
                if Ima is not None and len(Ima) > peak_idx:
                    y.append(Ima[peak_idx]/Imi)
                else:
                    y.append(np.nan)
            ax_I.plot(x, y, marker='o', label='Standard')

            ax_I.set_xticks(x, experiments, rotation=45)
            ax_I.set_title(f"Intensity of the Peak n°{peak_idx+1} ")
            ax_I.set_xlabel("Experiment")
            ax_I.set_ylabel("I ")
            ax_I.legend(fontsize='small')

        # Chi2 bar plot (left, last row)
        ax_chi2 = axs[max_peaks, 0]
        for i, method in enumerate(methods):
            y = [chi2.get(key, {}).get(method, np.nan) for key in experiments]
            ax_chi2.bar(x + i * width, y, width=width, label=method)
        ax_chi2.set_title("Chi2 Metric per Experiment")
        ax_chi2.set_xlabel("Experiment")
        ax_chi2.set_ylabel("Chi2")
        ax_chi2.set_xticks(x + width * (len(methods) - 1) / 2)
        ax_chi2.set_xticklabels(experiments, rotation=45)
        ax_chi2.legend(fontsize='small')

        # Z-score bar plot (right, last row)
        ax_z = axs[max_peaks, 1]
        for i, method in enumerate(methods):
            y = [z_score.get(key, {}).get(method, np.nan) for key in experiments]
            ax_z.bar(x + i * width, y, width=width, label=method)
        ax_z.set_title("Z-score Metric per Experiment")
        ax_z.set_xlabel("Experiment")
        ax_z.set_ylabel("Z-score")
        ax_z.set_xticks(x + width * (len(methods) - 1) / 2)
        ax_z.set_xticklabels(experiments, rotation=45)
        ax_z.legend(fontsize='small')


        # Hide unused subplots
        axs[max_peaks, 2].axis('off')
        
        plt.tight_layout()
        plt.show()

    """def overview(self, methods: list = None):
        
        Plot peak positions and peak ratios for each peak index, for each method, over the experiments.
        Each peak is represented by 2 graphs: 
        - Left: peak position for each method over the experiments
        - Middle: peak ratio for each method over the experiments
        - Right: FWHM for each method over the experiments.

        Additionally, two bar plots are generated:
        - Left: chi2 metric as bar plot over the experiments (per method)
        - Right: z-score metric as bar plot over the experiments (per method)

        Parameters
        ----------
        methods : list, optional
            List of method names to include in the plots. If None, all methods are included.
        

        positions = self.positions()
        ratios = self.ratios()
        chi2 = self.chi2()
        z_score = self.z_score()
        fwhm = self.fwhm()

        experiments = list(self.batch.experiments.keys())
        all_methods = set()
        # Collect all methods from all experiments
        for exp in self.batch.experiments.values():
            all_methods.update(exp.peaks.names())
        if methods is not None:
            methods = [m for m in methods if m in all_methods]
        else:
            methods = sorted(all_methods)
            
        x = np.arange(len(experiments))
        width = 0.8 / len(methods) if methods else 0.8

        # Determine the maximum number of peaks across all experiments and methods
        max_peaks = 0
        for key in experiments:
            for method in methods:
                pos = positions.get(key, {}).get(method, None)
                if pos is not None:
                    max_peaks = max(max_peaks, len(pos))

        fig, axs = plt.subplots(max_peaks + 1, 3, figsize=(14, 4 * (max_peaks + 1)))
        if max_peaks == 1:
            axs = np.array([axs])

        for peak_idx in range(max_peaks):
            # Left: Peak positions for each method over experiments
            ax_pos = axs[peak_idx, 0]
            for method in methods:
                y = []
                for key in experiments:
                    pos = positions.get(key, {}).get(method, None)
                    if pos is not None and len(pos) > peak_idx:
                        y.append(pos[peak_idx])
                    else:
                        y.append(np.nan)
                ax_pos.plot(x, y, marker='o', label=method)
                
            ax_pos.set_xticks(x, experiments, rotation=45)
            ax_pos.set_title(f"Peak n°{peak_idx+1} Position")
            ax_pos.set_xlabel("Experiment")
            ax_pos.set_ylabel("q (Å⁻¹)")
            ax_pos.legend(fontsize='small')

            # Middle: Peak ratios for each method over experiments
            if peak_idx != 0:  # Skip the first peak ratio plot
                ax_ratio = axs[peak_idx, 1]
                for method in methods:
                    y = []
                    for key in experiments:
                        ratio = ratios.get(key, {}).get(method, None)
                        if ratio is not None and len(ratio) > peak_idx:
                            y.append(ratio[peak_idx])
                        else:
                            y.append(np.nan)
                    ax_ratio.plot(x, y, marker='o', label=method)
                
                ax_ratio.set_xticks(x, experiments, rotation=45)
                ax_ratio.set_title(f"Peak n°{peak_idx+1} Ratio")
                ax_ratio.set_xlabel("Experiment")
                ax_ratio.set_ylabel("q/q0")
                ax_ratio.legend(fontsize='small')
            else:
                axs[peak_idx, 1].axis('off')
            
            # Right: FWHM for each method over experiments
            ax_fwhm = axs[peak_idx, 2]
            for method in methods:
                y = []
                for key in experiments:
                    f = fwhm.get(key, {}).get(method, None)
                    if f is not None and len(f) > peak_idx:
                        y.append(f[peak_idx])
                    else:
                        y.append(np.nan)
                ax_fwhm.plot(x, y, marker='o', label=method)
            
            ax_fwhm.set_xticks(x, experiments, rotation=45)
            ax_fwhm.set_title(f"Peak n°{peak_idx+1} FWHM")
            ax_fwhm.set_xlabel("Experiment")
            ax_fwhm.set_ylabel("FWHM (Å⁻¹)")
            ax_fwhm.legend(fontsize='small')

        # Chi2 bar plot (left, last row)
        ax_chi2 = axs[max_peaks, 0]
        for i, method in enumerate(methods):
            y = [chi2.get(key, {}).get(method, np.nan) for key in experiments]
            ax_chi2.bar(x + i * width, y, width=width, label=method)
        ax_chi2.set_title("Chi2 Metric per Experiment")
        ax_chi2.set_xlabel("Experiment")
        ax_chi2.set_ylabel("Chi2")
        ax_chi2.set_xticks(x + width * (len(methods) - 1) / 2)
        ax_chi2.set_xticklabels(experiments, rotation=45)
        ax_chi2.legend(fontsize='small')

        # Z-score bar plot (right, last row)
        ax_z = axs[max_peaks, 1]
        for i, method in enumerate(methods):
            y = [z_score.get(key, {}).get(method, np.nan) for key in experiments]
            ax_z.bar(x + i * width, y, width=width, label=method)
        ax_z.set_title("Z-score Metric per Experiment")
        ax_z.set_xlabel("Experiment")
        ax_z.set_ylabel("Z-score")
        ax_z.set_xticks(x + width * (len(methods) - 1) / 2)
        ax_z.set_xticklabels(experiments, rotation=45)
        ax_z.legend(fontsize='small')

        # Hide unused subplots
        axs[max_peaks, 2].axis('off')
        
        plt.tight_layout()
        plt.show()"""
        


class SAXSBatch:
    def __init__(self, dir_path, sample_name, type, file_patern, sort_key):
        """
        Initialize a SAXSBatch object.
        This class is used to manage a batch of SAXS experiments, loading data from files,
        and providing methods for preprocessing, peak finding, and analysis.

        Parameters
        ----------
        dir_path : str
            Path to the directory containing the sample data.
        sample_name : str
            Name of the sample directory within dir_path.
        file_patern : str
            Pattern to match the files containing SAXS data (e.g., '*.dat').
        sort_key : callable
            Function to sort the files. It should take a file path as input and return a sortable key.
            Example: `lambda f: int(os.path.basename(f).split('_')[1])` to sort by a numeric identifier in the filename.
        """
        self.dir_path = dir_path
        self.sample_name = sample_name
        self.file_patern = file_patern
        self.sort_key = sort_key
        self.type =type

        self.sample_path = os.path.join(dir_path, sample_name)

        self.experiments = {}  # key = file id ; value = SAXSExperiment

        # Pipeline controllers
        self._preprocess = PreprocessPipeline(self)
        self._findpeaks = PeakFinderPipeline(self)
        self._analyse = AnalysisPipeline(self)

    def load(self)-> SAXSBatch:
       
        
        """            Load SAXS data files from the specified directory, sort them using the provided sort_key,
            and create SAXSExperiment objects for each file.

            Returns
            -------
            SAXSBatch
                The current SAXSBatch instance, allowing for method chaining.
            """
        file_list = glob.glob(os.path.join(self.sample_path, self.type, self.file_patern))

        file_list = sorted([file_path.replace(os.sep, '/') for file_path in file_list])
        try:
            file_list = sorted(file_list, key=self.sort_key)
        except Exception as e:
            raise ValueError(f"Sorting failed: {e}")
        if not file_list:
            raise ValueError("No files found.")

        for f in file_list:
            key = self.sort_key(f)
            q, Iq, dIq = self._load_file(f)
            name = os.path.basename(f)
            self.experiments[key] = SAXSExperiment({'Iq': (q, Iq, dIq)}, name=name)

        return self

        
    
    def loadseq(self)-> SAXSBatch:
        """
        Charge les fichiers du SAXSBatch en leur attribuant
        des numéros séquentiels (1, 2, 3, …) comme clé.
        """
        file_list = glob.glob(os.path.join( self.sample_path, self.file_patern))
        print(file_list)
        file_list = sorted([file_path.replace(os.sep, '/') for file_path in file_list])
        print(file_list)

        if not file_list:
            raise ValueError("Aucun fichier trouvé.")

        for i, f in enumerate(file_list, start=1):
            q, Iq, dIq = self._load_file(f)
            # Le nom du fichier sans chemin
            name = os.path.basename(f)
            # Ajoute l’expérience dans le batch
            self.experiments[i] = SAXSExperiment({'Iq': (q, Iq, dIq)}, name=name)
        return self
      
        

    def _load_file(self, file_path: str) -> tuple:
        """
        Load a SAXS data file and return the q, I(q), and dI(q) arrays.

        Parameters
        ----------
        file_path : str
            Path to the SAXS data file.
        
        Returns
        -------
        tuple
            A tuple containing three numpy arrays: q, I(q), and dI(q).
        """
        try:
            data = np.genfromtxt(file_path, comments='#', delimiter=None)
            if data.ndim == 1:
                raise ValueError(f"Fichier {file_path} invalide ou vide.")
        except Exception as e:
            raise ValueError(f"Erreur lors de la lecture de {file_path} : {e}")

        n_cols = data.shape[1]
        if n_cols == 2:
            q, Iq = data.T
            dIq = np.ones_like(Iq)
        elif n_cols == 3:
            q, Iq, dIq = data.T
        else:
            raise ValueError(f"Format non reconnu ({n_cols} colonnes) pour {file_path}")
        return q, Iq, dIq
    
    def plot(self, **kwargs) -> SAXSBatch:
        """
        Plot the SAXS data for of selection of experiments in the batch.
        
        Parameters
        ----------
        partial : int, optional
            If specified, only every nth experiment will be plotted (default is False).
        subplot : bool, optional
            If True, create subplots for each experiment (default is False).
        curves : str or list of str or tuple of lists of str, optional
            List of curves to plot for each experiment. Each curve can be a single string or a list of strings.
            Default is (['Iq'],).
        peaks : bool or list of bool, optional
            Whether to show peaks for each curve. If bool, applies to all curves.
            If list, should match the length of curves.
            Default is (True,).
        plot_type : str or tuple of str, optional
            Type of plot to use for each curve. Can be 'loglog', 'semilogx', 'semilogy', or 'plot'.
            If str, applies to all curves. If tuple, should match the length of curves.
            Default is ('loglog',).
        legend : bool or list of bool, optional
            Whether to show legend for each curve. If bool, applies to all curves.
            If list, should match the length of curves.
            Default is (True,).
        
        Returns
        -------
        SAXSBatch
            The current SAXSBatch instance, allowing for method chaining.

        Example
        -------
        >>> batch.plot(partial=5, subplot=True, curves=(['Iq'], ['Iq_preprocess']), peaks=(True, True), plot_type=('loglog', 'semilogx'), legend=(True, True))
        Plots the I(q) and I(q) preprocessed curves for every 5th experiment in subplots, showing peaks and using different plot types for each curve.
        """
        
        partial = kwargs.get('partial', False)
        subplot = kwargs.get('subplot', False)
        curves = kwargs.get('curves', (['Iq'],))
        peaks = kwargs.get('peaks', (True,))
        plot_type = kwargs.get('plot_type', ('loglog',))
        legend = kwargs.get('legend', (True,))

        # Validate input types
        if isinstance(curves, str):
            curves = [curves]

        if isinstance(peaks, bool):
            peaks = [peaks] * len(curves)
        elif isinstance(peaks, (list, tuple)):
            assert len(peaks) == len(curves), "Length of peaks must match length of curves"

        if isinstance(plot_type, str):
            plot_type = [plot_type] * len(curves)
        elif isinstance(plot_type, (list, tuple)):
            assert len(plot_type) == len(curves), "Length of plot_type must match length of curves"

        if isinstance(legend, bool):
            legend = [legend] * len(curves)
        elif isinstance(legend, (list, tuple)):
            assert len(legend) == len(curves), "Length of legend must match length of curves"

        if subplot:
            exp_items = list(self.experiments.items())
            if partial:
                exp_items = [item for i, item in enumerate(exp_items) if i % partial == 0]
            nrows = len(exp_items)
            ncols = len(curves)
            _, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
            for row, (_, exp) in enumerate(exp_items):
                for col, curve in enumerate(curves):
                    show_peaks = peaks[col] if col < len(peaks) else False
                    ptype = plot_type[col] if col < len(plot_type) else 'loglog'
                    show_legend = legend[col] if col < len(legend) else True
                    ax = axs[row, col]
                    exp.plot(curves=curve, peaks=show_peaks, plot_type=ptype, legend=show_legend, ax=ax)
            # plt.tight_layout()
            plt.show()
        else:
            if partial:
                for i, exp in enumerate(self.experiments.values()):
                    if i % kwargs.get('partial') == 0:
                        exp.plot(**{k: v for k, v in kwargs.items() if k != 'partial'})
            else:
                for exp in self.experiments.values():
                    exp.plot(**kwargs)
        return self
    
    def plot_raw_all(self, select=None, log=True):
        """
        Plot raw I(q) for all experiments (or a selected subset)
        before preprocessing.

        Parameters
        ----------
        select : list of int or list of str
            List of experiment indices or names to plot.
            - If None: plot all
        log : bool
            True → log-log plot. False → linear axes.
        """

        # Liste des expériences
        names = list(self.experiments.keys())
        exps = list(self.experiments.values())

        # Sélection des indices/noms
        if select is None:
            indices = range(len(exps))
        else:
            indices = []
            for s in select:
                if isinstance(s, int):     # sélection par index
                    indices.append(s)
                else:                     # sélection par nom
                    indices.append(names.index(s))

        # Figure
        plt.figure(figsize=(10, 6))

        for idx in indices:
            exp = exps[idx]
            name = names[idx]

            # Extraction des données brutes
            q = exp.Iq.q
            I = exp.Iq.I

            plt.plot(q, I, label=name, alpha=0.8)

        plt.xlabel("q (Å⁻¹)")
        plt.ylabel("Intensity")
        plt.title("Raw I(q) before preprocessing")

        if log:
            plt.xscale("log")
            plt.yscale("log")

        plt.legend(fontsize='small')
        plt.tight_layout()
        plt.show()


    
    def apply_masks(self, qmin=None, qmax=None)-> SAXSBatch:
        """
        Apply masks to all experiments in the batch.
        This method will call the `apply_masks` method of each SAXSExperiment in the batch.

        Parameters
        ----------
        qmin : float, optional
            Minimum q value for the mask. If None, defaults to -np.inf.
        qmax : float, optional
            Maximum q value for the mask. If None, defaults to np.inf.

        Returns
        -------
        SAXSBatch
            The current SAXSBatch instance, allowing for method chaining.
        """
        for exp in self.experiments.values():
            exp.apply_masks(qmin, qmax)
        return self
    
    def filter(self, cond: lambda i, k, v: bool) -> SAXSBatch:
        """
        Filter the experiments based on a condition function.
        
        Parameters
        ----------
        cond : callable
            Parameters
            ----------
            - i : int
                Index of the experiment in the experiment dictionnary
            - k : Any
                Key of the experiment
            - v : SAXSExperiment
                Value of the experiment

            Returns
            -------
            bool :
                Whether or not to keep the experiment
        
        Returns
        -------
        SAXSBatch
            The current SAXSBatch instance, allowing for method chaining.

        Examples
        -------
        >>> batch.filter(lambda i, k, v: i % 5 == 0)
        Keep one in five experiment in the experiment dictionnary

        >>> batch.filter(lambda i, k, v: 'SPV' in v.peaks.names())
        Keeps each experiment for which peaks were detected by the spv() method
        """
        filtered_experiments = {
            k: v for i, (k, v) in enumerate(self.experiments.items())
            if cond(i, k, v)
        }
        self.experiments = filtered_experiments
        return self
    

    @property
    def preprocess(self) -> PreprocessPipeline:
        """
        Call of the preprocess pipeline
        The first time this pipeline is called an Iq_preprocess attribute is created by copying the Iq attribute

        Returns
        -------
        self._preprocess : PreprocessPipeline
            The current PreprocessPipeline instance, allowing for method chaining.

        Raises
        ------
        AttributeError
            If 'Iq_preprocess' attribute couldn't be found in an experiment and this experiment doesn't have 'Iq' attribute
        """
        for exp in self.experiments.values():
            if not hasattr(exp, 'Iq_preprocess'):
                if hasattr(exp, 'Iq'):
                    exp.Iq_preprocess = exp.Iq.copy()
                else:
                    raise AttributeError(f"Couldn't find suitable data in {exp.__class__.__name__} to create 'Iq_preprocess'")

        return self._preprocess

    @property
    def findpeaks(self) -> PreprocessPipeline:
        """
        Call of the peakfinder pipeline
        The first time this pipeline is called an 'Iq_peakfinder' attribute is created.
        The creation is performed by copying the 'Iq_preprocess' attribute.
        If it doesn't exist, the copy is made over the 'Iq' attribute

        Returns
        -------
        self._peakfinder : PeakFinderPipeline
            The current PeakFinderPipeline instance, allowing for method chaining.

        Raises
        ------
        AttributeError
            If 'Iq_peakfinder' attribute couldn't be found in an experiment and this experiment doesn't have 'Iq_preprocess' attribute neither 'Iq' attribute
        """
        for exp in self.experiments.values():
            if not hasattr(exp, 'Iq_peakfinder'):
                if hasattr(exp, 'Iq_preprocess'):
                    exp.Iq_peakfinder = exp.Iq_preprocess.copy()
                elif hasattr(exp, 'Iq'):
                    exp.Iq_peakfinder = exp.Iq.copy()
                else:
                    raise AttributeError(f"Couldn't find suitable data in {exp.__class__.__name__} to create 'Iq_peakfinder'")

        return self._findpeaks
    
    @property
    def analyse(self) -> AnalysisPipeline:
        """
        Call of the analysis pipeline

        Returns
        -------
        self._analyse : AnalysisPipeline
            The current AnalysisPipeline instance, allowing for method chaining.
        """
        return self._analyse

    def __len__(self) -> int:
        return len(self.experiments)