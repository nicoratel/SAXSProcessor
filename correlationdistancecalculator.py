
from filereaders import h5File_ID02, h5File_SWING, EdfFile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from saxsprocessor import SAXSProcessor
from pathlib import Path
f
from scipy.signal import savgol_filter, find_peaks


# PyFAI imports
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import pyFAI.detectors

#

class CorrelationDistanceCalculator:
    """
    Dedicated class for peak detection and correlation distance calculation.
    Analyzes radial profiles to extract structural information.
    """
    
    def __init__(self, processor: SAXSProcessor):
        """
        Initialize correlation distance calculator.
        
        Parameters:
        -----------
        processor : SAXSDataProcessor
            SAXS data processor instance
        """
        self.processor = processor
        
    def detect_peaks(self, q, I,
                    nb_peaks=1,
                    window_length=15,
                    polyorder=3,
                    prominence=0.5,
                    distance_pts=20,
                    q_range=None,
                    plot=False):
        """
        Detect peaks in I(q) using second derivative method.
        
        Parameters:
        -----------
        q, I : arrays
            Radial profile data
        nb_peaks : int
            Number of peaks to detect
        window_length : int
            Savitzky-Golay filter window (must be odd)
        polyorder : int
            Polynomial order for smoothing
        prominence : float
            Minimum peak prominence
        distance_pts : int
            Minimum distance between peaks (points)
        q_range : tuple
            (qmin, qmax) to restrict search
        plot : bool
            Display results
            
        Returns:
        --------
        q_peaks : array
            Peak positions (Å⁻¹)
        """
        if window_length % 2 == 0:
            window_length += 1
            
        delta_q = q[1] - q[0]
        d2I = savgol_filter(I, window_length=window_length, polyorder=polyorder, deriv=2, delta=delta_q)
        inverted_d2I = -d2I

        mask = np.ones_like(q, dtype=bool)
        if q_range:
            mask &= (q >= q_range[0]) & (q <= q_range[1])

        peaks, properties = find_peaks(inverted_d2I[mask], prominence=prominence, distance=distance_pts)
        sorted_indices = np.argsort(properties["prominences"])[::-1]
        top_peaks = peaks[sorted_indices[:nb_peaks]]
        q_detected = q[mask][top_peaks]
        
        if plot:
            plt.figure(figsize=(10, 6))
            plt.loglog(q, I, label="I(q)", linewidth=2)
            colors = ['r', 'g', 'b', 'c', 'm', 'y']
            for i, qp in enumerate(q_detected[:nb_peaks]):
                plt.axvline(qp, color=colors[i % len(colors)], ls='--', 
                           label=f'Peak {i+1}: d = {2*np.pi/qp:.1f} Å', linewidth=2)
            plt.xlabel("q (Å⁻¹)", fontsize=12)
            plt.ylabel("I(q)", fontsize=12)
            plt.title("Peak Detection by Second Derivative Method", fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)
            plt.tight_layout()
            plt.show()
            
        return q_detected[:nb_peaks]
        
    def compute_correlation_distances(self, 
                                     nb_peaks=1, 
                                     azimuth: float = 90, 
                                     width: float = 40,
                                     window_length=15,
                                     polyorder=3,
                                     prominence=0.5,
                                     distance_pts=20,
                                     q_range=None,
                                     plot=False):
        """
        Compute correlation distances from radial profile peaks.
        
        Parameters:
        -----------
        nb_peaks : int
            Number of peaks to detect
        azimuth : float
            Azimuthal angle (°)
        width : float
            Angular sector width (°)
        window_length : int
            Savitzky-Golay filter window
        polyorder : int
            Polynomial order for smoothing
        prominence : float
            Minimum peak prominence
        distance_pts : int
            Minimum distance between peaks
        q_range : tuple
            (qmin, qmax) to restrict search
        plot : bool
            Display results
            
        Returns:
        --------
        results : dict
            Dictionary containing:
            - 'distances': array of correlation distances (Å)
            - 'q_peaks': array of peak positions (Å⁻¹)
            - 'q_profile': full q array
            - 'I_profile': full intensity array
        """
        # Extract radial profile
        q, I = self.processor.extract_radial_profile(azimuth=azimuth, width=width, save=False)
        
        # Detect peaks
        q_peaks = self.detect_peaks(
            q, I, 
            nb_peaks=nb_peaks,
            window_length=window_length,
            polyorder=polyorder,
            prominence=prominence,
            distance_pts=distance_pts,
            q_range=q_range,
            plot=plot
        )
        
        # Compute distances
        distances = 2 * np.pi / q_peaks
        
        # Print results
        print(f"\n{'=' * 60}")
        print(f"Correlation Distance Analysis")
        print(f"{'=' * 60}")
        print(f"Sample: {self.processor.samplename}")
        print(f"Azimuthal sector: {azimuth}° ± {width/2}°")
        print(f"\nDetected {len(q_peaks)} peak(s):")
        for i, (qp, d) in enumerate(zip(q_peaks, distances)):
            print(f"  Peak {i+1}: q = {qp:.4f} Å⁻¹ → d = {d:.1f} Å")
        print(f"{'=' * 60}")
        
        results = {
            'distances': distances,
            'q_peaks': q_peaks,
            'q_profile': q,
            'I_profile': I,
            'azimuth': azimuth,
            'width': width
        }
        
        return results
    
    def analyze_anisotropy(self, 
                          nb_peaks=1,
                          azimuth_list=[0, 45, 90, 135],
                          width=40,
                          q_range=None,
                          plot=True):
        """
        Analyze structural anisotropy by comparing correlation distances 
        in different azimuthal directions.
        
        Parameters:
        -----------
        nb_peaks : int
            Number of peaks to detect
        azimuth_list : list
            List of azimuthal angles to analyze (°)
        width : float
            Angular sector width (°)
        q_range : tuple
            (qmin, qmax) to restrict search
        plot : bool
            Display comparison plot
            
        Returns:
        --------
        anisotropy_results : dict
            Dictionary with results for each azimuth
        """
        anisotropy_results = {}
        
        for azimuth in azimuth_list:
            results = self.compute_correlation_distances(
                nb_peaks=nb_peaks,
                azimuth=azimuth,
                width=width,
                q_range=q_range,
                plot=False
            )
            anisotropy_results[azimuth] = results
        
        if plot and len(anisotropy_results) > 1:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot 1: Radial profiles
            for azimuth, results in anisotropy_results.items():
                axes[0].loglog(results['q_profile'], results['I_profile'], 
                              label=f"{azimuth}°", linewidth=2, alpha=0.7)
                for qp in results['q_peaks']:
                    axes[0].axvline(qp, linestyle='--', alpha=0.3)
            
            axes[0].set_xlabel("q (Å⁻¹)", fontsize=12)
            axes[0].set_ylabel("I(q)", fontsize=12)
            axes[0].set_title("Radial Profiles by Direction", fontsize=14, fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Correlation distances vs azimuth
            colors = ['r', 'g', 'b', 'c', 'm', 'y']
            for peak_idx in range(nb_peaks):
                azimuths = []
                distances = []
                for azimuth, results in anisotropy_results.items():
                    if peak_idx < len(results['distances']):
                        azimuths.append(azimuth)
                        distances.append(results['distances'][peak_idx])
                
                if len(azimuths) > 0:
                    axes[1].plot(azimuths, distances, 'o-', 
                               color=colors[peak_idx % len(colors)],
                               markersize=10, linewidth=2,
                               label=f"Peak {peak_idx+1}")
            
            axes[1].set_xlabel("Azimuthal Angle (°)", fontsize=12)
            axes[1].set_ylabel("Correlation Distance (Å)", fontsize=12)
            axes[1].set_title("Structural Anisotropy", fontsize=14, fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        return anisotropy_results