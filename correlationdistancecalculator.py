"""
CorrelationDistanceCalculator amélioré avec méthode SPV
Intégration de la détection de pics par Split Pseudo-Voigt
"""

from filereaders import h5File_ID02, h5File_SWING, EdfFile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from saxsprocessor import SAXSProcessor
from pathlib import Path

from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import gaussian_filter1d

# Import de la classe SAXSExperiment pour la méthode SPV
from saxs_analysis import SAXSExperiment


class CorrelationDistanceCalculator:
    """
    Calculateur de distances de corrélation avec détection de pics améliorée.
    
    Nouvelles fonctionnalités :
    - Méthode SPV (Split Pseudo-Voigt) pour une détection précise
    - Soustraction automatique de loi de puissance
    - Quantification des incertitudes
    - Support multi-pics avec fit global
    - Comparaison avec la méthode historique (dérivée seconde)
    """
    
    def __init__(self, processor: SAXSProcessor):
        """
        Initialize correlation distance calculator.
        
        Parameters:
        -----------
        processor : SAXSProcessor
            Instance du processeur SAXS
        """
        self.processor = processor
        
        # Définir les paramètres valides pour chaque méthode de détection
        self._valid_params = {
            'derivative': {
                'qmin', 'qmax', 'window_length', 'polyorder', 
                'prominence', 'distance_pts', 'plot'
            },
            'spv': {
                'qmin', 'qmax', 'dI', 'subtract_power_law',
                'power_law_method', 'power_law_order', 'power_law_range',
                'smooth', 'smooth_sigma', 'verbose', 'plot'
            },
            'hybrid': {
                'qmin', 'qmax', 'dI',
                # Détection initiale (dérivée)
                'window_length', 'polyorder', 'prominence', 'distance_pts',
                # Raffinement SPV
                'subtract_power_law', 'power_law_method', 'power_law_order',
                'power_law_range', 'smooth', 'smooth_sigma',
                'fit_window_width', 'verbose', 'plot'
            }
        }
    
    def _filter_kwargs(self, kwargs: dict, method: str) -> dict:
        """
        Filtre les kwargs en fonction de la méthode de détection.
        
        Parameters:
        -----------
        kwargs : dict
            Paramètres à filtrer
        method : str
            Méthode de détection ('derivative', 'spv', ou 'hybrid')
            
        Returns:
        --------
        filtered_kwargs : dict
            Paramètres filtrés et valides pour la méthode choisie
        """
        method_lower = method.lower()
        
        # Normaliser les alias de méthode
        if method_lower in ['second_derivative', 'deriv']:
            method_lower = 'derivative'
        
        if method_lower not in self._valid_params:
            raise ValueError(f"Méthode inconnue : {method}. "
                           f"Utilisez 'hybrid', 'spv', ou 'derivative'")
        
        valid_params = self._valid_params[method_lower]
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        
        # Avertir si des paramètres ont été ignorés
        ignored_params = set(kwargs.keys()) - valid_params
        if ignored_params:
            print(f"⚠ Paramètres ignorés pour la méthode '{method}' : "
                  f"{', '.join(sorted(ignored_params))}")
        
        return filtered_kwargs
    
    # =========================================================================
    # MÉTHODE HISTORIQUE (dérivée seconde) - conservée pour compatibilité
    # =========================================================================
    
    def detect_peaks_derivative(self, q, I,
                                nb_peaks=1,
                                qmin=None,
                                qmax=None,
                                window_length=15,
                                polyorder=3,
                                prominence=0.5,
                                distance_pts=20,
                                plot=False):
        """
        MÉTHODE HISTORIQUE : Détection de pics par dérivée seconde.
        
        Cette méthode est conservée pour compatibilité ascendante.
        Pour de meilleurs résultats, utilisez detect_peaks_spv().
        
        Principe : 
        - Lisse I(q) avec filtre Savitzky-Golay
        - Calcule d²I/dq²
        - Les pics correspondent aux minima de d²I/dq²
        
        Limitations :
        - Sensible au bruit
        - Pas de modélisation physique
        - Pas d'incertitudes
        - Paramètres arbitraires
        
        Parameters:
        -----------
        q, I : arrays
            Profil radial
        nb_peaks : int
            Nombre de pics à détecter
        qmin : float, optional
            Valeur minimale de q (Å⁻¹)
        qmax : float, optional
            Valeur maximale de q (Å⁻¹)
        window_length : int
            Fenêtre du filtre Savitzky-Golay (impair)
        polyorder : int
            Ordre polynomial pour lissage
        prominence : float
            Prominence minimale
        distance_pts : int
            Distance minimale entre pics (points)
        plot : bool
            Afficher les résultats
            
        Returns:
        --------
        q_peaks : array
            Positions des pics (Å⁻¹)
        """
        if window_length % 2 == 0:
            window_length += 1
            
        delta_q = q[1] - q[0]
        d2I = savgol_filter(I, window_length=window_length, 
                           polyorder=polyorder, deriv=2, delta=delta_q)
        inverted_d2I = -d2I

        # Créer masque à partir de qmin/qmax (uniformisé avec SPV)
        mask = np.ones_like(q, dtype=bool)
        if qmin is not None:
            mask &= (q >= qmin)
        if qmax is not None:
            mask &= (q <= qmax)

        peaks, properties = find_peaks(inverted_d2I[mask], 
                                       prominence=prominence, 
                                       distance=distance_pts)
        sorted_indices = np.argsort(properties["prominences"])[::-1]
        top_peaks = peaks[sorted_indices[:nb_peaks]]
        q_detected = q[mask][top_peaks]
        q_detected = np.sort(q_detected)
        
        if plot:
            plt.figure(figsize=(10, 6))
            plt.loglog(q, I, label="I(q)", linewidth=2)
            colors = ['r', 'g', 'b', 'c', 'm', 'y']
            for i, qp in enumerate(q_detected[:nb_peaks]):
                plt.axvline(qp, color=colors[i % len(colors)], ls='--', 
                           label=f'Peak {i+1}: d = {2*np.pi/qp:.1f} Å', linewidth=2)
            plt.xlabel("q (Å⁻¹)", fontsize=12)
            plt.ylabel("I(q)", fontsize=12)
            plt.title("Peak Detection - Second Derivative Method", 
                     fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)
            plt.tight_layout()
            plt.show()
            
        return q_detected[:nb_peaks]
    
    # =========================================================================
    # NOUVELLE MÉTHODE SPV (Split Pseudo-Voigt) - RECOMMANDÉE
    # =========================================================================
    
    def detect_peaks_spv(self, q, I, dI=None,
                        nb_peaks=1,
                        qmin=None,
                        qmax=None,
                        subtract_power_law=True,
                        power_law_method='cancel',
                        power_law_order=None,
                        power_law_range=(2.5, 4.0),
                        smooth=False,
                        smooth_sigma=2,
                        verbose=True,
                        plot=False):
        """
        NOUVELLE MÉTHODE : Détection de pics par Split Pseudo-Voigt (SPV).
        
        Méthode recommandée pour une détection précise et robuste.
        
        Avantages :
        ✅ Modélisation physique avec Split Pseudo-Voigt
        ✅ Soustraction automatique de loi de puissance I(q) ~ q^(-m)
        ✅ Quantification des incertitudes sur positions
        ✅ Gestion des pics asymétriques
        ✅ Support multi-pics avec fit global cohérent
        
        Principe :
        1. Prétraitement : soustraction loi de puissance (optionnel)
        2. Détection : fit avec modèle Split Pseudo-Voigt
        3. Pour multi-pics : fit global simultané
        
        Parameters:
        -----------
        q, I : arrays
            Profil radial
        dI : array, optional
            Erreurs sur I (si None, ignorées)
        nb_peaks : int
            Nombre de pics à détecter
        qmin, qmax : float, optional
            Limites en q pour la recherche
        subtract_power_law : bool, default=True
            Soustraire la loi de puissance I(q) ~ q^(-m)
        power_law_method : str, default='cancel'
            'cancel' (standard) ou 'feat' (avancée)
        power_law_order : float, optional
            Ordre m fixe. Si None, optimisé automatiquement
        power_law_range : tuple, default=(2.5, 4.0)
            Plage de recherche pour m
        smooth : bool, default=False
            Lisser les données avec filtre gaussien
        smooth_sigma : float, default=2
            Paramètre sigma du lissage
        verbose : bool, default=True
            Afficher les informations détaillées
        plot : bool, default=False
            Afficher les graphiques
            
        Returns:
        --------
        results : dict
            Dictionnaire contenant :
            - 'q_peaks' : positions des pics (Å⁻¹)
            - 'q_peaks_std' : incertitudes sur positions (Å⁻¹)
            - 'FWHM' : largeurs à mi-hauteur (Å⁻¹)
            - 'd_spacings' : distances de corrélation (Å)
            - 'd_spacings_std' : incertitudes sur distances (Å)
            - 'power_law_order' : ordre m de la loi de puissance
            - 'spv_params' : paramètres du fit SPV
            - 'q_data' : vecteur q complet
            - 'I_processed' : intensités après prétraitement
            - 'I_fit' : courbe du fit SPV
            - 'I_residuals' : résidus du fit
        """
        # Créer une expérience SAXS
        if dI is None:
            dI = np.ones_like(I)
        
        exp = SAXSExperiment(
            data_dict={'Iq': (q, I, dI)},
            name=f'{self.processor.samplename}_correlation'
        )
        
        # Appliquer masques
        if qmin is not None or qmax is not None:
            exp.apply_masks(qmin=qmin, qmax=qmax)
        
        # Prétraitement
        exp.Iq_preprocess = exp.Iq.copy()
        
        power_law_order_value = None
        if subtract_power_law:
            try:
                if power_law_method == 'feat':
                    exp.feat_power_law(
                        input_attr='Iq_preprocess',
                        output_attr='Iq_preprocess',
                        init_order=power_law_order,
                        order_range=power_law_range if power_law_order is None else None,
                        verbose=verbose
                    )
                else:
                    exp.cancel_power_law(
                        input_attr='Iq_preprocess',
                        output_attr='Iq_preprocess',
                        init_order=power_law_order,
                        order_range=power_law_range if power_law_order is None else None,
                        verbose=verbose
                    )
                power_law_order_value = exp.Iq_preprocess.infos.get('order', None)
            except Exception as e:
                if verbose:
                    print(f"⚠ Soustraction loi de puissance échouée : {e}")
                    print("→ Poursuite sans soustraction")
        
        # Lissage optionnel
        exp.Iq_peakfinder = exp.Iq_preprocess.copy()
        if smooth:
            if verbose:
                print(f"   → Application du lissage gaussien (sigma={smooth_sigma})...")
            
            # Lisser les données BRUTES (pas les filtrées)
            # pour préserver les dimensions
            I_smooth = gaussian_filter1d(exp.Iq_peakfinder.y, sigma=smooth_sigma)
            exp.Iq_peakfinder.y = I_smooth
            
            if verbose:
                print(f"   → Lissage appliqué")
        
        # Détection SPV
        try:
            if nb_peaks == 1:
                exp.find_peaks_spv(
                    input_attr='Iq_peakfinder',
                    output_attr='SPV',
                    verbose=verbose,
                    plot=plot
                )
                output_key = 'SPV'
            else:
                exp.find_peaks_spv_batch(
                    n_expected_peaks=nb_peaks,
                    input_attr='Iq_peakfinder',
                    output_attr='SPV',
                    verbose=verbose,
                    plot=plot
                )
                output_key = 'SPV'
        except Exception as e:
            raise RuntimeError(f"Erreur SPV : {e}")
        
        # Extraire résultats
        q_peaks = exp.peaks[output_key]['q_values']
        q_peaks_std = exp.peaks[output_key]['q_values_std']
        FWHM = exp.peaks[output_key]['FWHM']
        
        # Propriétés SPV (déballées directement)
        all_keys = exp.peaks[output_key].keys()
        standard_keys = {'q_values', 'q_values_std', 'FWHM', 'Imin'}
        spv_params = {k: exp.peaks[output_key][k] for k in all_keys if k not in standard_keys}
        
        # Calcul des distances et incertitudes
        d_spacings = 2 * np.pi / q_peaks
        d_spacings_std = d_spacings * (q_peaks_std / q_peaks)  # Propagation d'erreur
        
        # Récupérer courbes
        q_data, I_processed, _ = exp.Iq_peakfinder.get_filtered_data()
        I_fit = exp.SPV_theory.y
        I_residuals = exp.SPV_residuals.y
        
        results = {
            'q_peaks': q_peaks,
            'q_peaks_std': q_peaks_std,
            'FWHM': FWHM,
            'd_spacings': d_spacings,
            'd_spacings_std': d_spacings_std,
            'power_law_order': power_law_order_value,
            'spv_params': spv_params,
            'q_data': q_data,
            'I_processed': I_processed,
            'I_fit': I_fit,
            'I_residuals': I_residuals,
            'exp': exp  # Pour accès avancé
        }
        
        return results
    
    # =========================================================================
    # NOUVELLE MÉTHODE HYBRIDE - MEILLEUR DES DEUX MONDES
    # =========================================================================
    
    def detect_peaks_hybrid(self, q, I, dI=None,
                           nb_peaks=1,
                           qmin=None,
                           qmax=None,
                           # Détection initiale (dérivée)
                           window_length=15,
                           polyorder=3,
                           prominence=0.5,
                           distance_pts=20,
                           # Raffinement SPV
                           subtract_power_law=True,
                           power_law_method='cancel',
                           power_law_order=None,
                           power_law_range=(2.5, 4.0),
                           smooth=False,
                           smooth_sigma=2,
                           fit_window_width=3,
                           verbose=True,
                           plot=False):
        """
        MÉTHODE HYBRIDE : Détection par dérivée + Raffinement SPV.
        
        ⭐ RECOMMANDÉ pour la plupart des cas ⭐
        
        Combine les avantages des deux approches :
        
        ✅ Détection robuste (dérivée) : trouve tous les pics
        ✅ Raffinement précis (SPV) : positions exactes avec incertitudes
        
        Workflow :
        1. Détection grossière avec dérivée seconde (robuste)
        2. Soustraction loi de puissance globale
        3. Pour chaque pic détecté :
           - Isolation fenêtre locale
           - Fit SPV à 1 pic
           - Extraction position + incertitude
        
        Parameters:
        -----------
        Voir detect_peaks_hybrid() dans utilities.py
        pour documentation complète.
        
        Returns:
        --------
        results : dict avec positions raffinées et incertitudes
        """
        # Import de la fonction hybrid
        from utilities import detect_peaks_hybrid as hybrid_func
        
        return hybrid_func(
            q, I, dI,
            nb_peaks=nb_peaks,
            qmin=qmin,
            qmax=qmax,
            window_length=window_length,
            polyorder=polyorder,
            prominence=prominence,
            distance_pts=distance_pts,
            subtract_power_law=subtract_power_law,
            power_law_method=power_law_method,
            power_law_order=power_law_order,
            power_law_range=power_law_range,
            smooth=smooth,
            smooth_sigma=smooth_sigma,
            fit_window_width=fit_window_width,
            verbose=verbose,
            plot=plot
        )
    
    # =========================================================================
    # MÉTHODE UNIFIÉE - Choisit automatiquement SPV, dérivée, ou HYBRIDE
    # =========================================================================
    
    def detect_peaks(self, q, I, dI=None,
                    nb_peaks=1,
                    method='hybrid',
                    **kwargs):
        """
        Méthode unifiée de détection de pics.
        
        Choisit automatiquement entre :
        - 'hybrid' : Dérivée + SPV (⭐ RECOMMANDÉ)
        - 'spv' : Split Pseudo-Voigt seul
        - 'derivative' : Dérivée seconde seule (historique)
        
        Parameters:
        -----------
        q, I : arrays
            Profil radial
        dI : array, optional
            Erreurs sur I
        nb_peaks : int
            Nombre de pics à détecter
        method : str, default='hybrid'
            'hybrid', 'spv', ou 'derivative'
        **kwargs : 
            Paramètres spécifiques à la méthode choisie
            
        Returns:
        --------
        Si method='hybrid' ou 'spv' : dict avec résultats complets
        Si method='derivative' : array avec positions des pics
        """
        if method.lower() == 'hybrid':
            return self.detect_peaks_hybrid(q, I, dI, nb_peaks=nb_peaks, **kwargs)
        elif method.lower() == 'spv':
            return self.detect_peaks_spv(q, I, dI, nb_peaks=nb_peaks, **kwargs)
        elif method.lower() in ['derivative', 'second_derivative', 'deriv']:
            return self.detect_peaks_derivative(q, I, nb_peaks=nb_peaks, **kwargs)
        else:
            raise ValueError(f"Méthode inconnue : {method}. "
                           f"Utilisez 'hybrid' (recommandé), 'spv', ou 'derivative'")
    
    # =========================================================================
    # CALCUL DES DISTANCES DE CORRÉLATION - Version améliorée
    # =========================================================================
    
    def compute_correlation_distances(self, 
                                     nb_peaks=1, 
                                     azimuth: float = 90, 
                                     width: float = 40,
                                     method='hybrid',
                                     **detection_params):
        """
        Calcule les distances de corrélation à partir du profil radial.
        
        Version améliorée avec support de la méthode HYBRIDE et filtrage des kwargs.
        
        Parameters:
        -----------
        nb_peaks : int
            Nombre de pics à détecter
        azimuth : float
            Angle azimutal (°)
        width : float
            Largeur du secteur angulaire (°)
        method : str, default='hybrid'
            Méthode de détection :
            - 'hybrid' : Dérivée + SPV (⭐ RECOMMANDÉ)
            - 'spv' : SPV seul
            - 'derivative' : Dérivée seule (historique)
        **detection_params :
            Paramètres pour la méthode de détection choisie
            Les paramètres non reconnus par la méthode seront ignorés avec un avertissement
            
        Returns:
        --------
        results : dict
            Dictionnaire contenant :
            - 'distances' : distances de corrélation (Å)
            - 'distances_std' : incertitudes (Å) [si SPV ou hybrid]
            - 'q_peaks' : positions des pics (Å⁻¹)
            - 'q_peaks_std' : incertitudes [si SPV ou hybrid]
            - 'q_peaks_initial' : positions initiales [si hybrid]
            - 'FWHM' : largeurs [si SPV ou hybrid]
            - 'method' : méthode utilisée
            - 'azimuth', 'width' : paramètres azimutaux
            - Autres paramètres selon la méthode
        """
        # Filtrer les kwargs en fonction de la méthode
        filtered_params = self._filter_kwargs(detection_params, method)
        
        # Extraire profil radial
        q, I = self.processor.extract_radial_profile(
            azimuth=azimuth, 
            width=width, 
            save=False
        )
        
        # Détecter pics avec la méthode choisie
        if method.lower() in ['hybrid', 'spv']:
            if method.lower() == 'hybrid':
                peak_results = self.detect_peaks_hybrid(
                    q, I, 
                    nb_peaks=nb_peaks,
                    **filtered_params
                )
            else:  # spv
                peak_results = self.detect_peaks_spv(
                    q, I, 
                    nb_peaks=nb_peaks,
                    **filtered_params
                )
            
            # Préparer résultats
            results = {
                'distances': peak_results['d_spacings'],
                'distances_std': peak_results['d_spacings_std'],
                'q_peaks': peak_results['q_peaks'],
                'q_peaks_std': peak_results['q_peaks_std'],
                'FWHM': peak_results['FWHM'],
                'power_law_order': peak_results.get('power_law_order'),
                'method': method,
                'azimuth': azimuth,
                'width': width,
                'q_profile': peak_results['q_data'],
                'I_profile': peak_results['I_processed'],
            }
            
            # Ajouter champs spécifiques hybrid
            if method.lower() == 'hybrid':
                results['q_peaks_initial'] = peak_results.get('q_peaks_initial')
            
            # Ajouter fit et résidus si disponibles
            if 'I_fit' in peak_results:
                results['I_fit'] = peak_results['I_fit']
            if 'I_residuals' in peak_results:
                results['I_residuals'] = peak_results['I_residuals']
            if 'spv_params' in peak_results:
                results['spv_params'] = peak_results['spv_params']
            
        else:  # méthode derivative
            q_peaks = self.detect_peaks_derivative(
                q, I,
                nb_peaks=nb_peaks,
                **filtered_params
            )
            
            distances = 2 * np.pi / q_peaks
            
            results = {
                'distances': distances,
                'q_peaks': q_peaks,
                'method': 'derivative',
                'azimuth': azimuth,
                'width': width,
                'q_profile': q,
                'I_profile': I
            }
        
        # Afficher résultats
        print(f"\n{'=' * 70}")
        print(f"Analyse de Distance de Corrélation")
        print(f"{'=' * 70}")
        print(f"Échantillon : {self.processor.samplename}")
        print(f"Méthode     : {method.upper()}")
        print(f"Secteur     : {azimuth}° ± {width/2}°")
        print(f"\n{len(results['q_peaks'])} pic(s) détecté(s) :")
        
        for i in range(len(results['q_peaks'])):
            q_peak = results['q_peaks'][i]
            d = results['distances'][i]
            
            # Pour hybrid, montrer aussi position initiale
            if method.lower() == 'hybrid' and 'q_peaks_initial' in results:
                q_init = results['q_peaks_initial'][i]
                delta = abs(q_peak - q_init)
                print(f"  Pic {i+1}:")
                print(f"    Dérivée : q = {q_init:.4f} Å⁻¹")
                if 'q_peaks_std' in results:
                    q_std = results['q_peaks_std'][i]
                    d_std = results['distances_std'][i]
                    print(f"    SPV     : q = {q_peak:.4f} ± {q_std:.4f} Å⁻¹ "
                          f"→ d = {d:.1f} ± {d_std:.1f} Å")
                    print(f"    Δq = {delta:.4f} Å⁻¹ ({delta/q_init*100:.2f}%)")
                else:
                    print(f"    SPV     : q = {q_peak:.4f} Å⁻¹ → d = {d:.1f} Å")
                if 'FWHM' in results:
                    print(f"    FWHM = {results['FWHM'][i]:.4f} Å⁻¹")
            
            elif 'q_peaks_std' in results:
                q_std = results['q_peaks_std'][i]
                d_std = results['distances_std'][i]
                print(f"  Pic {i+1}: q = {q_peak:.4f} ± {q_std:.4f} Å⁻¹ "
                      f"→ d = {d:.1f} ± {d_std:.1f} Å")
                if 'FWHM' in results:
                    print(f"          FWHM = {results['FWHM'][i]:.4f} Å⁻¹")
            else:
                print(f"  Pic {i+1}: q = {q_peak:.4f} Å⁻¹ → d = {d:.1f} Å")
        
        if 'power_law_order' in results and results['power_law_order']:
            print(f"\nOrdre loi de puissance : m = {results['power_law_order']:.2f}")
        
        print(f"{'=' * 70}")
        
        return results['distances'], results['distances_std'] if method.lower() in ['spv', 'hybrid'] else None, results
    
    # =========================================================================
    # ANALYSE D'ANISOTROPIE - Version améliorée
    # =========================================================================
    
    def analyze_anisotropy(self, 
                          nb_peaks=1,
                          azimuth_list=[0, 45, 90, 135],
                          width=40,
                          method='spv',
                          plot=True,
                          **detection_params):
        """
        Analyse l'anisotropie structurale en comparant les distances 
        de corrélation dans différentes directions azimutales.
        
        Version améliorée avec support méthode SPV et barres d'erreur.
        
        Parameters:
        -----------
        nb_peaks : int
            Nombre de pics à détecter
        azimuth_list : list
            Liste des angles azimutaux à analyser (°)
        width : float
            Largeur du secteur angulaire (°)
        method : str, default='spv'
            Méthode de détection : 'spv' ou 'derivative'
        plot : bool
            Afficher les graphiques comparatifs
        **detection_params :
            Paramètres pour la méthode de détection
            
        Returns:
        --------
        anisotropy_results : dict
            Dictionnaire avec résultats pour chaque azimut
        """
        anisotropy_results = {}
        
        print(f"\n{'=' * 70}")
        print(f"Analyse d'Anisotropie Structurale")
        print(f"{'=' * 70}")
        print(f"Méthode : {method.upper()}")
        print(f"Directions azimutales : {azimuth_list}")
        print(f"{'=' * 70}\n")
        
        for azimuth in azimuth_list:
            print(f"\n--- Direction {azimuth}° ---")
            distances, distances_std, results = self.compute_correlation_distances(
                nb_peaks=nb_peaks,
                azimuth=azimuth,
                width=width,
                method=method,
                verbose=False,
                plot=False,
                **detection_params
            )
            anisotropy_results[azimuth] = results
        
        if plot and len(anisotropy_results) > 1:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Plot 1: Profils radiaux
            for azimuth, results in anisotropy_results.items():
                axes[0].loglog(results['q_profile'], results['I_profile'], 
                              label=f"{azimuth}°", linewidth=2, alpha=0.7)
                # Marquer les pics
                for qp in results['q_peaks']:
                    axes[0].axvline(qp, linestyle='--', alpha=0.3)
            
            axes[0].set_xlabel("q (Å⁻¹)", fontsize=13)
            axes[0].set_ylabel("I(q)", fontsize=13)
            axes[0].set_title("Profils Radiaux par Direction", 
                            fontsize=14, fontweight='bold')
            axes[0].legend(fontsize=11)
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Distances de corrélation vs azimut
            colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33']
            
            for peak_idx in range(nb_peaks):
                azimuths = []
                distances = []
                distances_std = []
                
                for azimuth, results in anisotropy_results.items():
                    if peak_idx < len(results['distances']):
                        azimuths.append(azimuth)
                        distances.append(results['distances'][peak_idx])
                        if 'distances_std' in results:
                            distances_std.append(results['distances_std'][peak_idx])
                
                if len(azimuths) > 0:
                    color = colors[peak_idx % len(colors)]
                    
                    if len(distances_std) > 0 and method.lower() == 'spv':
                        # Avec barres d'erreur (SPV)
                        axes[1].errorbar(azimuths, distances, yerr=distances_std,
                                       fmt='o-', color=color,
                                       markersize=10, linewidth=2, capsize=5,
                                       label=f"Pic {peak_idx+1}")
                    else:
                        # Sans barres d'erreur (derivative)
                        axes[1].plot(azimuths, distances, 'o-', 
                                   color=color,
                                   markersize=10, linewidth=2,
                                   label=f"Pic {peak_idx+1}")
            
            axes[1].set_xlabel("Angle Azimutal (°)", fontsize=13)
            axes[1].set_ylabel("Distance de Corrélation (Å)", fontsize=13)
            axes[1].set_title(f"Anisotropie Structurale - Méthode {method.upper()}", 
                            fontsize=14, fontweight='bold')
            axes[1].legend(fontsize=11)
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        return anisotropy_results
    
    # =========================================================================
    # MÉTHODE DE COMPARAISON - SPV vs Dérivée
    # =========================================================================
    
    def compare_methods(self, 
                       nb_peaks=1,
                       azimuth=90,
                       width=40,
                       spv_params=None,
                       derivative_params=None):
        """
        Compare les résultats de détection entre méthode SPV et dérivée seconde.
        
        Utile pour :
        - Valider la transition vers SPV
        - Identifier les différences
        - Évaluer la robustesse
        
        Parameters:
        -----------
        nb_peaks : int
            Nombre de pics à détecter
        azimuth : float
            Angle azimutal (°)
        width : float
            Largeur secteur (°)
        spv_params : dict, optional
            Paramètres pour méthode SPV
        derivative_params : dict, optional
            Paramètres pour méthode dérivée
            
        Returns:
        --------
        comparison : dict
            Résultats des deux méthodes
        """
        if spv_params is None:
            spv_params = {'subtract_power_law': True, 'verbose': False}
        if derivative_params is None:
            derivative_params = {'window_length': 15, 'prominence': 0.5}
        
        print(f"\n{'=' * 70}")
        print(f"COMPARAISON DES MÉTHODES DE DÉTECTION")
        print(f"{'=' * 70}\n")
        
        # Méthode 1 : Dérivée seconde
        print("--- Méthode Dérivée Seconde (historique) ---")
        results_deriv = self.compute_correlation_distances(
            nb_peaks=nb_peaks,
            azimuth=azimuth,
            width=width,
            method='derivative',
            **derivative_params
        )
        
        # Méthode 2 : SPV
        print("\n--- Méthode SPV (recommandée) ---")
        results_spv = self.compute_correlation_distances(
            nb_peaks=nb_peaks,
            azimuth=azimuth,
            width=width,
            method='spv',
            **spv_params
        )
        
        # Comparaison
        print(f"\n{'=' * 70}")
        print("COMPARAISON DES RÉSULTATS")
        print(f"{'=' * 70}")
        
        for i in range(min(len(results_deriv['q_peaks']), len(results_spv['q_peaks']))):
            q_deriv = results_deriv['q_peaks'][i]
            d_deriv = results_deriv['distances'][i]
            
            q_spv = results_spv['q_peaks'][i]
            q_spv_std = results_spv['q_peaks_std'][i]
            d_spv = results_spv['distances'][i]
            d_spv_std = results_spv['distances_std'][i]
            
            delta_q = abs(q_spv - q_deriv)
            delta_d = abs(d_spv - d_deriv)
            
            print(f"\nPic {i+1} :")
            print(f"  Dérivée : q = {q_deriv:.4f} Å⁻¹  →  d = {d_deriv:.1f} Å")
            print(f"  SPV     : q = {q_spv:.4f} ± {q_spv_std:.4f} Å⁻¹  "
                  f"→  d = {d_spv:.1f} ± {d_spv_std:.1f} Å")
            print(f"  Δq = {delta_q:.4f} Å⁻¹  ({delta_q/q_spv*100:.1f}%)")
            print(f"  Δd = {delta_d:.1f} Å  ({delta_d/d_spv*100:.1f}%)")
        
        print(f"{'=' * 70}")
        
        # Graphique comparatif
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        q_deriv = results_deriv['q_profile']
        I_deriv = results_deriv['I_profile']
        q_spv = results_spv['q_profile']
        I_spv = results_spv['I_profile']
        
        # Plot 1: Profils superposés
        axes[0, 0].loglog(q_deriv, I_deriv, 'o', alpha=0.3, 
                         markersize=3, label='Données brutes')
        axes[0, 0].set_xlabel("q (Å⁻¹)", fontsize=12)
        axes[0, 0].set_ylabel("I(q)", fontsize=12)
        axes[0, 0].set_title("Données Brutes", fontsize=13, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Plot 2: Détection dérivée
        axes[0, 1].loglog(q_deriv, I_deriv, 'k-', alpha=0.5, linewidth=1.5, label='I(q)')
        for i, qp in enumerate(results_deriv['q_peaks']):
            axes[0, 1].axvline(qp, color='red', linestyle='--', linewidth=2,
                              label=f'Pic {i+1}' if i == 0 else '')
        axes[0, 1].set_xlabel("q (Å⁻¹)", fontsize=12)
        axes[0, 1].set_ylabel("I(q)", fontsize=12)
        axes[0, 1].set_title("Méthode Dérivée Seconde", fontsize=13, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Plot 3: Détection SPV
        axes[1, 0].loglog(q_spv, I_spv, 'o', alpha=0.3, markersize=3, label='Données prétraitées')
        axes[1, 0].loglog(q_spv, results_spv['I_fit'], 'r-', linewidth=2.5, label='Fit SPV')
        for i, qp in enumerate(results_spv['q_peaks']):
            axes[1, 0].axvline(qp, color='green', linestyle='--', linewidth=2,
                              label=f'Pic {i+1}' if i == 0 else '')
        axes[1, 0].set_xlabel("q (Å⁻¹)", fontsize=12)
        axes[1, 0].set_ylabel("I(q) × q^m", fontsize=12)
        axes[1, 0].set_title("Méthode SPV (avec fit)", fontsize=13, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Plot 4: Comparaison positions
        peak_indices = np.arange(1, min(len(results_deriv['q_peaks']), 
                                       len(results_spv['q_peaks'])) + 1)
        
        width = 0.35
        axes[1, 1].bar(peak_indices - width/2, results_deriv['distances'][:len(peak_indices)], 
                      width, label='Dérivée', alpha=0.7, color='red')
        
        if 'distances_std' in results_spv:
            axes[1, 1].bar(peak_indices + width/2, results_spv['distances'][:len(peak_indices)], 
                          width, yerr=results_spv['distances_std'][:len(peak_indices)],
                          label='SPV', alpha=0.7, color='green', capsize=5)
        else:
            axes[1, 1].bar(peak_indices + width/2, results_spv['distances'][:len(peak_indices)], 
                          width, label='SPV', alpha=0.7, color='green')
        
        axes[1, 1].set_xlabel("Numéro du Pic", fontsize=12)
        axes[1, 1].set_ylabel("Distance de Corrélation (Å)", fontsize=12)
        axes[1, 1].set_title("Comparaison des Distances", fontsize=13, fontweight='bold')
        axes[1, 1].set_xticks(peak_indices)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        comparison = {
            'derivative': results_deriv,
            'spv': results_spv
        }
        
        return comparison


# =============================================================================
# EXEMPLES D'UTILISATION
# =============================================================================

if __name__ == "__main__":
    """
    Exemples d'utilisation du CorrelationDistanceCalculator amélioré
    """
    
    # Supposons que vous avez déjà un processeur SAXS initialisé
    # processor = SAXSProcessor(...)
    # calc = CorrelationDistanceCalculator(processor)
    
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║  CorrelationDistanceCalculator - Guide d'Utilisation             ║
    ╚═══════════════════════════════════════════════════════════════════╝
    
    1. MÉTHODE SPV (RECOMMANDÉE) :
    --------------------------------
    results = calc.compute_correlation_distances(
        nb_peaks=2,
        azimuth=90,
        width=40,
        method='spv',
        subtract_power_law=True,
        verbose=True,
        plot=True
    )
    
    → Retourne distances avec incertitudes
    → Soustraction automatique de fond
    → Fit physiquement motivé
    
    
    2. MÉTHODE DÉRIVÉE (HISTORIQUE) :
    -----------------------------------
    results = calc.compute_correlation_distances(
        nb_peaks=2,
        azimuth=90,
        width=40,
        method='derivative',
        window_length=15,
        prominence=0.5
    )
    
    → Méthode rapide mais moins précise
    → Conservée pour compatibilité
    
    
    3. ANALYSE D'ANISOTROPIE :
    ---------------------------
    aniso = calc.analyze_anisotropy(
        nb_peaks=2,
        azimuth_list=[0, 45, 90, 135],
        method='spv',
        subtract_power_law=True,
        plot=True
    )
    
    → Compare distances dans différentes directions
    → Barres d'erreur automatiques avec SPV
    
    
    4. COMPARAISON DES MÉTHODES :
    ------------------------------
    comparison = calc.compare_methods(
        nb_peaks=2,
        azimuth=90,
        width=40
    )
    
    → Compare SPV vs dérivée seconde
    → Graphiques comparatifs détaillés
    → Quantifie les différences
    
    """)