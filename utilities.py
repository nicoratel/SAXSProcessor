from nematicordercalculator import CylinderFormFactor,NematicOrderCalculator
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import gaussian_filter1d
from saxs_analysis import SAXSExperiment


def detect_incomplete_azimuthal_profile(chi, I_az, threshold=0.1):
    """
    Detect if an azimuthal profile is incomplete by identifying regions with missing data (NaN or 0).
    
    An azimuthal profile is considered incomplete if there are significant gaps
    where intensities are NaN or zero, indicating data was not acquired or is invalid.
    
    Parameters:
    -----------
    chi : ndarray
        Azimuthal angles (typically -180 to 180)
    I_az : ndarray
        Intensity values
    threshold : float
        Threshold ratio for considering profile incomplete (default: 0.1)
        If > 10% of consecutive points are NaN/0, profile is considered incomplete
    
    Returns:
    --------
    bool
        True if profile is incomplete (has significant gaps), False if continuous
    str
        Description of gaps found
    """
    
    # Identify invalid data (NaN or <= 0)
    invalid_mask = np.isnan(I_az) | (I_az <= 0)
    
    if np.sum(invalid_mask) == 0:
        return False, "Profile is complete (no NaN/zero values)"
    
    # Check if too many points are invalid
    fraction_invalid = np.sum(invalid_mask) / len(I_az)
    
    if fraction_invalid > threshold:
        return True, f"Too many invalid points ({fraction_invalid*100:.1f}% are NaN/0, threshold={threshold*100:.1f}%)"
    
    # Look for consecutive NaN/0 regions
    valid_mask = ~invalid_mask
    
    if np.sum(valid_mask) == 0:
        return True, "All data points are invalid (NaN or 0)"
    
    # Find transitions from valid to invalid data
    transitions = np.diff(valid_mask.astype(int))
    gap_starts = np.where(transitions == -1)[0]  # Valid → Invalid
    gap_ends = np.where(transitions == 1)[0]     # Invalid → Valid
    
    if len(gap_starts) == 0:
        return False, "Profile is complete (no significant gaps in data)"
    
    # Describe gaps
    gap_description = f"Found {len(gap_starts)} gap(s) with invalid data:\n"
    for i, (start, end) in enumerate(zip(gap_starts, gap_ends)):
        chi_start = chi[start]
        chi_end = chi[end] if end < len(chi) else chi[-1]
        gap_size = end - start
        gap_description += f"    Gap {i+1}: between χ={chi_start:.1f}° and χ={chi_end:.1f}° ({gap_size} points)\n"
    
    return True, gap_description.strip()
   

def compute_nematic_parameter(
        processor = None,
        qvalue = 0.034,
        threshold = 0.05,
        radius = 78,
        L = 840,
        radius_pd = 0.3,
        L_pd = 0.75,
        plot=False,
        apply_mirror=None,
        verbose=True
        ):
    
    """
    processor: SAXSProceesor instance
        SAXSProcessor instance (mandatory)
    qvalue: float
        q value of interest at which azimuthal profile is extracted
    threshold: float
        relative value to defin the q range for azimuthal profile extraction
    radius: float
        Cylinder radius for form factor caclulation
    L: float
        Cylinder length for form factor calculation
    radius_pd, L_pd: float
        Polydispersity ratio for radius and length
    plot : bool
        Plot form factor and fitting results
    apply_mirror: bool or None
        Apply mirror symmetry to incomplete exp azimuthal profile
        If None (default), automatically detect if profile is incomplete

    *** Returns ****
    results : dict
        {'I0': I0_opt,
        'm': m_opt,
        'x0': x0_opt,
        'a': a_opt,
        'b': b_opt,
        'S': S_opt,
        'R2': r_squared,
        'I_model': I_model}
    """
    # 1. Extract azimuthal profile
    chi_exp, I_az_exp = processor.extract_azimuthal_profile(qvalue=qvalue, threshold = threshold)
    
    # Auto-detect if mirror symmetry should be applied
    if apply_mirror is None:
        is_incomplete, reason = detect_incomplete_azimuthal_profile(chi_exp, I_az_exp)
        apply_mirror = is_incomplete
        if verbose:
            print(f"\n{'=' * 60}")
            print("Azimuthal profile completeness check:")
            print(f"  Status: {'INCOMPLETE' if is_incomplete else 'COMPLETE'}")
            print(f"  {reason}")
            print(f"  → Mirror symmetry: {'ENABLED' if apply_mirror else 'DISABLED'}")
            print(f"{'=' * 60}")
    
    # 2. determine main orientation
    chi = processor.find_main_orientation(qvalue=qvalue, threshold= threshold)
    angle_folded = ((-chi + 45) % 90) - 45 # on utilise - chi car pyfai interprete les angles dans le sens horaire
    # ajout de 3 lignes pour corriger l'orientation
    chi_labo = -chi
    theta_fibril = chi_labo + 90
    angle_folded = ((theta_fibril + 45) % 90) - 45
    if verbose:        
        print('Main nanorod orientation is', angle_folded)
    # 3. Cylinder form factor calculation
    if verbose:
        print(f"\n{'=' * 60}")
        print('Computing cylinder form factor...')
    form_factor = CylinderFormFactor(
        processor=processor, 
        radius=radius,
        L=L,
        theta=90,
        phi=angle_folded,
        radius_pd=radius_pd,
        L_pd=L_pd,
        phi_pd=0,
        theta_pd=0,
        background=0.0001,
        scale=1,
        plot = plot)
    # 4. Nematic order calculation
    if verbose:
        print(f"\n{'=' * 60}")
        print('Fitting azimuthal profile to extract nematic order parameter...')
    nematic_calc = NematicOrderCalculator(form_factor=form_factor)
    results = nematic_calc.fit_azimuthal_profile(
        chi_exp,
        I_az_exp,
        qvalue_ff=qvalue,
        threshold_ff=threshold,
        plot=plot,
        target=chi,
        apply_mirror=apply_mirror,
        processor = processor,
        verbose=verbose
        )
        
    # --- gestion des résultats invalides ---
    if results is None or results["S"] is None:
        if verbose:
            print("⚠️ Fit failed → returning S = NaN")
        return np.nan, angle_folded, {
            "S": np.nan,
            "x0": np.nan,
            "m": np.nan,
            "R2": 0,
            "I_model": None
        }
    

    # --- sinon fit OK ---
    return results["S"], angle_folded, results


"""
MÉTHODE HYBRIDE : Détection par dérivée + Raffinement SPV
===========================================================


- Détection initiale robuste avec dérivée seconde
- Raffinement précis avec Split Pseudo-Voigt
"""




def detect_peaks_hybrid(q, I, dI=None,
                        nb_peaks=1,
                        qmin=None,
                        qmax=None,
                        # Paramètres de détection initiale (dérivée)
                        window_length=15,
                        polyorder=3,
                        prominence=0.5,
                        distance_pts=20,
                        # Paramètres de raffinement SPV
                        subtract_power_law=True,
                        power_law_method='cancel',
                        power_law_order=None,
                        power_law_range=(2.5, 4.0),
                        smooth=False,
                        smooth_sigma=2,
                        # Paramètres de fit local
                        fit_window_width=3,  # En nombre de FWHM
                        # Visualisation
                        verbose=True,
                        plot=False):
    """
    MÉTHODE HYBRIDE : Détection par dérivée seconde + Raffinement SPV.
    
    Cette méthode combine les avantages des deux approches :
    
    ÉTAPE 1 - DÉTECTION (dérivée seconde) :
    ----------------------------------------
    ✅ Robuste pour trouver tous les pics
    ✅ Fonctionne même avec pics faibles
    ✅ Bonne séparation des pics proches
    ✅ Rapide
    
    ÉTAPE 2 - RAFFINEMENT (SPV) :
    ------------------------------
    ✅ Précision sub-pixel sur la position
    ✅ Quantification des incertitudes
    ✅ Modélisation physique
    ✅ Gestion de l'asymétrie
    
    Workflow :
    ----------
    1. Soustraction loi de puissance (optionnel)
    2. Détection grossière par dérivée seconde
    3. Pour chaque pic détecté :
       a. Isoler une fenêtre locale autour du pic
       b. Fitter avec SPV à 1 pic
       c. Extraire position et incertitude
    4. Retourner résultats raffinés
    
    Parameters:
    -----------
    q, I : arrays
        Profil radial
    dI : array, optional
        Erreurs sur I
    nb_peaks : int
        Nombre de pics à détecter
    qmin, qmax : float, optional
        Limites en q
        
    # Détection initiale
    window_length : int, default=15
        Fenêtre Savitzky-Golay (impair)
    polyorder : int, default=3
        Ordre polynomial
    prominence : float, default=0.5
        Prominence minimale
    distance_pts : int, default=20
        Distance minimale entre pics (points)
        
    # Raffinement SPV
    subtract_power_law : bool, default=True
        Soustraire loi de puissance
    power_law_method : str, default='cancel'
        'cancel' ou 'feat'
    power_law_order : float, optional
        Ordre m (None=auto)
    power_law_range : tuple, default=(2.5, 4.0)
        Plage pour m
    smooth : bool, default=False
        Lissage gaussien
    smooth_sigma : float, default=2
        Paramètre lissage
    fit_window_width : float, default=3
        Largeur fenêtre de fit en nombre de FWHM estimés
        
    # Visualisation
    verbose : bool, default=True
        Afficher détails
    plot : bool, default=False
        Graphiques
        
    Returns:
    --------
    results : dict
        - 'q_peaks' : positions raffinées (Å⁻¹)
        - 'q_peaks_std' : incertitudes (Å⁻¹)
        - 'q_peaks_initial' : positions initiales (dérivée)
        - 'FWHM' : largeurs (Å⁻¹)
        - 'd_spacings' : distances (Å)
        - 'd_spacings_std' : incertitudes (Å)
        - 'power_law_order' : ordre m
        - 'method' : 'hybrid'
        - Autres paramètres SPV
    """
    
    if verbose:
        print("\n" + "="*70)
        print("MÉTHODE HYBRIDE : Dérivée + SPV")
        print("="*70)
    
    # =========================================================================
    # ÉTAPE 1 : DÉTECTION INITIALE PAR DÉRIVÉE SECONDE
    # =========================================================================
    
    if verbose:
        print("\n[Étape 1/3] Détection initiale par dérivée seconde...")
    
    # Préparer données
    if dI is None:
        dI = np.ones_like(I)
    
    # Masquer
    mask = np.ones_like(q, dtype=bool)
    if qmin is not None:
        mask &= (q >= qmin)
    if qmax is not None:
        mask &= (q <= qmax)
    
    q_masked = q[mask]
    I_masked = I[mask]
    Inew = I_masked*q_masked**(3)
    
    # Calculer dérivée seconde
    if window_length % 2 == 0:
        window_length += 1
    
    delta_q = q_masked[1] - q_masked[0]
    d2I = savgol_filter(Inew, window_length=window_length, 
                       polyorder=polyorder, deriv=2, delta=delta_q)
    inverted_d2I = -d2I
    
    # Détecter pics
    peaks_idx, properties = find_peaks(inverted_d2I, 
                                       prominence=prominence, 
                                       distance=distance_pts)
    
    # Trier par prominence
    sorted_indices = np.argsort(properties["prominences"])[::-1]
    top_peaks_idx = peaks_idx[sorted_indices[:nb_peaks]]
    top_peaks_idx = np.sort(top_peaks_idx)
    
    q_peaks_initial = q_masked[top_peaks_idx]
    
    if len(q_peaks_initial) == 0:
        raise ValueError("Aucun pic détecté par la méthode dérivée. "
                        "Essayez de réduire 'prominence' ou 'window_length'")
    
    if verbose:
        print(f"   → {len(q_peaks_initial)} pic(s) détecté(s)")
        for i, qp in enumerate(q_peaks_initial):
            print(f"      Pic {i+1}: q ≈ {qp:.4f} Å⁻¹")
    
    # =========================================================================
    # ÉTAPE 2 : PRÉTRAITEMENT GLOBAL (loi de puissance)
    # =========================================================================
    
    if verbose:
        print(f"\n[Étape 2/3] Prétraitement global...")
    
    # Créer expérience
    exp_global = SAXSExperiment(
        data_dict={'Iq': (q, I, dI)},
        name='hybrid_global'
    )
    
    # Masquer
    if qmin is not None or qmax is not None:
        exp_global.apply_masks(qmin=qmin, qmax=qmax)
    
    # Soustraction loi de puissance
    exp_global.Iq_preprocess = exp_global.Iq.copy()
    power_law_order_value = None
    
    if subtract_power_law:
        try:
            if power_law_method == 'feat':
                exp_global.feat_power_law(
                    input_attr='Iq_preprocess',
                    output_attr='Iq_preprocess',
                    init_order=power_law_order,
                    order_range=power_law_range if power_law_order is None else None,
                    verbose=verbose
                )
            else:
                exp_global.cancel_power_law(
                    input_attr='Iq_preprocess',
                    output_attr='Iq_preprocess',
                    init_order=power_law_order,
                    order_range=power_law_range if power_law_order is None else None,
                    verbose=verbose
                )
            power_law_order_value = exp_global.Iq_preprocess.infos.get('order', None)
            
            if verbose:
                print(f"   → Loi de puissance : m = {power_law_order_value:.2f}")
        
        except Exception as e:
            if verbose:
                print(f"   ⚠ Soustraction échouée : {e}")
    
    # Lissage
    exp_global.Iq_peakfinder = exp_global.Iq_preprocess.copy()
    if smooth:
        I_smooth = gaussian_filter1d(exp_global.Iq_peakfinder.y, sigma=smooth_sigma)
        exp_global.Iq_peakfinder.y = I_smooth
        if verbose:
            print(f"   → Lissage appliqué (σ={smooth_sigma})")
    
    # Données prétraitées
    q_preprocessed, I_preprocessed, dI_preprocessed = exp_global.Iq_peakfinder.get_filtered_data()
    
    # =========================================================================
    # ÉTAPE 3 : RAFFINEMENT SPV LOCAL POUR CHAQUE PIC
    # =========================================================================
    
    if verbose:
        print(f"\n[Étape 3/3] Raffinement SPV pour chaque pic...")
    
    q_peaks_refined = []
    q_peaks_std = []
    FWHM_list = []
    spv_params_list = []
    exclusion_zones = []  # Tracker les zones masquées pour éviter les repics
    
    for i, q_peak_init in enumerate(q_peaks_initial):
        if verbose:
            print(f"\n   → Pic {i+1}/{len(q_peaks_initial)} : q ≈ {q_peak_init:.4f} Å⁻¹")
        
        # Estimer FWHM initial (pour définir fenêtre)
        # Trouver l'index du pic
        idx_peak = np.argmin(np.abs(q_preprocessed - q_peak_init))
        
        # Estimer FWHM grossièrement (distance au premier point < 0.5*max)
        I_peak = I_preprocessed[idx_peak]
        half_max = I_peak / 2
        
        # Chercher à gauche
        idx_left = idx_peak
        while idx_left > 0 and I_preprocessed[idx_left] > half_max:
            idx_left -= 1
        
        # Chercher à droite
        idx_right = idx_peak
        while idx_right < len(I_preprocessed) - 1 and I_preprocessed[idx_right] > half_max:
            idx_right += 1
        
        FWHM_estimate = q_preprocessed[idx_right] - q_preprocessed[idx_left]
        FWHM_estimate = max(FWHM_estimate, 0.01)  # Minimum raisonnable
        
        if verbose:
            print(f"      FWHM estimé : {FWHM_estimate:.4f} Å⁻¹")
        
        # Définir fenêtre locale - RESTER CENTRÉ SUR LE PIC
        q_window_half = fit_window_width * FWHM_estimate
        q_window_min = max(q_peak_init - q_window_half, q_preprocessed.min())
        q_window_max = min(q_peak_init + q_window_half, q_preprocessed.max())
        
        if verbose:
            print(f"      Fenêtre brute : [{q_window_min:.4f}, {q_window_max:.4f}] Å⁻¹")
        
        # Extraire données dans la fenêtre
        window_mask = (q_preprocessed >= q_window_min) & (q_preprocessed <= q_window_max)
        q_window = q_preprocessed[window_mask]
        I_window = I_preprocessed[window_mask]
        dI_window = dI_preprocessed[window_mask]
        
        # Appliquer exclusions des pics précédents EN MASQUANT les points contaminés
        # Mais garder la fenêtre centrée - on ne supprime QUE les points en zone d'exclusion
        for q_min_excl, q_max_excl in exclusion_zones:
            # Créer masque pour exclure cette zone
            excl_mask = ~((q_window >= q_min_excl) & (q_window <= q_max_excl))
            
            # Appliquer le masque
            if np.sum(excl_mask) > 0:  # S'il reste des points
                q_window = q_window[excl_mask]
                I_window = I_window[excl_mask]
                dI_window = dI_window[excl_mask]
                
                if verbose and np.sum(~excl_mask) > 0:
                    print(f"      → Exclusion zone [{q_min_excl:.4f}, {q_max_excl:.4f}] : "
                          f"{np.sum(~excl_mask)} points supprimés")
        
        if verbose:
            print(f"      Fenêtre nettoyée : {len(q_window)} points sur "
                  f"[{q_window.min():.4f}, {q_window.max():.4f}] Å⁻¹")
        
        if len(q_window) < 10:
            if verbose:
                print(f"      ⚠ Fenêtre trop petite ({len(q_window)} points), pic ignoré")
            continue
        
        if verbose:
            print(f"      Fenêtre : [{q_window_min:.4f}, {q_window_max:.4f}] Å⁻¹ "
                  f"({len(q_window)} points)")
        
        # Créer expérience locale
        exp_local = SAXSExperiment(
            data_dict={'Iq_local': (q_window, I_window, dI_window)},
            name=f'hybrid_peak_{i+1}'
        )
        
        # Fitter avec SPV (1 seul pic)
        try:
            exp_local.find_peaks_spv(
                input_attr='Iq_local',
                output_attr='SPV_local',
                verbose=False,
                plot=False
            )
            
            # Extraire résultats
            q_refined = exp_local.peaks['SPV_local']['q_values'][0]
            q_std = exp_local.peaks['SPV_local']['q_values_std'][0]
            FWHM_refined = exp_local.peaks['SPV_local']['FWHM'][0]
            
            # Propriétés SPV
            all_keys = exp_local.peaks['SPV_local'].keys()
            standard_keys = {'q_values', 'q_values_std', 'FWHM', 'Imin'}
            spv_params = {k: exp_local.peaks['SPV_local'][k] for k in all_keys 
                         if k not in standard_keys}
            
            q_peaks_refined.append(q_refined)
            q_peaks_std.append(q_std)
            FWHM_list.append(FWHM_refined)
            spv_params_list.append(spv_params)
            
            # NOUVEAU : Ajouter la zone du pic raffiné à la liste d'exclusion
            # pour éviter que ce pic soit refitté dans les itérations suivantes
            exclusion_margin = FWHM_refined * 1.5  # 1.5× FWHM de marge
            exclusion_zones.append((q_refined - exclusion_margin, 
                                    q_refined + exclusion_margin))
            
            if verbose:
                print(f"      → Zone d'exclusion : [{q_refined - exclusion_margin:.4f}, "
                      f"{q_refined + exclusion_margin:.4f}] Å⁻¹ (±{exclusion_margin:.4f})")
            
            delta_q = abs(q_refined - q_peak_init)
            
            if verbose:
                print(f"      ✓ Raffiné : q = {q_refined:.4f} ± {q_std:.4f} Å⁻¹")
                print(f"                  FWHM = {FWHM_refined:.4f} Å⁻¹")
                print(f"                  Δq = {delta_q:.4f} Å⁻¹ "
                      f"({delta_q/q_peak_init*100:.2f}%)")
        
        except Exception as e:
            if verbose:
                print(f"      ✗ Échec SPV : {e}")
            # Garder position initiale sans raffinement
            q_peaks_refined.append(q_peak_init)
            q_peaks_std.append(q_peak_init * 0.05)  # 5% d'incertitude par défaut
            FWHM_list.append(FWHM_estimate)
            spv_params_list.append({})
            
            # NOUVEAU : Ajouter la zone à l'exclusion même en cas d'échec
            exclusion_margin = FWHM_estimate * 1.5
            exclusion_zones.append((q_peak_init - exclusion_margin, 
                                    q_peak_init + exclusion_margin))
            
            if verbose:
                print(f"      → Zone d'exclusion : [{q_peak_init - exclusion_margin:.4f}, "
                      f"{q_peak_init + exclusion_margin:.4f}] Å⁻¹ (±{exclusion_margin:.4f})")
    
    # Convertir en arrays
    q_peaks_refined = np.array(q_peaks_refined)
    q_peaks_std = np.array(q_peaks_std)
    FWHM_list = np.array(FWHM_list)
    
    # =========================================================================
    # DÉDUPLATION : Supprimer les pics trop proches (refittés 2 fois)
    # =========================================================================
    
    if len(q_peaks_refined) > 1:
        # Calculer distances entre pics consécutifs
        valid_indices = list(range(len(q_peaks_refined)))
        
        for i in range(len(q_peaks_refined) - 1):
            if i not in valid_indices:
                continue  # Pic déjà marqué comme dupliqué
            
            for j in range(i + 1, len(q_peaks_refined)):
                if j not in valid_indices:
                    continue
                
                delta_q = abs(q_peaks_refined[j] - q_peaks_refined[i])
                # Seuil de déduplation absolu : pics séparés de < 0.001 Å⁻¹ sont considérés comme doublons
                # (c'est un vrai doublon, pas juste des pics proches)
                min_separation = 0.001  # Å⁻¹
                
                if delta_q < min_separation:
                    # Les pics sont trop proches - supprimer le moins intense
                    # (On utilise la "hauteur" estimée = 1/FWHM comme proxy)
                    intensity_proxy_i = 1.0 / FWHM_list[i]
                    intensity_proxy_j = 1.0 / FWHM_list[j]
                    
                    if intensity_proxy_i > intensity_proxy_j:
                        # Garder i, supprimer j
                        valid_indices.remove(j)
                        if verbose:
                            print(f"\n   ⚠ DÉDUPLATION : Pic {j+1} (q={q_peaks_refined[j]:.4f}) "
                                  f"trop proche de Pic {i+1} (q={q_peaks_refined[i]:.4f}) "
                                  f"[Δq={delta_q:.4f} < {min_separation:.4f}] → SUPPRIMÉ")
                    else:
                        # Garder j, supprimer i
                        valid_indices.remove(i)
                        if verbose:
                            print(f"\n   ⚠ DÉDUPLATION : Pic {i+1} (q={q_peaks_refined[i]:.4f}) "
                                  f"trop proche de Pic {j+1} (q={q_peaks_refined[j]:.4f}) "
                                  f"[Δq={delta_q:.4f} < {min_separation:.4f}] → SUPPRIMÉ")
                        break  # Passer au pic suivant
        
        # Appliquer la déduplation
        if len(valid_indices) < len(q_peaks_refined):
            q_peaks_refined = q_peaks_refined[valid_indices]
            q_peaks_std = q_peaks_std[valid_indices]
            FWHM_list = FWHM_list[valid_indices]
            spv_params_list = [spv_params_list[i] for i in valid_indices]
            
            if verbose:
                print(f"\n   → {len(q_peaks_refined)} pic(s) restant(s) après déduplation")
    
    # Calculer distances
    d_spacings = 2 * np.pi / q_peaks_refined
    d_spacings_std = d_spacings * (q_peaks_std / q_peaks_refined)
    
    # =========================================================================
    # RÉSULTATS
    # =========================================================================
    
    if verbose:
        print("\n" + "="*70)
        print("RÉSULTATS - MÉTHODE HYBRIDE")
        print("="*70)
        print(f"{len(q_peaks_refined)} pic(s) raffiné(s) :\n")
        
        for i in range(len(q_peaks_refined)):
            q_init = q_peaks_initial[i] if i < len(q_peaks_initial) else None
            q_ref = q_peaks_refined[i]
            q_std = q_peaks_std[i]
            d = d_spacings[i]
            d_std = d_spacings_std[i]
            
            print(f"Pic {i+1} :")
            if q_init is not None:
                delta = abs(q_ref - q_init)
                print(f"  Dérivée : q = {q_init:.4f} Å⁻¹")
                print(f"  SPV     : q = {q_ref:.4f} ± {q_std:.4f} Å⁻¹")
                print(f"  Δq      : {delta:.4f} Å⁻¹ ({delta/q_init*100:.2f}%)")
            else:
                print(f"  SPV     : q = {q_ref:.4f} ± {q_std:.4f} Å⁻¹")
            print(f"  Distance: d = {d:.1f} ± {d_std:.1f} Å")
            print(f"  FWHM    : {FWHM_list[i]:.4f} Å⁻¹")
            print()
        
        if power_law_order_value:
            print(f"Loi de puissance : m = {power_law_order_value:.2f}")
        print("="*70)
    
    # =========================================================================
    # VISUALISATION
    # =========================================================================
    
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot 1: Dérivée seconde
        ax1 = axes[0, 0]
        ax1.plot(q_masked, inverted_d2I, 'b-', linewidth=1.5, label='-d²I/dq²')
        ax1.plot(q_peaks_initial, inverted_d2I[top_peaks_idx], 
                'ro', markersize=10, label='Pics détectés')
        ax1.set_xlabel('q (Å⁻¹)', fontsize=12)
        ax1.set_ylabel('-d²I/dq²', fontsize=12)
        ax1.set_title('Étape 1 : Détection par Dérivée Seconde', 
                     fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Données prétraitées + détections initiales
        ax2 = axes[0, 1]
        ax2.loglog(q_preprocessed, I_preprocessed, 'k-', 
                  linewidth=1.5, alpha=0.7, label='Données prétraitées')
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33']
        for i, qp in enumerate(q_peaks_initial):
            color = colors[i % len(colors)]
            ax2.axvline(qp, color=color, linestyle='--', linewidth=2, 
                       alpha=0.5, label=f'Détection {i+1}')
        ax2.set_xlabel('q (Å⁻¹)', fontsize=12)
        ax2.set_ylabel('I(q) [prétraitée]', fontsize=12)
        ax2.set_title('Étape 2 : Prétraitement Global', 
                     fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Résultats raffinés
        ax3 = axes[1, 0]
        ax3.loglog(q_preprocessed, I_preprocessed, 'k-', 
                  linewidth=1.5, alpha=0.5, label='Données')
        for i, (qp, qstd) in enumerate(zip(q_peaks_refined, q_peaks_std)):
            color = colors[i % len(colors)]
            ax3.axvline(qp, color=color, linestyle='-', linewidth=2.5, 
                       label=f'Pic {i+1} (SPV)')
            ax3.axvspan(qp - qstd, qp + qstd, color=color, alpha=0.2)
        ax3.set_xlabel('q (Å⁻¹)', fontsize=12)
        ax3.set_ylabel('I(q) [prétraitée]', fontsize=12)
        ax3.set_title('Étape 3 : Raffinement SPV', 
                     fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Comparaison positions
        ax4 = axes[1, 1]
        x_pos = np.arange(len(q_peaks_refined))
        width = 0.35
        
        if len(q_peaks_initial) == len(q_peaks_refined):
            ax4.bar(x_pos - width/2, q_peaks_initial, width, 
                   label='Dérivée (initial)', alpha=0.7, color='gray')
        
        ax4.bar(x_pos + width/2, q_peaks_refined, width, 
               yerr=q_peaks_std, capsize=5,
               label='SPV (raffiné)', alpha=0.7, color='green')
        
        ax4.set_xlabel('Numéro du Pic', fontsize=12)
        ax4.set_ylabel('Position q (Å⁻¹)', fontsize=12)
        ax4.set_title('Comparaison : Dérivée vs SPV Raffiné', 
                     fontsize=13, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f'Pic {i+1}' for i in range(len(q_peaks_refined))])
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    # =========================================================================
    # RETOUR
    # =========================================================================
    
    results = {
        'q_peaks': q_peaks_refined,
        'q_peaks_std': q_peaks_std,
        'q_peaks_initial': q_peaks_initial,
        'FWHM': FWHM_list,
        'd_spacings': d_spacings,
        'd_spacings_std': d_spacings_std,
        'power_law_order': power_law_order_value,
        'spv_params': spv_params_list,
        'method': 'hybrid',
        'q_data': q_preprocessed,
        'I_processed': I_preprocessed
    }
    
    return results





