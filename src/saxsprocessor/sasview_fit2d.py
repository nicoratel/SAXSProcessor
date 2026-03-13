"""
Module pour le fitting 2D avec SASView/Bumps
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

import sasmodels
import sasmodels.core
import sasmodels.data
import sasmodels.bumps_model
from sasmodels.data import plot_theory
import sasmodels.weights
sasmodels.weights.load_weights('maier_saupe.py')
sasmodels.weights.load_weights('maier_saupe_eq.py')

import bumps
import bumps.fitters
import bumps.fitproblem

# define matplotlib default values for better visualization
plt.rc('font', size=14) # default fontsize
plt.rc('axes', titlesize=16)
plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14)
plt.rc('legend', title_fontsize=14)
plt.rc('figure', titlesize=18)
plt.rc("figure", figsize=(5,5))
plt.rc("lines", linewidth=3)
plt.rc('image', cmap='jet')



def fit_2d_data(
    filename,
    model_name,
    params,
    roi_params=None,
    fit_ranges=None,
    slicing=10,
    error_coeff=0.05,
    I_min=0.0001,
    fit_method='lm',
    fit_steps=20,
    ftol=1.5e-08,
    xtol=1.5e-08,
    verbose=True,
    plot_results=True
):
    """
    Effectue un ajustement 2D avec SASView/Bumps.
    
    Paramètres
    ----------
    filename : str
        Chemin vers le fichier de données 2D (.dat)
    model_name : str
        Nom du modèle SASView (ex: 'elliptical_cylinder')
    params : dict
        Dictionnaire des paramètres du modèle
    roi_params : dict, optional
        Paramètres pour définir la région d'intérêt (ROI):
        - 'qmax_x': float, limite en qx (défaut: 0.08)
        - 'qmax_y': float, limite en qy (défaut: 0.03)
        - 'q_center': float, rayon du masque circulaire central (défaut: 0.01)
    fit_ranges : dict, optional
        Dictionnaire définissant les plages de variation pour le fit.
        Format: {'param_name': (min, max)}
        Ex: {'scale': (0.005, 0.1), 'background': (0.01, 10)}
    slicing : int, optional
        Facteur de réduction des points de données (défaut: 10)
        slicing=1 garde tous les points
    error_coeff : float, optional
        Coefficient d'erreur (défaut: 0.05 = 5%)
    I_min : float, optional
        Intensité minimale pour filtrer les données (défaut: 0.0001)
    fit_method : str, optional
        Méthode d'optimisation ('lm' pour Levenberg-Marquardt, défaut)
    fit_steps : int, optional
        Nombre d'étapes pour le fit (défaut: 20)
    ftol : float, optional
        Tolérance sur f(x) (défaut: 1.5e-08)
    xtol : float, optional
        Tolérance sur x (défaut: 1.5e-08)
    verbose : bool, optional
        Afficher les informations de progression (défaut: True)
    plot_results : bool, optional
        Afficher les graphiques (défaut: True)
    
    Retour
    ------
    dict
        Dictionnaire contenant:
        - 'model': le modèle fitté
        - 'experiment': l'expérience bumps
        - 'problem': le problème de fit
        - 'results': les résultats du fit
        - 'data_reduced': les données réduites utilisées
    """
    
    # Paramètres ROI par défaut
    if roi_params is None:
        roi_params = {
            'qmax_x': 0.08,
            'qmax_y': 0.03,
            'q_center': 0.01
        }
    
    # 1. Load data
    if verbose:
        print(f"Loading data from {filename}...")
    data2d = sasmodels.data.load_data(filename)
    
    I_all = data2d.data
    qx_all = data2d.qx_data
    qy_all = data2d.qy_data
    
    # 2. Apply ROI masks
    if verbose:
        print("Applying ROI masks...")
    
    qmax_x = roi_params.get('qmax_x', 0.08)
    qmax_y = roi_params.get('qmax_y', 0.03)
    q_center = roi_params.get('q_center', 0.01)
    
    # Créer les masques booléens (vectorisé, plus efficace)
    mask_intensity = I_all > I_min
    mask_roi_x = np.abs(qx_all) < qmax_x
    mask_roi_y = np.abs(qy_all) < qmax_y
    mask_center = (qx_all**2 + qy_all**2) > q_center**2
    
    # Combiner tous les masques
    mask_combined = mask_intensity & mask_roi_x & mask_roi_y & mask_center
    
    # Appliquer le masque
    qx_filtered = qx_all[mask_combined]
    qy_filtered = qy_all[mask_combined]
    I_filtered = I_all[mask_combined]
    error_filtered = error_coeff * I_filtered
    
    # Créer le tableau de données
    data_np = np.column_stack([qx_filtered, qy_filtered, I_filtered, error_filtered])
    
    if verbose:
        print(f"Number of points after filtering: {data_np.shape[0]}")
    
    # 3. Réduction des données (slicing)
    size = data_np.shape[0]
    data_reduced_ROI = sasmodels.data.Data2D(
        x=data_np[0:size:slicing, 0], 
        y=data_np[0:size:slicing, 1], 
        z=data_np[0:size:slicing, 2],
        dx=None, dy=None,
        dz=data_np[0:size:slicing, 3]
    )
    
    if verbose:
        print(f"Number of points after slicing (/{slicing}): {data_np[0:size:slicing, 0].shape[0]}")
    
    # 4. Create model
    if verbose:
        print(f"Loading model: {model_name}")
    kernel = sasmodels.core.load_model(model_name)
    model = sasmodels.bumps_model.Model(model=kernel, **params)
    
    # 5. Apply fitting ranges
    if fit_ranges is not None:
        if verbose:
            print("Applying fitting ranges...")
        for param_name, (min_val, max_val) in fit_ranges.items():
            if hasattr(model, param_name):
                getattr(model, param_name).range(min_val, max_val)
                if verbose:
                    print(f"  {param_name}: [{min_val}, {max_val}]")
    
    # 6. Création de l'expérience
    experiment = sasmodels.bumps_model.Experiment(data=data_reduced_ROI, model=model)
    
    # 7. Plot initial (before fit)
    if plot_results:
        plt.figure(figsize=(10, 5))
        plot_theory(data_reduced_ROI, experiment.theory(), view="log", limits=[0.1, 100])
        plt.title("Initial model (before fit)")
        plt.show()
    
    # 8. Fitting
    if verbose:
        print(f"\nStarting fit (method: {fit_method}, steps: {fit_steps})...")
    
    problem = bumps.fitproblem.FitProblem(experiment)
    results = bumps.fitters.fit(
        problem, 
        method=fit_method,
        steps=fit_steps, 
        ftol=ftol, 
        xtol=xtol, 
        verbose=verbose
    )
    
    # 9. Plot final (after fit)
    if plot_results:
        plt.figure(figsize=(10, 5))
        experiment.plot()
        plt.title("Fit result")
        plt.show()
    
    if verbose:
        print("\n=== Fit summary ===")
        print(problem.summarize())
        print(f"\nChi² = {problem.chisq():.4f}")
    
    # 10. Retour des résultats
    return {
        'model': model,
        'experiment': experiment,
        'problem': problem,
        'results': results,
        'data_reduced': data_reduced_ROI,
        'kernel': kernel
    }


def get_fitted_parameters(fit_result):
    """
    Extrait les paramètres fittés d'un résultat de fit.
    
    Paramètres
    ----------
    fit_result : dict
        Dictionnaire retourné par fit_2d_data()
    
    Retour
    ------
    dict
        Dictionnaire des paramètres fittés avec leurs valeurs
    """
    model = fit_result['model']
    problem = fit_result['problem']
    
    fitted_params = {}
    labels = problem.labels()
    values = fit_result['results'].x
    errors = fit_result['results'].dx
    
    for i, label in enumerate(labels):
        fitted_params[label] = {
            'value': values[i],
            'error': errors[i] if errors is not None else None
        }
    
    return fitted_params


def print_fitted_parameters(fit_result):
    """
    Affiche les paramètres fittés de manière formatée.
    
    Paramètres
    ----------
    fit_result : dict
        Dictionnaire retourné par fit_2d_data()
    """
    fitted_params = get_fitted_parameters(fit_result)
    
    print("\n=== Fitted parameters ===")
    for param_name, param_info in fitted_params.items():
        if param_info['error'] is not None:
            print(f"{param_name:20s} = {param_info['value']:10.4f} ± {param_info['error']:10.4f}")
        else:
            print(f"{param_name:20s} = {param_info['value']:10.4f}")


def calculate_nematic_order_parameter(distribution_type, pd_value, x0=90):
    """
    Calculate the nematic order parameter S for different angular distributions.
    
    The nematic order parameter is defined as:
    S = <P2(cos(theta))> where P2 is the second Legendre polynomial
    P2(x) = 0.5 * (3*x^2 - 1)
    
    Parameters
    ----------
    distribution_type : str
        Type of angular distribution:
        - 'gaussian': Gaussian distribution
        - 'maier_saupe': Maier-Saupe distribution
    pd_value : float
        Polydispersity parameter:
        - For 'gaussian': sigma in degrees (width of the distribution)
        - For 'maier_saupe': m parameter (strength of alignment)
    x0 : float, optional
        Central angle in degrees (default: 90)
        For orientation along the director axis
    
    Returns
    -------
    float
        Nematic order parameter S (between -0.5 and 1)
        S = 1: perfect alignment
        S = 0: isotropic (random orientation)
        S = -0.5: perpendicular alignment
    
    Examples
    --------
    >>> # Gaussian with sigma = 15 degrees
    >>> S = calculate_nematic_order_parameter('gaussian', 15)
    >>> print(f"S = {S:.4f}")
    
    >>> # Maier-Saupe with m = 10
    >>> S = calculate_nematic_order_parameter('maier_saupe', 10)
    >>> print(f"S = {S:.4f}")
    """
    
    # Legendre polynomial P2
    def P2(theta):
        return 0.5 * (3 * np.cos(theta)**2 - 1)
    
    if distribution_type.lower() == 'gaussian':
        # Convert sigma from degrees to radians
        sigma_rad = np.deg2rad(pd_value)
        x0_rad = np.deg2rad(x0)
        
        # Gaussian distribution
        def gaussian_dist(theta, sigma):
            return np.exp(-(theta)**2 / (2 * sigma**2))
        
        # Numerator: <P2> weighted by distribution
        def numerator_integrand(theta):
            return P2(theta) * gaussian_dist(theta, sigma_rad) * np.sin(theta)
        
        # Denominator: normalization
        def denominator_integrand(theta):
            return gaussian_dist(theta, sigma_rad) * np.sin(theta)
        
        numerator, _ = quad(numerator_integrand, 0, np.pi, epsabs=1e-10, epsrel=1e-10)
        denominator, _ = quad(denominator_integrand, 0, np.pi, epsabs=1e-10, epsrel=1e-10)
        
        return numerator / denominator
    
    elif distribution_type.lower() in ['maier_saupe', 'maier-saupe', 'maiersaupe']:
        # m is the Maier-Saupe parameter (strength)
        m = pd_value
        x0_rad = np.deg2rad(x0)
        
        # Maier-Saupe distribution: exp(m * cos^2(theta))
        def maier_saupe_dist(theta, m, x0):
            return np.exp(m * np.cos(theta - x0)**2)
        
        # Numerator
        def numerator_integrand(theta):
            return P2(theta - x0_rad) * maier_saupe_dist(theta, m, x0_rad) * np.sin(theta)
        
        # Denominator
        def denominator_integrand(theta):
            return maier_saupe_dist(theta, m, x0_rad) * np.sin(theta)
        
        numerator, _ = quad(numerator_integrand, 0, np.pi, epsabs=1e-10, epsrel=1e-10)
        denominator, _ = quad(denominator_integrand, 0, np.pi, epsabs=1e-10, epsrel=1e-10)
        
        return numerator / denominator
    
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}. "
                        f"Supported types: 'gaussian', 'maier_saupe'")


def calculate_nematic_order_from_fit(fit_result, angle_param='theta', distribution_type=None):
    """
    Calculate nematic order parameter from fit results.
    
    Parameters
    ----------
    fit_result : dict
        Dictionary returned by fit_2d_data()
    angle_param : str, optional
        Name of the angular parameter ('theta' or 'phi', default: 'theta')
    distribution_type : str, optional
        Type of distribution ('gaussian' or 'maier_saupe').
        If None, it will be detected from the model parameters.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'S': nematic order parameter
        - 'distribution_type': type of distribution used
        - 'pd_value': polydispersity parameter value
        - 'angle_value': central angle value
    
    Examples
    --------
    >>> result = fit_2d_data(...)
    >>> nematic_info = calculate_nematic_order_from_fit(result, angle_param='theta')
    >>> print(f"S_theta = {nematic_info['S']:.4f}")
    """
    model = fit_result['model']
    
    # Get the polydispersity type and value
    pd_type_attr = f"{angle_param}_pd_type"
    pd_value_attr = f"{angle_param}_pd"
    angle_attr = angle_param
    
    if not hasattr(model, pd_type_attr):
        raise ValueError(f"Model does not have parameter: {pd_type_attr}")
    
    # Helper function to get value from model parameter
    def get_param_value(obj):
        """Get value from parameter object or return directly if it's already a value"""
        if hasattr(obj, 'value'):
            return obj.value
        else:
            return obj
    
    # Auto-detect distribution type if not provided
    if distribution_type is None:
        distribution_type = get_param_value(getattr(model, pd_type_attr))
    
    pd_value = get_param_value(getattr(model, pd_value_attr))
    angle_value = get_param_value(getattr(model, angle_attr))
    
    # Calculate S
    S = calculate_nematic_order_parameter(distribution_type, pd_value, x0=angle_value)
    
    return {
        'S': S,
        'distribution_type': distribution_type,
        'pd_value': pd_value,
        'angle_value': angle_value,
        'angle_param': angle_param
    }


def print_nematic_order(fit_result, angle_params=['theta', 'phi']):
    """
    Print nematic order parameters for specified angles.
    
    Parameters
    ----------
    fit_result : dict
        Dictionary returned by fit_2d_data()
    angle_params : list of str, optional
        List of angle parameters to analyze (default: ['theta', 'phi'])
    
    Examples
    --------
    >>> result = fit_2d_data(...)
    >>> print_nematic_order(result)
    """
    print("\n=== Nematic Order Parameters ===")
    
    for angle_param in angle_params:
        try:
            nematic_info = calculate_nematic_order_from_fit(fit_result, angle_param=angle_param)
            
            print(f"\n{angle_param.upper()}:")
            print(f"  Distribution type: {nematic_info['distribution_type']}")
            print(f"  Central angle:     {nematic_info['angle_value']:.2f}°")
            print(f"  PD parameter:      {nematic_info['pd_value']:.4f}")
            print(f"  Order parameter S: {nematic_info['S']:.5f}")
            
        except (AttributeError, ValueError) as e:
            print(f"\n{angle_param.upper()}: Could not calculate (no polydispersity or {e})")


if __name__ == "__main__":
    # Exemple d'utilisation
    print("Module sasview_fit2d chargé avec succès!")
    print("\nExemple d'utilisation:")
    print("""
    from sasview_fit2d import fit_2d_data
    
    # Définir les paramètres du modèle
    params = {
        "phi": 90, "theta": 90, "psi": 0,
        "theta_pd_type": 'gaussian',
        "theta_pd": 15,
        "theta_pd_n": 100,
        "theta_pd_nsigma": 3,
        "phi_pd_type": 'gaussian',
        "phi_pd": 15,
        "phi_pd_n": 100,
        "phi_pd_nsigma": 3,
        "axis_ratio": 1,
        "radius_minor": 141,
        "radius_minor_pd_type": 'gaussian',
        "radius_minor_pd": 0.06,
        "radius_minor_pd_n": 8,
        "radius_minor_pd_nsigma": 4,
        "length": 10000,
        "background": 1,
        "scale": 0.02
    }
    
    # Définir les plages de fit
    fit_ranges = {
        'scale': (0.005, 0.1),
        'background': (0.01, 10),
        'theta_pd': (12, 16),
        'phi_pd': (12, 16)
    }
    
    # Définir la ROI
    roi_params = {
        'qmax_x': 0.08,
        'qmax_y': 0.03,
        'q_center': 0.01
    }
    
    # Lancer le fit
    result = fit_2d_data(
        filename="./E-field-isotropic/280V_240.dat",
        model_name="elliptical_cylinder",
        params=params,
        roi_params=roi_params,
        fit_ranges=fit_ranges,
        slicing=10
    )
    """)
