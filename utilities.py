from nematicordercalculator import CylinderFormFactor,NematicOrderCalculator
   

def compute_nematic_parameter(
        processor = None,
        qvalue = 0.034,
        threshold = 0.05,
        radius = 78,
        L = 840,
        radius_pd = 0.3,
        L_pd = 0.75,
        plot=False,
        apply_mirror=False,
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
    apply_mirror: bool
        Apply mirror symmetry to incomplete exp azimuthal profile

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
    chi_exp, I_az_exp = processor.extract_azimuthal_profile(qvalue=0.034, threshold = 0.05)
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
    return results["S"], angle_folded, results


