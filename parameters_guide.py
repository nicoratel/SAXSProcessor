"""
GUIDE COMPLET DES PARAMÈTRES POUR detect_peaks()
=================================================

Ce guide liste tous les paramètres acceptés via **detection_params
selon la méthode choisie (SPV ou derivative).
"""

# =============================================================================
# MÉTHODE SPV (method='spv') - PARAMÈTRES DISPONIBLES
# =============================================================================

SPV_PARAMETERS = {
    
    # --- PARAMÈTRES DE MASQUAGE EN Q ---
    'qmin': {
        'type': float,
        'default': None,
        'description': 'Valeur minimale de q à considérer (Å⁻¹)',
        'exemple': 'qmin=0.01',
        'note': 'Permet de restreindre la plage de q analysée'
    },
    
    'qmax': {
        'type': float,
        'default': None,
        'description': 'Valeur maximale de q à considérer (Å⁻¹)',
        'exemple': 'qmax=0.5',
        'note': 'Utile pour exclure les hauts q bruités'
    },
    
    # --- PARAMÈTRES DE LOI DE PUISSANCE ---
    'subtract_power_law': {
        'type': bool,
        'default': True,
        'description': 'Activer la soustraction de loi de puissance I(q) ~ q^(-m)',
        'exemple': 'subtract_power_law=True',
        'note': 'TRÈS RECOMMANDÉ pour des résultats précis'
    },
    
    'power_law_method': {
        'type': str,
        'default': 'cancel',
        'choices': ['cancel', 'feat'],
        'description': 'Méthode de soustraction de loi de puissance',
        'exemple': "power_law_method='cancel'",
        'note': {
            'cancel': 'Méthode standard (robuste)',
            'feat': 'Méthode avancée (utilise détection de pics pour égaliser minima)'
        }
    },
    
    'power_law_order': {
        'type': float,
        'default': None,
        'description': 'Ordre m fixe de la loi de puissance',
        'exemple': 'power_law_order=4  # Loi de Porod',
        'note': 'Si None, m est optimisé automatiquement. Valeurs typiques: 2-4'
    },
    
    'power_law_range': {
        'type': tuple,
        'default': (2.5, 4.0),
        'description': 'Plage de recherche pour optimiser m',
        'exemple': 'power_law_range=(3.0, 4.0)',
        'note': "N'est utilisé que si power_law_order=None"
    },
    
    # --- PARAMÈTRES DE LISSAGE ---
    'smooth': {
        'type': bool,
        'default': False,
        'description': 'Activer le lissage gaussien des données',
        'exemple': 'smooth=True',
        'note': 'Utile si données très bruitées'
    },
    
    'smooth_sigma': {
        'type': float,
        'default': 2,
        'description': 'Paramètre sigma du filtre gaussien',
        'exemple': 'smooth_sigma=1.5',
        'note': 'Plus grand = plus de lissage'
    },
    
    # --- PARAMÈTRES DE VISUALISATION ---
    'verbose': {
        'type': bool,
        'default': True,
        'description': 'Afficher les informations détaillées',
        'exemple': 'verbose=False',
        'note': 'Désactiver pour traitements en batch'
    },
    
    'plot': {
        'type': bool,
        'default': False,
        'description': 'Afficher les graphiques de fit',
        'exemple': 'plot=True',
        'note': 'Montre le fit SPV complet'
    },
}


# =============================================================================
# MÉTHODE DERIVATIVE (method='derivative') - PARAMÈTRES DISPONIBLES
# =============================================================================

DERIVATIVE_PARAMETERS = {
    
    # --- PARAMÈTRES DE MASQUAGE EN Q (UNIFORMISÉ AVEC SPV) ---
    'qmin': {
        'type': float,
        'default': None,
        'description': 'Valeur minimale de q à considérer (Å⁻¹)',
        'exemple': 'qmin=0.01',
        'note': 'Même interface que méthode SPV'
    },
    
    'qmax': {
        'type': float,
        'default': None,
        'description': 'Valeur maximale de q à considérer (Å⁻¹)',
        'exemple': 'qmax=0.5',
        'note': 'Même interface que méthode SPV'
    },
    
    # --- PARAMÈTRES DU FILTRE SAVITZKY-GOLAY ---
    'window_length': {
        'type': int,
        'default': 15,
        'description': 'Taille de la fenêtre de lissage (doit être impair)',
        'exemple': 'window_length=11',
        'note': 'Plus grand = plus de lissage, moins de sensibilité'
    },
    
    'polyorder': {
        'type': int,
        'default': 3,
        'description': 'Ordre du polynôme pour le lissage',
        'exemple': 'polyorder=2',
        'note': 'Généralement entre 2 et 4'
    },
    
    # --- PARAMÈTRES DE DÉTECTION DE PICS ---
    'prominence': {
        'type': float,
        'default': 0.5,
        'description': 'Prominence minimale des pics dans -d²I/dq²',
        'exemple': 'prominence=1.0',
        'note': 'Augmenter pour être plus sélectif'
    },
    
    'distance_pts': {
        'type': int,
        'default': 20,
        'description': 'Distance minimale entre pics (en points)',
        'exemple': 'distance_pts=30',
        'note': 'Évite de détecter des pics trop proches'
    },
    
    # --- PARAMÈTRES DE VISUALISATION ---
    'plot': {
        'type': bool,
        'default': False,
        'description': 'Afficher le graphique de détection',
        'exemple': 'plot=True',
        'note': 'Montre I(q) avec positions des pics'
    },
}


# =============================================================================
# EXEMPLES D'UTILISATION PRATIQUES
# =============================================================================

EXEMPLES_USAGE = """
╔═══════════════════════════════════════════════════════════════════════╗
║                    EXEMPLES D'UTILISATION                             ║
╚═══════════════════════════════════════════════════════════════════════╝

1. MÉTHODE SPV - CONFIGURATION DE BASE
────────────────────────────────────────
results = calc.compute_correlation_distances(
    nb_peaks=2,
    azimuth=90,
    width=40,
    method='spv',
    qmin=0.01,              # Exclure les petits q
    qmax=0.5,               # Exclure les grands q
    subtract_power_law=True,
    verbose=True,
    plot=True
)


2. MÉTHODE SPV - CONFIGURATION AVANCÉE
────────────────────────────────────────
results = calc.compute_correlation_distances(
    nb_peaks=3,
    azimuth=90,
    width=40,
    method='spv',
    qmin=0.02,
    qmax=0.4,
    subtract_power_law=True,
    power_law_method='cancel',     # 'cancel' ou 'feat'
    power_law_order=None,          # None = auto, ou fixer à 3, 4, etc.
    power_law_range=(2.5, 4.0),    # Plage si order=None
    smooth=True,                   # Lisser données bruitées
    smooth_sigma=2.0,              # Paramètre de lissage
    verbose=True,
    plot=True
)


3. MÉTHODE SPV - LOI DE PUISSANCE FIXÉE (POROD)
─────────────────────────────────────────────────
results = calc.compute_correlation_distances(
    nb_peaks=2,
    method='spv',
    qmin=0.05,
    qmax=0.3,
    subtract_power_law=True,
    power_law_order=4,           # Imposer loi de Porod
    verbose=True
)


4. MÉTHODE SPV - SANS SOUSTRACTION DE LOI DE PUISSANCE
────────────────────────────────────────────────────────
results = calc.compute_correlation_distances(
    nb_peaks=1,
    method='spv',
    qmin=0.08,
    qmax=0.25,
    subtract_power_law=False,    # Désactiver soustraction
    smooth=True,                 # Mais lisser
    smooth_sigma=1.5,
    verbose=True
)


5. MÉTHODE DERIVATIVE - CONFIGURATION STANDARD
────────────────────────────────────────────────
results = calc.compute_correlation_distances(
    nb_peaks=2,
    method='derivative',
    qmin=0.05,                   # ✅ Interface uniformisée
    qmax=0.3,                    # ✅ Interface uniformisée
    window_length=15,            # Fenêtre de lissage
    polyorder=3,                 # Ordre polynomial
    prominence=0.5,              # Sélectivité
    distance_pts=20,             # Distance min entre pics
    plot=True
)


6. MÉTHODE DERIVATIVE - TRÈS SÉLECTIF
───────────────────────────────────────
results = calc.compute_correlation_distances(
    nb_peaks=1,
    method='derivative',
    qmin=0.1,                    # ✅ Même interface que SPV
    qmax=0.25,                   # ✅ Même interface que SPV
    window_length=21,            # Plus de lissage
    polyorder=4,
    prominence=2.0,              # Plus sélectif
    distance_pts=50,             # Pics bien espacés
    plot=True
)


7. ANALYSE D'ANISOTROPIE AVEC SPV
───────────────────────────────────
aniso = calc.analyze_anisotropy(
    nb_peaks=2,
    azimuth_list=[0, 45, 90, 135],
    width=40,
    method='spv',
    qmin=0.02,
    qmax=0.4,
    subtract_power_law=True,
    power_law_method='cancel',
    smooth=False,
    verbose=False,               # Désactiver pour batch
    plot=True
)


8. COMPARAISON DES MÉTHODES
─────────────────────────────
comparison = calc.compare_methods(
    nb_peaks=2,
    azimuth=90,
    width=40,
    spv_params={
        'qmin': 0.02,
        'qmax': 0.4,
        'subtract_power_law': True,
        'smooth': False,
        'verbose': False
    },
    derivative_params={
        'qmin': 0.02,            # ✅ Uniformisé
        'qmax': 0.4,             # ✅ Uniformisé
        'window_length': 15,
        'prominence': 0.5,
        'distance_pts': 20
    }
)


═══════════════════════════════════════════════════════════════════════

RECOMMANDATIONS PAR CAS D'USAGE
═════════════════════════════════

📊 CAS 1 : Pics bien définis, signal propre
─────────────────────────────────────────────
method='spv'
subtract_power_law=True
power_law_order=None          # Optimisation auto
smooth=False                  # Pas besoin de lissage
qmin/qmax selon votre système


📊 CAS 2 : Données bruitées
────────────────────────────
method='spv'
subtract_power_law=True
smooth=True                   # Activer lissage
smooth_sigma=2.0              # Ajuster selon bruit
qmin/qmax pour éviter zones très bruitées


📊 CAS 3 : Pics larges/asymétriques
────────────────────────────────────
method='spv'                  # SPV gère l'asymétrie !
subtract_power_law=True
power_law_method='cancel'
smooth=False
Bien définir qmin/qmax


📊 CAS 4 : Fond avec loi de Porod connue
──────────────────────────────────────────
method='spv'
subtract_power_law=True
power_law_order=4             # Imposer Porod
qmin selon Guinier region


📊 CAS 5 : Plusieurs pics proches
───────────────────────────────────
method='spv'                  # Fit global !
nb_peaks=3 (ou plus)
subtract_power_law=True
qmin/qmax bien ajustés
smooth=False (éviter de fusionner pics)


📊 CAS 6 : Validation rapide / premiers tests
───────────────────────────────────────────────
method='derivative'           # Plus rapide
q_range selon région d'intérêt
window_length=15
prominence=0.5
Puis affiner avec SPV


═══════════════════════════════════════════════════════════════════════

VALEURS TYPIQUES SELON LE SYSTÈME
═══════════════════════════════════

🔬 Systèmes lamellaires (pics fins)
────────────────────────────────────
qmin=0.01, qmax=0.3
power_law_order=None (auto 2-4)
smooth=False
window_length=11 (derivative)


🔬 Micelles / systèmes isotropes (pics larges)
────────────────────────────────────────────────
qmin=0.05, qmax=0.5
power_law_order=None (auto 3-4)
smooth=True, smooth_sigma=2
window_length=15 (derivative)


🔬 Polymères / nanoparticules
───────────────────────────────
qmin=0.02, qmax=0.4
power_law_order=4 (Porod si interfaces nettes)
smooth selon qualité données
window_length=15-21 (derivative)


🔬 Systèmes très ordonnés (plusieurs ordres)
──────────────────────────────────────────────
nb_peaks=3-5
qmin très petit (0.005), qmax=0.6
power_law_order=None
smooth=False (garder structure fine)
distance_pts=30 (derivative, pics espacés)


═══════════════════════════════════════════════════════════════════════
"""


# =============================================================================
# TABLEAU RÉCAPITULATIF
# =============================================================================

RECAP_TABLE = """
╔═══════════════════════════════════════════════════════════════════════╗
║              TABLEAU RÉCAPITULATIF DES PARAMÈTRES                     ║
║                    ✅ INTERFACE UNIFORMISÉE                           ║
╚═══════════════════════════════════════════════════════════════════════╝

┌─────────────────────┬──────────┬───────────┬──────────────────────────┐
│ Paramètre           │ Méthode  │ Défaut    │ Description              │
├─────────────────────┼──────────┼───────────┼──────────────────────────┤
│ qmin                │ Both ✅  │ None      │ q min (Å⁻¹)              │
│ qmax                │ Both ✅  │ None      │ q max (Å⁻¹)              │
│ plot                │ Both ✅  │ False     │ Graphiques               │
├─────────────────────┼──────────┼───────────┼──────────────────────────┤
│ subtract_power_law  │ SPV      │ True      │ Soustraire I~q^(-m)      │
│ power_law_method    │ SPV      │ 'cancel'  │ 'cancel' ou 'feat'       │
│ power_law_order     │ SPV      │ None      │ m fixe ou None=auto      │
│ power_law_range     │ SPV      │ (2.5,4.0) │ Plage pour m             │
│ smooth              │ SPV      │ False     │ Lissage gaussien         │
│ smooth_sigma        │ SPV      │ 2         │ Paramètre lissage        │
│ verbose             │ SPV      │ True      │ Infos détaillées         │
├─────────────────────┼──────────┼───────────┼──────────────────────────┤
│ window_length       │ Deriv    │ 15        │ Fenêtre S-G (impair)     │
│ polyorder           │ Deriv    │ 3         │ Ordre polynomial         │
│ prominence          │ Deriv    │ 0.5       │ Prominence min           │
│ distance_pts        │ Deriv    │ 20        │ Distance min (pts)       │
└─────────────────────┴──────────┴───────────┴──────────────────────────┘

Note : "Both ✅" = INTERFACE UNIFORMISÉE pour les deux méthodes
       "SPV" = uniquement pour method='spv'
       "Deriv" = uniquement pour method='derivative'
       
💡 Avantage : qmin/qmax fonctionne de la même façon peu importe la méthode !
"""


# =============================================================================
# FONCTION HELPER POUR AFFICHER L'AIDE
# =============================================================================

def print_parameters_help(method='spv'):
    """
    Affiche l'aide détaillée sur les paramètres disponibles.
    
    Parameters:
    -----------
    method : str
        'spv', 'derivative', ou 'all'
    """
    print(RECAP_TABLE)
    print(EXEMPLES_USAGE)
    
    if method.lower() in ['spv', 'all']:
        print("\n" + "="*70)
        print("PARAMÈTRES DÉTAILLÉS - MÉTHODE SPV")
        print("="*70 + "\n")
        for param, info in SPV_PARAMETERS.items():
            print(f"📌 {param}")
            print(f"   Type    : {info['type'].__name__}")
            print(f"   Défaut  : {info['default']}")
            print(f"   Exemple : {info['exemple']}")
            print(f"   Note    : {info['note']}")
            print()
    
    if method.lower() in ['derivative', 'all']:
        print("\n" + "="*70)
        print("PARAMÈTRES DÉTAILLÉS - MÉTHODE DERIVATIVE")
        print("="*70 + "\n")
        for param, info in DERIVATIVE_PARAMETERS.items():
            print(f"📌 {param}")
            print(f"   Type    : {info['type'].__name__}")
            print(f"   Défaut  : {info['default']}")
            print(f"   Exemple : {info['exemple']}")
            print(f"   Note    : {info['note']}")
            print()


# =============================================================================
# TEST INTERACTIF
# =============================================================================

if __name__ == "__main__":
    print(RECAP_TABLE)
    print(EXEMPLES_USAGE)
    
    print("\n" + "="*70)
    print("Pour voir l'aide détaillée, utilisez :")
    print("="*70)
    print("from parameters_guide import print_parameters_help")
    print("print_parameters_help('spv')        # Aide SPV")
    print("print_parameters_help('derivative')  # Aide derivative")
    print("print_parameters_help('all')        # Aide complète")
