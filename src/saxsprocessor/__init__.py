"""saxsprocessor package"""

# Imports absolus pour charger les modules
from .saxsprocessor import SAXSProcessor
from .utilities import compute_correlation_distances
from .utilities import compute_nematic_parameter

# from .pdfanalysis import perform_automatic_pdf_analysis  # Module not found

__version__ = "0.1.1"

# Ce que les utilisateurs peuvent importer avec 'from saxsprocessor import *'
__all__ = [
    'SAXSProcessor',
    'compute_correlation_distances',
    'compute_nematic_parameter'
]