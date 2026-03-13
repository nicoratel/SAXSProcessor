"""
SAXSProcessor - Python toolkit for SAXS data reduction and analysis
"""

from .saxsprocessor import SAXSProcessor
from .saxs_analysis import (
    SAXSTools,
    SAXSData,
    SAXSPeakInfo,
    SAXSExperiment,
    PreprocessPipeline,
    PeakFinderPipeline,
    AnalysisPipeline,
    SAXSBatch,
)
from .nematicordercalculator import NematicOrderCalculator, BatchNematic
from .correlationdistancecalculator import CorrelationDistanceCalculator
from .filereaders import EdfFile, h5File_ID02, h5File_SWING
from .hybrid_peak_detection import detect_peaks_hybrid
from .utilities import (
    detect_incomplete_azimuthal_profile,
    compute_nematic_parameter,
)

__version__ = "0.1.0"

__all__ = [
    # Main processor
    "SAXSProcessor",
    # Analysis classes
    "SAXSTools",
    "SAXSData",
    "SAXSPeakInfo",
    "SAXSExperiment",
    "PreprocessPipeline",
    "PeakFinderPipeline",
    "AnalysisPipeline",
    "SAXSBatch",
    # Calculators
    "NematicOrderCalculator",
    "BatchNematic",
    "CorrelationDistanceCalculator",
    # File readers
    "EdfFile",
    "h5File_ID02",
    "h5File_SWING",
    # Utilities
    "detect_peaks_hybrid",
    "detect_incomplete_azimuthal_profile",
    "compute_nematic_parameter",
]
