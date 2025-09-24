"""
Quantitative Trading Research Platform

A comprehensive framework for quantitative analysis, strategy development,
backtesting, and machine learning model training for financial markets.
"""

__version__ = "1.0.0"
__author__ = "Quantitative Research Team"

from quant_research.config import settings
from quant_research.utils.core import setup_logging

setup_logging(settings.LOG_LEVEL)