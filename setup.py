"""Setup script for the Quantitative Research Platform."""

from setuptools import setup, find_packages

with open("README_QUANT_RESEARCH.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="quantitative-research-platform",
    version="1.0.0",
    author="Quantitative Research Team",
    author_email="quant@hivecapital.ai",
    description="Comprehensive quantitative trading research platform with advanced backtesting and ML capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/quantitative-research-platform",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.4",
            "pytest-asyncio>=0.21.2",
            "pytest-cov>=4.1.0",
            "black>=23.12.1",
            "isort>=5.13.2",
            "flake8>=6.1.0",
            "mypy>=1.17.1",
            "pre-commit>=3.8.0",
        ],
        "quant": [
            "QuantLib>=1.39",
            "QuantLib-Python>=1.18",
            "FinancePy>=1.0.1",
            "py-vollib>=1.0.1",
            "Riskfolio-Lib>=7.0.1",
            "empyrical-reloaded>=0.5.12",
            "QuantStats>=0.0.76",
        ],
        "ml": [
            "torch>=2.7.1",
            "transformers>=4.55.4",
            "stable-baselines3>=2.7.0",
            "gymnasium>=1.2.0",
            "FinRL>=0.3.7",
        ],
        "jupyter": [
            "jupyter>=1.1.1",
            "jupyterlab>=4.4.6",
            "ipython>=8.37.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "quant-research=quant_research.main:cli",
        ],
    },
    package_data={
        "quant_research": [
            "config/*.yaml",
            "examples/*.py",
            "docs/*.md",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)