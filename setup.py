"""
Financial Indicators Open Source Library Setup Script
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="financial-indicators",
    version="3.0.0",
    author="Financial Indicators Team",
    author_email="contact@financial-indicators.com",
    description="世界上最前沿的金融技术指标开源库",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RaphaelLcs-financial/financial-indicators",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=7.1.0", "black>=22.6.0", "flake8>=5.0.0"],
        "docs": ["sphinx>=5.0.0", "mkdocs>=1.3.0"],
        "ml": ["tensorflow>=2.10.0", "torch>=1.12.0", "xgboost>=1.6.0"],
        "crypto": ["ccxt>=2.0.0", "web3>=5.0.0"],
    },
    entry_points={
        "console_scripts": [
            "findicators=indicators.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="financial, indicators, quantitative, trading, machine-learning, quantum-finance",
    project_urls={
        "Bug Reports": "https://github.com/RaphaelLcs-financial/financial-indicators/issues",
        "Source": "https://github.com/RaphaelLcs-financial/financial-indicators",
        "Documentation": "https://financial-indicators.readthedocs.io/",
    },
)