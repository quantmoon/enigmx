# EnigMX

Welcome to EnigMX, a sophisticated Python library designed for quantitative finance, specifically focusing on feature engineering, backtesting, model tuning, and advanced data analysis techniques in financial markets.

## Modules Overview

### Core Components
- **DataBundle**: Scripts for data bundling and preprocessing.
- **EnigmxMain**: The main executable scripts that combine various components.
- **FeatureImportance**: Tools for evaluating and optimizing the importance of different data features.

### Analysis Tools
- **Backtester**: A module for backtesting trading strategies.
- **Feature Clustering**: Scripts for clustering features based on their importance and contributions to model performance.
- **ModelTunning**: Contains utilities for tuning and optimizing machine learning models.

### Utility Scripts
- **utils.py**: General utilities that support data handling and manipulation.
- **databundle_interface.py**: Interface for managing data bundles effectively.
- **metrics.py**: Metrics for evaluating model and strategy performance.

### Testing and Development
- **tests**: Directory containing test scripts ensuring the reliability of the core components.
- **alternative_methods.py**: Contains alternative approaches and methods for statistical analysis.

## Features
- **Triple Barrier Method**: Implementation of the triple barrier method in financial data analysis.
- **SADF**: Scripts for performing Sequential Augmented Dickey-Fuller tests on data series.
- **Microstructural Features**: Addition of microstructural features to the analysis toolkit.

## Installation

Clone the repository and install the required Python packages:

```bash
git clone https://github.com/quantmoon/enigmx.git
cd enigmx
pip install -r requirements.txt
