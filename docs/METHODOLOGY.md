# Methodology Documentation

## Overview

This document provides detailed information about the methodology used in the Marine ML Benchmark study for evaluating cross-dataset robustness of machine learning models in chlorophyll-a prediction.

## Research Design

### Objective
Evaluate the robustness of machine learning models across heterogeneous oceanographic datasets, focusing on model generalizability rather than data fusion approaches.

### Research Questions
1. How robust are ML models for chlorophyll-a prediction across different data types and scales?
2. Which models demonstrate the best cross-dataset generalization?
3. What data characteristics influence model performance and robustness?
4. Can we provide practical guidelines for model selection based on dataset properties?

## Dataset Selection and Preparation

### Inclusion Criteria
- **Temporal Coverage**: Minimum 1 year of data
- **Sample Size**: Minimum 1,000 observations after quality control
- **Data Quality**: <30% missing values in key variables
- **Target Variable**: Quantitative measure related to marine productivity
- **Geographic Scope**: Bohai Sea region for spatial consistency

### Exclusion Criteria
- **Insufficient Data**: <1,000 samples after preprocessing
- **Poor Quality**: >50% missing values or excessive outliers
- **Incompatible Format**: Cannot be standardized to common schema

### Final Dataset Selection
7 datasets passed validation criteria:
- **era5_daily**: Meteorological reanalysis (102,982 samples)
- **cast**: Oceanographic measurements (21,865 samples)
- **cleaned_data**: Multi-parameter observations (7,819 samples)
- **rolling_mean**: Smoothed chlorophyll-a data (8,855 samples)
- **processed_seq**: Time series sequences (8,039 samples)
- **biotoxin**: Harmful algal bloom data (5,076 samples)
- **hydrographic**: Hydrographic surveys (4,653 samples)

## Data Preprocessing Pipeline

### Stage 1: Data Loading and Standardization
```python
# Standardized loading process
for dataset in datasets:
    df = load_table(dataset_config)
    df = parse_dates(df, date_config)
    df = standardize_columns(df)
```

### Stage 2: Quality Control
- **Outlier Detection**: IQR method with 3σ threshold
- **Range Validation**: Domain-specific quality checks
- **Temporal Consistency**: Date format standardization
- **Missing Value Assessment**: Calculate missingness patterns

### Stage 3: Data Cleaning
- **Missing Value Imputation**: KNN imputation for <30% missing
- **Feature Removal**: Drop columns with ≥30% missing values
- **Outlier Treatment**: Remove observations outside valid ranges
- **Data Type Conversion**: Ensure numerical consistency

### Stage 4: Feature Engineering
- **Temporal Features**: Day of year, month, season indicators
- **Rolling Statistics**: 7-day and 30-day windows (mean, std, min, max)
- **Lag Features**: 1-7 day lags for time series data
- **Sequence Generation**: 30-step sequences for deep learning models

### Stage 5: Data Standardization
- **Numerical Scaling**: StandardScaler normalization
- **Target Transformation**: Log transformation where appropriate
- **Train/Test Splitting**: Temporal or stratified splits

## Model Selection and Configuration

### Traditional Machine Learning Models

#### Random Forest (RF)
- **Rationale**: Robust to overfitting, handles mixed data types well
- **Hyperparameters**: 
  - n_estimators: [100, 300, 500]
  - max_depth: [10, 20, None]
  - min_samples_split: [2, 5, 10]
  - min_samples_leaf: [1, 2, 4]

#### XGBoost (XGB)
- **Rationale**: Excellent performance on tabular data, built-in regularization
- **Hyperparameters**:
  - learning_rate: [0.01, 0.1, 0.2]
  - n_estimators: [100, 300, 500]
  - max_depth: [3, 6, 10]
  - subsample: [0.8, 0.9, 1.0]

#### Support Vector Regression (SVR)
- **Rationale**: Effective for non-linear relationships, kernel flexibility
- **Hyperparameters**:
  - C: [0.1, 1, 10, 100]
  - gamma: ['scale', 'auto', 0.001, 0.01, 0.1]
  - kernel: ['rbf', 'poly', 'sigmoid']

### Deep Learning Models

#### Long Short-Term Memory (LSTM)
- **Rationale**: Captures temporal dependencies in sequential data
- **Architecture**: 1-3 layers, 32-128 hidden units
- **Hyperparameters**:
  - hidden_size: [32, 64, 128]
  - num_layers: [1, 2, 3]
  - dropout: [0.1, 0.2, 0.3]
  - learning_rate: [0.001, 0.01, 0.1]

#### Transformer
- **Rationale**: Attention mechanism for sequence modeling
- **Architecture**: 1-3 layers, multi-head attention
- **Hyperparameters**:
  - d_model: [64, 128, 256]
  - nhead: [4, 8, 16]
  - num_layers: [1, 2, 3]
  - dropout: [0.1, 0.2, 0.3]

### Baseline Models
- **LASSO Regression**: L1 regularized linear regression
- **Ridge Regression**: L2 regularized linear regression
- **Mean Predictor**: Always predicts the training set mean

## Hyperparameter Optimization

### Optimization Strategy
- **Method**: Optuna Tree-structured Parzen Estimator (TPE)
- **Trials**: 30 trials per model-dataset combination
- **Objective**: Maximize R² score on validation set
- **Early Stopping**: For deep learning models (patience=10)

### Search Space Design
- **Logarithmic Scaling**: For learning rates and regularization parameters
- **Integer Parameters**: For tree-based model parameters
- **Categorical Parameters**: For kernel types and activation functions

### Validation Strategy
- **Time Series Data**: TimeSeriesSplit with 5 folds
- **Cross-sectional Data**: StratifiedKFold with 5 folds
- **Temporal Order**: Preserved for time-dependent datasets

## Evaluation Protocol

### Performance Metrics

#### Primary Metric: R² (Coefficient of Determination)
- **Formula**: R² = 1 - (SS_res / SS_tot)
- **Interpretation**: Proportion of variance explained by the model
- **Range**: (-∞, 1], where 1 indicates perfect prediction

#### Secondary Metrics
- **Mean Absolute Error (MAE)**: Average absolute prediction error
- **Root Mean Square Error (RMSE)**: Square root of mean squared error
- **Mean Absolute Percentage Error (MAPE)**: Percentage-based error metric

### Statistical Analysis

#### Confidence Intervals
- **Method**: Bootstrap resampling with 1,000 iterations
- **Confidence Level**: 95%
- **Application**: All performance metrics

#### Significance Testing
- **Paired t-test**: For comparing model performance
- **Bonferroni Correction**: For multiple comparisons
- **Effect Size**: Cohen's d for practical significance

#### Robustness Assessment
- **Cross-fold Stability**: Standard deviation across CV folds
- **Cross-dataset Consistency**: Performance variance across datasets
- **Sample Size Sensitivity**: Performance vs. dataset size analysis

### Model Comparison Framework

#### Individual Dataset Performance
- Best model identification per dataset
- Performance ranking across all models
- Statistical significance of differences

#### Cross-dataset Robustness
- Mean performance across all datasets
- Performance standard deviation (robustness indicator)
- Success rate (proportion of datasets with R² > 0.5)

#### Model Applicability
- Deep learning applicability assessment
- Data requirement analysis
- Computational efficiency comparison

## Feature Importance Analysis

### SHAP (SHapley Additive exPlanations)
- **Global Importance**: Average SHAP values across all samples
- **Local Explanations**: Individual prediction explanations
- **Feature Interactions**: Second-order SHAP interactions

### Permutation Importance
- **Method**: Permute feature values and measure performance drop
- **Validation**: Cross-validation to ensure stability
- **Ranking**: Consistent across different random seeds

### Cross-dataset Pattern Analysis
- **Common Features**: Important across multiple datasets
- **Dataset-specific Features**: Unique importance patterns
- **Feature Stability**: Consistency of importance rankings

## Validation and Quality Assurance

### Data Validation
- **Completeness Checks**: Verify all required fields present
- **Range Validation**: Ensure values within expected ranges
- **Temporal Consistency**: Check date ordering and gaps
- **Cross-reference Validation**: Compare with external sources

### Model Validation
- **Hyperparameter Sensitivity**: Test parameter stability
- **Random Seed Consistency**: Multiple runs with different seeds
- **Cross-validation Stability**: Consistent performance across folds
- **Overfitting Detection**: Training vs. validation performance

### Result Validation
- **Reproducibility**: All results reproducible with fixed seeds
- **Statistical Validity**: Appropriate statistical tests applied
- **Practical Significance**: Effect sizes reported alongside p-values
- **Robustness Checks**: Results stable across different preprocessing choices

## Limitations and Assumptions

### Methodological Limitations
- **Geographic Scope**: Limited to Bohai Sea region
- **Temporal Coverage**: Different time periods across datasets
- **Model Selection**: Limited to 5 mainstream algorithms
- **Hyperparameter Space**: Finite search space per computational constraints

### Data Limitations
- **Measurement Protocols**: Different instruments and methods
- **Spatial Resolution**: Varying spatial coverage and resolution
- **Temporal Resolution**: From daily to irregular sampling
- **Quality Control**: Different QC standards across data sources

### Statistical Assumptions
- **Independence**: Assumes samples are independent (may be violated in time series)
- **Stationarity**: Assumes statistical properties remain constant
- **Normality**: Some tests assume normal distribution of residuals
- **Homoscedasticity**: Assumes constant variance of errors

## Reproducibility Guidelines

### Computational Environment
- **Python Version**: 3.8+
- **Key Libraries**: scikit-learn 1.0+, XGBoost 1.4+, PyTorch 1.9+
- **Random Seeds**: Fixed at 42 for all random processes
- **Hardware**: Results obtained on standard CPU hardware

### Data Availability
- **Processed Data**: Available in repository
- **Raw Data Sources**: Documented with proper attribution
- **Preprocessing Scripts**: Complete pipeline provided
- **Quality Control Logs**: Detailed processing logs included

### Code Organization
- **Modular Design**: Separate modules for each processing stage
- **Configuration Files**: All parameters specified in YAML files
- **Documentation**: Comprehensive docstrings and comments
- **Testing**: Unit tests for all major functions

This methodology ensures rigorous, reproducible, and statistically sound evaluation of model robustness across heterogeneous oceanographic datasets.
