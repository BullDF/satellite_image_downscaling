# Satellite Image Downscaling for Air Quality Assessment

This repository contains the code and documentation for a project on calibrating and downscaling MERRA-2 reanalysis data to improve air quality assessments, with a focus on PM2.5 concentrations.

## Project Overview

The primary goal of this project is to develop a reliable model for predicting PM2.5 levels by combining satellite data from NASA's MERRA-2 with ground-truth measurements from OpenAQ. This calibrated model will then be used to downscale satellite imagery, providing more granular and accurate air quality data.

## Project Progress

The project has progressed through several key stages:

1.  **Data Collection and Preprocessing:** We have successfully gathered and merged data from two primary sources:
    *   **OpenAQ:** Ground-level PM2.5 measurements.
    *   **MERRA-2:** Reanalysis data from NASA.
    The data has been cleaned, and initial exploratory data analysis has been performed to understand the relationships between variables.

2.  **Model Exploration:** We have experimented with a wide range of machine learning models to find the most effective approach for this task. The models explored include:
    *   Traditional models: XGBoost, Random Forest.
    *   Deep Learning models: LSTMs, Transformers, and Variational Autoencoders (VAEs) implemented in PyTorch.

3.  **Feature Engineering:** We have developed several feature engineering techniques to improve model performance, including:
    *   Cyclical encoding for temporal features (month, day, hour).
    *   Embedding layers for categorical features (site, season).
    *   Inclusion of aridity information as a feature.

4.  **Model Development:** Based on our experiments, we have developed and implemented several custom PyTorch models, including an `LSTMCalibrationModel` and an `ASDM` (Artificial Neural Network Sequential Downscaling Method), which are the core of our current approach.

5.  **Out-of-Sample Validation:** We have begun the process of validating our models on out-of-sample data to assess their generalizability, which is a crucial step before proceeding to the downscaling phase.

## Repository Structure

- `.`
- `├── cedar/`: Core PyTorch model definitions, training scripts, and utilities.
- `├── code/`: Scripts and notebooks for data downloading, preprocessing, and validation.
- `└── MERRA2 Calibration-Downscale/`: LaTeX source for the project report ([main.pdf](MERRA2%20Calibration-Downscale/main.pdf)).

## How to Run the Code

1.  **Download Data:**

    ```bash
    python code/initial/download_merra2.py <category>
    ```

    Replace `<category>` with `aerosols`, `meteorology`, or `surface_flux`.

2.  **Train a Model:**

    ```bash
    python cedar/train.py --lr 0.001 --train_bs 512 --epochs 50
    ```

    Use `python cedar/train.py --help` to see all available training arguments.

3.  **Validate a Model:**

    The `code/validation/validation.ipynb` notebook can be used to evaluate trained models.
