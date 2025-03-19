# STELAR-GAM

**STELAR-GAM** stands for **Spatio-TEmporaL lAg-based Regression** Generalized Additive Model. This repository contains a single Python script demonstrating how to fit a spatiotemporal-lag GAM using B-spline bases and a restricted maximum likelihood (REML) approach for selecting smoothing parameters. The method is inspired by frameworks such as **mgcv** in R but remains self-contained in Python for illustrative purposes.

---

## Overview

- **Model Type**: Spatiotemporal-lag GAM  
- **Smoothing**: Cubic B-splines (or any degree) on latitude, longitude, time, and lag  
- **Penalty**: Second-difference penalty on spline coefficients  
- **Smoothing Parameter Selection**: REML-like objective function, optimized via `scipy.optimize.minimize`  
- **Interpretability**: Uses linear algebra and spline bases directly, rather than black-box machine learning methods  

---

## Features

- **Additive Structure**: Latitude, longitude, time, and lag components are modeled additively.  
- **Spline Construction**: Includes helper functions to generate B-spline design matrices and second-difference penalty matrices.  
- **REML Optimization**: A simplified negative log-REML objective is used to find an optimal smoothing parameter.  
- **Prediction**: Provides a `predict` method to generate predictions at new coordinates and time/lag points.  
- **Example Usage**: Demonstrates fitting the model to synthetic data and visualizing results.

---

## Getting Started

Clone or download this repository to access the `stelar_gam.py` script. The script includes all necessary functionality in a single file. Python dependencies are listed in the **Dependencies** section below. A quick usage example is included at the bottom of the script.

### Dependencies

- Python 3.7+ (earlier versions may work but are untested)  
- `numpy`  
- `pandas`  
- `matplotlib`  
- `scipy`  

These can be installed using pip:

```bash
pip install numpy pandas matplotlib scipy
