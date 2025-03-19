#!/usr/bin/env python3
"""
STELAR (Spatio-TEmporaL lAg-based Regression) GAM
-------------------------------------------------
A minimal, single-file Python "package" for fitting a spatiotemporal-lag
GAM using Restricted Maximum Likelihood (REML) to choose smoothing
parameters, inspired by R's mgcv approach.

This script demonstrates how to:
  1. Construct B-spline bases for (lat, lon, time, lag).
  2. Combine these into an additive model design matrix.
  3. Penalize the spline coefficients.
  4. Use a REML-like objective to pick the smoothing parameter(s).
  5. Fit the model and predict on new data.

Author: Jangho Lee
License: MIT 
"""

import numpy as np
import pandas as pd
from scipy.linalg import solve, det
from scipy.optimize import minimize
from typing import Optional, Union, Tuple

# ---------------------------------------------------------------------
# 1. HELPER FUNCTIONS: B-SPLINES AND PENALTY MATRICES
# ---------------------------------------------------------------------
def make_knots(x, n_splines, degree=3):
    """
    Creates a simple knot vector for B-splines:
      - The knots extend from min(x) to max(x), with extra boundary knots
        for 'degree' (e.g., cubic => 3) at each end.
    """
    x_min, x_max = np.min(x), np.max(x)
    # Internal knots are spaced between x_min and x_max
    n_internal = n_splines - (degree + 1)
    knots = np.linspace(x_min, x_max, n_internal)
    # Adds boundary knots
    extended_knots = np.concatenate((
        np.repeat(x_min, degree),
        knots,
        np.repeat(x_max, degree)
    ))
    return extended_knots

def b_spline_basis_1d(x, n_splines, degree=3):
    """
    Constructs a 1D B-spline basis for the vector x.

    Returns
    -------
    B : ndarray of shape (len(x), n_splines)
        Design matrix of spline basis functions.
    knots : ndarray
        Positions of the knot sequence used.
    """
    from scipy.interpolate import BSpline
    
    # Creates knot sequence
    knots = make_knots(x, n_splines, degree=degree)
    ncoef = n_splines
    B = np.zeros((len(x), ncoef))

    # Builds the basis by evaluating each coefficient as 1 in turn
    for i in range(ncoef):
        c = np.zeros(ncoef)
        c[i] = 1.0
        spline = BSpline(knots, c, degree)
        B[:, i] = spline(x)
    return B, knots

def second_difference_penalty(n_splines):
    """
    Builds a discrete second-difference penalty matrix of size
    (n_splines x n_splines). This approach penalizes curvature
    in the spline coefficients.
    """
    D = np.zeros((n_splines - 2, n_splines))
    for i in range(n_splines - 2):
        D[i, i]   = 1
        D[i, i+1] = -2
        D[i, i+2] = 1
    # Penalty matrix = D^T D
    return D.T @ D

# ---------------------------------------------------------------------
# 2. STELAR GAM CLASS
# ---------------------------------------------------------------------
class StelarGAM:
    """
    A minimal demonstration of a Spatio-Temporal-Lag GAM using
    B-splines and REML-based smoothing. This is a direct, single-penalty
    approach for illustration purposes.
    """

    def __init__(self,
                 n_splines_space=10,
                 n_splines_time=10,
                 n_splines_lag=10,
                 degree=3,
                 max_iter=50,
                 tol=1e-6,
                 verbose=True):
        """
        Initializes the STELAR-GAM model.

        Parameters
        ----------
        n_splines_space : int
            Number of basis splines for each spatial dimension (lat, lon).
        n_splines_time : int
            Number of basis splines for the time dimension.
        n_splines_lag : int
            Number of basis splines for the lag dimension.
        degree : int
            Polynomial degree for B-splines (3 is typical for cubic).
        max_iter : int
            Maximum iterations for smoothing-parameter optimization.
        tol : float
            Tolerance for convergence in smoothing-parameter optimization.
        verbose : bool
            Controls whether progress messages are printed.
        """
        self.n_splines_space = n_splines_space
        self.n_splines_time = n_splines_time
        self.n_splines_lag = n_splines_lag
        self.degree = degree
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        # Placeholders
        self.B_lat = None
        self.B_lon = None
        self.B_time = None
        self.B_lag = None
        self.S_lat = None
        self.S_lon = None
        self.S_time = None
        self.S_lag = None
        self.design_matrix = None
        self.penalty_matrix = None

        self.coef_ = None      # Fitted coefficients
        self.lambda_ = None    # Smoothing parameter
        self.sigma2_ = None    # Residual variance
        self.nobs_ = None
        self.ranks_ = None

    def _build_design_and_penalties(self, lat, lon, t, lag):
        """
        Builds the design matrix (B) and block-diagonal penalty matrix (S)
        for the four additive components: f(lat), f(lon), f(time), f(lag).
        """
        # 1) B-splines for lat
        self.B_lat, _ = b_spline_basis_1d(lat,
                                          n_splines=self.n_splines_space,
                                          degree=self.degree)
        self.S_lat = second_difference_penalty(self.n_splines_space)

        # 2) B-splines for lon
        self.B_lon, _ = b_spline_basis_1d(lon,
                                          n_splines=self.n_splines_space,
                                          degree=self.degree)
        self.S_lon = second_difference_penalty(self.n_splines_space)

        # 3) B-splines for time
        self.B_time, _ = b_spline_basis_1d(t,
                                           n_splines=self.n_splines_time,
                                           degree=self.degree)
        self.S_time = second_difference_penalty(self.n_splines_time)

        # 4) B-splines for lag
        self.B_lag, _ = b_spline_basis_1d(lag,
                                          n_splines=self.n_splines_lag,
                                          degree=self.degree)
        self.S_lag = second_difference_penalty(self.n_splines_lag)

        # Combines everything into an additive model:
        # design_matrix = [1, B_lat, B_lon, B_time, B_lag]
        intercept = np.ones((len(lat), 1))
        self.design_matrix = np.hstack([
            intercept,
            self.B_lat, self.B_lon, self.B_time, self.B_lag
        ])

        # Combines penalty matrices in block-diagonal form.
        # The intercept column remains unpenalized.
        n_int = 1
        block_dim = (n_int
                     + self.n_splines_space
                     + self.n_splines_space
                     + self.n_splines_time
                     + self.n_splines_lag)

        S = np.zeros((block_dim, block_dim))
        start_lat = n_int
        end_lat   = n_int + self.n_splines_space
        S[start_lat:end_lat, start_lat:end_lat] = self.S_lat

        start_lon = end_lat
        end_lon   = start_lon + self.n_splines_space
        S[start_lon:end_lon, start_lon:end_lon] = self.S_lon

        start_time = end_lon
        end_time   = start_time + self.n_splines_time
        S[start_time:end_time, start_time:end_time] = self.S_time

        start_lag = end_time
        end_lag   = start_lag + self.n_splines_lag
        S[start_lag:end_lag, start_lag:end_lag] = self.S_lag

        self.penalty_matrix = S

    def _penalized_coefs(self, lam, X, y, S):
        """
        Solves for coefficients (beta) in the penalized least squares system:
           beta_hat = (X^T X + lam * S)^(-1) X^T y
        """
        XtX = X.T @ X
        lhs = XtX + lam * S
        rhs = X.T @ y
        beta_hat = solve(lhs, rhs, assume_a='sym')
        return beta_hat

    def _reml_objective(self, log_lam, X, y, S):
        """
        Negative log-REML (approx) for a single smoothing parameter lam = exp(log_lam).

        The simplified approach:
          1) beta_hat = (X^T X + lam S)^(-1) X^T y
          2) residuals = y - X beta_hat
          3) edf = trace(A), A = X (X^T X + lam S)^(-1) X^T
          4) sigma^2 = sum of residuals^2 / (n - edf)
          5) -2 * log(REML) ~ (n - edf) * log(sigma^2) + log|X^T X + lam S|
             (plus constants that do not affect lam selection)

        The negative is returned, so minimization finds the best lam.

        Reference: "Generalized Additive Models: an Introduction with R" (Simon Wood).
        This implementation is a condensed demonstration.
        """
        lam = np.exp(log_lam)
        n = X.shape[0]

        # Solve for beta
        beta_hat = self._penalized_coefs(lam, X, y, S)
        residuals = y - X @ beta_hat

        # Hat matrix A
        # A = X (X^T X + lam S)^(-1) X^T
        # Only the trace is needed for edf.
        lhs = X.T @ X + lam * S
        try:
            M_inv = np.linalg.inv(lhs)
        except np.linalg.LinAlgError:
            return 1e12  # Large penalty if singular

        A = X @ (M_inv @ X.T)
        edf = np.trace(A)

        # Estimate sigma^2
        rss = np.dot(residuals, residuals)
        df_resid = n - edf
        if df_resid <= 0:
            return 1e12
        sigma2 = rss / df_resid

        # log|lhs| = log determinant of (X^T X + lam S)
        logdet_lhs = np.log(np.linalg.det(lhs) + 1e-300)  # Avoid log(0)

        # -2 log(REML) ~ (n - edf)*log(sigma^2) + log|lhs|
        neg2_log_reml = (n - edf) * np.log(sigma2) + logdet_lhs

        return neg2_log_reml / 2.0  # Halved for convenience

    def fit(self,
            lat: Union[np.ndarray, list],
            lon: Union[np.ndarray, list],
            time: Union[np.ndarray, list],
            lag: Union[np.ndarray, list],
            y: Union[np.ndarray, list]) -> None:
        """
        Fits the STELAR GAM to data.

        Parameters
        ----------
        lat : array-like
            Latitude values for each observation.
        lon : array-like
            Longitude values for each observation.
        time : array-like
            Time indices for each observation.
        lag : array-like
            Lag values for each observation (e.g., number of time steps behind).
        y : array-like
            The target/response variable.
        """
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        t   = np.asarray(time)
        lag = np.asarray(lag)
        y   = np.asarray(y)
        self.nobs_ = len(y)

        # Build design matrix and penalty
        self._build_design_and_penalties(lat, lon, t, lag)
        X = self.design_matrix
        S = self.penalty_matrix

        # 1) Optimize log(lambda) via REML
        init_log_lam = np.log(1.0)  # Start at lam=1
        opt_res = minimize(
            fun=self._reml_objective,
            x0=init_log_lam,
            args=(X, y, S),
            method='L-BFGS-B',
            options={'maxiter': self.max_iter, 'ftol': self.tol}
        )

        if self.verbose:
            print("Optimization success:", opt_res.success)
            print("Message:", opt_res.message)
            print("Final negative REML:", opt_res.fun)
            print("Estimated log(lambda):", opt_res.x)

        self.lambda_ = float(np.exp(opt_res.x))
        # 2) Solve for beta with chosen lambda
        self.coef_ = self._penalized_coefs(self.lambda_, X, y, S)

        # 3) Estimate final residual variance
        residuals = y - X @ self.coef_
        lhs = X.T @ X + self.lambda_ * S
        M_inv = np.linalg.inv(lhs)
        A = X @ (M_inv @ X.T)
        edf = np.trace(A)
        df_resid = self.nobs_ - edf
        self.sigma2_ = np.sum(residuals**2) / max(df_resid, 1.0)
        self.ranks_ = edf  # Approximate rank

    def predict(self,
                lat: Union[np.ndarray, list],
                lon: Union[np.ndarray, list],
                time: Union[np.ndarray, list],
                lag: Union[np.ndarray, list]) -> np.ndarray:
        """
        Generates predictions from the fitted STELAR GAM model.

        Parameters
        ----------
        lat : array-like
        lon : array-like
        time : array-like
        lag : array-like

        Returns
        -------
        yhat : ndarray
            Predicted values at the provided coordinates/indices.
        """
        if self.coef_ is None:
            raise RuntimeError("Model is not fit yet.")

        # Build design matrix for new data
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        t   = np.asarray(time)
        lag = np.asarray(lag)

        # In a production system, the same knots used at training would be stored
        # and reused here. This demonstration simply calls the same routine.
        B_lat, _ = b_spline_basis_1d(lat, self.n_splines_space, degree=self.degree)
        B_lon, _ = b_spline_basis_1d(lon, self.n_splines_space, degree=self.degree)
        B_time, _ = b_spline_basis_1d(t, self.n_splines_time, degree=self.degree)
        B_lag, _ = b_spline_basis_1d(lag, self.n_splines_lag, degree=self.degree)

        intercept = np.ones((len(lat), 1))
        Xnew = np.hstack([intercept, B_lat, B_lon, B_time, B_lag])

        # Predict
        yhat = Xnew @ self.coef_
        return yhat

# ---------------------------------------------------------------------
# 3. EXAMPLE USAGE (if run as a script)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generates a small synthetic dataset
    np.random.seed(123)

    n = 300
    lat_data = np.random.uniform(-10, 10, n)
    lon_data = np.random.uniform(30, 50, n)
    time_data = np.linspace(0, 365, n)  # Example: day of year
    lag_data = np.random.uniform(0, 5, n)

    # True underlying function (arbitrary example)
    f_lat = 0.05 * lat_data**2
    f_lon = 0.1 * np.sin(lon_data)
    f_time = 2.0 * np.cos((2*np.pi/365) * time_data)
    f_lag = -1.0 * np.log(1 + lag_data)

    # Combine
    y_true = 5.0 + f_lat + f_lon + f_time + f_lag
    # Add noise
    noise = np.random.normal(0, 1.0, n)
    y_obs = y_true + noise

    # Fit the STELAR GAM
    model = StelarGAM(
        n_splines_space=8,
        n_splines_time=8,
        n_splines_lag=8,
        degree=3,
        verbose=True
    )
    model.fit(lat_data, lon_data, time_data, lag_data, y_obs)

    print("\n--- Model Results ---")
    print(f"Lambda: {model.lambda_:.4f}")
    print(f"Sigma^2: {model.sigma2_:.4f}")
    print(f"EDF (approx): {model.ranks_:.2f}")

    # Predict
    y_hat = model.predict(lat_data, lon_data, time_data, lag_data)

    # Quick check: R^2
    ss_res = np.sum((y_obs - y_hat)**2)
    ss_tot = np.sum((y_obs - np.mean(y_obs))**2)
    r2 = 1 - ss_res/ss_tot
    print(f"R^2 on training set: {r2:.3f}")

    # Plot a quick comparison
    plt.figure(figsize=(6, 4))
    plt.scatter(y_obs, y_hat, alpha=0.6)
    plt.plot([y_obs.min(), y_obs.max()],
             [y_obs.min(), y_obs.max()],
             color='red', lw=2)
    plt.xlabel("Observed Y")
    plt.ylabel("Fitted Y")
    plt.title(f"STELAR GAM Fit (R^2={r2:.3f})")
    plt.tight_layout()
    plt.show()
