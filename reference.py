"""
Fit a univariate, K=1 exponential Hawkes process (MLE, analytic gradient)
to each of the top-N USDT pairs downloaded by data.py, and write the fitted
parameters (mu, alpha, gamma) for every symbol to a single CSV.

Assumes data.py has already been run and populated:
    binance_data/<SYMBOL>/<SYMBOL>-aggTrades-<DAY>.csv

Timestamps are left in raw Binance units (milliseconds) after zeroing to the
first event of the day -- NOT converted to seconds. This means fitted gamma
is in units of "per millisecond". Keep this in mind if you later compute a
half-life (ln(2)/gamma) or compare gamma across a version of this script
that does convert to seconds.
"""

import os
import glob
import csv

import numpy as np
import pandas as pd
from scipy.optimize import minimize

OUTPUT_DIR = "binance_data"
RESULTS_CSV = "hawkes_fit_results.csv"

COLUMNS = [
    "trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
    "timestamp", "is-buyer-maker", "is_best_match",
]

INITIAL_GUESS = [0.0001, 0.00001, 0.0001]
BOUNDS = [(1e-9, None), (1e-9, None), (1e-9, None)]


def neg_ll_and_grad(params, times, T):
    """K=1 univariate Hawkes negative log-likelihood and analytic gradient.

    lambda(t) = mu + alpha * sum_{j: t_j < t} exp(-gamma * (t - t_j))
    (unnormalized exponential kernel -- alpha/gamma appears in the
    compensator, matching the original mle.py convention).
    """
    mu, alpha, gamma = params
    N = len(times)

    temp_A = 0.0   # K(t_i): decayed event count
    temp_L = 0.0   # L(t_i): decayed, timestamp-weighted event sum

    lam0 = mu
    running_log_sum = np.log(lam0)
    grad_mu = 1.0 / lam0
    grad_alpha = 0.0
    grad_gamma = 0.0

    for i in range(1, N):
        dt = times[i] - times[i - 1]
        decay = np.exp(-gamma * dt)

        temp_A = decay * (1 + temp_A)
        temp_L = decay * (temp_L + times[i - 1])

        lam = mu + alpha * temp_A
        running_log_sum += np.log(lam)

        grad_mu += 1.0 / lam
        grad_alpha += temp_A / lam
        grad_gamma += alpha * (-times[i] * temp_A + temp_L) / lam

    grad_mu -= T

    compsum1 = 0.0  # sum (1 - exp(-gamma*(T-ti)))
    compsum2 = 0.0  # sum (T-ti) * exp(-gamma*(T-ti))
    for i in range(N):
        tau = T - times[i]
        e = np.exp(-gamma * tau)
        compsum1 += (1 - e)
        compsum2 += tau * e

    compensator = mu * T + (alpha / gamma) * compsum1

    grad_alpha -= compsum1 / gamma
    grad_gamma -= (alpha / gamma) * compsum2
    grad_gamma += (alpha / gamma ** 2) * compsum1

    neg_ll = -(running_log_sum - compensator)
    grad = -np.array([grad_mu, grad_alpha, grad_gamma])

    return neg_ll, grad


# Earlier attempt at fixing duplicate timestamps by nudging them apart --
# kept as a note, not used. Gamma just rescaled proportionally with any
# small fixed offset, so it didn't actually remove the distortion.
# Duplicates are dropped entirely instead (see dedupe below).


def load_symbol_times(symbol_dir):
    """Load and concatenate all daily aggTrades CSVs for one symbol,
    sort by timestamp, drop exact-duplicate timestamps, and zero to the
    first event.

    Returns (times, n_raw, n_after_dedup) or (None, 0, 0) if no data found.
    """
    csv_paths = sorted(glob.glob(os.path.join(symbol_dir, "*.csv")))
    if not csv_paths:
        return None, 0, 0

    frames = [pd.read_csv(p, header=None, names=COLUMNS) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)

    times = df["timestamp"].to_numpy()
    n_raw = len(times)

    times = np.sort(times)
    times = np.unique(times)  # sorted + dedupe in one step
    n_after_dedup = len(times)

    if n_after_dedup < 2:
        return None, n_raw, n_after_dedup

    times = times - times[0]
    return times, n_raw, n_after_dedup


def fit_symbol(times):
    T = times[-1]
    result = minimize(
        fun=neg_ll_and_grad,
        x0=INITIAL_GUESS,
        args=(times, T),
        jac=True,
        method="L-BFGS-B",
        bounds=BOUNDS,
    )
    return result, T


def main():
    symbol_dirs = sorted(
        d for d in glob.glob(os.path.join(OUTPUT_DIR, "*"))
        if os.path.isdir(d)
    )
    print(f"Found {len(symbol_dirs)} symbol directories under '{OUTPUT_DIR}/'.\n")

    rows = []
    for symbol_dir in symbol_dirs:
        symbol = os.path.basename(symbol_dir)

        times, n_raw, n_dedup = load_symbol_times(symbol_dir)
        if times is None:
            print(f"  {symbol}: skipped (no usable data, n_raw={n_raw})")
            rows.append({
                "symbol": symbol, "mu": None, "alpha": None, "gamma": None,
                "success": False, "message": "no usable data",
                "n_raw": n_raw, "n_after_dedup": n_dedup, "T": None, "neg_ll": None,
            })
            continue

        try:
            result, T = fit_symbol(times)
            mu_hat, alpha_hat, gamma_hat = result.x
            print(
                f"  {symbol}: mu={mu_hat:.6g} alpha={alpha_hat:.6g} "
                f"gamma={gamma_hat:.6g} success={result.success} "
                f"(n_raw={n_raw}, n_dedup={n_dedup})"
            )
            rows.append({
                "symbol": symbol, "mu": mu_hat, "alpha": alpha_hat, "gamma": gamma_hat,
                "success": bool(result.success), "message": result.message,
                "n_raw": n_raw, "n_after_dedup": n_dedup, "T": T, "neg_ll": result.fun,
            })
        except Exception as exc:
            print(f"  {symbol}: FAILED ({exc})")
            rows.append({
                "symbol": symbol, "mu": None, "alpha": None, "gamma": None,
                "success": False, "message": str(exc),
                "n_raw": n_raw, "n_after_dedup": n_dedup, "T": None, "neg_ll": None,
            })

    fieldnames = [
        "symbol", "mu", "alpha", "gamma", "success", "message",
        "n_raw", "n_after_dedup", "T", "neg_ll",
    ]
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_ok = sum(1 for r in rows if r["success"])
    print(f"\nDone. {n_ok}/{len(rows)} symbols fitted successfully.")
    print(f"Results written to '{RESULTS_CSV}'.")


if __name__ == "__main__":
    main()