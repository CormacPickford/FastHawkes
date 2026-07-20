import numpy as np
from hawkes_gpu import validate_against_cpu

rng = np.random.default_rng(0)
times_dict = {}
for sym, n in [("AAAUSDT", 300), ("BBBUSDT", 1200), ("CCCUSDT", 50)]:
    t = np.sort(rng.exponential(scale=150, size=n).cumsum().astype(np.int64))
    t = np.unique(t)
    times_dict[sym] = t

mu_arr = np.full(3, 0.0005)
alpha_arr = np.full(3, 0.0001)
gamma_arr = np.full(3, 0.002)

validate_against_cpu(times_dict, mu_arr, alpha_arr, gamma_arr)