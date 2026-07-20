import pandas as pd
import numpy as np

columns = ['trade_id','price','quantity','first_trade_id','last_trade_id','timestamp','is-buyer-maker','is_best_match']
btc_data = pd.read_csv('binance_data/BTCUSDT/BTCUSDT-aggTrades-2026-07-01.csv',header=None,names=columns)
print(btc_data.head())
times_btc = btc_data['timestamp']
print(times_btc.shape)

def neg_log_likelihood(params,times,T):
    mu = params[0]
    alpha = params[1]
    gamma = params[2]
    temp_A = 0
    running_log_sum = np.log(mu)
    for i in range(1,len(times)):
        temp_A = np.exp(-gamma * (times[i] - times[i-1])) * ( 1 + temp_A )
        running_log_sum += np.log(mu + alpha * temp_A)
    compsum = 0
    for i in range(0,len(times)):
        compsum += ( 1 - np.exp(-gamma * (T-times[i])))
    compensator =  mu * T + (alpha/gamma) * compsum
    return -(running_log_sum - compensator)

def neg_ll_and_grad(params, times, T):
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
        dt = times[i] - times[i-1]
        decay = np.exp(-gamma * dt)

        temp_A = decay * (1 + temp_A)
        temp_L = decay * (temp_L + times[i-1])

        lam = mu + alpha * temp_A
        running_log_sum += np.log(lam)

        grad_mu += 1.0 / lam
        grad_alpha += temp_A / lam
        grad_gamma += alpha * (-times[i] * temp_A + temp_L) / lam

    grad_mu -= T

    compsum1 = 0.0   # sum (1 - exp(-gamma*(T-ti)))
    compsum2 = 0.0   # sum (T-ti) * exp(-gamma*(T-ti))
    for i in range(N):
        tau = T - times[i]
        e = np.exp(-gamma * tau)
        compsum1 += (1 - e)
        compsum2 += tau * e

    compensator = mu * T + (alpha / gamma) * compsum1

    grad_alpha -= compsum1 / gamma
    grad_gamma -= (alpha / gamma) * compsum2
    grad_gamma += (alpha / gamma**2) * compsum1

    neg_ll = -(running_log_sum - compensator)
    grad = -np.array([grad_mu, grad_alpha, grad_gamma])

    return neg_ll, grad
'''
duplicate timestamps were skewing gamma values
idea was that we enforce a minimum difference between 2 timestamps by pushing timestamps forwards until the condition is satisfied across whole series. 
But gamma would just scale proportionally with any reasonably small diff so it was not really fixing distortion.
Instead, duplicates have been removed entirely. 



def anticlash(times):
    temp_times = times.copy()
    clash = True
    diff = 1000
    while clash:
        for i in range(1,len(temp_times)):
            if temp_times[i] <= temp_times[i-1] + diff:
                temp_times[i] += diff - (temp_times[i] - temp_times[i-1])
        clash = not (len(list(set(temp_times))) == len(temp_times))
    return temp_times
'''
initial_guess = [0.0001, 0.00001, 0.0001]

times = times_btc.to_numpy()
times = times-min(times)
times = times
print(len(set(times)))

# 54% of data is duplicate i.e set is 46% of original size
times = sorted(list(set(times)))

print(times)
T = times[-1]


from scipy.optimize import minimize

result = minimize(
    fun=neg_log_likelihood,   # function to minimize
    x0=initial_guess,          # starting point, e.g. [mu0, alpha0, gamma0]
    args=(times, T),     # extra fixed arguments passed to fun
    method="L-BFGS-B",         # optimization algorithm
    bounds=[(1e-9, None), (1e-9, None), (1e-9, None)],  # keep params positive
      # optional: analytic gradient
)

#mu_hat, alpha_hat, gamma_hat = result.x
#print(result.x)

result = minimize(
    fun=neg_ll_and_grad,
    x0=initial_guess,
    args=(times, T),
    jac=True,
    method='L-BFGS-B',
    bounds=[(1e-8, None)] * 3,
)
print(result.x)

