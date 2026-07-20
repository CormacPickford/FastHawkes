"""
GPU (CuPy RawKernel) batched evaluation of the K=1 univariate Hawkes
negative log-likelihood and analytic gradient, across many symbols at once.

Design: ONE CUDA THREAD PER SYMBOL. Each thread runs the exact same
sequential O(N) recursion as the CPU reference (mle.py's neg_ll_and_grad),
just for its own symbol's event sequence. This matches the "parallel across
symbols, sequential within a symbol" architecture decided on earlier --
within a single symbol the likelihood is inherently sequential (each event's
state depends on the previous one), so there is no attempt to parallelize
across time within one symbol.

Data layout (ragged array, CSR-style):
    times_flat : all symbols' (deduped, zeroed) timestamps concatenated
    offsets    : offsets[s], offsets[s+1] give the [start, end) slice in
                 times_flat belonging to symbol s. Length = num_symbols + 1.

KNOWN LIMITATION (flagged, not fixed here): symbols have very different N
(event counts). Threads in the same warp finish at very different times
(warp divergence / load imbalance), since a thread handling a 200k-event
liquid pair does far more loop iterations than one handling a 500-event
illiquid pair. This is a real performance issue to revisit later (e.g.
sorting symbols by N before assigning to warps to reduce the spread within
a warp) -- correctness first, this file does not attempt that optimization.

IMPORTANT: I do not have a GPU or CuPy available in the environment I'm
writing this in, so the actual .cu kernel below has NOT been compiled or
run on real hardware. What IS verified below (see _emulate_kernel_cpu and
the test at the bottom) is that the *batching/indexing logic* -- the ragged
array offsets, the per-symbol loop bounds -- exactly reproduces the scalar
CPU reference (mle.py's neg_ll_and_grad) on synthetic multi-symbol data.
The CUDA kernel source mirrors that emulator line-for-line, so the main
remaining risk is CUDA-specific (compilation, extern "C" mangling, dtype
mismatches) rather than a logic error -- but you still need to run
validate_against_cpu() on your actual GPU box before trusting it.
"""

import numpy as np


# ---------------------------------------------------------------------------
# CPU reference (identical to mle.py's neg_ll_and_grad) -- kept here so this
# file can validate itself without importing mle.py.
# ---------------------------------------------------------------------------

def neg_ll_and_grad_cpu(params, times, T):
    mu, alpha, gamma = params
    N = len(times)

    temp_A = 0.0
    temp_L = 0.0

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

    compsum1 = 0.0
    compsum2 = 0.0
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


# ---------------------------------------------------------------------------
# Batch construction: turn {symbol: times_array} into the ragged layout
# ---------------------------------------------------------------------------

def build_batch(times_dict):
    """
    times_dict: dict[str, np.ndarray[int64]] -- per-symbol deduped, sorted,
                zeroed timestamps (as produced by mle.py's load_symbol_times).

    Returns:
        symbols   : list[str], order matches all other arrays
        times_flat: np.ndarray[int64], all symbols concatenated
        offsets   : np.ndarray[int32], length len(symbols)+1
        T_arr     : np.ndarray[float64], per-symbol T = times[-1]
    """
    symbols = list(times_dict.keys())
    lengths = [len(times_dict[s]) for s in symbols]

    offsets = np.zeros(len(symbols) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(lengths)

    times_flat = np.concatenate([times_dict[s] for s in symbols]).astype(np.int64)
    T_arr = np.array([times_dict[s][-1] for s in symbols], dtype=np.float64)

    return symbols, times_flat, offsets, T_arr


# ---------------------------------------------------------------------------
# Pure-Python emulator of the CUDA kernel -- same logic, no GPU required.
# This is what gets validated against neg_ll_and_grad_cpu below. The actual
# .cu kernel source (further down) mirrors this function line-for-line.
# ---------------------------------------------------------------------------

def _emulate_kernel_cpu(times_flat, offsets, T_arr, mu_arr, alpha_arr, gamma_arr):
    num_symbols = len(mu_arr)
    negll_out = np.zeros(num_symbols)
    grad_mu_out = np.zeros(num_symbols)
    grad_alpha_out = np.zeros(num_symbols)
    grad_gamma_out = np.zeros(num_symbols)

    for s in range(num_symbols):
        start, end = offsets[s], offsets[s + 1]
        N = end - start
        if N < 2:
            continue

        mu, alpha, gamma = mu_arr[s], alpha_arr[s], gamma_arr[s]
        T = T_arr[s]

        temp_A = 0.0
        temp_L = 0.0
        running_log_sum = np.log(mu)
        grad_mu = 1.0 / mu
        grad_alpha = 0.0
        grad_gamma = 0.0

        t_prev = times_flat[start]
        for i in range(1, N):
            t_i = times_flat[start + i]
            dt = float(t_i - t_prev)
            decay = np.exp(-gamma * dt)

            temp_A = decay * (1.0 + temp_A)
            temp_L = decay * (temp_L + float(t_prev))

            lam = mu + alpha * temp_A
            running_log_sum += np.log(lam)

            grad_mu += 1.0 / lam
            grad_alpha += temp_A / lam
            grad_gamma += alpha * (-float(t_i) * temp_A + temp_L) / lam

            t_prev = t_i

        grad_mu -= T

        compsum1 = 0.0
        compsum2 = 0.0
        for i in range(N):
            tau = T - float(times_flat[start + i])
            e = np.exp(-gamma * tau)
            compsum1 += (1.0 - e)
            compsum2 += tau * e

        compensator = mu * T + (alpha / gamma) * compsum1

        grad_alpha -= compsum1 / gamma
        grad_gamma -= (alpha / gamma) * compsum2
        grad_gamma += (alpha / gamma ** 2) * compsum1

        negll_out[s] = -(running_log_sum - compensator)
        grad_mu_out[s] = -grad_mu
        grad_alpha_out[s] = -grad_alpha
        grad_gamma_out[s] = -grad_gamma

    return negll_out, grad_mu_out, grad_alpha_out, grad_gamma_out


# ---------------------------------------------------------------------------
# Actual CUDA kernel source. Mirrors _emulate_kernel_cpu above line-for-line.
# NOT compiled/run in this environment -- run validate_against_cpu() on your
# GPU box to confirm this actually matches once compiled.
# ---------------------------------------------------------------------------

KERNEL_SRC = r'''
extern "C" __global__
void hawkes_k1_negll_grad(
    const long long* times_flat,
    const int* offsets,
    const double* T_arr,
    const double* mu_arr,
    const double* alpha_arr,
    const double* gamma_arr,
    double* negll_out,
    double* grad_mu_out,
    double* grad_alpha_out,
    double* grad_gamma_out,
    int num_symbols)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_symbols) return;

    int start = offsets[s];
    int end   = offsets[s + 1];
    int N = end - start;

    if (N < 2) {
        negll_out[s] = 0.0;
        grad_mu_out[s] = 0.0;
        grad_alpha_out[s] = 0.0;
        grad_gamma_out[s] = 0.0;
        return;
    }

    double mu = mu_arr[s];
    double alpha = alpha_arr[s];
    double gamma = gamma_arr[s];
    double T = T_arr[s];

    double temp_A = 0.0;
    double temp_L = 0.0;

    double running_log_sum = log(mu);
    double grad_mu = 1.0 / mu;
    double grad_alpha = 0.0;
    double grad_gamma = 0.0;

    long long t_prev = times_flat[start];

    for (int i = 1; i < N; i++) {
        long long t_i = times_flat[start + i];
        double dt = (double)(t_i - t_prev);
        double decay = exp(-gamma * dt);

        temp_A = decay * (1.0 + temp_A);
        temp_L = decay * (temp_L + (double)t_prev);

        double lam = mu + alpha * temp_A;
        running_log_sum += log(lam);

        grad_mu += 1.0 / lam;
        grad_alpha += temp_A / lam;
        grad_gamma += alpha * (-(double)t_i * temp_A + temp_L) / lam;

        t_prev = t_i;
    }

    grad_mu -= T;

    double compsum1 = 0.0;
    double compsum2 = 0.0;
    for (int i = 0; i < N; i++) {
        double tau = T - (double)(times_flat[start + i]);
        double e = exp(-gamma * tau);
        compsum1 += (1.0 - e);
        compsum2 += tau * e;
    }

    double compensator = mu * T + (alpha / gamma) * compsum1;

    grad_alpha -= compsum1 / gamma;
    grad_gamma -= (alpha / gamma) * compsum2;
    grad_gamma += (alpha / (gamma * gamma)) * compsum1;

    double neg_ll = -(running_log_sum - compensator);

    negll_out[s]     = neg_ll;
    grad_mu_out[s]    = -grad_mu;
    grad_alpha_out[s] = -grad_alpha;
    grad_gamma_out[s] = -grad_gamma;
}
'''


# ---------------------------------------------------------------------------
# GPU launch wrapper. Requires cupy + a real GPU -- not run in this sandbox.
# ---------------------------------------------------------------------------

def run_gpu_batch(times_flat, offsets, T_arr, mu_arr, alpha_arr, gamma_arr,
                   block_size=128):
    import cupy as cp  # imported here so this module still loads without cupy

    num_symbols = len(mu_arr)

    times_gpu = cp.asarray(times_flat, dtype=cp.int64)
    offsets_gpu = cp.asarray(offsets, dtype=cp.int32)
    T_gpu = cp.asarray(T_arr, dtype=cp.float64)
    mu_gpu = cp.asarray(mu_arr, dtype=cp.float64)
    alpha_gpu = cp.asarray(alpha_arr, dtype=cp.float64)
    gamma_gpu = cp.asarray(gamma_arr, dtype=cp.float64)

    negll_gpu = cp.zeros(num_symbols, dtype=cp.float64)
    gmu_gpu = cp.zeros(num_symbols, dtype=cp.float64)
    galpha_gpu = cp.zeros(num_symbols, dtype=cp.float64)
    ggamma_gpu = cp.zeros(num_symbols, dtype=cp.float64)

    kernel = cp.RawKernel(KERNEL_SRC, "hawkes_k1_negll_grad")
    grid_size = (num_symbols + block_size - 1) // block_size

    kernel(
        (grid_size,), (block_size,),
        (times_gpu, offsets_gpu, T_gpu, mu_gpu, alpha_gpu, gamma_gpu,
         negll_gpu, gmu_gpu, galpha_gpu, ggamma_gpu, np.int32(num_symbols)),
    )
    cp.cuda.Stream.null.synchronize()

    return (cp.asnumpy(negll_gpu), cp.asnumpy(gmu_gpu),
            cp.asnumpy(galpha_gpu), cp.asnumpy(ggamma_gpu))


# ---------------------------------------------------------------------------
# Validation entry point -- run this on your GPU box after compiling.
# Compares GPU kernel output against the CPU scalar reference, per symbol.
# ---------------------------------------------------------------------------

def validate_against_cpu(times_dict, mu_arr, alpha_arr, gamma_arr, rtol=1e-6):
    symbols, times_flat, offsets, T_arr = build_batch(times_dict)

    gpu_negll, gpu_gmu, gpu_galpha, gpu_ggamma = run_gpu_batch(
        times_flat, offsets, T_arr, mu_arr, alpha_arr, gamma_arr
    )

    max_err = 0.0
    for s, sym in enumerate(symbols):
        times = times_dict[sym]
        if len(times) < 2:
            continue
        params = [mu_arr[s], alpha_arr[s], gamma_arr[s]]
        cpu_negll, cpu_grad = neg_ll_and_grad_cpu(params, times, T_arr[s])

        err = max(
            abs(gpu_negll[s] - cpu_negll) / max(abs(cpu_negll), 1e-12),
            abs(gpu_gmu[s] - cpu_grad[0]) / max(abs(cpu_grad[0]), 1e-12),
            abs(gpu_galpha[s] - cpu_grad[1]) / max(abs(cpu_grad[1]), 1e-12),
            abs(gpu_ggamma[s] - cpu_grad[2]) / max(abs(cpu_grad[2]), 1e-12),
        )
        max_err = max(max_err, err)
        status = "OK" if err < rtol else "MISMATCH"
        print(f"  {sym}: rel_err={err:.2e} [{status}]")

    print(f"\nMax relative error across all symbols: {max_err:.2e}")
    if max_err < rtol:
        print("GPU kernel matches CPU reference.")
    else:
        print("GPU kernel DOES NOT match CPU reference -- do not trust it yet.")


# ---------------------------------------------------------------------------
# Newton optimizer -- CPU reference. This is the ground truth the in-kernel
# Newton loop (NEWTON_KERNEL_SRC below) must match to ~1e-8 once written.
# Uses neg_ll_and_grad_cpu for the gradient (already validated above) and a
# central finite-difference Hessian, same scheme the kernel loop should use.
# ---------------------------------------------------------------------------

def newton_cpu(times, T, mu0, alpha0, gamma0,
               max_iters=50, tol=1e-8, fd_eps=1e-6, floor=1e-10):
    theta = np.array([mu0, alpha0, gamma0], dtype=np.float64)

    iters_used = 0
    converged = False

    for it in range(max_iters):
        iters_used = it + 1
        negll, grad = neg_ll_and_grad_cpu(theta, times, T)

        H = np.zeros((3, 3))
        for j in range(3):
            step = fd_eps * max(abs(theta[j]), 1.0)
            theta_p = theta.copy(); theta_p[j] += step
            theta_m = theta.copy(); theta_m[j] -= step
            _, grad_p = neg_ll_and_grad_cpu(theta_p, times, T)
            _, grad_m = neg_ll_and_grad_cpu(theta_m, times, T)
            H[:, j] = (grad_p - grad_m) / (2.0 * step)
        H = 0.5 * (H + H.T)  # symmetrize away FD noise

        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break

        theta_new = np.maximum(theta - delta, floor)

        if np.linalg.norm(theta_new - theta) < tol:
            theta = theta_new
            converged = True
            break
        theta = theta_new

    negll, grad = neg_ll_and_grad_cpu(theta, times, T)
    return theta, negll, iters_used, converged


# ---------------------------------------------------------------------------
# Newton optimizer -- CUDA kernel. Mirrors run_gpu_batch's launch pattern.
# The loop body is a STUB: fill in the gradient recursion (copy the body of
# hawkes_k1_negll_grad above) plus a finite-difference Hessian, 3x3 solve,
# and update step that matches newton_cpu line-for-line. Validate against
# newton_cpu with validate_newton_against_cpu() before trusting it.
# ---------------------------------------------------------------------------

NEWTON_KERNEL_SRC = r'''
extern "C" __global__
void hawkes_k1_newton(
    const long long* times_flat,
    const int* offsets,
    const double* T_arr,
    double* mu_arr,        // in: initial guess; out: fitted
    double* alpha_arr,
    double* gamma_arr,
    double* negll_out,
    int* iters_out,
    int* converged_out,
    int num_symbols,
    int max_iters,
    double tol,
    double fd_eps,
    double floor_val)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_symbols) return;

    int start = offsets[s];
    int end   = offsets[s + 1];
    int N = end - start;
    if (N < 2) {
        negll_out[s] = 0.0;
        iters_out[s] = 0;
        converged_out[s] = 0;
        return;
    }

    double mu = mu_arr[s], alpha = alpha_arr[s], gamma = gamma_arr[s];
    double T = T_arr[s];

    // =====================================================================
    // TODO (you write this): Newton loop, up to max_iters, matching
    // newton_cpu() above:
    //   1. evaluate grad(mu, alpha, gamma) -- inline the same recursion as
    //      hawkes_k1_negll_grad's loop body over times_flat[start:end]
    //   2. central-difference Hessian: perturb each of mu/alpha/gamma by
    //      +-fd_eps*max(|theta_j|,1), re-evaluate grad, H[:,j] = (g+ - g-)/(2*step)
    //      then symmetrize H
    //   3. solve H * delta = grad (3x3 -- Cramer's rule or explicit inverse,
    //      no cuBLAS/cuSOLVER call from inside a kernel)
    //   4. theta_new = max(theta - delta, floor_val) elementwise
    //   5. if ||theta_new - theta|| < tol: converged_out[s]=1; break
    //      else theta = theta_new
    // Write negll_out[s], iters_out[s], converged_out[s] before returning.
    // =====================================================================

    mu_arr[s] = mu;
    alpha_arr[s] = alpha;
    gamma_arr[s] = gamma;
}
'''


def run_gpu_newton(times_flat, offsets, T_arr, mu_arr, alpha_arr, gamma_arr,
                    max_iters=50, tol=1e-8, fd_eps=1e-6, floor_val=1e-10,
                    block_size=128):
    """Fits mu/alpha/gamma per symbol in place via Newton's method on GPU.

    mu_arr/alpha_arr/gamma_arr are the initial guesses on the way in;
    returns the fitted values (also mutates copies pulled back from GPU).
    """
    import cupy as cp  # imported here so this module still loads without cupy

    num_symbols = len(mu_arr)

    times_gpu = cp.asarray(times_flat, dtype=cp.int64)
    offsets_gpu = cp.asarray(offsets, dtype=cp.int32)
    T_gpu = cp.asarray(T_arr, dtype=cp.float64)
    mu_gpu = cp.asarray(mu_arr, dtype=cp.float64)
    alpha_gpu = cp.asarray(alpha_arr, dtype=cp.float64)
    gamma_gpu = cp.asarray(gamma_arr, dtype=cp.float64)

    negll_gpu = cp.zeros(num_symbols, dtype=cp.float64)
    iters_gpu = cp.zeros(num_symbols, dtype=cp.int32)
    converged_gpu = cp.zeros(num_symbols, dtype=cp.int32)

    kernel = cp.RawKernel(NEWTON_KERNEL_SRC, "hawkes_k1_newton")
    grid_size = (num_symbols + block_size - 1) // block_size

    kernel(
        (grid_size,), (block_size,),
        (times_gpu, offsets_gpu, T_gpu, mu_gpu, alpha_gpu, gamma_gpu,
         negll_gpu, iters_gpu, converged_gpu,
         np.int32(num_symbols), np.int32(max_iters),
         np.float64(tol), np.float64(fd_eps), np.float64(floor_val)),
    )
    cp.cuda.Stream.null.synchronize()

    return (cp.asnumpy(mu_gpu), cp.asnumpy(alpha_gpu), cp.asnumpy(gamma_gpu),
            cp.asnumpy(negll_gpu), cp.asnumpy(iters_gpu), cp.asnumpy(converged_gpu))


def validate_newton_against_cpu(times_dict, mu0_arr, alpha0_arr, gamma0_arr,
                                 max_iters=50, tol=1e-8, rtol=1e-6):
    """Compares GPU Newton fit against newton_cpu, per symbol. Run on your
    GPU box once NEWTON_KERNEL_SRC's loop body is filled in."""
    symbols, times_flat, offsets, T_arr = build_batch(times_dict)

    gpu_mu, gpu_alpha, gpu_gamma, gpu_negll, gpu_iters, gpu_conv = run_gpu_newton(
        times_flat, offsets, T_arr, mu0_arr, alpha0_arr, gamma0_arr,
        max_iters=max_iters, tol=tol,
    )

    max_err = 0.0
    for s, sym in enumerate(symbols):
        times = times_dict[sym]
        if len(times) < 2:
            continue
        cpu_theta, cpu_negll, cpu_iters, cpu_conv = newton_cpu(
            times, T_arr[s], mu0_arr[s], alpha0_arr[s], gamma0_arr[s],
            max_iters=max_iters, tol=tol,
        )
        gpu_theta = np.array([gpu_mu[s], gpu_alpha[s], gpu_gamma[s]])
        err = np.max(np.abs(gpu_theta - cpu_theta) / np.maximum(np.abs(cpu_theta), 1e-12))
        max_err = max(max_err, err)
        status = "OK" if err < rtol else "MISMATCH"
        print(f"  {sym}: rel_err={err:.2e} gpu_iters={gpu_iters[s]} "
              f"cpu_iters={cpu_iters} [{status}]")

    print(f"\nMax relative error across all symbols: {max_err:.2e}")
    if max_err < rtol:
        print("GPU Newton kernel matches CPU reference.")
    else:
        print("GPU Newton kernel DOES NOT match CPU reference -- do not trust it yet.")


# ---------------------------------------------------------------------------
# Self-test (no GPU/cupy required): validates the batching/indexing logic
# by comparing _emulate_kernel_cpu against neg_ll_and_grad_cpu, looped per
# symbol, on synthetic multi-symbol data with DIFFERENT lengths per symbol
# (this specifically exercises the offsets/ragged-array boundary logic).
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    times_dict = {}
    for sym, n in [("AAAUSDT", 300), ("BBBUSDT", 1200), ("CCCUSDT", 50)]:
        t = np.sort(rng.exponential(scale=150, size=n).cumsum().astype(np.int64))
        t = np.unique(t)  # dedupe, matching real pipeline
        times_dict[sym] = t

    symbols, times_flat, offsets, T_arr = build_batch(times_dict)

    mu_arr = np.full(len(symbols), 0.0005)
    alpha_arr = np.full(len(symbols), 0.0001)
    gamma_arr = np.full(len(symbols), 0.002)

    emu_negll, emu_gmu, emu_galpha, emu_ggamma = _emulate_kernel_cpu(
        times_flat, offsets, T_arr, mu_arr, alpha_arr, gamma_arr
    )

    print("Comparing batched emulator against scalar CPU reference:\n")
    max_err = 0.0
    for s, sym in enumerate(symbols):
        params = [mu_arr[s], alpha_arr[s], gamma_arr[s]]
        cpu_negll, cpu_grad = neg_ll_and_grad_cpu(params, times_dict[sym], T_arr[s])

        err = max(
            abs(emu_negll[s] - cpu_negll) / max(abs(cpu_negll), 1e-12),
            abs(emu_gmu[s] - cpu_grad[0]) / max(abs(cpu_grad[0]), 1e-12),
            abs(emu_galpha[s] - cpu_grad[1]) / max(abs(cpu_grad[1]), 1e-12),
            abs(emu_ggamma[s] - cpu_grad[2]) / max(abs(cpu_grad[2]), 1e-12),
        )
        max_err = max(max_err, err)
        print(f"  {sym} (N={len(times_dict[sym])}): "
              f"neg_ll emu={emu_negll[s]:.6f} cpu={cpu_negll:.6f}  rel_err={err:.2e}")

    print(f"\nMax relative error: {max_err:.2e}")
    assert max_err < 1e-9, "Batching/indexing logic does not match CPU reference!"
    print("Batching and indexing logic verified against CPU reference.")

    # -----------------------------------------------------------------------
    # newton_cpu sanity check (no GPU required): confirms the Newton harness
    # itself converges to a gradient ~0 stationary point on each symbol,
    # before there's any kernel to compare it against.
    # -----------------------------------------------------------------------
    print("\nSanity-checking newton_cpu (CPU-only, no kernel involved yet):\n")
    print("  NOTE: this is undamped Newton -- no line search / trust region.")
    print("  It is NOT expected to converge from an arbitrary starting point;")
    print("  this just confirms the harness runs and matches its own math,")
    print("  not that plain Newton is a good optimizer here (it may not be).\n")
    for sym in symbols:
        times = times_dict[sym]
        theta, negll, iters, converged = newton_cpu(
            times, times[-1], mu0=0.0005, alpha0=0.0001, gamma0=0.002
        )
        _, grad = neg_ll_and_grad_cpu(theta, times, times[-1])
        gnorm = np.linalg.norm(grad)
        status = "converged" if gnorm < 1e-4 else "DID NOT CONVERGE"
        print(f"  {sym} (N={len(times)}): iters={iters} flag={converged} [{status}] "
              f"theta={theta} |grad|={gnorm:.2e} negll={negll:.6f}")