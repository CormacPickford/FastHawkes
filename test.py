import cupy as cp
import numpy as np
import pandas as pd
'''
# Host: read CSV as before, keep using pandas
df = pd.read_csv("binance_data/BTCUSDT/BTCUSDT-aggTrades-2026-07-01.csv",
                  header=None, names=COLUMNS)
times_np = df["timestamp"].to_numpy(dtype=np.int64)
times_np = np.unique(times_np) - times_np.min()  # sort + dedupe, zero to start

# Move to GPU
times_gpu = cp.asarray(times_np)
'''
# Your actual CUDA C kernel, as a string (or read from a .cu file)
kernel_src = r'''
extern "C" __global__
void vector_add(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}
'''

vector_add = cp.RawKernel(kernel_src, "vector_add")

n = 1 << 20
a = cp.random.rand(n, dtype=cp.float32)
b = cp.random.rand(n, dtype=cp.float32)
out = cp.zeros(n, dtype=cp.float32)

block_size = 256
grid_size = (n + block_size - 1) // block_size
vector_add((grid_size,), (block_size,), (a, b, out, n))

result = cp.asnumpy(out)  # bring back to host as a NumPy array if needed
print(result)