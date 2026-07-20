import pandas as pd

data = pd.read_csv('hawkes_fit_results.csv')
print(data)
a_g = data[['alpha','gamma']]
arr = a_g.to_numpy()
branching_ratios = arr[:,0]/arr[:,1]
print(sorted(branching_ratios))