# Standard modules
from time import process_time_ns

# External modules
import numpy as np

# Internal modules
from cosmotd.utils import laplacian2D_iterative


if __name__ == "__main__":
    N = 1000
    num_trials = 1000
    count = 0
    arr = np.zeros(shape=(N, N))
    for _ in range(num_trials):
        start = process_time_ns()
        laplacian2D_iterative(arr, dx=0.1)
        count += process_time_ns() - start
    mean_time = count/num_trials
    print(f"The function took {mean_time:.2f} nanoseconds of process time.")
