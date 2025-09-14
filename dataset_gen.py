# dataset_gen.py
import numpy as np
from tqdm import tqdm
from rf_generator import generate_example

def generate_dataset(out_file='dataset.npz', n_examples=1000, seed=0):
    rng = np.random.RandomState(seed)
    # generate one example to get shape
    x0, y0, f_s, t_s = generate_example(rng=rng)
    F, T = x0.shape
    X = np.zeros((n_examples, F, T), dtype=np.float32)
    Y = np.zeros((n_examples, 4), dtype=np.int64)
    freqs = f_s
    times = t_s
    for i in tqdm(range(n_examples), desc='Generating dataset'):
        # random presence flags (you can tune probabilities)
        include = rng.rand(4) < 0.9  # 90% chance each band present
        x, y, f_s, t_s = generate_example(include_5g=bool(include[0]),
                                          include_wifi=bool(include[1]),
                                          include_bt=bool(include[2]),
                                          include_zb=bool(include[3]),
                                          rng=np.random.RandomState(rng.randint(0,2**31-1)))
        X[i] = x
        Y[i] = y
    np.savez_compressed(out_file, X=X, Y=Y, freqs=freqs, times=times)
    print("Saved", out_file)

if __name__ == "__main__":
    generate_dataset(out_file='dataset.npz', n_examples=1000, seed=42)
