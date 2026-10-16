import numpy as np
import matplotlib.pyplot as plt

# Best time (ms) per N for each implementation, taken from output.o and output_tc.o.
# nan = no run (Naive-2 is skipped for N >= 512: its rows/cols don't fit in shared memory)
Ns      = [64, 128, 512, 1024, 4096, 16384]
tensor  = [0.006, 0.007, 0.059, 0.409, 24.806, 3490.314]
blockMM = [0.008, 0.010, 0.141, 1.112, 90.470, 5748.420]
naive1  = [0.008, 0.021, 1.599, 15.584, 1880.279, 129076.414]
naive2  = [0.014, 0.061, np.nan, np.nan, np.nan, np.nan]

x = np.arange(len(Ns))
w = 0.2
plt.bar(x - 1.5 * w, tensor,  w, label="Tensor core", color="#2a78d6")
plt.bar(x - 0.5 * w, blockMM, w, label="Block MM",    color="#eb6834")
plt.bar(x + 0.5 * w, naive1,  w, label="Naive-1",     color="#1baf7a")
plt.bar(x + 1.5 * w, naive2,  w, label="Naive-2 (skipped for N ≥ 512)", color="#eda100")

plt.yscale("log")   # times span 0.006 ms to 129,000 ms
plt.xticks(x, Ns)
plt.xlabel("N")
plt.ylabel("Execution time (ms, log scale)")
plt.title("Tensor core vs best CUDA core implementations (RTX 2080 Ti)")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("tensor_vs_cuda.png", dpi=150)
plt.show()