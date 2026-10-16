import re
import matplotlib.pyplot as plt

# Parse the log: one dict per timed run (skipped / non-result lines ignored) 
rows = []
for line in open("output_log.o"):
    if "time=" not in line:
        continue
    kernel = line.split(",")[0]
    N = int(re.search(r"N=(\d+)", line).group(1))
    p = int(re.search(r", p=(\d+)", line).group(1))
    t = float(re.search(r"time=([\d.]+)", line).group(1))
    if kernel == "blockMM":
        bn, tn = (N // p) ** 2, p * p
    else:
        bn = int(re.search(r"bn=(\d+)", line).group(1))
        tn = int(re.search(r"tn=(\d+)", line).group(1))
    rows.append(dict(kernel=kernel, N=N, p=p, bn=bn, tn=tn, t=t))

kernels = ["naive1", "naive2", "blockMM"]
Ns = sorted(set(r["N"] for r in rows))

# Q1: scatter, every (N, p, bn, tn) combination vs time  
plt.figure(figsize=(16, 6))
for k in kernels:
    rs = [r for r in rows if r["kernel"] == k]
    labels = [f"{r['N']},{r['p']},{r['bn']},{r['tn']}" for r in rs]
    plt.scatter(labels, [r["t"] for r in rs], label=k)
plt.yscale("log")
plt.xticks(rotation=90, fontsize=6)
plt.xlabel("N, p, blockNum, threadNum")
plt.ylabel("time (ms, log)")
plt.title("Q1: execution time for every configuration")
plt.legend()
plt.tight_layout()
plt.savefig("q1_scatter.png", dpi=150)

# Q3: for fixed N, p, compare the two blockNum/threadNum splits (Naive-1/2 only) 
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, k in zip(axes, ["naive1", "naive2"]):
    rs = [r for r in rows if r["kernel"] == k and r["p"] > 1]
    few = [r for r in rs if r["tn"] > 32]     # few large blocks
    many = [r for r in rs if r["tn"] == 32]   # many small blocks (32 threads)
    ax.plot([f"N={r['N']},p={r['p']}" for r in few], [r["t"] for r in few], "o-", label="few large blocks (tn=256/1024)")
    ax.plot([f"N={r['N']},p={r['p']}" for r in many], [r["t"] for r in many], "s-", label="many small blocks (tn=32)")
    ax.set_yscale("log")
    ax.set_title(f"Q3: {k}")
    ax.set_ylabel("time (ms, log)")
    ax.tick_params(axis="x", rotation=90)
    ax.legend()
plt.tight_layout()
plt.savefig("q3_blocks_threads.png", dpi=150)

# Q2: best config per (N, p), x = p, one line per N, one subplot per kernel  
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, k in zip(axes, kernels):
    for N in Ns:
        rs = [r for r in rows if r["kernel"] == k and r["N"] == N]
        ps = sorted(set(r["p"] for r in rs))
        best = [min(r["t"] for r in rs if r["p"] == p) for p in ps]
        if ps:
            ax.plot(ps, best, "o-", label=f"N={N}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("p")
    ax.set_ylabel("best time (ms, log)")
    ax.set_title(f"Q2: {k}")
    ax.legend()
plt.tight_layout()
plt.savefig("q2_best_vs_p.png", dpi=150)

plt.show()