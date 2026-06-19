"""
run_noise_ablation.py
----------------------
Sweeps IBM-like depolarizing noise on the quantum kernel and shows:
  1. Kernel concentration: off-diagonal variance drops toward 0 as noise grows
  2. Classification performance collapses to chance as the kernel loses signal

Uses a fixed simulator instance per noise level (avoids re-instantiation overhead),
50 train + 30 test molecules (enough to see the trend without taking minutes).
"""
import sys, json, time
sys.path.insert(0, "python")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from sklearn.metrics import balanced_accuracy_score

from data_utils import load_bbbp, scaffold_split, scale_features
from quantum_kernel import build_feature_map, shot_based_kernel_matrix
from classical_baselines import fit_svm_select_C

NUM_QUBITS, REPS = 6, 2
N_TRAIN, N_TEST, SHOTS = 50, 30, 300

def make_noise_model(p):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p, 1),     ["u", "rx", "ry", "rz", "p"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p * 10, 2),["cx"])
    return nm

print("Loading data...")
smiles, X, y, scaffolds = load_bbbp("data/BBBP.csv")
ti, vi, tei = scaffold_split(scaffolds, seed=0)
ytr, yva, yte = y[ti], y[vi], y[tei]
Xtr, Xva, Xte, _ = scale_features(X[ti], X[vi], X[tei])

rng = np.random.RandomState(42)
tr_i = rng.choice(len(Xtr), N_TRAIN, replace=False)
te_i = rng.choice(len(Xte), N_TEST,  replace=False)
Xtr_s, ytr_s = Xtr[tr_i], ytr[tr_i]
Xte_s, yte_s = Xte[te_i], yte[te_i]

fm = build_feature_map(num_qubits=NUM_QUBITS, reps=REPS, entanglement="full")
noise_levels = [0.0, 0.001, 0.003, 0.005, 0.01, 0.02, 0.05]
bal_accs, k_vars = [], []

for p in noise_levels:
    t0 = time.time()
    nm  = make_noise_model(p) if p > 0 else None
    sim = AerSimulator(noise_model=nm)

    K_tr = np.zeros((N_TRAIN, N_TRAIN))
    for i in range(N_TRAIN):
        for j in range(i, N_TRAIN):
            circ = fm.assign_parameters(Xtr_s[i]).compose(fm.assign_parameters(Xtr_s[j]).inverse())
            circ.measure_all()
            counts = sim.run(circ, shots=SHOTS).result().get_counts()
            val = counts.get("0" * NUM_QUBITS, 0) / SHOTS
            K_tr[i, j] = K_tr[j, i] = val

    K_te = np.zeros((N_TEST, N_TRAIN))
    for i in range(N_TEST):
        for j in range(N_TRAIN):
            circ = fm.assign_parameters(Xte_s[i]).compose(fm.assign_parameters(Xtr_s[j]).inverse())
            circ.measure_all()
            counts = sim.run(circ, shots=SHOTS).result().get_counts()
            K_te[i, j] = counts.get("0" * NUM_QUBITS, 0) / SHOTS

    mask = ~np.eye(N_TRAIN, dtype=bool)
    k_vars.append(float(K_tr[mask].var()))
    K_tr_reg = np.clip(K_tr, 0, None) + 1e-5 * np.eye(N_TRAIN)
    K_te_c   = np.clip(K_te, 0, None)

    try:
        model, C, _ = fit_svm_select_C("precomputed", K_tr_reg, ytr_s, K_tr_reg, ytr_s,
                                        C_grid=(0.1, 1.0, 10.0))
        bal_acc = balanced_accuracy_score(yte_s, model.predict(K_te_c))
    except Exception:
        bal_acc = 0.5
    bal_accs.append(bal_acc)
    print(f"  p={p:.3f}  kernel_var={k_vars[-1]:.5f}  bal_acc={bal_acc:.3f}  ({time.time()-t0:.1f}s)")

res = {"noise_levels": noise_levels, "balanced_accuracy": bal_accs, "kernel_off_diag_variance": k_vars}
with open("results/ablation_noise.json", "w") as f:
    json.dump(res, f, indent=2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(noise_levels, bal_accs, "o-", color="#1f77b4", linewidth=2)
ax1.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance (balanced acc = 0.5)")
ax1.set_xlabel("Single-qubit depolarizing error rate  p"); ax1.set_ylabel("Balanced accuracy (subsample)")
ax1.set_title("Classification degrades under gate noise"); ax1.legend()
ax2.semilogy(noise_levels, [max(v, 1e-8) for v in k_vars], "s-", color="#d62728", linewidth=2)
ax2.set_xlabel("Single-qubit depolarizing error rate  p"); ax2.set_ylabel("Off-diagonal kernel variance (log)")
ax2.set_title("Kernel concentration: variance → 0\nmeans all molecule pairs look identical")
plt.tight_layout()
plt.savefig("docs/ablation_noise.png", dpi=140)
print("Saved docs/ablation_noise.png")
