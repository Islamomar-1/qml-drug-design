"""
run_main_experiment.py
-------------------------
End-to-end: load BBBP, scaffold-split, verify quantum kernel is a valid
Gram matrix, train classical baselines and quantum-kernel SVM on identical
data, and report balanced accuracy / ROC-AUC.
"""
import sys, json, time
sys.path.insert(0, "python")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score

from data_utils import load_bbbp, scaffold_split, scale_features
from quantum_kernel import build_feature_map, statevectors, exact_kernel_matrix, verify_kernel
from classical_baselines import majority_baseline, fit_svm_select_C, fit_logistic_regression, evaluate

NUM_QUBITS = 6
REPS = 2

def main():
    print("Loading BBBP and computing descriptors...")
    smiles, X, y, scaffolds = load_bbbp("data/BBBP.csv")
    train_idx, val_idx, test_idx = scaffold_split(scaffolds, seed=0)
    ytr, yva, yte = y[train_idx], y[val_idx], y[test_idx]
    Xtr, Xva, Xte, scaler = scale_features(X[train_idx], X[val_idx], X[test_idx])
    print(f"  {len(smiles)} molecules, split {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    results = {}

    print("\n--- Majority class baseline ---")
    results["Majority class"] = majority_baseline(ytr, yte)
    print(results["Majority class"])

    print("\n--- Logistic regression ---")
    lr_model, lr_C, _ = fit_logistic_regression(Xtr, ytr, Xva, yva)
    y_pred = lr_model.predict(Xte); y_score = lr_model.predict_proba(Xte)[:, 1]
    results["Logistic regression"] = evaluate(yte, y_pred, y_score)
    print(f"  C={lr_C}  test: {results['Logistic regression']}")

    print("\n--- RBF-kernel SVM ---")
    rbf_model, rbf_C, _ = fit_svm_select_C("rbf", Xtr, ytr, Xva, yva)
    y_pred = rbf_model.predict(Xte); y_score = rbf_model.predict_proba(Xte)[:, 1]
    results["RBF-kernel SVM"] = evaluate(yte, y_pred, y_score)
    print(f"  C={rbf_C}  test: {results['RBF-kernel SVM']}")

    print("\n--- Linear SVM ---")
    lin_model, lin_C, _ = fit_svm_select_C("linear", Xtr, ytr, Xva, yva)
    y_pred = lin_model.predict(Xte); y_score = lin_model.predict_proba(Xte)[:, 1]
    results["Linear SVM"] = evaluate(yte, y_pred, y_score)
    print(f"  C={lin_C}  test: {results['Linear SVM']}")

    print(f"\n--- Quantum kernel SVM ({NUM_QUBITS} qubits, ZZFeatureMap, reps={REPS}) ---")
    fm = build_feature_map(num_qubits=NUM_QUBITS, reps=REPS, entanglement="full")
    t0 = time.time()
    sv_tr = statevectors(fm, Xtr)
    sv_va = statevectors(fm, Xva)
    sv_te = statevectors(fm, Xte)
    print(f"  statevectors in {time.time()-t0:.1f}s")

    K_tr = exact_kernel_matrix(sv_tr, sv_tr)
    K_va = exact_kernel_matrix(sv_va, sv_tr)
    K_te = exact_kernel_matrix(sv_te, sv_tr)
    verify_kernel(K_tr, name="quantum train kernel")

    q_model, q_C, q_val = fit_svm_select_C("precomputed", K_tr, ytr, K_va, yva)
    y_pred = q_model.predict(K_te); y_score = q_model.predict_proba(K_te)[:, 1]
    results["Quantum kernel SVM"] = evaluate(yte, y_pred, y_score)
    print(f"  C={q_C} (val bal-acc={q_val:.3f})  test: {results['Quantum kernel SVM']}")

    print("\n=== Summary (test set) ===")
    print(f"{'Model':<25} {'Accuracy':>10} {'Balanced Acc':>14} {'ROC-AUC':>10}")
    for name, m in results.items():
        print(f"{name:<25} {m['accuracy']:>10.3f} {m['balanced_accuracy']:>14.3f} {m['roc_auc']:>10.3f}")

    with open("results/main_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Bar chart
    names = list(results.keys())
    bal_acc = [results[n]["balanced_accuracy"] for n in names]
    auc     = [results[n]["roc_auc"] for n in names]
    x_pos   = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(x_pos - 0.18, bal_acc, 0.35, label="Balanced accuracy", color="#1f77b4")
    ax.bar(x_pos + 0.18, auc,     0.35, label="ROC-AUC",           color="#ff7f0e")
    ax.set_xticks(x_pos); ax.set_xticklabels(names, rotation=18, ha="right")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance")
    ax.set_title("BBBP blood-brain barrier: quantum kernel SVM vs classical baselines")
    ax.legend(); plt.tight_layout()
    plt.savefig("docs/main_results.png", dpi=140)
    print("\nSaved docs/main_results.png")

    # Kernel heatmap
    order  = np.argsort(ytr[:80])
    K_sub  = K_tr[:80][:, :80][np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(K_sub, cmap="viridis", vmin=0, vmax=1)
    ax.set_title("Quantum kernel matrix\n(80 train molecules, sorted by BBB label)")
    plt.colorbar(im, ax=ax, label="K(x, y)")
    plt.tight_layout()
    plt.savefig("docs/kernel_heatmap.png", dpi=140)
    print("Saved docs/kernel_heatmap.png")

    return results

if __name__ == "__main__":
    main()
