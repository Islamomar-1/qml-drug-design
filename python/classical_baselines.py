"""
classical_baselines.py
-------------------------
Classical comparison points for the quantum kernel SVM, trained on the
exact same 6 descriptors, exact same scaffold split, and exact same SVM
algorithm (only the kernel differs, for the SVM baselines) -- the same
"change one thing at a time" discipline used in the C++ project's
ablations.

Because BBBP is imbalanced (about 76% penetrant), plain accuracy is a
misleading headline number -- a constant "always predict penetrant"
classifier already scores ~76%. Balanced accuracy and ROC-AUC are
reported throughout instead.
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score


def evaluate(y_true, y_pred, y_score):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score),
    }


def majority_baseline(y_train, y_val_or_test):
    majority_class = int(np.round(y_train.mean()))
    y_pred = np.full_like(y_val_or_test, majority_class)
    y_score = np.full_like(y_val_or_test, float(majority_class), dtype=float)
    return evaluate(y_val_or_test, y_pred, y_score)


def fit_svm_select_C(kernel, Xtr_or_Ktr, ytr, Xva_or_Kva, yva, C_grid=(0.1, 0.3, 1.0, 3.0, 10.0, 30.0)):
    """
    kernel: 'rbf', 'linear', or 'precomputed' (for quantum kernels, where
    Xtr_or_Ktr / Xva_or_Kva are already-computed Gram matrices, not raw features).
    Selects C by validation balanced accuracy, mirroring how the C++
    project picked its checkpoint by validation RMSE rather than training loss.
    """
    best_C, best_score, best_model = None, -1.0, None
    for C in C_grid:
        model = SVC(kernel=kernel, C=C, probability=True, class_weight="balanced", random_state=0)
        model.fit(Xtr_or_Ktr, ytr)
        y_pred = model.predict(Xva_or_Kva)
        score = balanced_accuracy_score(yva, y_pred)
        if score > best_score:
            best_C, best_score, best_model = C, score, model
    return best_model, best_C, best_score


def fit_logistic_regression(Xtr, ytr, Xva, yva, C_grid=(0.01, 0.1, 1.0, 10.0)):
    best_C, best_score, best_model = None, -1.0, None
    for C in C_grid:
        model = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=0)
        model.fit(Xtr, ytr)
        score = balanced_accuracy_score(yva, model.predict(Xva))
        if score > best_score:
            best_C, best_score, best_model = C, score, model
    return best_model, best_C, best_score
