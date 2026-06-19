"""
quantum_kernel.py
--------------------
Computes quantum kernels: K(x, y) = |<phi(x)|phi(y)>|^2, where phi is a
quantum feature map that encodes a classical vector into the state of an
n-qubit circuit. This is the approach introduced by Havlicek et al.,
"Supervised learning with quantum-enhanced feature spaces" (Nature, 2019):
rather than using a quantum computer to train a classifier directly, use
it to evaluate a kernel that a classical SVM can't easily approximate
classically, then hand the resulting kernel matrix to an ordinary
classical SVM.

Two ways to compute the kernel are implemented, deliberately:

1. EXACT (statevector): get the full statevector for every data point
   once, then compute every pairwise overlap as a single vectorized
   matrix multiplication. This is fast and exact, and is what's used for
   the main experiment and the entanglement ablation.

2. SHOT-BASED (measurement circuit): build phi(x) followed by the INVERSE
   of phi(y), measure all qubits, and estimate K(x,y) as the fraction of
   shots landing on the all-zeros bitstring. This is how a quantum kernel
   actually has to be evaluated on real hardware -- you cannot read out a
   statevector's amplitudes directly, only sample measurement outcomes --
   and it's the only way to meaningfully inject hardware noise, since a
   noise model acts on circuits, not on inner products. Used only for the
   noise ablation, since it requires one circuit execution per pair of
   points rather than one simulation per point.

verify_kernel() checks that the resulting kernel matrix is symmetric,
has a unit diagonal, and is positive semi-definite (a valid Gram matrix)
-- the way check-grad verified the backward pass in the C++ project.
"""
import numpy as np
from qiskit.circuit.library import zz_feature_map, z_feature_map
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


def build_feature_map(num_qubits, reps=2, entanglement="full"):
    """
    entanglement="full"/"linear"/"circular": a ZZFeatureMap -- single-qubit
      rotations interleaved with entangling ZZ interactions between qubits,
      reps times. This is the standard quantum-kernel feature map.
    entanglement="none": a ZFeatureMap -- single-qubit rotations only, no
      entangling gates at all. The resulting state factorizes into a
      tensor product over qubits, so the kernel reduces to a PRODUCT of
      independent single-feature kernels: exactly the kind of feature map
      that confers no advantage over a classical product kernel. This is
      the control condition for the entanglement ablation.
    """
    if entanglement == "none":
        return z_feature_map(feature_dimension=num_qubits, reps=reps)
    return zz_feature_map(feature_dimension=num_qubits, reps=reps, entanglement=entanglement)


def statevectors(feature_map, X):
    """One simulation per data point; returns an (n_samples, 2**n_qubits) complex array."""
    dim = 2 ** feature_map.num_qubits
    out = np.empty((len(X), dim), dtype=complex)
    for i, x in enumerate(X):
        out[i] = Statevector(feature_map.assign_parameters(x)).data
    return out


def exact_kernel_matrix(sv_a, sv_b):
    """|<a|b>|^2 for every pair, via one matrix multiplication. sv_a, sv_b: (n, dim) complex arrays."""
    overlaps = sv_a.conj() @ sv_b.T
    return np.abs(overlaps) ** 2


def shot_based_kernel_value(feature_map, x, y, shots=2000, noise_model=None, simulator=None):
    """Estimate K(x, y) by measuring P(all-zeros) after phi(x) then phi(y)^-1."""
    circ = feature_map.assign_parameters(x).compose(feature_map.assign_parameters(y).inverse())
    circ.measure_all()
    sim = simulator if simulator is not None else AerSimulator(noise_model=noise_model)
    counts = sim.run(circ, shots=shots).result().get_counts()
    zero_string = "0" * feature_map.num_qubits
    return counts.get(zero_string, 0) / shots


def shot_based_kernel_matrix(feature_map, X_a, X_b, shots=2000, noise_model=None, symmetric=False):
    """
    Full kernel matrix via the shot-based method. O(n_a * n_b) circuit
    executions -- only practical for the smaller subsample used in the
    noise ablation, not the full dataset.
    """
    sim = AerSimulator(noise_model=noise_model)
    n_a, n_b = len(X_a), len(X_b)
    K = np.zeros((n_a, n_b))
    for i in range(n_a):
        j_range = range(i, n_b) if symmetric else range(n_b)
        for j in j_range:
            val = shot_based_kernel_value(feature_map, X_a[i], X_b[j], shots=shots, simulator=sim)
            K[i, j] = val
            if symmetric and j < n_a:
                K[j, i] = val
    return K


def verify_kernel(K, tol=1e-6, name="kernel"):
    """Checks symmetry, unit diagonal (for a self-kernel), and positive semi-definiteness."""
    issues = []
    eigvals = None
    if K.shape[0] == K.shape[1]:
        asym = np.max(np.abs(K - K.T))
        if asym > tol:
            issues.append(f"not symmetric (max asymmetry {asym:.2e})")
        diag_err = np.max(np.abs(np.diag(K) - 1.0))
        if diag_err > 1e-3:
            issues.append(f"diagonal not ~1.0 (max deviation {diag_err:.2e})")
        eigvals = np.linalg.eigvalsh((K + K.T) / 2)
        min_eig = eigvals.min()
        if min_eig < -1e-6:
            issues.append(f"not PSD (min eigenvalue {min_eig:.2e})")
    if issues:
        print(f"[FAIL] {name}: " + "; ".join(issues))
        return False
    print(f"[PASS] {name}: symmetric, unit diagonal, positive semi-definite "
          f"(min eigenvalue {eigvals.min():.2e})")
    return True


if __name__ == "__main__":
    # Quick self-test: exact vs shot-based agreement, and Gram matrix validity.
    rng = np.random.RandomState(0)
    X = rng.uniform(0, np.pi, size=(6, 6))
    fm = build_feature_map(num_qubits=6, reps=2, entanglement="full")

    sv = statevectors(fm, X)
    K_exact = exact_kernel_matrix(sv, sv)
    verify_kernel(K_exact, name="exact kernel (6 random points)")

    i, j = 1, 4
    exact_val = K_exact[i, j]
    shot_val = shot_based_kernel_value(fm, X[i], X[j], shots=20000)
    print(f"exact K[{i},{j}]={exact_val:.4f}  shot-based estimate={shot_val:.4f} "
          f"(should agree within shot noise)")
