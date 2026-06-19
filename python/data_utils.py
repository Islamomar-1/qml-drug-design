"""
data_utils.py
---------------
Loads the BBBP (blood-brain barrier penetration) dataset and turns each
molecule into a small, fixed-size feature vector suitable for encoding
into a handful of qubits.

Feature choice is not arbitrary: MolWt, MolLogP, TPSA, NumHDonors,
NumHAcceptors, and NumRotatableBonds are essentially Lipinski/"rule of
five"-style descriptors, and lower molecular weight, higher lipophilicity
(LogP), and lower polar surface area are textbook predictors of whether a
compound crosses the blood-brain barrier. Six features also maps onto
exactly six qubits for the feature map -- one classical descriptor per
qubit, the standard convention in quantum-kernel literature.

Splitting uses Bemis-Murcko scaffold splitting (grouping molecules by
their core ring scaffold and keeping each scaffold entirely within one
split), which is the split MoleculeNet itself recommends for BBBP
specifically, since a random split tends to overestimate how well a
model generalizes to structurally novel compounds.
"""
import csv
from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
from sklearn.preprocessing import MinMaxScaler

RDLogger.DisableLog("rdApp.*")

FEATURE_NAMES = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
]


def compute_descriptors(mol):
    return np.array([
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
    ], dtype=np.float64)


def load_bbbp(csv_path):
    """Returns (smiles_list, X_raw, y, scaffolds) for every molecule that parses."""
    smiles_list, X_raw, y, scaffolds = [], [], [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles = row["smiles"].strip()
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            X_raw.append(compute_descriptors(mol))
            y.append(int(row["p_np"]))
            smiles_list.append(smiles)
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            except Exception:
                scaffold = ""
            scaffolds.append(scaffold)
    return smiles_list, np.array(X_raw), np.array(y), scaffolds


def scaffold_split(scaffolds, frac_train=0.7, frac_val=0.15, seed=0):
    """
    Bemis-Murcko scaffold split, following the standard greedy convention
    (used by MoleculeNet/DeepChem): group molecule indices by scaffold,
    order scaffold groups largest-first, then fill the train split first,
    then validation, then whatever's left goes to test. This keeps every
    occurrence of a given scaffold in only one split, so the model is
    evaluated on structurally novel compounds rather than near-duplicates
    of training molecules.
    """
    groups = defaultdict(list)
    for idx, scaf in enumerate(scaffolds):
        groups[scaf].append(idx)

    rng = np.random.RandomState(seed)
    group_list = list(groups.values())
    # Shuffle within size-tiers so the split isn't sensitive to CSV row order,
    # while still processing largest scaffold groups first.
    rng.shuffle(group_list)
    group_list.sort(key=len, reverse=True)

    n = len(scaffolds)
    n_train_target = int(frac_train * n)
    n_val_target = int(frac_val * n)

    train_idx, val_idx, test_idx = [], [], []
    for group in group_list:
        if len(train_idx) < n_train_target:
            train_idx.extend(group)
        elif len(val_idx) < n_val_target:
            val_idx.extend(group)
        else:
            test_idx.extend(group)
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def scale_features(X_train, X_val, X_test, feature_range=(0.0, np.pi)):
    """
    Fits the scaler on TRAIN only (no leakage), then applies it everywhere.
    Scaling to [0, pi] is a standard convention for angle-encoding features
    into rotation gates: it uses the full useful range of a rotation
    without wrapping around past 2*pi, which would make two genuinely
    different feature values look identical to the quantum circuit.
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    scaler.fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test), scaler


if __name__ == "__main__":
    smiles, X, y, scaffolds = load_bbbp("data/BBBP.csv")
    print(f"Loaded {len(smiles)} molecules, {len(set(scaffolds))} unique scaffolds")
    print(f"Label balance: {np.bincount(y)} (0=non-penetrant, 1=penetrant)")

    train_idx, val_idx, test_idx = scaffold_split(scaffolds)
    print(f"Scaffold split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        print(f"  {name} label balance: {np.bincount(y[idx], minlength=2)}")

    Xtr, Xva, Xte, scaler = scale_features(X[train_idx], X[val_idx], X[test_idx])
    print(f"Feature ranges after scaling (train): min={Xtr.min(axis=0)}, max={Xtr.max(axis=0)}")
