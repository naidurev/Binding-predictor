#!/usr/bin/env python3
"""
Evaluate trained model on COACH420 benchmark.
Compares per-residue predictions against P2Rank baseline.

Input:  ../models/best_model.pt
        ../data/coach420_combined/*.npz
        ../data/coach420_ids.txt
        ../data/coach420/p2rank/*.csv
Output: ../results/coach420_metrics.json
        ../results/coach420_comparison.csv
"""

import os
import sys
import csv
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    matthews_corrcoef
)

sys.path.append(os.path.dirname(__file__))
from importlib import import_module
model_arch = import_module('06_model_architecture')

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = '/content/models/best_model.pt'
DATA_DIR     = '/content/data/coach420_combined'
IDS_PATH     = '/content/data/coach420_ids.txt'
COACH420_DIR = '/content/data/coach420'   # flat directory (Windows zip)
RESULTS_DIR  = '/content/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Build a lookup: protein_id → path of its p2rank CSV in the flat directory
# Files may be named like 'coach420\p2rank\148lE.pdb_predictions.csv' (Windows-flattened)
# or '148lE.pdb_predictions.csv' if already extracted correctly.
_p2rank_index = {}
for _f in os.listdir(COACH420_DIR):
    if 'p2rank' in _f and _f.endswith('.pdb_predictions.csv'):
        _basename = _f.replace('\\', '/').split('/')[-1]
        _pid = _basename.replace('.pdb_predictions.csv', '')
        _p2rank_index[_pid] = os.path.join(COACH420_DIR, _f)
print(f"Indexed {len(_p2rank_index)} p2rank CSV files.")

# ── Dataset ───────────────────────────────────────────────────────────────────
class ProteinDataset(Dataset):
    def __init__(self, protein_ids, data_dir):
        self.protein_ids = protein_ids
        self.data_dir    = data_dir

    def __len__(self):
        return len(self.protein_ids)

    def __getitem__(self, idx):
        pid  = self.protein_ids[idx]
        data = np.load(os.path.join(self.data_dir, f"{pid}.npz"))
        return {
            'id':       pid,
            'features': torch.from_numpy(data['features']),
            'labels':   torch.from_numpy(data['labels'])
        }

def collate_fn(batch):
    features = [item['features'] for item in batch]
    labels   = [item['labels']   for item in batch]
    ids      = [item['id']       for item in batch]
    lengths  = torch.tensor([len(f) for f in features])

    features_padded = pad_sequence(features, batch_first=True, padding_value=0)
    labels_padded   = pad_sequence(labels,   batch_first=True, padding_value=-1)

    max_len = features_padded.size(1)
    mask = torch.arange(max_len)[None, :] >= lengths[:, None]

    return features_padded, labels_padded, mask, ids

# ── Load model ────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Try best_model.pt first, then fall back to best_model_v3.pt
if not os.path.exists(MODEL_PATH):
    alt = MODEL_PATH.replace('best_model.pt', 'best_model_v3.pt')
    if os.path.exists(alt):
        MODEL_PATH = alt
model = model_arch.BindingSitePredictor().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Loaded: {MODEL_PATH}")

# ── Load protein IDs ──────────────────────────────────────────────────────────
with open(IDS_PATH) as f:
    all_ids = [l.strip() for l in f if l.strip()]

# Filter to proteins that exist
protein_ids = [pid for pid in all_ids
               if os.path.exists(os.path.join(DATA_DIR, f"{pid}.npz"))]
print(f"Evaluating {len(protein_ids)} COACH420 proteins\n")

dataset = ProteinDataset(protein_ids, DATA_DIR)
loader  = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

# ── Model inference ───────────────────────────────────────────────────────────
per_protein_preds  = {}   # pid -> {'probs': array, 'labels': array}

with torch.no_grad():
    for features, labels, mask, ids in loader:
        features = features.to(device)
        mask     = mask.to(device)
        logits   = model(features, mask)
        probs    = torch.sigmoid(logits).cpu().numpy()
        labels_np = labels.numpy()

        for i, pid in enumerate(ids):
            valid = labels_np[i] != -1
            per_protein_preds[pid] = {
                'probs':  probs[i][valid],
                'labels': labels_np[i][valid].astype(int),
                'n_residues': int(valid.sum())
            }

print(f"Inference done on {len(per_protein_preds)} proteins.")

# ── Parse P2Rank predictions ──────────────────────────────────────────────────
def parse_p2rank(protein_id, n_residues):
    """
    Returns binary array of length n_residues.
    Residues in P2Rank's top pocket are marked 1.
    P2Rank residue IDs are 1-indexed residue numbers (not array indices).
    We map them to array indices conservatively: index = resnum - 1.
    """
    csv_file = _p2rank_index.get(protein_id)
    if csv_file is None or not os.path.exists(csv_file):
        return None

    preds = np.zeros(n_residues, dtype=np.int8)
    try:
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            rows   = list(reader)

        if not rows:
            return preds

        # Use only the top-ranked pocket (rank 1)
        top_row = sorted(rows, key=lambda r: int(r['rank'].strip()))[0]
        res_ids_str = top_row['residue_ids'].strip()
        if not res_ids_str:
            return preds

        for res_str in res_ids_str.split():
            try:
                idx = int(res_str.strip()) - 1  # 1-indexed → 0-indexed
                if 0 <= idx < n_residues:
                    preds[idx] = 1
            except ValueError:
                continue
    except Exception:
        return None

    return preds

# ── Compute metrics ───────────────────────────────────────────────────────────
our_all_probs  = []
our_all_labels = []
p2rank_all_preds  = []
p2rank_all_labels = []

per_protein_results = []

for pid, d in per_protein_preds.items():
    probs  = d['probs']
    labels = d['labels']
    n      = d['n_residues']

    if len(np.unique(labels)) < 2:
        continue  # skip proteins with single class

    our_all_probs.append(probs)
    our_all_labels.append(labels)

    p2rank_preds = parse_p2rank(pid, n)

    row = {'protein': pid, 'n_residues': n, 'n_binding': int(labels.sum())}

    # Our model metrics
    try:
        row['our_auc']  = round(float(roc_auc_score(labels, probs)), 4)
        row['our_f1']   = round(float(f1_score(labels, (probs >= 0.70).astype(int), zero_division=0)), 4)
        row['our_mcc']  = round(float(matthews_corrcoef(labels, (probs >= 0.70).astype(int))), 4)
    except Exception:
        row['our_auc'] = row['our_f1'] = row['our_mcc'] = None

    # P2Rank metrics
    if p2rank_preds is not None:
        p2rank_all_preds.append(p2rank_preds)
        p2rank_all_labels.append(labels)
        try:
            row['p2rank_f1']  = round(float(f1_score(labels, p2rank_preds, zero_division=0)), 4)
            row['p2rank_mcc'] = round(float(matthews_corrcoef(labels, p2rank_preds)), 4)
            row['p2rank_prec'] = round(float(precision_score(labels, p2rank_preds, zero_division=0)), 4)
            row['p2rank_rec']  = round(float(recall_score(labels, p2rank_preds, zero_division=0)), 4)
        except Exception:
            row['p2rank_f1'] = row['p2rank_mcc'] = None
    else:
        row['p2rank_f1'] = row['p2rank_mcc'] = None

    per_protein_results.append(row)

# ── Global metrics ────────────────────────────────────────────────────────────
our_probs_all  = np.concatenate(our_all_probs)
our_labels_all = np.concatenate(our_all_labels)
our_preds_all  = (our_probs_all >= 0.70).astype(int)

our_global = {
    'auc_roc':    round(float(roc_auc_score(our_labels_all, our_probs_all)), 4),
    'avg_prec':   round(float(average_precision_score(our_labels_all, our_probs_all)), 4),
    'f1':         round(float(f1_score(our_labels_all, our_preds_all, zero_division=0)), 4),
    'precision':  round(float(precision_score(our_labels_all, our_preds_all, zero_division=0)), 4),
    'recall':     round(float(recall_score(our_labels_all, our_preds_all, zero_division=0)), 4),
    'mcc':        round(float(matthews_corrcoef(our_labels_all, our_preds_all)), 4),
}

p2rank_global = {}
if p2rank_all_preds:
    p2r_preds  = np.concatenate(p2rank_all_preds)
    p2r_labels = np.concatenate(p2rank_all_labels)
    p2rank_global = {
        'f1':        round(float(f1_score(p2r_labels, p2r_preds, zero_division=0)), 4),
        'precision': round(float(precision_score(p2r_labels, p2r_preds, zero_division=0)), 4),
        'recall':    round(float(recall_score(p2r_labels, p2r_preds, zero_division=0)), 4),
        'mcc':       round(float(matthews_corrcoef(p2r_labels, p2r_preds)), 4),
    }

# ── Print results ─────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  COACH420 Evaluation ({len(per_protein_results)} proteins)")
print(f"{'='*55}")
print(f"\n  {'Metric':<20} {'Our Model':>12} {'P2Rank':>12}")
print(f"  {'-'*44}")
print(f"  {'AUC-ROC':<20} {our_global['auc_roc']:>12.4f} {'N/A':>12}")
print(f"  {'Avg Precision':<20} {our_global['avg_prec']:>12.4f} {'N/A':>12}")
print(f"  {'F1':<20} {our_global['f1']:>12.4f} {p2rank_global.get('f1', 'N/A'):>12}")
print(f"  {'Precision':<20} {our_global['precision']:>12.4f} {p2rank_global.get('precision', 'N/A'):>12}")
print(f"  {'Recall':<20} {our_global['recall']:>12.4f} {p2rank_global.get('recall', 'N/A'):>12}")
print(f"  {'MCC':<20} {our_global['mcc']:>12.4f} {p2rank_global.get('mcc', 'N/A'):>12}")
print(f"{'='*55}\n")

our_aucs = [r['our_auc'] for r in per_protein_results if r['our_auc'] is not None]
print(f"  Per-protein AUC  — Mean: {np.mean(our_aucs):.4f}  Median: {np.median(our_aucs):.4f}")
print(f"                      Min: {np.min(our_aucs):.4f}  Max: {np.max(our_aucs):.4f}\n")

# ── Save results ──────────────────────────────────────────────────────────────
results = {
    'n_proteins': len(per_protein_results),
    'our_model':  our_global,
    'p2rank':     p2rank_global,
    'per_protein_auc_summary': {
        'mean': round(float(np.mean(our_aucs)), 4),
        'median': round(float(np.median(our_aucs)), 4),
        'min': round(float(np.min(our_aucs)), 4),
        'max': round(float(np.max(our_aucs)), 4),
    },
    'per_protein': per_protein_results
}

json_path = os.path.join(RESULTS_DIR, 'coach420_metrics.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved metrics → {json_path}")

# CSV comparison table
csv_path = os.path.join(RESULTS_DIR, 'coach420_comparison.csv')
with open(csv_path, 'w', newline='') as f:
    fieldnames = ['protein', 'n_residues', 'n_binding',
                  'our_auc', 'our_f1', 'our_mcc',
                  'p2rank_f1', 'p2rank_prec', 'p2rank_rec', 'p2rank_mcc']
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(per_protein_results)
print(f"Saved comparison table → {csv_path}")

# ── Plot: per-protein AUC distribution ───────────────────────────────────────
plt.figure(figsize=(7, 4))
plt.hist(our_aucs, bins=20, edgecolor='white', color='steelblue')
plt.axvline(np.mean(our_aucs), color='red', linestyle='--', label=f'Mean={np.mean(our_aucs):.3f}')
plt.xlabel('Per-Protein AUC-ROC')
plt.ylabel('Count')
plt.title(f'COACH420 — Per-Protein AUC Distribution (n={len(our_aucs)})')
plt.legend()
plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, 'coach420_auc_dist.png')
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Saved plot → {plot_path}")
print("\nDone.")
