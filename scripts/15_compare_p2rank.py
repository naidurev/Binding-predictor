#!/usr/bin/env python3
"""
Compare our model vs P2Rank on COACH420.

P2Rank residues CSV  → per-residue probability (positional match with .npz)
Our model            → per-residue sigmoid probability from best_model_v5.pt
Ground truth         → labels in .npz

Output:
  results/comparison_per_protein.csv   — per-protein metrics for both models
  results/comparison_global.json       — aggregate metrics
  results/comparison_plots/            — scatter, ROC overlay, histogram
"""

import os, sys, json, csv
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    matthews_corrcoef, roc_curve
)

sys.path.append(os.path.dirname(__file__))
from importlib import import_module
model_arch = import_module('06_model_architecture')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(BASE, 'models', 'best_model_v5.pt')
DATA_DIR     = os.path.join(BASE, 'data', 'coach420_combined')
P2RANK_DIR   = os.path.join(BASE, 'p2rank_output')
RESULTS_DIR  = os.path.join(BASE, 'results', 'comparison')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load protein IDs from available .npz files ─────────────────────────────────
protein_ids = sorted([
    f.replace('.npz', '')
    for f in os.listdir(DATA_DIR) if f.endswith('.npz')
])
print(f"Found {len(protein_ids)} proteins in coach420_combined")

# ── Dataset ────────────────────────────────────────────────────────────────────
class ProteinDataset(Dataset):
    def __init__(self, ids, data_dir):
        self.ids      = ids
        self.data_dir = data_dir

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        pid  = self.ids[idx]
        data = np.load(os.path.join(self.data_dir, f'{pid}.npz'))
        return {
            'id':       pid,
            'features': torch.from_numpy(data['features']),
            'labels':   torch.from_numpy(data['labels'].astype(np.int8))
        }

def collate_fn(batch):
    features = [b['features'] for b in batch]
    labels   = [b['labels']   for b in batch]
    ids      = [b['id']       for b in batch]
    lengths  = torch.tensor([len(f) for f in features])
    feat_pad = pad_sequence(features, batch_first=True, padding_value=0)
    lab_pad  = pad_sequence(labels,   batch_first=True, padding_value=-1)
    max_len  = feat_pad.size(1)
    mask     = torch.arange(max_len)[None, :] >= lengths[:, None]
    return feat_pad, lab_pad, mask, ids

# ── Load model ─────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
model = model_arch.BindingSitePredictor().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()
print(f"Loaded model: {MODEL_PATH}")

# ── Run inference ──────────────────────────────────────────────────────────────
loader = DataLoader(ProteinDataset(protein_ids, DATA_DIR),
                    batch_size=4, shuffle=False, collate_fn=collate_fn)

our_preds = {}   # pid → {'probs': np.array, 'labels': np.array}

with torch.no_grad():
    for feat, lab, mask, ids in loader:
        feat = feat.to(device)
        mask = mask.to(device)
        logits = model(feat, mask)
        probs  = torch.sigmoid(logits).cpu().numpy()
        lab_np = lab.numpy()
        for i, pid in enumerate(ids):
            valid = lab_np[i] != -1
            our_preds[pid] = {
                'probs':  probs[i][valid],
                'labels': lab_np[i][valid].astype(int)
            }

print(f"Inference done: {len(our_preds)} proteins")

# ── Parse P2Rank residues CSV ──────────────────────────────────────────────────
def load_p2rank_residues(pid, n_residues):
    """
    Reads {pid}.pdb_residues.csv → returns (probs_array, binary_array) of length n_residues.
    Matches positionally (row order = PDB residue order).
    If CSV has more/fewer rows, truncate or pad with 0.
    """
    path = os.path.join(P2RANK_DIR, f'{pid}.pdb_residues.csv')
    if not os.path.exists(path):
        return None, None

    probs  = []
    binary = []
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = float(row[' probability'].strip())
                pocket = int(row[' pocket'].strip())
                probs.append(p)
                binary.append(1 if pocket > 0 else 0)
    except Exception as e:
        print(f"  Parse error {pid}: {e}")
        return None, None

    probs  = np.array(probs,  dtype=np.float32)
    binary = np.array(binary, dtype=np.int8)

    # Align length with our model's residue count
    if len(probs) > n_residues:
        probs  = probs[:n_residues]
        binary = binary[:n_residues]
    elif len(probs) < n_residues:
        pad = n_residues - len(probs)
        probs  = np.pad(probs,  (0, pad))
        binary = np.pad(binary, (0, pad))

    return probs, binary

# ── Per-protein comparison ─────────────────────────────────────────────────────
rows = []
our_probs_all,    our_labels_all    = [], []
p2r_probs_all,    p2r_labels_all    = [], []
p2r_binary_all,   p2r_bin_labels    = [], []

skipped_p2rank = 0

for pid, d in our_preds.items():
    probs  = d['probs']
    labels = d['labels']
    n      = len(labels)

    if len(np.unique(labels)) < 2:
        continue   # skip single-class proteins

    our_probs_all.append(probs)
    our_labels_all.append(labels)

    row = {'protein': pid, 'n_residues': n, 'n_binding': int(labels.sum())}

    # Our model metrics
    try:
        row['our_auc']  = round(float(roc_auc_score(labels, probs)), 4)
        row['our_ap']   = round(float(average_precision_score(labels, probs)), 4)
        row['our_f1']   = round(float(f1_score(labels, (probs >= 0.5).astype(int), zero_division=0)), 4)
        row['our_prec'] = round(float(precision_score(labels, (probs >= 0.5).astype(int), zero_division=0)), 4)
        row['our_rec']  = round(float(recall_score(labels, (probs >= 0.5).astype(int), zero_division=0)), 4)
        row['our_mcc']  = round(float(matthews_corrcoef(labels, (probs >= 0.5).astype(int))), 4)
    except Exception:
        for k in ['our_auc','our_ap','our_f1','our_prec','our_rec','our_mcc']:
            row[k] = None

    # P2Rank metrics
    p2r_probs, p2r_bin = load_p2rank_residues(pid, n)
    if p2r_probs is not None:
        p2r_probs_all.append(p2r_probs)
        p2r_labels_all.append(labels)
        p2r_binary_all.append(p2r_bin)
        p2r_bin_labels.append(labels)
        try:
            row['p2r_auc']  = round(float(roc_auc_score(labels, p2r_probs)), 4)
            row['p2r_ap']   = round(float(average_precision_score(labels, p2r_probs)), 4)
            row['p2r_f1']   = round(float(f1_score(labels, p2r_bin, zero_division=0)), 4)
            row['p2r_prec'] = round(float(precision_score(labels, p2r_bin, zero_division=0)), 4)
            row['p2r_rec']  = round(float(recall_score(labels, p2r_bin, zero_division=0)), 4)
            row['p2r_mcc']  = round(float(matthews_corrcoef(labels, p2r_bin)), 4)
        except Exception:
            for k in ['p2r_auc','p2r_ap','p2r_f1','p2r_prec','p2r_rec','p2r_mcc']:
                row[k] = None
    else:
        skipped_p2rank += 1
        for k in ['p2r_auc','p2r_ap','p2r_f1','p2r_prec','p2r_rec','p2r_mcc']:
            row[k] = None

    rows.append(row)

print(f"Processed {len(rows)} proteins | P2Rank missing: {skipped_p2rank}")

# ── Global metrics ─────────────────────────────────────────────────────────────
our_p  = np.concatenate(our_probs_all)
our_l  = np.concatenate(our_labels_all)
p2r_p  = np.concatenate(p2r_probs_all)   if p2r_probs_all  else None
p2r_l  = np.concatenate(p2r_labels_all)  if p2r_labels_all else None
p2r_b  = np.concatenate(p2r_binary_all)  if p2r_binary_all else None

def global_metrics(labels, probs, binary=None, thresh=0.5):
    if binary is None:
        binary = (probs >= thresh).astype(int)
    return {
        'auc_roc':   round(float(roc_auc_score(labels, probs)), 4),
        'avg_prec':  round(float(average_precision_score(labels, probs)), 4),
        'f1':        round(float(f1_score(labels, binary, zero_division=0)), 4),
        'precision': round(float(precision_score(labels, binary, zero_division=0)), 4),
        'recall':    round(float(recall_score(labels, binary, zero_division=0)), 4),
        'mcc':       round(float(matthews_corrcoef(labels, binary)), 4),
    }

our_global = global_metrics(our_l, our_p)
p2r_global = global_metrics(p2r_l, p2r_p, binary=p2r_b) if p2r_p is not None else {}

our_aucs = [r['our_auc'] for r in rows if r.get('our_auc') is not None]
p2r_aucs = [r['p2r_auc'] for r in rows if r.get('p2r_auc') is not None]

# ── Print ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*58}")
print(f"  COACH420 Comparison ({len(rows)} proteins)")
print(f"{'='*58}")
print(f"  {'Metric':<20} {'Our Model':>12} {'P2Rank':>12}")
print(f"  {'-'*44}")
for metric in ['auc_roc','avg_prec','f1','precision','recall','mcc']:
    our_val = our_global.get(metric, 'N/A')
    p2r_val = p2r_global.get(metric, 'N/A')
    our_str = f"{our_val:.4f}" if isinstance(our_val, float) else our_val
    p2r_str = f"{p2r_val:.4f}" if isinstance(p2r_val, float) else p2r_val
    print(f"  {metric.upper():<20} {our_str:>12} {p2r_str:>12}")
print(f"{'='*58}")
print(f"\n  Per-protein AUC:")
print(f"  Ours   — Mean: {np.mean(our_aucs):.4f}  Median: {np.median(our_aucs):.4f}")
if p2r_aucs:
    print(f"  P2Rank — Mean: {np.mean(p2r_aucs):.4f}  Median: {np.median(p2r_aucs):.4f}")

# ── Save JSON ──────────────────────────────────────────────────────────────────
summary = {
    'n_proteins': len(rows),
    'our_model':  our_global,
    'p2rank':     p2r_global,
    'per_protein_auc': {
        'our':    {'mean': round(float(np.mean(our_aucs)),4), 'median': round(float(np.median(our_aucs)),4)},
        'p2rank': {'mean': round(float(np.mean(p2r_aucs)),4), 'median': round(float(np.median(p2r_aucs)),4)} if p2r_aucs else {},
    },
    'per_protein': rows
}
with open(os.path.join(RESULTS_DIR, 'comparison_global.json'), 'w') as f:
    json.dump(summary, f, indent=2)

# ── Save CSV ───────────────────────────────────────────────────────────────────
fieldnames = ['protein','n_residues','n_binding',
              'our_auc','our_ap','our_f1','our_prec','our_rec','our_mcc',
              'p2r_auc','p2r_ap','p2r_f1','p2r_prec','p2r_rec','p2r_mcc']
with open(os.path.join(RESULTS_DIR, 'comparison_per_protein.csv'), 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved → {RESULTS_DIR}/comparison_global.json")
print(f"Saved → {RESULTS_DIR}/comparison_per_protein.csv")

# ── Plots ──────────────────────────────────────────────────────────────────────
# 1. Per-protein AUC scatter: ours vs p2rank
paired = [(r['our_auc'], r['p2r_auc']) for r in rows
          if r.get('our_auc') and r.get('p2r_auc')]
if paired:
    x = [p[1] for p in paired]  # p2rank
    y = [p[0] for p in paired]  # ours
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.4, s=15, color='steelblue')
    plt.plot([0,1],[0,1], 'r--', lw=1, label='Equal')
    plt.xlabel('P2Rank AUC')
    plt.ylabel('Our Model AUC')
    plt.title(f'Per-protein AUC: Ours vs P2Rank (n={len(paired)})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'auc_scatter.png'), dpi=150)
    plt.close()

# 2. AUC histogram overlay
plt.figure(figsize=(8, 4))
plt.hist(our_aucs, bins=20, alpha=0.6, label=f'Ours (mean={np.mean(our_aucs):.3f})',  color='steelblue', edgecolor='white')
if p2r_aucs:
    plt.hist(p2r_aucs, bins=20, alpha=0.6, label=f'P2Rank (mean={np.mean(p2r_aucs):.3f})', color='darkorange', edgecolor='white')
plt.xlabel('Per-Protein AUC-ROC')
plt.ylabel('Count')
plt.title('COACH420 — AUC Distribution')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'auc_histogram.png'), dpi=150)
plt.close()

# 3. Global ROC curve overlay
plt.figure(figsize=(6, 6))
fpr_o, tpr_o, _ = roc_curve(our_l, our_p)
plt.plot(fpr_o, tpr_o, color='steelblue', lw=2, label=f'Ours (AUC={our_global["auc_roc"]:.3f})')
if p2r_p is not None:
    fpr_p, tpr_p, _ = roc_curve(p2r_l, p2r_p)
    plt.plot(fpr_p, tpr_p, color='darkorange', lw=2, label=f'P2Rank (AUC={p2r_global["auc_roc"]:.3f})')
plt.plot([0,1],[0,1],'k--',lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('COACH420 — Global ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'roc_overlay.png'), dpi=150)
plt.close()

print(f"Saved plots → {RESULTS_DIR}/")
print("\nDone.")
