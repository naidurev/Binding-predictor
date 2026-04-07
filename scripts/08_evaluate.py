#!/usr/bin/env python3

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    matthews_corrcoef, confusion_matrix,
    roc_curve, precision_recall_curve
)

sys.path.append(os.path.dirname(__file__))
from importlib import import_module
model_arch = import_module('06_model_architecture')

# ── Paths ────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, 'models', 'best_model.pt')
DATA_DIR   = os.path.join(BASE, 'data', 'combined')
VAL_SPLIT  = os.path.join(BASE, 'data', 'splits', 'val.txt')
RESULTS    = os.path.join(BASE, 'results')
PLOTS_DIR  = os.path.join(RESULTS, 'val_plots')

os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Dataset (same as 07_train.py) ────────────────────────────────────────────
class ProteinDataset(Dataset):
    def __init__(self, protein_ids, data_dir):
        self.protein_ids = protein_ids
        self.data_dir = data_dir

    def __len__(self):
        return len(self.protein_ids)

    def __getitem__(self, idx):
        pid = self.protein_ids[idx]
        data = np.load(os.path.join(self.data_dir, f"{pid}.npz"))
        return {
            'id': pid,
            'features': torch.from_numpy(data['features']),
            'labels': torch.from_numpy(data['labels'])
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

model = model_arch.BindingSitePredictor().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Loaded model from {MODEL_PATH}")

# ── Load val set ──────────────────────────────────────────────────────────────
with open(VAL_SPLIT) as f:
    val_ids = f.read().splitlines()

# Filter to proteins that exist in combined/
val_ids = [pid for pid in val_ids if os.path.exists(os.path.join(DATA_DIR, f"{pid}.npz"))]
print(f"Evaluating on {len(val_ids)} val proteins")

dataset = ProteinDataset(val_ids, DATA_DIR)
loader  = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

# ── Inference ─────────────────────────────────────────────────────────────────
all_probs  = []
all_labels = []
per_protein = {}

with torch.no_grad():
    for features, labels, mask, ids in loader:
        features = features.to(device)
        mask     = mask.to(device)
        logits   = model(features, mask)
        probs    = torch.sigmoid(logits).cpu().numpy()
        labels_np = labels.numpy()

        for i, pid in enumerate(ids):
            valid = labels_np[i] != -1
            p = probs[i][valid]
            l = labels_np[i][valid].astype(int)

            all_probs.append(p)
            all_labels.append(l)

            # Per-protein AUC (skip if only one class present)
            if len(np.unique(l)) == 2:
                auc = roc_auc_score(l, p)
                f1  = f1_score(l, (p >= 0.5).astype(int), zero_division=0)
                per_protein[pid] = {'auc': round(float(auc), 4), 'f1_05': round(float(f1), 4)}
            else:
                per_protein[pid] = {'auc': None, 'f1_05': None, 'note': 'single class'}

all_probs  = np.concatenate(all_probs)
all_labels = np.concatenate(all_labels)

print(f"\nTotal residues evaluated: {len(all_labels):,}")
print(f"Binding residues: {all_labels.sum():,} ({100*all_labels.mean():.1f}%)")

# ── Global metrics at threshold 0.5 ──────────────────────────────────────────
preds_05 = (all_probs >= 0.5).astype(int)

auc_roc = roc_auc_score(all_labels, all_probs)
avg_prec = average_precision_score(all_labels, all_probs)
f1_05   = f1_score(all_labels, preds_05, zero_division=0)
prec_05 = precision_score(all_labels, preds_05, zero_division=0)
rec_05  = recall_score(all_labels, preds_05, zero_division=0)
mcc_05  = matthews_corrcoef(all_labels, preds_05)
cm      = confusion_matrix(all_labels, preds_05)

print(f"\n── Global Metrics (threshold=0.5) ──────────────────")
print(f"  AUC-ROC:          {auc_roc:.4f}")
print(f"  Avg Precision:    {avg_prec:.4f}")
print(f"  F1:               {f1_05:.4f}")
print(f"  Precision:        {prec_05:.4f}")
print(f"  Recall:           {rec_05:.4f}")
print(f"  MCC:              {mcc_05:.4f}")
print(f"  Confusion matrix:\n{cm}")

# ── Optimal threshold (maximise F1) ──────────────────────────────────────────
thresholds = np.arange(0.05, 0.95, 0.05)
f1_scores  = [f1_score(all_labels, (all_probs >= t).astype(int), zero_division=0) for t in thresholds]
best_idx   = int(np.argmax(f1_scores))
best_thresh = float(thresholds[best_idx])
best_f1     = float(f1_scores[best_idx])

preds_best = (all_probs >= best_thresh).astype(int)
prec_best  = precision_score(all_labels, preds_best, zero_division=0)
rec_best   = recall_score(all_labels, preds_best, zero_division=0)
mcc_best   = matthews_corrcoef(all_labels, preds_best)

print(f"\n── Optimal Threshold ───────────────────────────────")
print(f"  Best threshold:   {best_thresh:.2f}")
print(f"  F1:               {best_f1:.4f}")
print(f"  Precision:        {prec_best:.4f}")
print(f"  Recall:           {rec_best:.4f}")
print(f"  MCC:              {mcc_best:.4f}")

# ── Per-protein summary ───────────────────────────────────────────────────────
valid_aucs = [v['auc'] for v in per_protein.values() if v['auc'] is not None]
print(f"\n── Per-Protein AUC ({len(valid_aucs)} proteins) ─────────────────")
print(f"  Mean: {np.mean(valid_aucs):.4f}  Median: {np.median(valid_aucs):.4f}")
print(f"  Min:  {np.min(valid_aucs):.4f}  Max:    {np.max(valid_aucs):.4f}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
results = {
    'global': {
        'auc_roc': round(auc_roc, 4),
        'avg_precision': round(avg_prec, 4),
        'threshold_0.5': {
            'f1': round(f1_05, 4),
            'precision': round(prec_05, 4),
            'recall': round(rec_05, 4),
            'mcc': round(mcc_05, 4),
            'confusion_matrix': cm.tolist()
        },
        'optimal_threshold': {
            'threshold': round(best_thresh, 2),
            'f1': round(best_f1, 4),
            'precision': round(prec_best, 4),
            'recall': round(rec_best, 4),
            'mcc': round(mcc_best, 4)
        }
    },
    'per_protein_summary': {
        'n_proteins': len(valid_aucs),
        'mean_auc': round(float(np.mean(valid_aucs)), 4),
        'median_auc': round(float(np.median(valid_aucs)), 4),
        'min_auc': round(float(np.min(valid_aucs)), 4),
        'max_auc': round(float(np.max(valid_aucs)), 4)
    },
    'per_protein': per_protein
}

out_path = os.path.join(RESULTS, 'val_metrics.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved metrics → {out_path}")

# ── Plots ─────────────────────────────────────────────────────────────────────

# 1. ROC curve
fpr, tpr, _ = roc_curve(all_labels, all_probs)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, lw=2, label=f'AUC = {auc_roc:.3f}')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Validation Set)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'roc_curve.png'), dpi=150)
plt.close()

# 2. Precision-Recall curve
prec_arr, rec_arr, _ = precision_recall_curve(all_labels, all_probs)
plt.figure(figsize=(6, 5))
plt.plot(rec_arr, prec_arr, lw=2, label=f'AP = {avg_prec:.3f}')
plt.axhline(y=all_labels.mean(), color='k', linestyle='--', lw=1, label='Baseline')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Validation Set)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'pr_curve.png'), dpi=150)
plt.close()

# 3. Confusion matrix
plt.figure(figsize=(5, 4))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar()
plt.xticks([0, 1], ['Non-binding', 'Binding'])
plt.yticks([0, 1], ['Non-binding', 'Binding'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (threshold=0.5)')
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                 color='white' if cm[i, j] > cm.max() / 2 else 'black')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), dpi=150)
plt.close()

# 4. F1 vs threshold
plt.figure(figsize=(6, 5))
plt.plot(thresholds, f1_scores, lw=2)
plt.axvline(x=best_thresh, color='r', linestyle='--', label=f'Best={best_thresh:.2f} (F1={best_f1:.3f})')
plt.axvline(x=0.5, color='gray', linestyle=':', label='Default=0.5')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.title('F1 Score vs Threshold')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'f1_vs_threshold.png'), dpi=150)
plt.close()

print(f"Saved plots → {PLOTS_DIR}/")
print("\nDone.")
