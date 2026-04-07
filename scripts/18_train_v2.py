#!/usr/bin/env python3
"""
Training v2 — improved hyperparameters.

Changes from v1 (12_train_combined.py):
  - pos_weight: 11.0 → 5.0   (better precision/recall balance)
  - lr: 0.0001 → 0.00005      (more stable, trains longer)
  - patience: 10 → 15         (don't stop too early)
  - scheduler patience: 5 → 7

Saves: ../models/best_model_v6.pt
"""

import os, sys, torch, torch.nn as nn, numpy as np, random, json
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

sys.path.append(os.path.dirname(__file__))
from importlib import import_module
model_arch = import_module('06_model_architecture')

# ── Hyperparameters ────────────────────────────────────────────────────────────
POS_WEIGHT    = 5.0       # was 11.0
LR            = 0.00005   # was 0.0001
PATIENCE      = 15        # was 10
SCHED_PATIENCE= 7         # was 5
BATCH_SIZE    = 4
EPOCHS        = 100
WEIGHT_DECAY  = 0.001
SEED          = 42

# ── Paths ──────────────────────────────────────────────────────────────────────
CHEN11_DIR = '../data/combined'
HOLO4K_DIR = '../data/holo4k_combined'
MODEL_OUT  = '../models/best_model_v6.pt'
CKPT_DIR   = '../models/checkpoints_v2'
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs('../models', exist_ok=True)
os.makedirs('../results', exist_ok=True)

# ── Dataset ────────────────────────────────────────────────────────────────────
class ProteinDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        pid, data_dir = self.samples[idx]
        data = np.load(os.path.join(data_dir, f"{pid}.npz"))
        return {'features': torch.from_numpy(data['features']),
                'labels':   torch.from_numpy(data['labels'])}

def collate_fn(batch):
    features = [b['features'] for b in batch]
    labels   = [b['labels']   for b in batch]
    lengths  = torch.tensor([len(f) for f in features])
    feat_pad = pad_sequence(features, batch_first=True, padding_value=0)
    lab_pad  = pad_sequence(labels,   batch_first=True, padding_value=-1)
    mask     = torch.arange(feat_pad.size(1))[None, :] >= lengths[:, None]
    return feat_pad, lab_pad, mask

# ── Data splits ────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"\nHyperparameters:")
print(f"  pos_weight = {POS_WEIGHT}  (was 11.0)")
print(f"  lr         = {LR}       (was 0.0001)")
print(f"  patience   = {PATIENCE}         (was 10)")

with open('../data/splits/train.txt') as f:
    chen11_train = [(p, CHEN11_DIR) for p in f.read().splitlines()]
with open('../data/splits/val.txt') as f:
    chen11_val   = [(p, CHEN11_DIR) for p in f.read().splitlines()]

with open('../data/holo4k_ids.txt') as f:
    holo4k_ids = f.read().splitlines()
random.seed(SEED)
random.shuffle(holo4k_ids)
split = int(len(holo4k_ids) * 0.8)
holo4k_train = [(p, HOLO4K_DIR) for p in holo4k_ids[:split]]
holo4k_val   = [(p, HOLO4K_DIR) for p in holo4k_ids[split:]]

train_samples = chen11_train + holo4k_train
val_samples   = chen11_val   + holo4k_val

print(f"\nTrain: {len(chen11_train)} CHEN11 + {len(holo4k_train)} HOLO4K = {len(train_samples)}")
print(f"Val:   {len(chen11_val)} CHEN11 + {len(holo4k_val)} HOLO4K = {len(val_samples)}")

train_loader = DataLoader(ProteinDataset(train_samples), batch_size=BATCH_SIZE,
                          shuffle=True,  collate_fn=collate_fn, num_workers=2)
val_loader   = DataLoader(ProteinDataset(val_samples),   batch_size=BATCH_SIZE,
                          shuffle=False, collate_fn=collate_fn, num_workers=2)

# ── Model ──────────────────────────────────────────────────────────────────────
model     = model_arch.BindingSitePredictor().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=SCHED_PATIENCE
)
criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([POS_WEIGHT]).to(device), reduction='none'
)
scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"\n{'='*65}")
print(f"{'Epoch':>6} {'Train':>10} {'Val':>10} {'AUC':>8} {'F1@0.5':>8} {'Prec':>8} {'Rec':>8}")
print(f"{'='*65}")

# ── Training loop ──────────────────────────────────────────────────────────────
best_val_loss = float('inf')
best_val_auc  = 0.0
no_improve    = 0
history       = []

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0.0
    for features, labels, mask in train_loader:
        features, labels, mask = features.to(device), labels.to(device), mask.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits    = model(features, mask)
            loss_mask = (labels != -1)
            loss      = (criterion(logits, labels.float()) * loss_mask).sum() / loss_mask.sum()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

    # Validate
    model.eval()
    val_loss   = 0.0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for features, labels, mask in val_loader:
            features, labels, mask = features.to(device), labels.to(device), mask.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits    = model(features, mask)
                loss_mask = (labels != -1)
                loss      = (criterion(logits, labels.float()) * loss_mask).sum() / loss_mask.sum()
            val_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            lbls  = labels.cpu().numpy()
            valid = lbls != -1
            all_probs.append(probs[valid])
            all_labels.append(lbls[valid].astype(int))

    train_loss /= len(train_loader)
    val_loss   /= len(val_loader)
    probs_all   = np.concatenate(all_probs)
    labels_all  = np.concatenate(all_labels)
    preds_all   = (probs_all >= 0.5).astype(int)

    try:
        val_auc  = roc_auc_score(labels_all, probs_all)
        val_f1   = f1_score(labels_all, preds_all, zero_division=0)
        val_prec = precision_score(labels_all, preds_all, zero_division=0)
        val_rec  = recall_score(labels_all, preds_all, zero_division=0)
    except Exception:
        val_auc = val_f1 = val_prec = val_rec = 0.0

    marker = " ✓" if val_loss < best_val_loss else ""
    print(f"{epoch+1:>6} {train_loss:>10.4f} {val_loss:>10.4f} {val_auc:>8.4f} "
          f"{val_f1:>8.4f} {val_prec:>8.4f} {val_rec:>8.4f}{marker}")

    history.append({'epoch': epoch+1, 'train_loss': round(train_loss,4),
                    'val_loss': round(val_loss,4), 'val_auc': round(val_auc,4),
                    'val_f1': round(val_f1,4), 'val_prec': round(val_prec,4),
                    'val_rec': round(val_rec,4)})

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_auc  = val_auc
        no_improve    = 0
        torch.save(model.state_dict(), MODEL_OUT)
    else:
        no_improve += 1

    if (epoch + 1) % 10 == 0:
        torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                    'val_loss': val_loss, 'val_auc': val_auc},
                   f'{CKPT_DIR}/epoch_{epoch+1}.pt')

    if no_improve >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
        break

print(f"\n{'='*65}")
print(f"Training complete.")
print(f"  Best val loss: {best_val_loss:.4f}")
print(f"  Best val AUC:  {best_val_auc:.4f}")
print(f"  Saved → {MODEL_OUT}")

with open('../results/training_history_v2.json', 'w') as f:
    json.dump(history, f, indent=2)
print(f"  History → ../results/training_history_v2.json")
