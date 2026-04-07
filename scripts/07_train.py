#!/usr/bin/env python3

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import roc_auc_score
import json

sys.path.append(os.path.dirname(__file__))
from importlib import import_module
model_arch = import_module('06_model_architecture')

class ProteinDataset(Dataset):
    def __init__(self, protein_ids, data_dir):
        self.protein_ids = protein_ids
        self.data_dir = data_dir

    def __len__(self):
        return len(self.protein_ids)

    def __getitem__(self, idx):
        protein_id = self.protein_ids[idx]
        data = np.load(os.path.join(self.data_dir, f"{protein_id}.npz"))
        return {
            'features': torch.from_numpy(data['features']),
            'labels': torch.from_numpy(data['labels'])
        }

def collate_fn(batch):
    features = [item['features'] for item in batch]
    labels   = [item['labels']   for item in batch]
    lengths  = torch.tensor([len(f) for f in features])

    features_padded = pad_sequence(features, batch_first=True, padding_value=0)
    labels_padded   = pad_sequence(labels,   batch_first=True, padding_value=-1)

    max_len = features_padded.size(1)
    mask = torch.arange(max_len)[None, :] >= lengths[:, None]

    return features_padded, labels_padded, mask

# ── Setup ─────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

with open('../data/splits/train.txt') as f:
    train_ids = f.read().splitlines()
with open('../data/splits/val.txt') as f:
    val_ids = f.read().splitlines()

train_dataset = ProteinDataset(train_ids, '../data/combined')
val_dataset   = ProteinDataset(val_ids,   '../data/combined')

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,  collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset,   batch_size=4, shuffle=False, collate_fn=collate_fn)

# ── Compute pos_weight from actual training data ──────────────────────────────
print("Computing class weights from training data...")
n_pos, n_neg = 0, 0
for pid in train_ids:
    path = f'../data/combined/{pid}.npz'
    if os.path.exists(path):
        labels = np.load(path)['labels']
        n_pos += int(labels.sum())
        n_neg += int((labels == 0).sum())
pos_weight = n_neg / max(n_pos, 1)
print(f"  Positives: {n_pos:,}  Negatives: {n_neg:,}  pos_weight: {pos_weight:.1f}")

# ── Model, optimizer, scheduler ──────────────────────────────────────────────
model     = model_arch.BindingSitePredictor().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([pos_weight]).to(device), reduction='none'
)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

os.makedirs('../models/checkpoints', exist_ok=True)

# ── Training loop ─────────────────────────────────────────────────────────────
best_val_loss  = float('inf')
patience       = 10
no_improve     = 0
epochs         = 100
history        = []

for epoch in range(epochs):
    # — Train —
    model.train()
    train_loss = 0

    for features, labels, mask in train_loader:
        features, labels, mask = features.to(device), labels.to(device), mask.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            logits    = model(features, mask)
            loss_mask = (labels != -1)
            loss      = criterion(logits, labels.float())
            loss      = (loss * loss_mask).sum() / loss_mask.sum()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    # — Validate —
    model.eval()
    val_loss   = 0
    val_probs  = []
    val_labels = []

    with torch.no_grad():
        for features, labels, mask in val_loader:
            features, labels, mask = features.to(device), labels.to(device), mask.to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                logits    = model(features, mask)
                loss_mask = (labels != -1)
                loss      = criterion(logits, labels.float())
                loss      = (loss * loss_mask).sum() / loss_mask.sum()

            val_loss += loss.item()

            probs  = torch.sigmoid(logits).cpu().numpy()
            lbls   = labels.cpu().numpy()
            valid  = lbls != -1
            val_probs.append(probs[valid])
            val_labels.append(lbls[valid].astype(int))

    train_loss /= len(train_loader)
    val_loss   /= len(val_loader)

    val_probs_all  = np.concatenate(val_probs)
    val_labels_all = np.concatenate(val_labels)
    try:
        val_auc = roc_auc_score(val_labels_all, val_probs_all)
    except Exception:
        val_auc = 0.0

    print(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, AUC: {val_auc:.4f}")

    history.append({
        'epoch': epoch + 1,
        'train_loss': round(train_loss, 4),
        'val_loss': round(val_loss, 4),
        'val_auc': round(val_auc, 4)
    })

    scheduler.step(val_loss)

    # — Save best model —
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve    = 0
        torch.save(model.state_dict(), '../models/best_model.pt')
        print(f"  ✓ Best model saved (val_loss={best_val_loss:.4f})")
    else:
        no_improve += 1

    # — Checkpoint every 10 epochs —
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_auc': val_auc
        }, f'../models/checkpoints/epoch_{epoch+1}.pt')

    # — Early stopping —
    if no_improve >= patience:
        print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
        break

# ── Save training history ─────────────────────────────────────────────────────
os.makedirs('../results', exist_ok=True)
with open('../results/training_history.json', 'w') as f:
    json.dump(history, f, indent=2)

print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
print(f"Training history saved → ../results/training_history.json")
