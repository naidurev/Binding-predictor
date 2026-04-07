#!/usr/bin/env python3

import os
import numpy as np
from sklearn.model_selection import train_test_split

labels_dir = "../data/labels"
geometric_dir = "../data/geometric"
embeddings_dir = "../data/embeddings"
output_dir = "../data/combined"
splits_dir = "../data/splits"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(splits_dir, exist_ok=True)

label_files = sorted(os.listdir(labels_dir))
protein_ids = []

for label_file in label_files:
    protein_id = label_file.replace('.npy', '')
    emb_file = f"{protein_id}A.pdb.seq.npy"
    
    if not os.path.exists(os.path.join(embeddings_dir, emb_file)):
        emb_file = f"{protein_id}B.pdb.seq.npy"
    if not os.path.exists(os.path.join(embeddings_dir, emb_file)):
        emb_file = f"{protein_id}C.pdb.seq.npy"
    if not os.path.exists(os.path.join(embeddings_dir, emb_file)):
        print(f"Skip {protein_id}: no embedding")
        continue
    
    labels = np.load(os.path.join(labels_dir, label_file))
    geometric = np.load(os.path.join(geometric_dir, f"{protein_id}.npy"))
    embeddings = np.load(os.path.join(embeddings_dir, emb_file))
    
    if not (len(labels) == len(geometric) == len(embeddings)):
        print(f"Skip {protein_id}: length mismatch {len(labels)} {len(geometric)} {len(embeddings)}")
        continue
    
    combined = np.concatenate([geometric, embeddings], axis=1)
    
    np.savez_compressed(
        os.path.join(output_dir, f"{protein_id}.npz"),
        features=combined.astype(np.float32),
        labels=labels.astype(np.int8)
    )
    
    protein_ids.append(protein_id)

train_ids, val_ids = train_test_split(protein_ids, test_size=0.2, random_state=42)

with open(os.path.join(splits_dir, 'train.txt'), 'w') as f:
    f.write('\n'.join(train_ids))

with open(os.path.join(splits_dir, 'val.txt'), 'w') as f:
    f.write('\n'.join(val_ids))

print(f"Combined: {len(protein_ids)} proteins")
print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")
print(f"Output: {output_dir}")
