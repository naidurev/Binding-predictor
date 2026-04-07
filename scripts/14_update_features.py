#!/usr/bin/env python3
"""
Script 14: Merge DSSP features into existing .npz files.
Updates features from shape (N, 1285) → (N, 1291).
Saves updated .npz files back to the same directories.
"""

import os
import numpy as np

COMBINED_DIR    = '../data/combined'
HOLO4K_DIR      = '../data/holo4k_combined'
COACH420_DIR    = '../data/coach420_combined'
DSSP_DIR        = '../data/dssp'

def update_dataset(combined_dir, label):
    npz_files = sorted([f for f in os.listdir(combined_dir) if f.endswith('.npz')])
    updated, skipped = 0, 0

    for fname in npz_files:
        protein_id = fname.replace('.npz', '')
        npz_path  = os.path.join(combined_dir, fname)
        dssp_path = os.path.join(DSSP_DIR, protein_id + '.npy')

        if not os.path.exists(dssp_path):
            skipped += 1
            continue

        data = np.load(npz_path)
        features = data['features']   # (N, 1285)
        labels   = data['labels']     # (N,)

        # Already updated?
        if features.shape[1] == 1291:
            updated += 1
            continue

        dssp_feats = np.load(dssp_path)  # (N, 6)

        if dssp_feats.shape[0] != features.shape[0]:
            skipped += 1
            continue

        new_features = np.concatenate([features, dssp_feats], axis=1)  # (N, 1291)
        np.savez_compressed(npz_path, features=new_features, labels=labels)
        updated += 1

    print(f"  {label}: Updated {updated}, Skipped {skipped} (no DSSP file)")
    return updated, skipped


if __name__ == '__main__':
    print("Merging DSSP features into .npz files...\n")

    update_dataset(COMBINED_DIR,  'CHEN11')
    update_dataset(HOLO4K_DIR,    'HOLO4K')
    update_dataset(COACH420_DIR,  'COACH420')

    # Verify
    sample = sorted([f for f in os.listdir(COMBINED_DIR) if f.endswith('.npz')])[0]
    d = np.load(os.path.join(COMBINED_DIR, sample))
    print(f"\nVerification — sample feature shape: {d['features'].shape}")
    print("Expected: (N, 1291)")
    print("Done.")
