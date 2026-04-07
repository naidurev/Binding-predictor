#!/usr/bin/env python3
"""
Preprocess COACH420 benchmark dataset.
Extracts labels, geometric features, and ESM-2 embeddings,
then combines them into .npz files ready for model inference.

Output: ../data/coach420_combined/{protein_id}.npz
        ../data/coach420_ids.txt
"""

import os
import sys
import torch
import numpy as np
from Bio.PDB import PDBParser, NeighborSearch, PPBuilder
from transformers import AutoTokenizer, EsmModel

COACH420_DIR = '/content/data/coach420'
OUT_DIR      = '/content/data/coach420_combined'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load ESM-2 once ───────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print("Loading ESM-2 model...")
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
esm_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)
esm_model.eval()
print("ESM-2 loaded.\n")

# ── Helpers ───────────────────────────────────────────────────────────────────
parser  = PDBParser(QUIET=True)
builder = PPBuilder()

AMINO_ACIDS = set([
    'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'
])

def extract_geometric(residue):
    """5 geometric features: center x,y,z, volume, atom count."""
    atoms = list(residue.get_atoms())
    if not atoms:
        return None
    coords = np.array([a.coord for a in atoms])
    center = coords.mean(axis=0)
    ranges = coords.max(axis=0) - coords.min(axis=0)
    volume = float(np.prod(ranges + 1e-6))
    return np.array([center[0], center[1], center[2], volume, len(atoms)], dtype=np.float32)

def get_esm_embedding(sequence):
    """Returns (seq_len, 1280) ESM-2 embedding."""
    seq = sequence[:1022]  # ESM max length (1024 - 2 special tokens)
    inputs = tokenizer(seq, return_tensors='pt', add_special_tokens=True).to(device)
    with torch.no_grad():
        out = esm_model(**inputs)
    emb = out.last_hidden_state[0, 1:-1].cpu().numpy().astype(np.float32)
    return emb

# ── Identify the 420 correct protein PDB files ────────────────────────────────
# The coach420 zip was extracted with a flat structure where Windows paths
# became literal filenames (e.g. 'coach420\p2rank\148lE.pdb_predictions.csv').
# We use p2rank CSV filenames as the authoritative list of the 420 protein IDs.
all_files = os.listdir(COACH420_DIR)

# Find p2rank CSV files — they contain the official 420 protein IDs
p2rank_csvs = [f for f in all_files
               if 'p2rank' in f and f.endswith('.pdb_predictions.csv')]
print(f"Found {len(p2rank_csvs)} p2rank CSV files.")

if p2rank_csvs:
    # Extract protein ID: handles both 'coach420\p2rank\148lE.pdb_predictions.csv'
    # (Windows-flattened) and plain '148lE.pdb_predictions.csv'
    protein_id_set = set()
    for f in p2rank_csvs:
        # Take last component after any path separator (\ or /)
        basename = f.replace('\\', '/').split('/')[-1]
        pid = basename.replace('.pdb_predictions.csv', '')
        protein_id_set.add(pid)
    print(f"Extracted {len(protein_id_set)} unique protein IDs from p2rank CSVs.\n")

    # Build list of PDB filenames that match those protein IDs exactly
    # Root-level PDB files are either 'coach420\148lE.pdb' (flattened) or '148lE.pdb'
    pdb_map = {}  # protein_id → actual filename in COACH420_DIR
    for f in all_files:
        if not f.endswith('.pdb'):
            continue
        basename = f.replace('\\', '/').split('/')[-1]
        pid = basename[:-4]  # strip .pdb
        if pid in protein_id_set:
            pdb_map[pid] = f

    pdb_files = sorted(pdb_map.values())
    print(f"Matched {len(pdb_files)} PDB files to p2rank protein IDs.\n")
else:
    # Fallback: use all PDB files that look like root-level ones (no path separators)
    pdb_files = sorted([f for f in all_files
                        if f.endswith('.pdb') and '\\' not in f and '/' not in f])
    print(f"No p2rank CSVs found. Using {len(pdb_files)} root-level PDB files.\n")
    pdb_map = {f[:-4]: f for f in pdb_files}

print(f"Processing {len(pdb_files)} PDB files.\n")

processed, skipped = 0, 0
protein_ids = []

for idx, pdb_file in enumerate(pdb_files):
    # protein_id is the key without .pdb; pdb_file is the actual filename in dir
    basename   = pdb_file.replace('\\', '/').split('/')[-1]
    protein_id = basename[:-4]
    out_path   = os.path.join(OUT_DIR, f"{protein_id}.npz")

    if os.path.exists(out_path):
        protein_ids.append(protein_id)
        processed += 1
        continue

    pdb_path  = os.path.join(COACH420_DIR, pdb_file)
    structure = parser.get_structure(protein_id, pdb_path)

    # Collect residues and atoms
    residues_list  = []
    protein_atoms  = []
    ligand_atoms   = []

    for model in structure:
        for chain in model:
            for residue in chain:
                hetflag = residue.id[0]
                if hetflag == ' ' and residue.resname in AMINO_ACIDS:
                    residues_list.append(residue)
                    protein_atoms.extend(list(residue.get_atoms()))
                elif hetflag.startswith('H_'):
                    ligand_atoms.extend(list(residue.get_atoms()))

    if not residues_list:
        print(f"  Skip {protein_id}: no standard residues")
        skipped += 1
        continue

    if not ligand_atoms:
        print(f"  Skip {protein_id}: no ligand atoms (no binding site labels possible)")
        skipped += 1
        continue

    # Labels: 1 if residue within 4Å of any ligand atom
    ns = NeighborSearch(protein_atoms)
    binding_set = set()
    for lig_atom in ligand_atoms:
        for res in ns.search(lig_atom.coord, 4.0, level='R'):
            binding_set.add(id(res))

    labels = np.array(
        [1 if id(r) in binding_set else 0 for r in residues_list],
        dtype=np.int8
    )

    # Geometric features
    geom_feats = []
    valid_mask  = []
    for r in residues_list:
        g = extract_geometric(r)
        if g is not None:
            geom_feats.append(g)
            valid_mask.append(True)
        else:
            valid_mask.append(False)

    if not all(valid_mask):
        # Filter out residues with no atoms
        residues_list = [r for r, v in zip(residues_list, valid_mask) if v]
        labels        = labels[valid_mask]

    geom_feats = np.array(geom_feats, dtype=np.float32)  # (N, 5)

    # Sequence for ESM
    seq_parts = []
    for pp in builder.build_peptides(structure, aa_only=True):
        seq_parts.append(str(pp.get_sequence()))
    sequence = ''.join(seq_parts)

    if len(sequence) == 0:
        print(f"  Skip {protein_id}: empty sequence")
        skipped += 1
        continue

    # ESM embedding
    try:
        emb = get_esm_embedding(sequence)
    except Exception as e:
        print(f"  Skip {protein_id}: ESM error - {e}")
        skipped += 1
        continue

    # Truncate to ESM max length if needed
    n = min(len(residues_list), len(geom_feats), len(emb), len(labels))
    if n == 0:
        print(f"  Skip {protein_id}: zero length after truncation")
        skipped += 1
        continue

    geom_feats = geom_feats[:n]
    emb        = emb[:n]
    labels     = labels[:n]

    features = np.concatenate([geom_feats, emb], axis=1)  # (N, 1285)

    np.savez_compressed(out_path, features=features, labels=labels)
    protein_ids.append(protein_id)
    processed += 1

    if (idx + 1) % 20 == 0:
        print(f"[{idx+1}/{len(pdb_files)}] Processed: {processed}, Skipped: {skipped}")

# Save protein ID list
ids_path = '/content/data/coach420_ids.txt'
with open(ids_path, 'w') as f:
    f.write('\n'.join(protein_ids))

print(f"\nDone. Processed: {processed}, Skipped: {skipped}")
print(f"IDs saved → {ids_path}")
print(f"Data saved → {OUT_DIR}")
