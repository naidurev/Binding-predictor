#!/usr/bin/env python3
"""
Preprocess HOLO4K training dataset.
Extracts labels, geometric features, and ESM-2 embeddings,
then combines them into .npz files ready for model training.

Output: ../data/holo4k_combined/{protein_id}.npz
        ../data/holo4k_ids.txt
"""

import os
import sys
import torch
import numpy as np
from Bio.PDB import PDBParser, NeighborSearch, PPBuilder
from transformers import AutoTokenizer, EsmModel

HOLO4K_DIR = '../data/holo4k'
OUT_DIR    = '../data/holo4k_combined'
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
    atoms = list(residue.get_atoms())
    if not atoms:
        return None
    coords = np.array([a.coord for a in atoms])
    center = coords.mean(axis=0)
    ranges = coords.max(axis=0) - coords.min(axis=0)
    volume = float(np.prod(ranges + 1e-6))
    return np.array([center[0], center[1], center[2], volume, len(atoms)], dtype=np.float32)

def get_esm_embedding(sequence):
    seq = sequence[:1022]
    inputs = tokenizer(seq, return_tensors='pt', add_special_tokens=True).to(device)
    with torch.no_grad():
        out = esm_model(**inputs)
    emb = out.last_hidden_state[0, 1:-1].cpu().numpy().astype(np.float32)
    return emb

# ── Process each PDB ──────────────────────────────────────────────────────────
pdb_files = sorted([f for f in os.listdir(HOLO4K_DIR) if f.endswith('.pdb')])
print(f"Found {len(pdb_files)} PDB files.\n")

processed, skipped = 0, 0
protein_ids = []

for idx, pdb_file in enumerate(pdb_files):
    protein_id = pdb_file[:-4]
    out_path   = os.path.join(OUT_DIR, f"{protein_id}.npz")

    if os.path.exists(out_path):
        protein_ids.append(protein_id)
        processed += 1
        continue

    pdb_path  = os.path.join(HOLO4K_DIR, pdb_file)
    structure = parser.get_structure(protein_id, pdb_path)

    residues_list = []
    protein_atoms = []
    ligand_atoms  = []

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
        skipped += 1
        continue

    if not ligand_atoms:
        skipped += 1
        continue

    ns = NeighborSearch(protein_atoms)
    binding_set = set()
    for lig_atom in ligand_atoms:
        for res in ns.search(lig_atom.coord, 4.0, level='R'):
            binding_set.add(id(res))

    labels = np.array(
        [1 if id(r) in binding_set else 0 for r in residues_list],
        dtype=np.int8
    )

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
        residues_list = [r for r, v in zip(residues_list, valid_mask) if v]
        labels        = labels[np.array(valid_mask)]

    geom_feats = np.array(geom_feats, dtype=np.float32)

    seq_parts = []
    for pp in builder.build_peptides(structure, aa_only=True):
        seq_parts.append(str(pp.get_sequence()))
    sequence = ''.join(seq_parts)

    if len(sequence) == 0:
        skipped += 1
        continue

    try:
        emb = get_esm_embedding(sequence)
    except Exception as e:
        print(f"  Skip {protein_id}: ESM error - {e}")
        skipped += 1
        continue

    n = min(len(residues_list), len(geom_feats), len(emb), len(labels))
    if n == 0:
        skipped += 1
        continue

    geom_feats = geom_feats[:n]
    emb        = emb[:n]
    labels     = labels[:n]

    features = np.concatenate([geom_feats, emb], axis=1)  # (N, 1285)

    np.savez_compressed(out_path, features=features, labels=labels)
    protein_ids.append(protein_id)
    processed += 1

    if (idx + 1) % 50 == 0:
        print(f"[{idx+1}/{len(pdb_files)}] Processed: {processed}, Skipped: {skipped}")

ids_path = '../data/holo4k_ids.txt'
with open(ids_path, 'w') as f:
    f.write('\n'.join(protein_ids))

print(f"\nDone. Processed: {processed}, Skipped: {skipped}")
print(f"IDs saved → {ids_path}")
print(f"Data saved → {OUT_DIR}")
