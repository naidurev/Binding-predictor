#!/usr/bin/env python3
"""
Script 13: Compute DSSP features for all proteins.
Outputs 6 features per residue:
  is_helix, is_sheet, is_loop, rsa, phi_norm, psi_norm
Saves to data/dssp/{protein_id}.npy
"""

import os
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

# ── Paths ────────────────────────────────────────────────────────────────────
COMBINED_DIR      = '../data/combined'
HOLO4K_DIR        = '../data/holo4k_combined'
COACH420_DIR      = '../data/coach420_combined'

CHEN11_PDB_DIR    = '../data/chen11'
HOLO4K_PDB_DIR    = '../data/holo4k'
COACH420_PDB_DIR  = '../data/coach420'

DSSP_OUT_DIR      = '../data/dssp'
os.makedirs(DSSP_OUT_DIR, exist_ok=True)

# Max RSA values per residue type (Sander & Rost 1994) for normalization
MAX_RSA = {
    'ALA': 106, 'ARG': 248, 'ASN': 157, 'ASP': 163, 'CYS': 135,
    'GLN': 198, 'GLU': 194, 'GLY':  84, 'HIS': 184, 'ILE': 169,
    'LEU': 164, 'LYS': 205, 'MET': 188, 'PHE': 197, 'PRO': 136,
    'SER': 130, 'THR': 142, 'TRP': 227, 'TYR': 222, 'VAL': 142,
}
DEFAULT_MAX_RSA = 180  # fallback for non-standard residues

parser = PDBParser(QUIET=True)

def ss_to_onehot(ss_char):
    """Convert DSSP secondary structure character to [is_helix, is_sheet, is_loop]."""
    if ss_char in ('H', 'G', 'I'):   # helix types
        return [1, 0, 0]
    elif ss_char in ('E', 'B'):       # sheet types
        return [0, 1, 0]
    else:                              # loop / coil / turn
        return [0, 0, 1]

def normalize_angle(angle):
    """Normalize angle from [-180, 180] to [-1, 1]. Handle 360.0 (missing)."""
    if angle is None or abs(angle) > 360:
        return 0.0
    return float(angle) / 180.0

CRYST1_DUMMY = "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n"

def compute_dssp_features(pdb_path, protein_id):
    """
    Returns numpy array of shape (num_residues, 6):
    [is_helix, is_sheet, is_loop, rsa, phi_norm, psi_norm]
    Returns None if DSSP fails.
    """
    import tempfile
    try:
        # mkdssp requires a CRYST1 record; p2rank PDB files start with ATOM.
        # Write a temp file with a dummy CRYST1 prepended if needed.
        with open(pdb_path, 'r') as f:
            content = f.read()
        # Keep only ATOM/TER/END lines + add CRYST1 if missing.
        # Stripping HETATM avoids mkdssp mmCIF errors on blank chain IDs.
        clean_lines = [l for l in content.splitlines(keepends=True)
                       if l.startswith(('ATOM  ', 'TER', 'END', 'CRYST1'))]
        clean_content = ''.join(clean_lines)
        if not clean_content.startswith('CRYST1'):
            clean_content = CRYST1_DUMMY + clean_content
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False)
        tmp.write(clean_content)
        tmp.close()
        dssp_input = tmp.name

        structure = parser.get_structure(protein_id, dssp_input)
        model = structure[0]
        dssp = DSSP(model, dssp_input, dssp='mkdssp')

        if dssp_input != pdb_path:
            import os as _os
            _os.unlink(dssp_input)
    except Exception as e:
        print(f"  DSSP failed for {protein_id}: {e}")
        return None

    features = []
    for key in dssp.keys():
        d = dssp[key]
        aa       = d[1]   # amino acid
        ss       = d[2]   # secondary structure character
        asa      = d[3]   # accessible surface area (absolute)
        phi      = d[4]   # phi angle
        psi      = d[5]   # psi angle

        onehot = ss_to_onehot(ss)
        max_asa = MAX_RSA.get(aa, DEFAULT_MAX_RSA)
        rsa = min(float(asa) / max_asa, 1.0) if max_asa > 0 else 0.0
        phi_n = normalize_angle(phi)
        psi_n = normalize_angle(psi)

        features.append(onehot + [rsa, phi_n, psi_n])

    return np.array(features, dtype=np.float32)  # (N, 6)


def find_pdb(protein_id, dataset):
    """Return PDB file path for a given protein ID and dataset."""
    if dataset == 'chen11':
        # ID format: 'a.001.001.001_1s69a' → PDB = 'a.001.001.001_1s69a.pdb'
        return os.path.join(CHEN11_PDB_DIR, protein_id + '.pdb')
    elif dataset == 'holo4k':
        return os.path.join(HOLO4K_PDB_DIR, protein_id + '.pdb')
    elif dataset == 'coach420':
        return os.path.join(COACH420_PDB_DIR, protein_id + '.pdb')
    return None


def process_dataset(combined_dir, dataset_name):
    npz_files = sorted([f for f in os.listdir(combined_dir) if f.endswith('.npz')])
    processed, skipped = 0, 0

    for i, fname in enumerate(npz_files):
        protein_id = fname.replace('.npz', '')
        out_path = os.path.join(DSSP_OUT_DIR, protein_id + '.npy')

        if os.path.exists(out_path):
            processed += 1
            continue

        pdb_path = find_pdb(protein_id, dataset_name)
        if pdb_path is None or not os.path.exists(pdb_path):
            print(f"  PDB not found: {pdb_path}")
            skipped += 1
            continue

        # Load existing features to get expected residue count
        data = np.load(os.path.join(combined_dir, fname))
        expected_n = data['features'].shape[0]

        dssp_feats = compute_dssp_features(pdb_path, protein_id)
        if dssp_feats is None:
            skipped += 1
            continue

        # DSSP may skip HETATMs or non-standard residues — align lengths
        if dssp_feats.shape[0] != expected_n:
            # Pad or truncate to match
            if dssp_feats.shape[0] > expected_n:
                dssp_feats = dssp_feats[:expected_n]
            else:
                pad = np.zeros((expected_n - dssp_feats.shape[0], 6), dtype=np.float32)
                dssp_feats = np.vstack([dssp_feats, pad])

        np.save(out_path, dssp_feats)
        processed += 1

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(npz_files)}] Processed: {processed}, Skipped: {skipped}")

    print(f"  Done — Processed: {processed}, Skipped: {skipped}")
    return processed, skipped


if __name__ == '__main__':
    print("Computing DSSP features...\n")

    print(f"CHEN11 ({len([f for f in os.listdir(COMBINED_DIR) if f.endswith('.npz')])} proteins):")
    process_dataset(COMBINED_DIR, 'chen11')

    print(f"\nHOLO4K ({len([f for f in os.listdir(HOLO4K_DIR) if f.endswith('.npz')])} proteins):")
    process_dataset(HOLO4K_DIR, 'holo4k')

    print(f"\nCOACH420 ({len([f for f in os.listdir(COACH420_DIR) if f.endswith('.npz')])} proteins):")
    process_dataset(COACH420_DIR, 'coach420')

    print(f"\nAll DSSP features saved to {DSSP_OUT_DIR}")
