#!/usr/bin/env python3
"""
Compute DSSP features for COACH420 proteins only.
Calls mkdssp directly via subprocess (works with mkdssp 4.x).
Saves .npy files to data/dssp/{protein_id}.npy
"""

import os, sys, subprocess, tempfile, numpy as np
from Bio.PDB import PDBParser

BASE         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COACH420_NPZ = os.path.join(BASE, 'data', 'coach420_combined')
COACH420_PDB = os.path.join(BASE, 'data', 'coach420')
DSSP_OUT     = os.path.join(BASE, 'data', 'dssp')
os.makedirs(DSSP_OUT, exist_ok=True)

MAX_RSA = {
    'ALA':106,'ARG':248,'ASN':157,'ASP':163,'CYS':135,
    'GLN':198,'GLU':194,'GLY':84, 'HIS':184,'ILE':169,
    'LEU':164,'LYS':205,'MET':188,'PHE':197,'PRO':136,
    'SER':130,'THR':142,'TRP':227,'TYR':222,'VAL':142,
}
AA3TO1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
    'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
}
DEFAULT_MAX_RSA = 180
HEADER_DUMMY = "HEADER                                            01-JAN-00   XXXX              \n"
CRYST1_DUMMY = "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n"

def ss_to_onehot(c):
    if c in ('H','G','I'): return [1,0,0]
    if c in ('E','B'):     return [0,1,0]
    return [0,0,1]

def norm_angle(a):
    if a is None or abs(a) > 360: return 0.0
    return float(a) / 180.0

def parse_dssp_file(dssp_path):
    """Parse classic DSSP output file → list of (aa, ss, asa, phi, psi)."""
    residues = []
    in_residues = False
    with open(dssp_path) as f:
        for line in f:
            if line.startswith('  #  RESIDUE'):
                in_residues = True
                continue
            if not in_residues:
                continue
            if len(line) < 38:
                continue
            # Skip chain breaks
            if line[13] == '!':
                continue
            aa  = line[13]
            ss  = line[16] if line[16].strip() else 'C'
            try:
                asa = float(line[34:38].strip())
            except ValueError:
                asa = 0.0
            try:
                phi = float(line[103:109].strip())
            except (ValueError, IndexError):
                phi = 360.0
            try:
                psi = float(line[109:115].strip())
            except (ValueError, IndexError):
                psi = 360.0
            residues.append((aa, ss, asa, phi, psi))
    return residues

def compute_dssp(pdb_path, protein_id, expected_n):
    # Write clean temp PDB (ATOM only + CRYST1)
    with open(pdb_path) as f:
        content = f.read()
    clean = [l for l in content.splitlines(keepends=True)
             if l.startswith(('ATOM  ','TER','END','CRYST1'))]
    clean_str = ''.join(clean)
    if not clean_str.startswith('CRYST1'):
        clean_str = CRYST1_DUMMY + clean_str
    clean_str = HEADER_DUMMY + clean_str

    tmp_pdb  = tempfile.NamedTemporaryFile(suffix='.pdb',  delete=False, mode='w')
    tmp_dssp = tempfile.NamedTemporaryFile(suffix='.dssp', delete=False)
    tmp_pdb.write(clean_str)
    tmp_pdb.close()
    tmp_dssp.close()

    try:
        result = subprocess.run(
            ['mkdssp', tmp_pdb.name, tmp_dssp.name],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            print(f"  mkdssp error for {protein_id}: {result.stderr.strip()[:200]}")
            return None

        residues = parse_dssp_file(tmp_dssp.name)
    except Exception as e:
        print(f"  DSSP failed for {protein_id}: {e}")
        return None
    finally:
        os.unlink(tmp_pdb.name)
        try: os.unlink(tmp_dssp.name)
        except: pass

    if not residues:
        print(f"  DSSP produced no residues for {protein_id}")
        return None

    features = []
    for aa, ss, asa, phi, psi in residues:
        # Convert 1-letter to 3-letter for MAX_RSA lookup
        aa3 = next((k for k,v in AA3TO1.items() if v == aa), None)
        max_asa = MAX_RSA.get(aa3, DEFAULT_MAX_RSA) if aa3 else DEFAULT_MAX_RSA
        rsa = min(asa / max_asa, 1.0) if max_asa > 0 else 0.0
        features.append(ss_to_onehot(ss) + [rsa, norm_angle(phi), norm_angle(psi)])

    feats = np.array(features, dtype=np.float32)

    # Align to expected length
    if feats.shape[0] > expected_n:
        feats = feats[:expected_n]
    elif feats.shape[0] < expected_n:
        pad = np.zeros((expected_n - feats.shape[0], 6), dtype=np.float32)
        feats = np.vstack([feats, pad])

    return feats


# ── Main ───────────────────────────────────────────────────────────────────────
npz_files = sorted(f for f in os.listdir(COACH420_NPZ) if f.endswith('.npz'))
print(f"Processing {len(npz_files)} COACH420 proteins...\n")

processed = skipped = already_done = 0

for i, fname in enumerate(npz_files):
    pid      = fname.replace('.npz', '')
    out_path = os.path.join(DSSP_OUT, pid + '.npy')

    if os.path.exists(out_path):
        already_done += 1
        continue

    pdb_path = os.path.join(COACH420_PDB, pid + '.pdb')
    if not os.path.exists(pdb_path):
        print(f"  PDB not found: {pid}")
        skipped += 1
        continue

    data = np.load(os.path.join(COACH420_NPZ, fname))
    expected_n = data['features'].shape[0]

    feats = compute_dssp(pdb_path, pid, expected_n)
    if feats is None:
        skipped += 1
        continue

    np.save(out_path, feats)
    processed += 1

    if (i + 1) % 50 == 0:
        print(f"  [{i+1}/{len(npz_files)}] processed={processed} skipped={skipped} already={already_done}")

print(f"\nDone — Processed: {processed}, Skipped: {skipped}, Already done: {already_done}")
print(f"DSSP files saved to: {DSSP_OUT}")
