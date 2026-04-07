#!/usr/bin/env python3
"""
Inference script: PDB file → per-residue binding scores

Usage:
    python3 16_inference.py path/to/protein.pdb [--threshold 0.5] [--output results.csv]

Output CSV columns:
    chain, resnum, resname, score, prediction
"""

import os, sys, argparse, subprocess, tempfile, csv
import numpy as np
import torch
from Bio.PDB import PDBParser
from importlib import import_module

# ── Config ─────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, 'models', 'best_model_v5.pt')

AA3TO1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
    'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
}
MAX_RSA = {
    'ALA':106,'ARG':248,'ASN':157,'ASP':163,'CYS':135,'GLN':198,'GLU':194,
    'GLY':84, 'HIS':184,'ILE':169,'LEU':164,'LYS':205,'MET':188,'PHE':197,
    'PRO':136,'SER':130,'THR':142,'TRP':227,'TYR':222,'VAL':142,
}
HEADER_DUMMY = "HEADER                                            01-JAN-00   XXXX              \n"
CRYST1_DUMMY = "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n"

# ── Step 1: Parse PDB → residue list ──────────────────────────────────────────
def parse_pdb_residues(pdb_path):
    """
    Returns list of dicts: {chain, resnum, resname, atoms}
    Only standard amino acid residues (ATOM records).
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    residues = []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.id[0] != ' ':        # skip HETATM and water
                    continue
                resname = res.resname.strip()
                if resname not in AA3TO1:   # skip non-standard AA
                    continue
                atoms = list(res.get_atoms())
                if not atoms:
                    continue
                residues.append({
                    'chain':   chain.id,
                    'resnum':  res.id[1],
                    'resname': resname,
                    'atoms':   atoms,
                })
    return residues

# ── Step 2: Geometric features ─────────────────────────────────────────────────
def compute_geometric(residues):
    """Returns (N, 5) array: center_x, center_y, center_z, volume, atom_count"""
    feats = []
    for r in residues:
        coords   = np.array([a.coord for a in r['atoms']])
        center   = coords.mean(axis=0)
        bbox     = coords.max(axis=0) - coords.min(axis=0)
        volume   = float(bbox.prod())
        feats.append([center[0], center[1], center[2], volume, len(r['atoms'])])
    return np.array(feats, dtype=np.float32)

# ── Step 3: ESM-2 embeddings ───────────────────────────────────────────────────
def compute_esm2(residues, device):
    """Returns (N, 1280) array. Handles truncation at 1022 residues."""
    from transformers import AutoTokenizer, EsmModel

    sequence = ''.join(AA3TO1[r['resname']] for r in residues)
    n        = len(sequence)

    print(f"  Loading ESM-2 (facebook/esm2_t33_650M_UR50D)...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    esm_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)
    esm_model.eval()

    MAX_LEN = 1022   # ESM-2 max (excl. BOS/EOS tokens)

    if n <= MAX_LEN:
        inputs = tokenizer(sequence, return_tensors='pt',
                           truncation=True, max_length=MAX_LEN + 2).to(device)
        with torch.no_grad():
            out = esm_model(**inputs)
        emb = out.last_hidden_state[0, 1:-1].cpu().numpy()  # (N, 1280)
    else:
        # Sliding window: chunk with 50-residue overlap, average overlapping positions
        print(f"  Protein has {n} residues — using sliding window (max {MAX_LEN})...")
        emb    = np.zeros((n, 1280), dtype=np.float32)
        counts = np.zeros(n, dtype=np.float32)
        stride = MAX_LEN - 50
        start  = 0
        while start < n:
            end  = min(start + MAX_LEN, n)
            chunk = sequence[start:end]
            inputs = tokenizer(chunk, return_tensors='pt').to(device)
            with torch.no_grad():
                out = esm_model(**inputs)
            chunk_emb = out.last_hidden_state[0, 1:-1].cpu().numpy()
            emb[start:end]    += chunk_emb[:end-start]
            counts[start:end] += 1
            if end == n:
                break
            start += stride
        emb = emb / counts[:, None]

    return emb.astype(np.float32)

# ── Step 4: DSSP features ──────────────────────────────────────────────────────
def compute_dssp(pdb_path, n_residues):
    """Returns (N, 6) array: is_helix, is_sheet, is_loop, RSA, phi_norm, psi_norm"""

    def ss_onehot(c):
        if c in ('H','G','I'): return [1,0,0]
        if c in ('E','B'):     return [0,1,0]
        return [0,0,1]

    def norm_angle(a):
        if a is None or abs(a) > 360: return 0.0
        return float(a) / 180.0

    # Write clean temp PDB with HEADER + CRYST1
    with open(pdb_path) as f:
        content = f.read()
    clean = [l for l in content.splitlines(keepends=True)
             if l.startswith(('ATOM  ','TER','END','CRYST1'))]
    clean_str = HEADER_DUMMY + CRYST1_DUMMY + ''.join(clean)

    tmp_pdb  = tempfile.NamedTemporaryFile(suffix='.pdb',  delete=False, mode='w')
    tmp_dssp = tempfile.NamedTemporaryFile(suffix='.dssp', delete=False)
    tmp_pdb.write(clean_str); tmp_pdb.close(); tmp_dssp.close()

    try:
        result = subprocess.run(
            ['mkdssp', tmp_pdb.name, tmp_dssp.name],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            print(f"  DSSP warning: {result.stderr.strip()[:100]} — using zeros")
            return np.zeros((n_residues, 6), dtype=np.float32)

        # Parse DSSP output
        residues = []
        in_res   = False
        with open(tmp_dssp.name) as f:
            for line in f:
                if line.startswith('  #  RESIDUE'):
                    in_res = True; continue
                if not in_res or len(line) < 38: continue
                if line[13] == '!': continue
                aa  = line[13]
                ss  = line[16].strip() or 'C'
                try:    asa = float(line[34:38])
                except: asa = 0.0
                try:    phi = float(line[103:109])
                except: phi = 360.0
                try:    psi = float(line[109:115])
                except: psi = 360.0
                aa3 = next((k for k,v in AA3TO1.items() if v == aa), None)
                max_asa = MAX_RSA.get(aa3, 180) if aa3 else 180
                rsa = min(asa / max_asa, 1.0)
                residues.append(ss_onehot(ss) + [rsa, norm_angle(phi), norm_angle(psi)])

    finally:
        os.unlink(tmp_pdb.name)
        try: os.unlink(tmp_dssp.name)
        except: pass

    feats = np.array(residues, dtype=np.float32) if residues else np.zeros((n_residues, 6), dtype=np.float32)

    # Align length
    if len(feats) > n_residues:
        feats = feats[:n_residues]
    elif len(feats) < n_residues:
        feats = np.vstack([feats, np.zeros((n_residues - len(feats), 6), dtype=np.float32)])

    return feats

# ── Step 5: Run model ──────────────────────────────────────────────────────────
def run_model(features, device):
    """features: (N, 1291) numpy array → returns (N,) probability array"""
    sys.path.insert(0, os.path.dirname(__file__))
    model_arch = import_module('06_model_architecture')

    model = model_arch.BindingSitePredictor().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    feat_tensor = torch.from_numpy(features).unsqueeze(0).to(device)  # (1, N, 1291)
    mask        = torch.zeros(1, features.shape[0], dtype=torch.bool).to(device)

    with torch.no_grad():
        logits = model(feat_tensor, mask)
        probs  = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    return probs

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Predict binding residues from PDB file')
    parser.add_argument('pdb',        help='Input PDB file path')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold for binding prediction (default: 0.5)')
    parser.add_argument('--output',   default=None,
                        help='Output CSV path (default: <pdb_name>_binding.csv)')
    args = parser.parse_args()

    if not os.path.exists(args.pdb):
        print(f"Error: PDB file not found: {args.pdb}")
        sys.exit(1)

    pdb_name   = os.path.splitext(os.path.basename(args.pdb))[0]
    output_csv = args.output or f"{pdb_name}_binding.csv"
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*50}")
    print(f"  Binding Site Prediction")
    print(f"  PDB:       {args.pdb}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Device:    {device}")
    print(f"{'='*50}\n")

    # Step 1: Parse residues
    print("[1/5] Parsing PDB...")
    residues = parse_pdb_residues(args.pdb)
    n = len(residues)
    print(f"  Found {n} standard residues")

    # Step 2: Geometric features
    print("[2/5] Computing geometric features...")
    geo_feats = compute_geometric(residues)       # (N, 5)
    print(f"  Shape: {geo_feats.shape}")

    # Step 3: ESM-2 embeddings
    print("[3/5] Computing ESM-2 embeddings...")
    esm_feats = compute_esm2(residues, device)    # (N, 1280)
    print(f"  Shape: {esm_feats.shape}")

    # Step 4: DSSP features
    print("[4/5] Computing DSSP features...")
    dssp_feats = compute_dssp(args.pdb, n)        # (N, 6)
    print(f"  Shape: {dssp_feats.shape}")

    # Combine: [geo | esm2 | dssp] = 1291
    features = np.concatenate([geo_feats, esm_feats, dssp_feats], axis=1)
    print(f"\n  Combined feature shape: {features.shape}  (expected N x 1291)")

    # Step 5: Model inference
    print("[5/5] Running model...")
    probs = run_model(features, device)

    # ── Output ────────────────────────────────────────────────────────────────
    binding_residues = [(r, p) for r, p in zip(residues, probs) if p >= args.threshold]
    n_binding = len(binding_residues)

    print(f"\n{'='*50}")
    print(f"  Results (threshold={args.threshold})")
    print(f"{'='*50}")
    print(f"  Total residues:   {n}")
    print(f"  Predicted binding:{n_binding} ({100*n_binding/n:.1f}%)")
    print(f"\n  Top 10 binding residues:")
    print(f"  {'Chain':>5} {'ResNum':>7} {'ResName':>8} {'Score':>8}")
    print(f"  {'-'*35}")

    sorted_all = sorted(zip(residues, probs), key=lambda x: -x[1])
    for r, p in sorted_all[:10]:
        print(f"  {r['chain']:>5} {r['resnum']:>7} {r['resname']:>8} {p:>8.4f}")

    # Save CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['chain', 'resnum', 'resname', 'score', 'prediction'])
        for r, p in zip(residues, probs):
            writer.writerow([
                r['chain'],
                r['resnum'],
                r['resname'],
                round(float(p), 4),
                'BINDING' if p >= args.threshold else 'non-binding'
            ])

    print(f"\n  Full results saved → {output_csv}")
    print(f"\nDone.")

if __name__ == '__main__':
    main()
