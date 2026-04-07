#!/usr/bin/env python3
"""
Binding Site Predictor
======================
Predicts per-residue binding site probabilities from a PDB file.

Usage:
    python predict.py protein.pdb
    python predict.py protein.pdb --threshold 0.8
    python predict.py protein.pdb --threshold 0.7 --output my_results.csv

Outputs:
    <pdb>_binding.csv   - per-residue scores (CSV)
    <pdb>_binding.pml   - PyMOL script to visualize predictions

Requirements:
    pip install -r requirements.txt
    mkdssp (sudo apt install dssp  OR  brew install brewsci/bio/dssp)

Model weights:
    Download best_model_v6.pt from releases and place in models/best_model_v6.pt
"""

import os, sys
# Add scripts/ to path for model architecture
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

import argparse
import csv
import subprocess
import tempfile
import numpy as np
import torch
from Bio.PDB import PDBParser
from importlib import import_module

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best_model_v6.pt')

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

# ── Feature computation ────────────────────────────────────────────────────────

def parse_residues(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    residues = []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.id[0] != ' ':
                    continue
                if res.resname.strip() not in AA3TO1:
                    continue
                atoms = list(res.get_atoms())
                if not atoms:
                    continue
                residues.append({
                    'chain':   chain.id,
                    'resnum':  res.id[1],
                    'resname': res.resname.strip(),
                    'atoms':   atoms,
                })
    return residues


def geometric_features(residues):
    feats = []
    for r in residues:
        coords  = np.array([a.coord for a in r['atoms']])
        center  = coords.mean(axis=0)
        bbox    = coords.max(axis=0) - coords.min(axis=0)
        feats.append([center[0], center[1], center[2], float(bbox.prod()), len(r['atoms'])])
    return np.array(feats, dtype=np.float32)    # (N, 5)


def esm2_embeddings(residues, device):
    from transformers import AutoTokenizer, EsmModel

    sequence = ''.join(AA3TO1[r['resname']] for r in residues)
    n        = len(sequence)
    MAX_LEN  = 1022

    print(f"  Loading ESM-2...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    esm       = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)
    esm.eval()

    if n <= MAX_LEN:
        inputs = tokenizer(sequence, return_tensors='pt',
                           truncation=True, max_length=MAX_LEN + 2).to(device)
        with torch.no_grad():
            emb = esm(**inputs).last_hidden_state[0, 1:-1].cpu().numpy()
    else:
        print(f"  Long protein ({n} residues) — sliding window...")
        emb    = np.zeros((n, 1280), dtype=np.float32)
        counts = np.zeros(n, dtype=np.float32)
        stride = MAX_LEN - 50
        start  = 0
        while start < n:
            end   = min(start + MAX_LEN, n)
            inp   = tokenizer(sequence[start:end], return_tensors='pt').to(device)
            with torch.no_grad():
                chunk = esm(**inp).last_hidden_state[0, 1:-1].cpu().numpy()
            emb[start:end]    += chunk[:end-start]
            counts[start:end] += 1
            if end == n: break
            start += stride
        emb = emb / counts[:, None]

    return emb.astype(np.float32)    # (N, 1280)


def dssp_features(pdb_path, n):
    def ss_onehot(c):
        if c in ('H','G','I'): return [1,0,0]
        if c in ('E','B'):     return [0,1,0]
        return [0,0,1]

    def norm_angle(a):
        return 0.0 if (a is None or abs(a) > 360) else float(a) / 180.0

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
            print(f"  DSSP warning — using zeros")
            return np.zeros((n, 6), dtype=np.float32)

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

    feats = np.array(residues, dtype=np.float32) if residues else np.zeros((n, 6), dtype=np.float32)
    if len(feats) > n:   feats = feats[:n]
    elif len(feats) < n: feats = np.vstack([feats, np.zeros((n - len(feats), 6), dtype=np.float32)])
    return feats    # (N, 6)


def run_model(features, device):
    model_arch = import_module('model')
    model      = model_arch.BindingSitePredictor().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    feat_t = torch.from_numpy(features).unsqueeze(0).to(device)
    mask   = torch.zeros(1, features.shape[0], dtype=torch.bool).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(feat_t, mask)).squeeze(0).cpu().numpy()
    return probs

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Predict binding site residues from a PDB file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py protein.pdb
  python predict.py protein.pdb --threshold 0.8
  python predict.py protein.pdb --threshold 0.7 --output results.csv

Score interpretation:
  >= 0.8  High confidence binding residue
  0.5-0.8 Possible binding residue
  < 0.5   Likely non-binding
        """
    )
    parser.add_argument('pdb',         help='Input PDB file')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5)')
    parser.add_argument('--output',    default=None,
                        help='Output CSV file (default: <pdb>_binding.csv)')
    parser.add_argument('--no-pymol',  action='store_true',
                        help='Skip PyMOL script generation')
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.pdb):
        print(f"Error: File not found: {args.pdb}")
        sys.exit(1)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model weights not found at {MODEL_PATH}")
        print(f"Download best_model_v6.pt from the releases page.")
        sys.exit(1)

    pdb_name   = os.path.splitext(os.path.basename(args.pdb))[0]
    output_csv = args.output or f"{pdb_name}_binding.csv"
    output_pml = output_csv.replace('.csv', '.pml')
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*52}")
    print(f"  Binding Site Predictor")
    print(f"  Input:     {args.pdb}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Device:    {device}")
    print(f"{'='*52}\n")

    print("[1/5] Parsing residues...")
    residues = parse_residues(args.pdb)
    n = len(residues)
    if n == 0:
        print("Error: No standard amino acid residues found in PDB.")
        sys.exit(1)
    print(f"  {n} residues found")

    print("[2/5] Geometric features...")
    geo  = geometric_features(residues)

    print("[3/5] ESM-2 embeddings...")
    esm  = esm2_embeddings(residues, device)

    print("[4/5] DSSP features...")
    dssp = dssp_features(args.pdb, n)

    features = np.concatenate([geo, esm, dssp], axis=1)   # (N, 1291)

    print("[5/5] Running model...")
    probs = run_model(features, device)

    # ── Results ───────────────────────────────────────────────────────────────
    binding = [(r, p) for r, p in zip(residues, probs) if p >= args.threshold]
    binding.sort(key=lambda x: -x[1])

    print(f"\n{'='*52}")
    print(f"  Results  (threshold = {args.threshold})")
    print(f"{'='*52}")
    print(f"  Total residues:    {n}")
    print(f"  Predicted binding: {len(binding)} ({100*len(binding)/n:.1f}%)")

    if binding:
        print(f"\n  Top binding residues:")
        print(f"  {'Chain':>5} {'ResNum':>7} {'AA':>5} {'Score':>7}")
        print(f"  {'-'*30}")
        for r, p in binding[:15]:
            bar = '█' * int(p * 10)
            print(f"  {r['chain']:>5} {r['resnum']:>7} {r['resname']:>5} {p:>7.3f}  {bar}")
        if len(binding) > 15:
            print(f"  ... and {len(binding)-15} more (see {output_csv})")
    else:
        print(f"\n  No residues above threshold {args.threshold}.")
        print(f"  Try lowering --threshold (e.g. --threshold 0.3)")

    # Save CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['chain', 'resnum', 'resname', 'score', 'prediction'])
        for r, p in zip(residues, probs):
            pred = 'BINDING' if p >= args.threshold else 'non-binding'
            writer.writerow([r['chain'], r['resnum'], r['resname'],
                             round(float(p), 4), pred])

    print(f"\n  Full results → {output_csv}")

    # ── PyMOL script ──────────────────────────────────────────────────────────
    if not args.no_pymol:
        pdb_abs = os.path.abspath(args.pdb)

        # Group residues by chain and confidence tier
        high   = {}   # score >= 0.8
        medium = {}   # 0.5 <= score < 0.8
        low    = {}   # score < threshold but shown for context (skipped)

        for r, p in zip(residues, probs):
            chain = r['chain']
            rnum  = str(r['resnum'])
            if p >= 0.8:
                high.setdefault(chain, []).append(rnum)
            elif p >= args.threshold:
                medium.setdefault(chain, []).append(rnum)

        def sel_string(rdict):
            parts = []
            for chain, rnums in rdict.items():
                parts.append(f"(chain {chain} and resi {'+'.join(rnums)})")
            return ' or '.join(parts) if parts else 'none'

        high_sel   = sel_string(high)
        medium_sel = sel_string(medium)

        all_binding_chains = set(list(high.keys()) + list(medium.keys()))
        binding_sel = sel_string(
            {c: high.get(c, []) + medium.get(c, []) for c in all_binding_chains}
        )

        pml_lines = [
            f"# Binding Site Predictor — PyMOL visualization",
            f"# Threshold: {args.threshold}",
            f"# Generated by predict.py",
            f"",
            f"load {pdb_abs}",
            f"",
            f"# Base representation",
            f"hide everything",
            f"show cartoon",
            f"color grey80, all",
            f"",
            f"# Binding residues",
        ]

        if high_sel != 'none':
            pml_lines += [
                f"select binding_high, {high_sel}",
                f"show sticks, binding_high",
                f"color red, binding_high",
                f"",
            ]
        if medium_sel != 'none':
            pml_lines += [
                f"select binding_medium, {medium_sel}",
                f"show sticks, binding_medium",
                f"color orange, binding_medium",
                f"",
            ]
        if binding_sel != 'none':
            pml_lines += [
                f"select binding_all, {binding_sel}",
                f"",
            ]

        pml_lines += [
            f"# Surface of binding region",
            f"show surface, all",
            f"set transparency, 0.5",
            f"",
            f"# Labels",
            f"set label_size, 14",
            f"label binding_high and name CA, \"%s%s\" % (oneletter, resi)",
            f"",
            f"zoom binding_all" if binding_sel != 'none' else "zoom all",
            f"",
            f"# Color key:",
            f"# RED    = high confidence binding (score >= 0.8)",
            f"# ORANGE = medium confidence binding (score >= {args.threshold})",
            f"# GREY   = non-binding",
        ]

        with open(output_pml, 'w') as f:
            f.write('\n'.join(pml_lines))

        print(f"  PyMOL script  → {output_pml}")
        print(f"  Open in PyMOL: pymol {output_pml}")

    print(f"\nDone.\n")


if __name__ == '__main__':
    main()
