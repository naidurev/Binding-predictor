#!/usr/bin/env python3
"""
Batch test: download PDBs, get ground truth, run inference, compare.
Saves summary table to results/batch_test/summary.csv

Usage:
    python3 17_batch_test.py
"""

import os, sys, csv, json, subprocess, tempfile
import numpy as np
import torch
from Bio.PDB import PDBParser
from importlib import import_module

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, 'results', 'batch_test')
TMP_DIR     = os.path.join(BASE, 'data', 'batch_test_pdbs')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# ── Test proteins ──────────────────────────────────────────────────────────────
TEST_PROTEINS = [
    {'id': '7KX5', 'chain': 'A', 'family': 'Viral protease',     'note': 'SARS-CoV-2 Mpro'},
    {'id': '5VBM', 'chain': 'A', 'family': 'GTPase oncogene',    'note': 'KRAS G12C (shallow pocket)'},
    {'id': '4EBV', 'chain': 'A', 'family': 'Chaperone',          'note': 'HSP90 ATP pocket'},
    {'id': '5CSW', 'chain': 'A', 'family': 'Apoptosis',          'note': 'BCL-2 hydrophobic groove'},
    {'id': '6OIM', 'chain': 'A', 'family': 'DNA repair',         'note': 'PARP1 NAD+ site'},
    {'id': '4HJO', 'chain': 'A', 'family': 'Kinase (PI3K path)', 'note': 'AKT1'},
    {'id': '1QXO', 'chain': 'A', 'family': 'Metalloenzyme',      'note': 'Carbonic anhydrase II'},
    {'id': '3ERT', 'chain': 'A', 'family': 'Nuclear receptor',   'note': 'Estrogen receptor'},
    {'id': '6Y2F', 'chain': 'A', 'family': 'Viral protease',     'note': 'SARS-CoV-2 Mpro v2'},
    {'id': '4GV1', 'chain': 'A', 'family': 'Metabolism enzyme',  'note': 'IDO1 heme enzyme'},
]

AA3 = {'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
       'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'}

parser_pdb = PDBParser(QUIET=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
def download_pdb(pdb_id, out_path):
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    result = subprocess.run(['wget', '-q', url, '-O', out_path],
                            capture_output=True, timeout=30)
    return os.path.exists(out_path) and os.path.getsize(out_path) > 0

def get_ground_truth(pdb_path, chain_id='A', cutoff=4.0):
    """Returns dict: ligand_name -> set of (resnum, resname) tuples"""
    struct = parser_pdb.get_structure('prot', pdb_path)
    gt = {}
    for model in struct:
        # Collect ligands
        ligands = {}
        for chain in model:
            for res in chain:
                if res.id[0] in (' ', 'W'): continue
                rname = res.resname.strip()
                if rname in ('HOH', 'WAT', 'DOD'): continue
                key = f"{chain.id}_{rname}_{res.id[1]}"
                ligands[key] = res

        if not ligands:
            continue

        # For each ligand find binding residues on target chain
        for lig_key, lig_res in ligands.items():
            lig_coords = np.array([a.coord for a in lig_res.get_atoms()])
            if len(lig_coords) == 0: continue

            binding = set()
            if chain_id not in model: continue
            for res in model[chain_id]:
                if res.id[0] != ' ': continue
                if res.resname.strip() not in AA3: continue
                for atom in res:
                    dists = np.linalg.norm(lig_coords - atom.coord, axis=1)
                    if dists.min() <= cutoff:
                        binding.add((res.id[1], res.resname.strip()))
                        break
            if binding:
                gt[lig_key] = binding

    return gt

def run_inference(pdb_path):
    """Run inference script, return dict of resnum->score"""
    tmp_out = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
    tmp_out.close()
    result = subprocess.run(
        [sys.executable,
         os.path.join(BASE, 'scripts', '16_inference.py'),
         pdb_path,
         '--threshold', '0.0',
         '--output', tmp_out.name],
        capture_output=True, text=True, timeout=300
    )
    scores = {}
    if os.path.exists(tmp_out.name):
        with open(tmp_out.name) as f:
            for row in csv.DictReader(f):
                scores[int(row['resnum'])] = float(row['score'])
        os.unlink(tmp_out.name)
    return scores, result.stdout + result.stderr

def compute_metrics(scores, gt_residues, thresh):
    predicted = {r for r, s in scores.items() if s >= thresh}
    gt_nums   = {r for r, _ in gt_residues}
    tp = predicted & gt_nums
    fp = predicted - gt_nums
    fn = gt_nums - predicted
    prec = len(tp)/(len(tp)+len(fp)) if predicted else 0
    rec  = len(tp)/(len(tp)+len(fn)) if gt_nums   else 0
    f1   = 2*prec*rec/(prec+rec)     if (prec+rec) > 0 else 0
    return {'tp': len(tp), 'fp': len(fp), 'fn': len(fn),
            'precision': round(prec, 3), 'recall': round(rec, 3), 'f1': round(f1, 3)}

# ── Main loop ──────────────────────────────────────────────────────────────────
summary_rows = []
detail_log   = {}

print(f"\n{'='*65}")
print(f"  Batch Test — {len(TEST_PROTEINS)} proteins")
print(f"{'='*65}\n")

for p in TEST_PROTEINS:
    pid   = p['id']
    chain = p['chain']
    print(f"[{pid}] {p['note']} ({p['family']})")

    pdb_path = os.path.join(TMP_DIR, f"{pid}.pdb")

    # 1. Download
    if not os.path.exists(pdb_path):
        print(f"  Downloading...")
        ok = download_pdb(pid, pdb_path)
        if not ok:
            print(f"  ✗ Download failed — skipping")
            continue
    else:
        print(f"  Already downloaded.")

    # 2. Ground truth
    gt = get_ground_truth(pdb_path, chain_id=chain)
    if not gt:
        print(f"  ✗ No ligand found on chain {chain} — skipping")
        continue

    # Use the ligand with most binding residues (main binding site)
    main_lig = max(gt, key=lambda k: len(gt[k]))
    gt_residues = gt[main_lig]
    lig_name = main_lig.split('_')[1]
    print(f"  Ligand: {lig_name} → {len(gt_residues)} binding residues (4Å)")

    # 3. Inference
    print(f"  Running inference...")
    scores, log = run_inference(pdb_path)
    if not scores:
        print(f"  ✗ Inference failed")
        print(log[-500:])
        continue
    print(f"  Inference done: {len(scores)} residues")

    # 4. Metrics at multiple thresholds
    row = {
        'pdb_id':    pid,
        'family':    p['family'],
        'note':      p['note'],
        'ligand':    lig_name,
        'n_gt':      len(gt_residues),
        'n_residues': len(scores),
    }

    for thresh in [0.5, 0.7, 0.8]:
        m = compute_metrics(scores, gt_residues, thresh)
        row[f't{int(thresh*10)}_tp']   = m['tp']
        row[f't{int(thresh*10)}_fp']   = m['fp']
        row[f't{int(thresh*10)}_prec'] = m['precision']
        row[f't{int(thresh*10)}_rec']  = m['recall']
        row[f't{int(thresh*10)}_f1']   = m['f1']

    # GT residue scores
    gt_scores = sorted([(r, n, scores.get(r, 0.0)) for r, n in gt_residues],
                        key=lambda x: -x[2])
    row['gt_min_score']  = round(min(s for _,_,s in gt_scores), 3)
    row['gt_mean_score'] = round(sum(s for _,_,s in gt_scores)/len(gt_scores), 3)
    row['gt_max_score']  = round(max(s for _,_,s in gt_scores), 3)
    row['truly_missed']  = sum(1 for _,_,s in gt_scores if s < 0.5)

    summary_rows.append(row)
    detail_log[pid] = {'gt': [(r,n,round(s,4)) for r,n,s in gt_scores], 'all_ligands': list(gt.keys())}

    # Print summary
    m5 = compute_metrics(scores, gt_residues, 0.5)
    m8 = compute_metrics(scores, gt_residues, 0.8)
    print(f"  thresh=0.5 → Prec:{m5['precision']:.2f} Rec:{m5['recall']:.2f} F1:{m5['f1']:.2f}")
    print(f"  thresh=0.8 → Prec:{m8['precision']:.2f} Rec:{m8['recall']:.2f} F1:{m8['f1']:.2f}")
    print(f"  GT scores  → min:{row['gt_min_score']} mean:{row['gt_mean_score']} max:{row['gt_max_score']}")
    print(f"  Truly missed (<0.5): {row['truly_missed']}/{len(gt_residues)}\n")

# ── Save summary ───────────────────────────────────────────────────────────────
csv_path = os.path.join(RESULTS_DIR, 'summary.csv')
if summary_rows:
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

json_path = os.path.join(RESULTS_DIR, 'detail.json')
with open(json_path, 'w') as f:
    json.dump(detail_log, f, indent=2)

# ── Print final table ──────────────────────────────────────────────────────────
print(f"\n{'='*75}")
print(f"  FINAL SUMMARY")
print(f"{'='*75}")
print(f"{'PDB':>6} {'Family':<20} {'Lig':>5} {'GT':>4} | {'Rec@0.5':>8} {'Rec@0.8':>8} {'F1@0.5':>7} | {'Missed':>7}")
print(f"  {'-'*70}")
for r in summary_rows:
    print(f"{r['pdb_id']:>6} {r['family']:<20} {r['ligand']:>5} {r['n_gt']:>4} | "
          f"{r['t5_rec']:>8.2f} {r['t8_rec']:>8.2f} {r['t5_f1']:>7.2f} | "
          f"{r['truly_missed']:>4}/{r['n_gt']:<3}")

print(f"\nSaved → {csv_path}")
print(f"Saved → {json_path}")
print(f"\nDone.")
