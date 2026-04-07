#!/usr/bin/env python3
"""
Large-scale test: 1000 unseen proteins from BioLiP
====================================================
Downloads 1000 proteins not seen during training/validation,
runs inference, compares against BioLiP ground truth,
and generates comprehensive stats + plots.

Usage (Colab):
    Upload BioLiP_results.txt to /content/
    !cd /content/scripts && python 19_large_scale_test.py

Outputs:
    /content/results/largescale_metrics.json
    /content/results/largescale_per_protein.csv
    /content/results/largescale_*.png  (plots)
"""

import os, sys, json, csv, random, subprocess, tempfile, time, requests
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    matthews_corrcoef, roc_curve, precision_recall_curve
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
model_arch = import_module('06_model_architecture')

# ── Paths ──────────────────────────────────────────────────────────────────────
BIOLIP_PATH   = '/content/BioLiP_results.txt'
MODEL_PATH    = '/content/models/best_model_v6.pt'
RESULTS_DIR   = '/content/results/largescale'
PDB_CACHE_DIR = '/content/data/largescale_pdbs'
PROGRESS_FILE = '/content/results/largescale/progress.json'

HOLO4K_IDS_PATH  = '/content/data/holo4k_ids.txt'
COACH420_IDS_PATH = '/content/data/coach420_ids.txt'
SPLITS_TRAIN     = '/content/data/splits/train.txt'
SPLITS_VAL       = '/content/data/splits/val.txt'

os.makedirs(RESULTS_DIR,   exist_ok=True)
os.makedirs(PDB_CACHE_DIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
N_PROTEINS   = 1000
RANDOM_SEED  = 42
MIN_SEQ_LEN  = 50
MAX_SEQ_LEN  = 800   # keep ESM2 fast
MIN_BINDING  = 3     # at least 3 binding residues

EXCLUDE_LIGANDS = {
    'rna', 'dna', 'peptide',
    'ZN', 'MG', 'CA', 'MN', 'FE', 'FE2', 'CU', 'NA', 'K', 'CL', 'NI', 'CO',
    'SO4', 'PO4', 'GOL', 'EDO', 'PEG', 'MPD', 'BME', 'DTT', 'IOD',
    'MSE', 'SEP', 'TPO', 'CSO', 'PTR',
    'HOH', 'DOD', 'WAT',
}

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


# ── Step 1: Load existing IDs to exclude ───────────────────────────────────────
def load_existing_pdbs():
    existing = set()

    # HOLO4K: format "11bg" → 4-letter PDB ID
    if os.path.exists(HOLO4K_IDS_PATH):
        with open(HOLO4K_IDS_PATH) as f:
            for line in f:
                pid = line.strip()
                if pid: existing.add(pid[:4].lower())

    # COACH420: format "148lE" → first 4 chars
    if os.path.exists(COACH420_IDS_PATH):
        with open(COACH420_IDS_PATH) as f:
            for line in f:
                pid = line.strip()
                if pid: existing.add(pid[:4].lower())

    # CHEN11 train/val: format "c.001.008.008_2j8fa" → last 4 before chain
    for path in [SPLITS_TRAIN, SPLITS_VAL]:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    pid = line.strip()
                    if '_' in pid:
                        pdb_part = pid.split('_')[-1]
                        existing.add(pdb_part[:4].lower())

    print(f"Excluding {len(existing)} known PDB IDs")
    return existing


# ── Step 2: Parse BioLiP and sample candidates ─────────────────────────────────
def parse_biolip(existing_pdbs):
    """
    Returns dict: (pdb, chain) -> {'binding_pos': set, 'sequence': str}
    Uses col 9 (sequential numbering) and col 21 (sequence).
    """
    print("Parsing BioLiP_results.txt ...")
    protein_data = defaultdict(lambda: {'binding_pos': set(), 'sequence': ''})

    with open(BIOLIP_PATH, encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if i % 100000 == 0 and i > 0:
                print(f"  {i:,} rows parsed...")

            parts = line.strip().split('\t')
            if len(parts) < 9: continue

            pdb     = parts[0].strip().lower()
            chain   = parts[1].strip()
            ligand  = parts[4].strip()
            binding = parts[8].strip()   # col 9 (0-indexed: col 8) — sequential numbering
            seq     = parts[20].strip() if len(parts) > 20 else ''

            # Filters
            if pdb in existing_pdbs:            continue
            if ligand.lower() in EXCLUDE_LIGANDS: continue
            if not binding:                     continue
            if not seq:                         continue

            seq_len = len(seq)
            if seq_len < MIN_SEQ_LEN or seq_len > MAX_SEQ_LEN: continue

            # Parse binding positions: "F43 R45 H93" → {43, 45, 93}
            positions = set()
            for res_str in binding.split():
                try:
                    pos = int(res_str[1:])
                    if 1 <= pos <= seq_len:
                        positions.add(pos)
                except (ValueError, IndexError):
                    continue

            if len(positions) < MIN_BINDING: continue

            key = (pdb, chain)
            protein_data[key]['sequence'] = seq
            protein_data[key]['binding_pos'].update(positions)

    # Filter by binding site size
    candidates = {
        k: v for k, v in protein_data.items()
        if len(v['binding_pos']) >= MIN_BINDING
    }
    print(f"  Valid candidates: {len(candidates):,}")
    return candidates


def sample_proteins(candidates, n, seed):
    random.seed(seed)
    keys = list(candidates.keys())
    if len(keys) <= n:
        return keys
    return random.sample(keys, n)


# ── Step 3: Download PDB ────────────────────────────────────────────────────────
def download_pdb(pdb_id, cache_dir, timeout=30):
    pdb_path = os.path.join(cache_dir, f"{pdb_id}.pdb")
    if os.path.exists(pdb_path):
        return pdb_path
    try:
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200 and len(r.content) > 1000:
            with open(pdb_path, 'w') as f:
                f.write(r.text)
            return pdb_path
    except Exception:
        pass
    return None


# ── Step 4: Feature extraction ─────────────────────────────────────────────────
def parse_residues_from_pdb(pdb_path, chain_id):
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('p', pdb_path)
    residues = []
    for model in structure:
        for chain in model:
            if chain.id != chain_id: continue
            for res in chain:
                if res.id[0] != ' ': continue
                if res.resname.strip() not in AA3TO1: continue
                atoms = list(res.get_atoms())
                if not atoms: continue
                residues.append({
                    'chain': chain.id, 'resnum': res.id[1],
                    'resname': res.resname.strip(), 'atoms': atoms,
                })
    return residues


def geometric_features(residues):
    feats = []
    for r in residues:
        coords = np.array([a.coord for a in r['atoms']])
        center = coords.mean(axis=0)
        bbox   = coords.max(axis=0) - coords.min(axis=0)
        feats.append([center[0], center[1], center[2],
                      float(bbox.prod()), len(r['atoms'])])
    return np.array(feats, dtype=np.float32)


def esm2_embeddings_batch(sequences, esm_model, tokenizer, device, max_len=1022):
    embeddings = []
    for seq in sequences:
        n = len(seq)
        if n <= max_len:
            inputs = tokenizer(seq, return_tensors='pt',
                               truncation=True, max_length=max_len+2).to(device)
            with torch.no_grad():
                emb = esm_model(**inputs).last_hidden_state[0, 1:-1].cpu().numpy()
        else:
            emb    = np.zeros((n, 1280), dtype=np.float32)
            counts = np.zeros(n, dtype=np.float32)
            stride = max_len - 50
            start  = 0
            while start < n:
                end = min(start + max_len, n)
                inp = tokenizer(seq[start:end], return_tensors='pt').to(device)
                with torch.no_grad():
                    chunk = esm_model(**inp).last_hidden_state[0, 1:-1].cpu().numpy()
                emb[start:end]    += chunk[:end-start]
                counts[start:end] += 1
                if end == n: break
                start += stride
            emb = emb / counts[:, None]
        embeddings.append(emb.astype(np.float32))
    return embeddings


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
        result = subprocess.run(['mkdssp', tmp_pdb.name, tmp_dssp.name],
                                capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return np.zeros((n, 6), dtype=np.float32)

        res_list = []
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
                aa3     = next((k for k,v in AA3TO1.items() if v == aa), None)
                max_asa = MAX_RSA.get(aa3, 180) if aa3 else 180
                rsa     = min(asa / max_asa, 1.0)
                res_list.append(ss_onehot(ss) + [rsa, norm_angle(phi), norm_angle(psi)])
    finally:
        os.unlink(tmp_pdb.name)
        try: os.unlink(tmp_dssp.name)
        except: pass

    feats = np.array(res_list, dtype=np.float32) if res_list else np.zeros((n, 6), dtype=np.float32)
    if len(feats) > n:   feats = feats[:n]
    elif len(feats) < n: feats = np.vstack([feats, np.zeros((n-len(feats), 6))])
    return feats


# ── Step 5: Inference ──────────────────────────────────────────────────────────
def run_inference(features, model, device):
    feat_t = torch.from_numpy(features).unsqueeze(0).to(device)
    mask   = torch.zeros(1, features.shape[0], dtype=torch.bool).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(feat_t, mask)).squeeze(0).cpu().numpy()
    return probs


# ── Step 6: Build ground truth from BioLiP ────────────────────────────────────
def build_labels(binding_positions, n_residues):
    """binding_positions: set of 1-indexed sequential positions"""
    labels = np.zeros(n_residues, dtype=np.int8)
    for pos in binding_positions:
        idx = pos - 1
        if 0 <= idx < n_residues:
            labels[idx] = 1
    return labels


# ── Step 7: Compute metrics ────────────────────────────────────────────────────
def compute_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    metrics = {}
    try: metrics['auc']       = float(roc_auc_score(labels, probs))
    except: metrics['auc']    = None
    try: metrics['avg_prec']  = float(average_precision_score(labels, probs))
    except: metrics['avg_prec'] = None
    metrics['f1']        = float(f1_score(labels, preds, zero_division=0))
    metrics['precision'] = float(precision_score(labels, preds, zero_division=0))
    metrics['recall']    = float(recall_score(labels, preds, zero_division=0))
    try: metrics['mcc']  = float(matthews_corrcoef(labels, preds))
    except: metrics['mcc'] = 0.0
    metrics['n_binding'] = int(labels.sum())
    metrics['n_total']   = len(labels)
    metrics['binding_pct'] = float(labels.sum() / len(labels) * 100)
    return metrics


# ── Step 8: Generate plots ─────────────────────────────────────────────────────
def generate_plots(results, all_probs, all_labels):
    print("\nGenerating plots...")

    valid = [r for r in results if r.get('auc') is not None]
    aucs  = [r['auc']  for r in valid]
    f1s   = [r['f1']   for r in valid]
    precs = [r['precision'] for r in valid]
    recs  = [r['recall']    for r in valid]
    lens  = [r['n_total']   for r in valid]
    bs    = [r['n_binding'] for r in valid]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'Large-Scale Test — {len(valid)} Unseen Proteins from BioLiP',
                 fontsize=14, fontweight='bold')

    # 1. AUC distribution
    ax = axes[0, 0]
    ax.hist(aucs, bins=30, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(np.mean(aucs), color='red', linestyle='--',
               label=f'Mean={np.mean(aucs):.3f}')
    ax.axvline(np.median(aucs), color='orange', linestyle='-.',
               label=f'Median={np.median(aucs):.3f}')
    ax.set_xlabel('AUC-ROC'); ax.set_ylabel('Count')
    ax.set_title('AUC-ROC Distribution'); ax.legend(fontsize=9)

    # 2. F1 distribution
    ax = axes[0, 1]
    ax.hist(f1s, bins=30, color='seagreen', edgecolor='white', alpha=0.85)
    ax.axvline(np.mean(f1s), color='red', linestyle='--',
               label=f'Mean={np.mean(f1s):.3f}')
    ax.set_xlabel('F1 Score'); ax.set_ylabel('Count')
    ax.set_title('F1 Score Distribution'); ax.legend(fontsize=9)

    # 3. Precision vs Recall scatter
    ax = axes[0, 2]
    sc = ax.scatter(recs, precs, c=aucs, cmap='RdYlGn', s=15, alpha=0.6,
                    vmin=0.5, vmax=1.0)
    plt.colorbar(sc, ax=ax, label='AUC')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('Precision vs Recall\n(color = AUC)')
    ax.plot([0,1],[0,1], 'k--', alpha=0.3)

    # 4. Aggregate ROC curve
    ax = axes[1, 0]
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    agg_auc = roc_auc_score(all_labels, all_probs)
    ax.plot(fpr, tpr, color='steelblue', lw=2,
            label=f'AUC = {agg_auc:.3f}')
    ax.plot([0,1],[0,1], 'k--', alpha=0.4)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('Aggregate ROC Curve'); ax.legend()

    # 5. Aggregate Precision-Recall curve
    ax = axes[1, 1]
    prec_curve, rec_curve, _ = precision_recall_curve(all_labels, all_probs)
    avg_p = average_precision_score(all_labels, all_probs)
    ax.plot(rec_curve, prec_curve, color='seagreen', lw=2,
            label=f'Avg Prec = {avg_p:.3f}')
    baseline = all_labels.mean()
    ax.axhline(baseline, color='grey', linestyle='--',
               label=f'Baseline = {baseline:.3f}')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('Aggregate Precision-Recall Curve'); ax.legend()

    # 6. AUC vs protein length
    ax = axes[1, 2]
    bins    = [0, 100, 200, 300, 400, 500, 600, 800]
    bin_auc = []
    bin_labels_plot = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = [lo <= l < hi for l in lens]
        if sum(mask) > 5:
            bin_auc.append(np.mean([a for a, m in zip(aucs, mask) if m]))
            bin_labels_plot.append(f'{lo}-{hi}')
    if bin_auc:
        ax.bar(range(len(bin_auc)), bin_auc, color='mediumpurple',
               edgecolor='white', alpha=0.85)
        ax.set_xticks(range(len(bin_auc)))
        ax.set_xticklabels(bin_labels_plot, rotation=30, ha='right', fontsize=8)
        ax.set_ylim(0, 1); ax.axhline(0.5, color='grey', linestyle='--')
        ax.set_xlabel('Protein Length'); ax.set_ylabel('Mean AUC')
        ax.set_title('AUC by Protein Length')

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, 'largescale_summary.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out}")

    # Bonus: binding site size vs F1
    plt.figure(figsize=(7, 5))
    plt.scatter(bs, f1s, alpha=0.4, s=15, color='coral')
    plt.xlabel('Number of Binding Residues')
    plt.ylabel('F1 Score')
    plt.title('Binding Site Size vs F1')
    plt.tight_layout()
    out2 = os.path.join(RESULTS_DIR, 'largescale_bindinsize_f1.png')
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"  Saved → {out2}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print("Loading model...")
    model = model_arch.BindingSitePredictor().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Load ESM2
    print("Loading ESM-2...")
    from transformers import AutoTokenizer, EsmModel
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    esm_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)
    esm_model.eval()

    # Load existing IDs
    existing_pdbs = load_existing_pdbs()

    # Parse BioLiP
    candidates = parse_biolip(existing_pdbs)

    # Sample 1000
    sampled_keys = sample_proteins(candidates, N_PROTEINS, RANDOM_SEED)
    print(f"\nSampled {len(sampled_keys)} proteins")

    # Load progress (resume if crashed)
    completed = {}
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            completed = json.load(f)
        print(f"Resuming — {len(completed)} already done")

    # ── Main processing loop ───────────────────────────────────────────────────
    results       = []
    all_probs_agg = []
    all_labels_agg= []
    skipped       = 0

    for idx, (pdb, chain) in enumerate(sampled_keys):
        key_str = f"{pdb}_{chain}"

        if key_str in completed:
            results.append(completed[key_str])
            if completed[key_str].get('auc') is not None:
                all_probs_agg.append(np.array(completed[key_str]['_probs']))
                all_labels_agg.append(np.array(completed[key_str]['_labels']))
            continue

        print(f"[{idx+1}/{len(sampled_keys)}] {pdb} chain {chain}", end=' ')

        # Download PDB
        pdb_path = download_pdb(pdb, PDB_CACHE_DIR)
        if pdb_path is None:
            print("→ download failed, skip")
            skipped += 1
            continue

        try:
            # Parse residues for this chain
            residues = parse_residues_from_pdb(pdb_path, chain)
            if len(residues) < MIN_SEQ_LEN:
                print(f"→ too short ({len(residues)} res), skip")
                skipped += 1
                continue

            n = len(residues)
            seq = ''.join(AA3TO1[r['resname']] for r in residues)

            # Ground truth from BioLiP
            binding_pos = candidates[(pdb, chain)]['binding_pos']
            labels = build_labels(binding_pos, n)

            if labels.sum() < MIN_BINDING:
                print(f"→ too few binding residues, skip")
                skipped += 1
                continue

            # Features
            geo  = geometric_features(residues)
            esm  = esm2_embeddings_batch([seq], esm_model, tokenizer, device)[0]
            dssp = dssp_features(pdb_path, n)
            features = np.concatenate([geo, esm, dssp], axis=1)  # (N, 1291)

            # Inference
            probs = run_inference(features, model, device)

            # Metrics
            m = compute_metrics(labels, probs)
            m['pdb']   = pdb
            m['chain'] = chain
            m['_probs']  = probs.tolist()
            m['_labels'] = labels.tolist()

            results.append(m)
            all_probs_agg.append(probs)
            all_labels_agg.append(labels)

            # Save progress
            completed[key_str] = m
            with open(PROGRESS_FILE, 'w') as pf:
                json.dump(completed, pf)

            print(f"→ AUC={m['auc']:.3f} F1={m['f1']:.3f} "
                  f"Prec={m['precision']:.3f} Rec={m['recall']:.3f} "
                  f"({m['n_binding']}/{n} binding)")

        except Exception as e:
            print(f"→ ERROR: {e}, skip")
            skipped += 1
            continue

    # ── Aggregate results ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Large-Scale Test Complete")
    print(f"  Processed: {len(results)}  |  Skipped: {skipped}")
    print(f"{'='*60}")

    valid = [r for r in results if r.get('auc') is not None]
    if not valid:
        print("No valid results!")
        return

    aucs  = [r['auc']       for r in valid]
    f1s   = [r['f1']        for r in valid]
    precs = [r['precision'] for r in valid]
    recs  = [r['recall']    for r in valid]
    mccs  = [r['mcc']       for r in valid]

    print(f"\n  {'Metric':<20} {'Mean':>8} {'Median':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*60}")
    for name, vals in [('AUC-ROC', aucs), ('F1', f1s),
                        ('Precision', precs), ('Recall', recs), ('MCC', mccs)]:
        print(f"  {name:<20} {np.mean(vals):>8.4f} {np.median(vals):>8.4f} "
              f"{np.std(vals):>8.4f} {np.min(vals):>8.4f} {np.max(vals):>8.4f}")

    # Global aggregate metrics
    all_p = np.concatenate(all_probs_agg)
    all_l = np.concatenate(all_labels_agg)
    print(f"\n  Global aggregate (all residues pooled):")
    print(f"    AUC-ROC:       {roc_auc_score(all_l, all_p):.4f}")
    print(f"    Avg Precision: {average_precision_score(all_l, all_p):.4f}")
    print(f"    F1 (@0.5):     {f1_score(all_l, (all_p>=0.5).astype(int), zero_division=0):.4f}")
    print(f"    Precision:     {precision_score(all_l, (all_p>=0.5).astype(int), zero_division=0):.4f}")
    print(f"    Recall:        {recall_score(all_l, (all_p>=0.5).astype(int), zero_division=0):.4f}")
    print(f"{'='*60}\n")

    # Save JSON
    summary = {
        'n_processed': len(valid),
        'n_skipped': skipped,
        'per_protein_stats': {
            'auc':       {'mean': np.mean(aucs),  'median': np.median(aucs),
                          'std': np.std(aucs),    'min': np.min(aucs),   'max': np.max(aucs)},
            'f1':        {'mean': np.mean(f1s),   'median': np.median(f1s),
                          'std': np.std(f1s),     'min': np.min(f1s),    'max': np.max(f1s)},
            'precision': {'mean': np.mean(precs), 'median': np.median(precs)},
            'recall':    {'mean': np.mean(recs),  'median': np.median(recs)},
        },
        'global_aggregate': {
            'auc':       float(roc_auc_score(all_l, all_p)),
            'avg_prec':  float(average_precision_score(all_l, all_p)),
            'f1':        float(f1_score(all_l, (all_p>=0.5).astype(int), zero_division=0)),
            'precision': float(precision_score(all_l, (all_p>=0.5).astype(int), zero_division=0)),
            'recall':    float(recall_score(all_l, (all_p>=0.5).astype(int), zero_division=0)),
        }
    }
    json_path = os.path.join(RESULTS_DIR, 'largescale_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved → {json_path}")

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, 'largescale_per_protein.csv')
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['pdb','chain','n_total','n_binding','binding_pct',
                      'auc','avg_prec','f1','precision','recall','mcc']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(valid)
    print(f"Saved → {csv_path}")

    # Plots
    generate_plots(valid, all_p, all_l)

    print("\nDone!")


if __name__ == '__main__':
    main()
