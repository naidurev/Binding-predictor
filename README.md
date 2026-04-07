# Protein Binding Site Predictor

Per-residue binding site prediction using ESM-2 embeddings + geometric features + DSSP secondary structure, with a 3-layer transformer classifier.

## Quick Start

```bash
git clone https://github.com/naidurev/Binding-predictor
cd Binding-predictor

pip install -r requirements.txt          # Python dependencies
sudo apt install dssp                    # mkdssp (Linux)
# brew install brewsci/bio/dssp          # mkdssp (Mac)

# Download model weights
mkdir -p models
wget https://github.com/naidurev/Binding-predictor/releases/download/v2.0/best_model_v6.pt -O models/best_model_v6.pt

python predict.py your_protein.pdb
```

## Usage

```bash
# Basic (threshold 0.5)
python predict.py protein.pdb

# Stricter predictions (fewer false positives)
python predict.py protein.pdb --threshold 0.8

# Custom output file
python predict.py protein.pdb --threshold 0.7 --output results.csv

# Skip PyMOL script generation
python predict.py protein.pdb --no-pymol
```

### Output Files

| File | Description |
|------|-------------|
| `<pdb>_binding.csv` | Per-residue scores and predictions |
| `<pdb>_binding.pml` | PyMOL script — open with `pymol <pdb>_binding.pml` |

**PyMOL color scheme:**
- 🔴 **Red** — high confidence binding residue (score ≥ 0.8)
- 🟠 **Orange** — medium confidence binding residue (score ≥ threshold)
- ⬜ **Grey** — non-binding

### Score Interpretation

| Score | Meaning |
|-------|---------|
| ≥ 0.8 | High confidence binding residue |
| 0.5 – 0.8 | Possible binding residue |
| < 0.5 | Likely non-binding |

## Model

| Feature | Dimension |
|---------|-----------|
| ESM-2 embeddings (facebook/esm2_t33_650M_UR50D) | 1280 |
| Geometric (center xyz, volume, atom count) | 5 |
| DSSP (helix/sheet/loop, RSA, φ, ψ) | 6 |
| **Total** | **1291** |

**Architecture**: LayerNorm → Linear(1291→1280) → TransformerEncoder(3 layers, 8 heads) → FFN(1280→512→128→1)

## Performance

Evaluated on COACH420 benchmark (420 held-out proteins):

| Metric | v5 | v6 (latest) | P2Rank |
|--------|-----|-------------|--------|
| AUC-ROC | 0.902 | **0.903** | 0.895 |
| Avg Precision | 0.417 | **0.450** | — |
| F1 (thresh=0.5) | 0.352 | **0.416** | 0.438 |
| Precision | 0.220 | **0.283** | — |
| Recall | 0.780 | **0.782** | — |
| MCC | — | **0.419** | — |

Tested on 9 diverse unseen proteins (viral proteases, kinases, nuclear receptors, etc.) — average recall 0.91 at threshold=0.5.

## Training Data

- **CHEN11** — 220 proteins
- **HOLO4K** — 4541 proteins
- Labels: residues within 4Å of any bound ligand atom

## Project Structure

```
binding_predictor/
├── predict.py              ← main inference script
├── requirements.txt
├── models/
│   └── best_model_v6.pt    ← model weights (download separately)
└── scripts/
    ├── 06_model_architecture.py
    ├── 18_train_v2.py          ← retrain script (v6)
    ├── 10_eval_coach420.py
    ├── 15_compare_p2rank.py
    └── ...
```

## Requirements

- Python 3.9+
- CUDA GPU recommended (CPU works, slower)
- `mkdssp` system binary (`sudo apt install dssp` on Linux)
- ~3GB RAM for ESM-2 model


