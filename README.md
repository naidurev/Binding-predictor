# Protein Binding Site Predictor

Per-residue ligand binding site prediction from a PDB file, using ESM-2 language model embeddings, geometric features, and DSSP secondary structure — classified by a 3-layer Transformer.

## Quick Start

```bash
git clone https://github.com/naidurev/Binding-predictor
cd Binding-predictor

# Install Python dependencies
pip install -r requirements.txt

# Install mkdssp
sudo apt install dssp                   # Linux
brew install brewsci/bio/dssp           # Mac
# Windows: download from https://github.com/PDB-REDO/dssp/releases

# Download model weights
mkdir -p models
wget https://github.com/naidurev/Binding-predictor/releases/download/v2.0/best_model_v6.pt -O models/best_model_v6.pt

# Run on any PDB file
python predict.py your_protein.pdb
```

## Usage

```bash
# Basic prediction (threshold 0.5)
python predict.py protein.pdb

# Stricter — fewer false positives
python predict.py protein.pdb --threshold 0.8

# Custom output file
python predict.py protein.pdb --threshold 0.7 --output results.csv

# Skip PyMOL script generation
python predict.py protein.pdb --no-pymol
```

### Output Files

| File | Description |
|------|-------------|
| `<pdb>_binding.csv` | Per-residue scores and BINDING/non-binding labels |
| `<pdb>_binding.pml` | PyMOL script for 3D visualization |

**PyMOL visualization:**
```bash
pymol protein_binding.pml          # Linux/Mac
# Windows: File → Run Script → select the .pml file
# ⚠️ The .pml file must be in the same folder as the PDB file
```

**Color scheme:**
- 🔴 **Red** — high confidence binding residue (score ≥ 0.8)
- 🟠 **Orange** — medium confidence binding residue (score ≥ threshold)
- ⬜ **Grey** — non-binding

### Score Interpretation

| Score | Meaning |
|-------|---------|
| ≥ 0.8 | High confidence binding residue |
| 0.5 – 0.8 | Possible binding residue |
| < 0.5 | Likely non-binding |

## Example

```bash
$ python predict.py 7PCD.pdb --threshold 0.7

====================================================
  Binding Site Predictor
  Input:     7PCD.pdb
  Threshold: 0.7
  Device:    cuda
====================================================

[1/5] Parsing residues...
  914 residues found
[2/5] Geometric features...
[3/5] ESM-2 embeddings...
[4/5] DSSP features...
[5/5] Running model...

====================================================
  Results  (threshold = 0.7)
====================================================
  Total residues:    914
  Predicted binding: 34 (3.7%)

  Top binding residues:
  Chain  ResNum    AA   Score
      A     727   SER   0.923  █████████
      A     730   GLY   0.901  █████████
      A     732   TYR   0.897  ████████
      A     800   ASP   0.874  ████████
      A     804   TRP   0.856  ████████
      A     862   GLY   0.841  ████████
      ...

  Full results → 7PCD_binding.csv
  PyMOL script  → 7PCD_binding.pml
  Open in PyMOL: pymol 7PCD_binding.pml
```

## Model

### Input Features (per residue)

| Feature | Dimension | Description |
|---------|-----------|-------------|
| ESM-2 embeddings | 1280 | Per-residue contextual embedding (facebook/esm2_t33_650M_UR50D) |
| Geometric | 5 | Residue center (x,y,z), bounding box volume, atom count |
| DSSP | 6 | Secondary structure (helix/sheet/loop), RSA, φ, ψ angles |
| **Total** | **1291** | |

### Architecture

```
Input (N, 1291)
    → LayerNorm
    → Linear(1291 → 1280)
    → TransformerEncoder(3 layers, 8 heads, dropout=0.3)
    → FFN: Linear(1280→512) → ReLU → Linear(512→128) → ReLU → Linear(128→1)
Output: binding probability per residue ∈ [0, 1]
```

**Parameters:** ~41.7M | **Training loss:** BCEWithLogitsLoss (pos_weight=5.0)

## Performance

### COACH420 Benchmark (420 held-out proteins)

| Metric | Value |
|--------|-------|
| AUC-ROC | **0.903** |
| Avg Precision | **0.450** |
| F1 (thresh=0.5) | **0.416** |
| Precision | **0.283** |
| Recall | **0.782** |
| MCC | **0.419** |

### Large-Scale Validation (858 unseen proteins from BioLiP)

| Metric | Mean | Median |
|--------|------|--------|
| AUC-ROC | 0.852 | **0.913** |
| F1 | 0.260 | 0.250 |
| Recall | 0.640 | 0.733 |
| MCC | 0.278 | 0.300 |

Validated on **858 unseen proteins** from BioLiP spanning diverse protein families — median AUC-ROC **0.913**.

## Training Data

| Dataset | Proteins | Source |
|---------|----------|--------|
| CHEN11 | 220 | BioLiP benchmark set |
| HOLO4K | 4541 | BioLiP, clustered at 30% sequence identity |
| **Total** | **4761** | |

Labels: residues within **4Å** of any bound ligand atom (excluding ions, solvents, crystallographic artifacts).

## Project Structure

```
binding_predictor/
├── predict.py               ← main inference script (any PDB → binding residues)
├── requirements.txt         ← Python dependencies
├── models/
│   └── best_model_v6.pt     ← model weights (download from releases)
└── scripts/
    ├── model.py             ← Transformer model architecture
    └── training_pipeline.py ← full training script (reproduce v6)
```

## Requirements

- Python 3.9+
- CUDA GPU recommended (CPU works, slower — ESM-2 takes ~5 min/protein on CPU)
- `mkdssp` system binary
- ~3 GB RAM for ESM-2 model loading
