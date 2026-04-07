#!/usr/bin/env python3

import os
import sys
import numpy as np
from Bio.PDB import PDBParser

data_dir = sys.argv[1] if len(sys.argv) > 1 else "../data/chen11"
output_dir = "../data/geometric"
os.makedirs(output_dir, exist_ok=True)

pdb_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pdb')])
parser = PDBParser(QUIET=True)

for idx, pdb_file in enumerate(pdb_files):
    pdb_path = os.path.join(data_dir, pdb_file)
    structure = parser.get_structure(pdb_file[:-4], pdb_path)
    
    residues_data = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != ' ':
                    continue
                
                atoms = list(residue.get_atoms())
                if not atoms:
                    continue
                
                coords = np.array([atom.coord for atom in atoms])
                center = coords.mean(axis=0)
                
                distances = np.linalg.norm(coords - center, axis=1)
                bbox_size = coords.max(axis=0) - coords.min(axis=0)
                
                features = np.array([
                    center[0], center[1], center[2],
                    bbox_size.prod(),
                    len(atoms)
                ], dtype=np.float32)
                
                residues_data.append(features)
    
    features_array = np.array(residues_data, dtype=np.float32)
    output_file = os.path.join(output_dir, f"{pdb_file[:-4]}.npy")
    np.save(output_file, features_array)
    
    if (idx + 1) % 50 == 0:
        print(f"Processed: {idx + 1}")

print(f"\nComplete: {len(pdb_files)} proteins")
print(f"Output: {output_dir}")
