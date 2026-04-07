#!/usr/bin/env python3

import os
import sys
import numpy as np
from Bio.PDB import PDBParser, NeighborSearch

data_dir = sys.argv[1] if len(sys.argv) > 1 else "../data/chen11"
output_dir = "../data/labels"
os.makedirs(output_dir, exist_ok=True)

pdb_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pdb')])
parser = PDBParser(QUIET=True)
cutoff = 4.0

processed = 0
skipped = 0

for pdb_file in pdb_files:
    pdb_path = os.path.join(data_dir, pdb_file)
    structure = parser.get_structure(pdb_file[:-4], pdb_path)
    
    protein_atoms = []
    ligand_atoms = []
    residues_list = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                hetflag = residue.id[0]
                if hetflag == ' ':
                    residues_list.append((chain.id, residue.id, residue))
                    protein_atoms.extend(list(residue.get_atoms()))
                elif hetflag.startswith('H_'):
                    ligand_atoms.extend(list(residue.get_atoms()))
    
    if not ligand_atoms:
        skipped += 1
        continue
    
    binding_residues = set()
    ns = NeighborSearch(protein_atoms)
    
    for lig_atom in ligand_atoms:
        nearby = ns.search(lig_atom.coord, cutoff, level='R')
        for res in nearby:
            binding_residues.add((res.parent.id, res.id))
    
    labels = np.array([1 if (chain_id, res_id) in binding_residues else 0 
                       for chain_id, res_id, _ in residues_list], dtype=np.int8)
    
    output_file = os.path.join(output_dir, f"{pdb_file[:-4]}.npy")
    np.save(output_file, labels)
    processed += 1
    
    if processed % 50 == 0:
        print(f"Processed: {processed}")

print(f"\nComplete: {processed} proteins")
print(f"Skipped: {skipped} (no ligands)")
print(f"Output: {output_dir}")
