#!/usr/bin/env python3

import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, EsmModel
from Bio import SeqIO

data_dir = sys.argv[1] if len(sys.argv) > 1 else "../data/chen11"
output_dir = "../data/embeddings"
os.makedirs(output_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)
model.eval()

torch.set_num_threads(6)

fasta_dir = os.path.join(data_dir, "fasta")
fasta_files = sorted([f for f in os.listdir(fasta_dir) if f.endswith('.fasta')])

for idx, fasta_file in enumerate(fasta_files):
    fasta_path = os.path.join(fasta_dir, fasta_file)
    record = next(SeqIO.parse(fasta_path, "fasta"))
    sequence = str(record.seq)
    
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[0, 1:-1].cpu().numpy()
    
    protein_id = fasta_file.replace('.fasta', '')
    output_file = os.path.join(output_dir, f"{protein_id}.npy")
    np.save(output_file, embeddings.astype(np.float32))
    
    if (idx + 1) % 10 == 0:
        print(f"Processed: {idx + 1}/{len(fasta_files)}")

print(f"\nComplete: {len(fasta_files)} proteins")
print(f"Output: {output_dir}")
