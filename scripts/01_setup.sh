#!/bin/bash

python3 -c "
import sys
packages = ['torch', 'biopython', 'numpy', 'scipy', 'transformers']
missing = []
for pkg in packages:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f'Installing: {\" \".join(missing)}')
    import os
    os.system(f'{sys.executable} -m pip install --break-system-packages {\" \".join(missing)}')
else:
    print('All dependencies installed')
"

DATASET_DIR="/mnt/e/UPF/2nd-trimester/SBI/project/p2rank-datasets-master/p2rank-datasets-master"
ln -sf "$DATASET_DIR/chen11" ../data/chen11
ln -sf "$DATASET_DIR/coach420" ../data/coach420

echo "Setup complete"
echo "chen11: $(ls ../data/chen11/*.pdb 2>/dev/null | wc -l) proteins"
echo "coach420: $(ls ../data/coach420/*.pdb 2>/dev/null | wc -l) proteins"
