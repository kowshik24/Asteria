# Asteria Installation Guide

## Core Dependencies

Install the core dependencies with:
```bash
pip install -r requirements.txt
```

## Optional Dependencies for Enhanced Baselines

### 1. NearPy (LSH-based similarity search)

NearPy can be tricky to install. Try these approaches in order:

**Option 1: Standard pip install**
```bash
pip install nearpy
```

**Option 2: From source (recommended if Option 1 fails)**
```bash
git clone https://github.com/pixelogik/NearPy.git
cd NearPy
pip install -e .
```

**Option 3: Alternative LSH library**
If NearPy still fails, you can use LSHash as an alternative:
```bash
pip install lshash
```

### 2. ScaNN (Google's similarity search library)

ScaNN is Google's high-performance similarity search library:

```bash
pip install scann
```

Note: ScaNN may require manual compilation on some systems.

### 3. FAISS GPU Support

For GPU acceleration with FAISS:
```bash
# Replace faiss-cpu with faiss-gpu in requirements.txt
pip uninstall faiss-cpu
pip install faiss-gpu
```

### 4. LaTeX Support (for publication-ready plots)

On Ubuntu/Debian:
```bash
sudo apt-get install texlive-latex-base dvipng
```

On macOS:
```bash
brew install --cask mactex
```

On Windows:
Install MiKTeX from https://miktex.org/

## Verification

Test your installation:
```bash
python -c "import torch, numpy, matplotlib, seaborn; print('Core dependencies OK')"
python -c "import torchvision, faiss, annoy; print('Baseline dependencies OK')"
```

Test optional dependencies:
```bash
python -c "try: import nearpy; print('NearPy: OK'); except: print('NearPy: Not available')"
python -c "try: import scann; print('ScaNN: OK'); except: print('ScaNN: Not available')"
```

## Troubleshooting

### Common Issues:

1. **CUDA/GPU Issues**: Ensure CUDA drivers are installed for GPU support
2. **Matplotlib Backend**: If plots don't show, try: `export MPLBACKEND=Agg`
3. **Memory Issues**: Reduce dataset sizes in experiments if you encounter OOM errors
4. **Permission Issues**: Use `--user` flag: `pip install --user -r requirements.txt`

### Minimal Installation (CPU-only, no optional baselines):

```bash
pip install torch numpy matplotlib seaborn torchvision scipy scikit-learn pandas tqdm
```

This minimal setup will run all Asteria experiments but skip optional baseline comparisons.
