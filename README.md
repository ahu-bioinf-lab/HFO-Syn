# HFO-Syn


## Requirements

- **Python** = 3.11
- **CUDA** 12.8 (GPU recommended) or CPU
- **NVIDIA GPU**  NVIDIA GeForce RTX 5080

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.7.1+cu128 | Deep learning framework |
| torch-geometric | 2.6.1 | Graph neural networks |
| torch-scatter | 2.1.2 | Scatter aggregation operations |
| torch-sparse | 0.6.18 | Sparse matrix support for PyG |
| NumPy | 1.26.4 | Numerical computing |
| Pandas | 2.3.1 | Data I/O and manipulation |
| SciPy | 1.16.0 | Statistical computations (Pearson r) |
| scikit-learn | 1.7.1 | Metrics, data splitting, imputation |
| tqdm | 4.67.1 | Progress bars |
| RDKit | 2025.3.3 | Molecular fingerprints (MACCS) |
| DeepChem | 2.8.0 | Drug molecular feature extraction |
| NetworkX | 3.3 | Graph algorithms (Steiner tree) |
| openpyxl | 3.1.5 | Excel file reading backend |

---

## Installation

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 2. Install PyTorch with CUDA 12.8

```bash
pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install PyTorch Geometric and scatter/sparse

```bash
pip install torch-geometric==2.6.1
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.7.1+cu128.html
```

### 4. Install remaining dependencies

```bash
pip install numpy==1.26.4 pandas==2.3.1 scipy==1.16.0 scikit-learn==1.7.1 \
    tqdm==4.67.1 rdkit==2025.3.3 deepchem==2.8.0 networkx==3.3 openpyxl==3.1.5
```
