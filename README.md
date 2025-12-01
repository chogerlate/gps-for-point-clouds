# GP-PCS: One-Shot Feature-Preserving Point Cloud Simplification with Gaussian Processes on Riemannian Manifolds [ICPR 2024 (Oral)]

Read our paper here:

[![Published Paper](https://img.shields.io/badge/Published-Paper-blue)](https://doi.org/10.1007/978-3-031-78456-9_28)
[![arXiv](https://img.shields.io/badge/arXiv-2303.15225-b31b1b.svg)](https://arxiv.org/abs/2303.15225)

## 🔍 Overview

We propose a novel, one-shot point cloud simplification method which preserves both the salient structural features and the overall shape of a point cloud without any prior surface reconstruction step. Our method employs Gaussian processes suitable for functions defined on Riemannian manifolds, allowing us to model the surface variation function across any given point cloud. A simplified version of the original cloud is obtained by sequentially selecting points using a greedy sparsification scheme. The selection criterion used for this scheme ensures that the simplified cloud best represents the surface variation of the original point cloud.

### Key Features

- 🎯 **Feature-Preserving**: Maintains structural features and overall shape
- 🎨 **Color Preservation**: Automatically preserves color information from input PLY files
- ⚡ **One-Shot**: No prior surface reconstruction required
- 🔬 **Riemannian Manifolds**: Uses geometric kernels for accurate modeling
- 💾 **Multiple Formats**: Supports PLY, XYZ, and NPZ output formats 
 
 Below you can see the simplified representations of the Stanford Dragon (top row) and associated reconstructed meshes (bottom row) for all evaluated simplification techniques. Here GP is the simplification technique implemented in our paper.

![Teaser](./teaser.png)

## 🛠️ Setup

### Quick Setup (Recommended)

**Python 3.10** is required. Follow these steps for a clean installation:

#### 1. Create Conda Environment

```bash
conda create -n pcs python=3.10
conda activate pcs
```

#### 2. Install PyTorch with CUDA Support

For CUDA 11.8 (recommended):
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

For CUDA 11.6 (alternative):
```bash
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
```

#### 3. Install Core Dependencies

```bash
# Gaussian Process and Kernel Libraries
pip install gpytorch
pip install git+https://github.com/gpflow/geometrickernels.git

# Critical: Fix numpy version compatibility
pip install numpy==1.26.4

# 3D Processing Libraries
pip install open3d jakteristics matplotlib

# PyTorch3D (from source)
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

#### 4. Install Deep Graph Library (DGL)

For CUDA 11.8:
```bash
pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu118/repo.html
```

For CUDA 11.7:
```bash
pip install --pre dgl -f https://data.dgl.ai/wheels/cu117/repo.html
```

#### 5. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import gpytorch; import geometric_kernels; print('Dependencies OK')"
```

### Troubleshooting

**CUDA Library Issues (Linux):**
If you encounter `libcusparse.so.11` not found errors:
```bash
# Find CUDA library path
find /usr/local/cuda*/lib64 -name "libcusparse.so.11" 2>/dev/null

# Add to LD_LIBRARY_PATH (replace with your actual path)
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```

**Numpy/Scipy Compatibility:**
If you see `ValueError: All ufuncs must have type numpy.ufunc`:
```bash
pip install "numpy<2.0,>=1.22.0" "scipy>=1.9.0,<2.0.0"
```

### Alternative: Google Colab Setup

For Google Colab users, see `notebooks/colab_setup.ipynb` for a complete setup script that handles all dependencies automatically.

## 🚀 Quick Start

### Clone and Run

```bash
git clone https://github.com/stutipathak5/gps-for-point-clouds.git
cd gps-for-point-clouds
conda activate pcs

# Run demo with default settings
python demo.py

# Or use the command-line interface
python run_gp_pcs.py resources/clouds/bun_zipper.ply --ratio 0.1 --output_dir output/
```

### Command-Line Usage

```bash
python run_gp_pcs.py <input_file.ply> [options]

Options:
  --output_dir DIR          Output directory (default: output)
  --neigh_size N            Neighbourhood size for curvature (default: 30)
  --target_points N          Target number of points
  --ratio FLOAT              Simplification ratio 0.0-1.0 (e.g., 0.1 for 10%)
  --random_cloud_size N      Random subset size for processing (default: 25000)
  --opt_subset_size N       Subset size for hyperparameter optimization (default: 300)
  --n_iter N                Number of optimization iterations (default: 100)
  --gpu                      Use GPU if available
```

**Example:**
```bash
# Simplify to 10% of original points
python run_gp_pcs.py input.ply --ratio 0.1 --gpu

# Simplify to exactly 5000 points
python run_gp_pcs.py input.ply --target_points 5000 --output_dir results/
```

### Features

- ✅ **Color Preservation**: Automatically preserves colors from input PLY files
- ✅ **Multiple Output Formats**: Saves as `.ply`, `.xyz`, and `.npz`
- ✅ **GPU Support**: Use `--gpu` flag for faster processing
- ✅ **Flexible Simplification**: Specify by ratio or exact point count

### Batch Processing

For multiple point clouds:
```bash
python run_all.py
```

## 📤 Output Files

The algorithm generates three output files for each simplified point cloud:

1. **`.ply`** - Point cloud with colors preserved (if input had colors)
2. **`.xyz`** - Simple text format with coordinates only
3. **`.npz`** - NumPy archive containing:
   - `org_coords`: Original point coordinates
   - `org_faces`: Original mesh faces (if available)
   - `simp_coords`: Simplified point coordinates
   - `org_curv`: Original curvature values
   - `original_indices`: Mapping from simplified to original point indices

**Color Preservation**: If your input PLY file contains color information, the output PLY will automatically preserve these colors mapped to the simplified points. This enables immediate visualization in tools like CloudCompare, MeshLab, or Blender without any post-processing.

## 📝 Citation

Please consider citing the following if you find this work useful:

```bibtex
@inproceedings{pathak2025gp,
  title={GP-PCS: One-shot Feature-Preserving Point Cloud Simplification with Gaussian Processes on Riemannian Manifolds},
  author={Pathak, Stuti and Baldwin-McDonald, Thomas and Sels, Seppe and Penne, Rudi},
  booktitle={International Conference on Pattern Recognition},
  pages={436--452},
  year={2025},
  organization={Springer}
}
```





