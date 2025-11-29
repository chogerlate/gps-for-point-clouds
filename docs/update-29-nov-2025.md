# Engineering Report: Optimization and Stabilization of GP-based Point Cloud Simplification

**Date:** November 29, 2025
**Subject:** Resolution of Backend Dispatch, Numerical Stability, and Memory Scalability Issues
**Context:** `gps-for-point-clouds` repository

## 1. Executive Summary

We have successfully refactored the core Gaussian Process (GP) kernel implementations and algorithmic selection logic. The primary objectives were to resolve backend dispatch conflicts between PyTorch and the `geometric_kernels` library (built on `lab`/`plum`), fix critical numerical instability during hyperparameter estimation, and reduce memory complexity from $\mathcal{O}(N^2)$ to $\mathcal{O}(N)$ for the selection metric.

The simplified point cloud algorithm is now functional within the Google Colab environment, capable of handling dense point clouds (~35k points) on standard GPU hardware without hitting memory limits.

## 2. Technical Deep Dive

### 2.1. Backend Interoperability & Dispatch Resolution
**Problem:** The `geometric_kernels` library relies on `plum` for multiple dispatch and `lab` for backend abstraction. In the current environment, `lab` failed to correctly resolve PyTorch tensors for operations like `cast`, `where`, and `sum`, resulting in `plum.resolver.NotFoundLookupError`.

**Solution:**
*   **Kernel Overrides:** In `gp_point_clouds/kernel.py`, we subclassed `MaternKarhunenLoeveKernel` and overrode critical methods (`spectrum`, `eigenvalues`, `K`).
*   **Explicit PyTorch Logic:** Inside these overrides, we implemented explicit `is_torch` checks. When PyTorch tensors are detected, we bypass `lab` abstractions and use direct `torch.*` operations (e.g., `torch.where`, `torch.matmul`), ensuring robust execution.
*   **Safe Eigenfunction Calls:** The library's `eigenfunctions` call failed when passed PyTorch tensors. We implemented a `get_eigenfunctions_safe` helper that seamlessly bridges the gap by detaching tensors to NumPy (and reshaping to $[N, 1]$ to satisfy `np.take_along_axis` requirements), invoking the library code, and converting results back to the correct CUDA device.
*   **Bypassing Interception:** To prevent `plum` from intercepting our overrides via parent class registration, we implemented `K_diag_safe` (a uniquely named method) to guarantee our optimized diagonal computation path is executed.

### 2.2. Memory Complexity Reduction (OOM Fix)
**Problem:** The original greedy selection algorithm computed the full posterior covariance matrix $\Sigma_t = K_{rr} - K_{ri} K_{ii}^{-1} K_{ri}^T$ for the remainder set. For a remainder set of size $N \approx 24,000$, storing $K_{rr}$ required $\approx 4.6$ GB of contiguous GPU memory, causing `torch.cuda.OutOfMemoryError`.

**Solution:**
*   **Diagonalization:** The greedy selection metric only requires the variance (the diagonal of $\Sigma_t$). We refactored `gp_point_clouds/algorithm.py` to compute only the diagonal elements.
*   **Vectorized Computation:**
    *   $K_{rr}$ diagonal is computed directly via `K_diag_safe`.
    *   The reduction term diagonal is computed via row-wise summation: $\text{sum}((K_{ri} L^{-T})^2, \text{dim}=1)$, where $L$ is the Cholesky factor.
*   **Impact:** Reduced space complexity for the selection step from quadratic $\mathcal{O}(N^2)$ to linear $\mathcal{O}(N)$, effectively saving ~9GB of VRAM per iteration.

### 2.3. Numerical Stability
**Problem:** The Cholesky decomposition frequently failed with `NotPSDError` (Matrix not positive semi-definite) due to an ill-conditioned covariance matrix.

**Solution:**
*   **Hyperparameter Initialization:** The initial smoothness parameter $\nu$ was set to `10000` (approximating an RBF kernel), which induces stiffness. We relaxed this to $\nu=2.5$ (Matern 5/2), providing a more stable starting point.
*   **Noise Floor:** We increased the likelihood noise constraint from $10^{-6}$ to $10^{-4}$ and the initial noise from $10^{-5}$ to $10^{-2}$.
*   **Jitter:** Wrapped the optimization loop in `gpytorch.settings.cholesky_jitter(1e-3)` to stabilize decomposition during early training epochs.

## 3. Codebase Hygiene & Usability

*   **Abstract Class Fixes:** The `PointCloud` class in `gp_point_clouds/spaces.py` was missing implementations for `random`, `element_shape`, and `element_dtype`, which prevented instantiation. These were implemented to satisfy the `DiscreteSpectrumSpace` interface.
*   **Pathing:** `demo.py` was updated to handle absolute paths specific to the Colab environment (`/content/gps-for-point-clouds/resources/results/`) with robust directory creation logic.
*   **Visualization:** A new `visualize_results.py` script was added to parse `.ply`/`.xyz` outputs and render them using `matplotlib` (static) and `plotly` (interactive), aiding in immediate result verification.

## 4. Conclusion

The system is now stable and optimized. The greedy subset algorithm correctly leverages the spectral properties of the Laplacian on the point cloud manifold without memory overflows or dispatch crashes. The transition to diagonal-only computation for the uncertainty metric was the key enabler for scalability.