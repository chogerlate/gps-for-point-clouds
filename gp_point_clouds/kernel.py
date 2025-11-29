import lab as B
import numpy as np
import torch
from geometric_kernels.kernels import MaternKarhunenLoeveKernel
from geometric_kernels.lab_extras import from_numpy


class MaternKarhunenLoeveKernelDeviceAgnostic(MaternKarhunenLoeveKernel):
    def __init__(self, space, num_eigenfunctions, device):
        super().__init__(space, num_eigenfunctions)
        self.device = device

    @staticmethod
    def spectrum(
        s: B.Numeric, nu: B.Numeric, lengthscale: B.Numeric, dimension: int
    ) -> B.Numeric:
        """
        Static method computing the spectrum of the Matérn kernel with
        hyperparameters `nu` and `lengthscale` on the space with eigenvalues `s`
        and dimension `dimension`.
        
        Overridden to fix casting issues when mixing numpy and torch tensors.
        """
        # Check if lengthscale is a torch tensor
        is_torch = hasattr(lengthscale, 'cpu')
        
        # Use from_numpy to convert numpy arrays to the correct backend
        # This fixes the issue where np.r_[1.0] (numpy array) can't be cast to torch dtype
        if is_torch:
            # Convert np.r_[1.0] to torch tensor before using in B.where
            # Use torch.tensor directly to ensure proper conversion
            one_value = torch.tensor(1.0, dtype=lengthscale.dtype, device=lengthscale.device)
            
            # Ensure nu is on the correct device
            if isinstance(nu, torch.Tensor) and nu.device != lengthscale.device:
                nu = nu.to(lengthscale.device)
            
            safe_nu = torch.where(nu == np.inf, one_value, nu)
        else:
            # For numpy backend, use original approach
            safe_nu = B.where(nu == np.inf, B.cast(B.dtype(lengthscale), np.r_[1.0]), nu)

        # for nu == np.inf
        # Convert s to the correct backend if it's numpy
        if is_torch and not hasattr(s, 'cpu'):
            # s is numpy, convert to torch
            s_cast = torch.tensor(s, dtype=lengthscale.dtype, device=lengthscale.device)
        else:
            s_cast = B.cast(B.dtype(lengthscale), s)
        
        if is_torch:
            spectral_values_nu_infinite = torch.exp(
                -(lengthscale**2) / 2.0 * s_cast
            )
        else:
            spectral_values_nu_infinite = B.exp(
                -(lengthscale**2) / 2.0 * s_cast
            )

        # for nu < np.inf
        power = -safe_nu - dimension / 2.0
        # Convert s to the correct backend if it's numpy
        if is_torch and not hasattr(s, 'cpu'):
            # s is numpy, convert to torch
            s_cast = torch.tensor(s, dtype=safe_nu.dtype, device=safe_nu.device)
        else:
            s_cast = B.cast(B.dtype(safe_nu), s)
            
        base = 2.0 * safe_nu / lengthscale**2 + s_cast
        spectral_values_nu_finite = base**power

        if is_torch:
            return torch.where(
                nu == np.inf, spectral_values_nu_infinite, spectral_values_nu_finite
            )
        else:
            return B.where(
                nu == np.inf, spectral_values_nu_infinite, spectral_values_nu_finite
            )

    def eigenvalues(self, params):
        """
        Override eigenvalues to use our fixed spectrum method.
        """
        from geometric_kernels.kernels.karhunen_loeve import _check_field_in_params, _check_1_vector
        
        _check_field_in_params(params, "lengthscale")
        _check_1_vector(params["lengthscale"], 'params["lengthscale"]')
        _check_field_in_params(params, "nu")
        _check_1_vector(params["nu"], 'params["nu"]')

        spectral_values = self.spectrum(
            self.eigenvalues_laplacian,
            nu=params["nu"],
            lengthscale=params["lengthscale"],
            dimension=self.space.dimension,
        )

        if self.normalize:
            if hasattr(spectral_values, 'cpu'):
                normalizer = torch.sum(spectral_values)
            else:
                normalizer = B.sum(spectral_values)
            spectral_values = spectral_values / normalizer

        return spectral_values

    def K(self, params, X, X2=None, **kwargs):
        """
        Override K to avoid B.cast issues with torch tensors in geometric_kernels.
        """
        weights = self.eigenvalues(params)  # [L, 1]
        
        # Helper to safely call eigenfunctions
        def get_eigenfunctions_safe(indices):
            is_torch = hasattr(indices, 'cpu')
            if is_torch:
                # Move to CPU numpy to avoid plum dispatch error in library
                indices_np = indices.detach().cpu().numpy().astype(int) # Ensure int for indices
                if indices_np.ndim == 1:
                    indices_np = indices_np[:, None]
                # Call library function
                ef_np = self.eigenfunctions(indices_np, **kwargs)
                # Convert back to torch on correct device
                # Ensure we match dtype of weights/params if possible, or float
                # Usually eigenfunctions are float
                return torch.tensor(ef_np, device=indices.device, dtype=weights.dtype if hasattr(weights, 'dtype') else torch.float64)
            else:
                return self.eigenfunctions(indices, **kwargs)

        eigenfunctions = get_eigenfunctions_safe(X)  # [N, L]

        if X2 is None:
            eigenfunctions2 = eigenfunctions
        else:
            eigenfunctions2 = get_eigenfunctions_safe(X2)  # [M, L]

        if hasattr(weights, 'cpu'):
            # Torch path
            # Avoid B.cast casting issues by using torch directly
            weights = weights.to(dtype=params["nu"].dtype) if hasattr(params["nu"], 'dtype') else weights
            
            # Perform kernel calculation: K = Phi * Sigma * Phi^T
            # weights is [L, 1], eigenfunctions is [N, L]
            # element-wise mult broadcasts weights: [N, L] * [1, L] (transposed logic) -> [N, L]
            # But weights is [L, 1], so we want to scale columns of eigenfunctions?
            # The formula is sum_l S_l phi_l(x) phi_l(x')
            # = sum_l phi_l(x) * S_l * phi_l(x')
            # In matrix form: Phi * diag(S) * Phi^T
            
            # weights is already the spectrum S_l (or sqrt(S_l) if feature map?? No, K uses full spectrum)
            # In geometric_kernels K implementation:
            # weights = B.cast(..., eigenvalues)
            # return B.matmul(eigenfunctions * weights[None, :], B.transpose(eigenfunctions2))
            # weights[None, :] makes it [1, L]
            # eigenfunctions * weights[None, :] -> [N, L] (broadcasting)
            # then matmul with [L, M]
            
            # Reshape weights to [1, L] for broadcasting
            weights_t = weights.T  # [1, L]
            
            # Scaled eigenfunctions
            scaled_eigenfunctions = eigenfunctions * weights_t
            
            # Result
            return torch.matmul(scaled_eigenfunctions, eigenfunctions2.T)
            
        else:
            # Fallback to original implementation for non-torch
            weights = B.cast(B.dtype(params["nu"]), weights)
            return B.matmul(
                eigenfunctions * weights[None, :],
                B.transpose(eigenfunctions2),
            )

    def K_diag_safe(self, params, X, **kwargs):
        """
        Override K_diag to avoid B.cast issues with torch tensors and for memory efficiency.
        Renamed to K_diag_safe to avoid plum dispatch interception.
        """
        weights = self.eigenvalues(params)  # [L, 1]
        
        # Helper to safely call eigenfunctions (copied from K)
        def get_eigenfunctions_safe(indices):
            is_torch = hasattr(indices, 'cpu')
            if is_torch:
                indices_np = indices.detach().cpu().numpy().astype(int)
                if indices_np.ndim == 1:
                    indices_np = indices_np[:, None]
                ef_np = self.eigenfunctions(indices_np, **kwargs)
                return torch.tensor(ef_np, device=indices.device, dtype=weights.dtype if hasattr(weights, 'dtype') else torch.float64)
            else:
                return self.eigenfunctions(indices, **kwargs)

        eigenfunctions = get_eigenfunctions_safe(X)  # [N, L]
        
        if hasattr(weights, 'cpu'):
             # Torch path
             weights = weights.to(dtype=params["nu"].dtype) if hasattr(params["nu"], 'dtype') else weights
             # weights.T is [1, L]
             # eigenfunctions**2 is [N, L]
             # elementwise multiply by weights (broadcast), then sum over L
             return torch.sum((eigenfunctions**2) * weights.T, dim=1)
        else:
             # Fallback
             weights = B.cast(B.dtype(params["nu"]), weights)
             return B.sum((eigenfunctions**2) * weights[None, :], axis=1)

    def _spectrum(
        self, s: B.Numeric, nu: B.Numeric, lengthscale: B.Numeric
    ) -> B.Numeric:
        """
        NOTE - modified to add cast of base to device; wasn't previously
               working in a device agnostic fashion across CPU/GPU.

        Matern or RBF spectrum evaluated at `s`.
        Depends on the `lengthscale` parameters.
        """
        if nu == np.inf:
            return B.exp(-(lengthscale**2) / 2.0 * from_numpy(lengthscale, s**2))
        elif nu > 0:
            power = -nu - self.space.dimension / 2.0
            base = 2.0 * nu / lengthscale**2 + B.cast(
                B.dtype(nu), from_numpy(nu, s**2)
            ).to(self.device)
            return base**power
        else:
            raise NotImplementedError
