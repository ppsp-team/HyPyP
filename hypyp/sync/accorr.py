try:
    import torch
    TORCH_AVAILABLE = True
    MPS_AVAILABLE = torch.backends.mps.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    MPS_AVAILABLE = False

try:    
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

import numpy as np
from tqdm import tqdm
from numpy.typing import NDArray
from hypyp.sync.utils import _multiply_conjugate, _multiply_product
from typing import Optional 

NUMBA_OPTIMIZATION = 'numba'
TORCH_CPU_OPTIMIZATION = 'torch_cpu'
TORCH_MPS_OPTIMIZATION = 'torch_mps'

def accorr(complex_signal: np.ndarray, epochs_average: bool = True, 
           show_progress: bool = True, optimization: Optional[str] = None) -> np.ndarray:
    """
    Computes Adjusted Circular Correlation.

    Parameters
    ----------
    complex_signal : np.ndarray
        Complex analytic signals with shape (n_epochs, n_freq, 2*n_channels, n_times)
        
    epochs_average : bool, optional
        If True, connectivity values are averaged across epochs (default)
        If False, epoch-by-epoch connectivity is preserved
        
    show_progress : bool, optional
        If True, display a progress bar during computation (default)
        If False, no progress bar is shown (progress bar is lighter in this version)

    optimization : str, optional
        If None, execution is done in cpu with no additional python libraries
        If 'numba', execution is done in cpu using numba optimization 
                    (just-in-time compilation and parallelization)
        If 'torch_cpu', execution is parallelized with pytorch numeric library 
                        in cpu
        If 'torch_mps', execution is parallelized with pytorch numeric library 
                        using MPS [Apple’s Metal Performance Shaders](https://huggingface.co/docs/accelerate/usage_guides/mps)
    
    Returns
    -------
    con : np.ndarray
        Adjusted circular correlation matrix with shape:
        - If epochs_average=True: (n_freq, 2*n_channels, 2*n_channels)
        - If epochs_average=False: (n_freq, n_epochs, 2*n_channels, 2*n_channels)

    References
    ----------
    Zimmermann, M., Schultz-Nielsen, K., Dumas, G., & Konvalinka, I. (2024).
    Arbitrary methodological decisions skew inter-brain synchronization estimates
    in hyperscanning-EEG studies. Imaging Neuroscience, 2.
    https://doi.org/10.1162/imag_a_00350
    """

    if optimization == None:
        return _accorr_hybrid_precompute(complex_signal, epochs_average, show_progress)
    elif optimization == NUMBA_OPTIMIZATION:
        if NUMBA_AVAILABLE:
            return _accorr_hybrid_precompute_numba(complex_signal, 
                                                   epochs_average)
        else:
            raise ValueError('Numba library not available for selected optimization')
    elif optimization in [TORCH_CPU_OPTIMIZATION, TORCH_MPS_OPTIMIZATION]:
        if not TORCH_AVAILABLE:
            raise ValueError('Torch library not available for selected optimization')
        if optimization == TORCH_MPS_OPTIMIZATION:
            if MPS_AVAILABLE:
                return _accorr_hybrid_precompute_torch_loop(complex_signal,
                                                            epochs_average,
                                                            show_progress,
                                                            device='mps')
            else:
                raise ValueError('MPS not available on this device for the selected optimization')
        elif optimization == TORCH_CPU_OPTIMIZATION:
            return _accorr_hybrid_precompute_torch_loop(complex_signal,
                                                        epochs_average,
                                                        show_progress,
                                                        device='cpu')
    else:
        raise ValueError(
            f'Optimization parameter is none of the accepted ('
            f'{NUMBA_OPTIMIZATION}, {TORCH_CPU_OPTIMIZATION}, '
            f'{TORCH_MPS_OPTIMIZATION})'
        )
            
                                            
def _accorr_hybrid_precompute(
    complex_signal: NDArray[np.complexfloating], 
    epochs_average: bool = True, 
    show_progress: bool = True
) -> NDArray[np.floating]:
    """
    Computes Adjusted Circular Correlation using an optimized hybrid approach.
    
    This is an optimized version that pre-computes m_adj and n_adj for ALL pairs
    by reusing the cross_conj and cross_prod matrices, reducing computation in the
    denominator loop.

    See Also
    --------
    _accorr : Main function with full parameter and return value descriptions
    
    Notes
    -----
    Key optimization: Instead of computing mean_diff and mean_sum for each pair
    in the loop, we pre-compute them for all pairs at once by reusing:
    - cross_conj / n_times gives the mean of exp(i*(alpha1 - alpha2))
    - cross_prod / n_times gives the mean of exp(i*(alpha1 + alpha2))
    
    This significantly reduces the number of exp() and mean() operations.
    """
    n_epochs, n_freq, n_ch_total, n_times = complex_signal.shape
    transpose_axes = (0, 1, 3, 2)
    
    # Numerator (vectorized)
    z = complex_signal / np.abs(complex_signal)
    c, s = np.real(z), np.imag(z)
    
    cross_conj = _multiply_conjugate(c, s, transpose_axes)
    cross_prod = _multiply_product(c, s, transpose_axes)
    
    r_minus = np.abs(cross_conj)
    r_plus = np.abs(cross_prod)
    num = r_minus - r_plus
    
    # === OPTIMIZATION: Pre-compute m_adj and n_adj for ALL pairs ===
    # cross_conj[i,j] = sum(z_i * conj(z_j)) = sum(exp(i*(alpha_i - alpha_j)))
    # cross_prod[i,j] = sum(z_i * z_j) = sum(exp(i*(alpha_i + alpha_j)))
    mean_diff_all = np.angle(cross_conj / n_times)  # Reuses cross_conj!
    mean_sum_all = np.angle(cross_prod / n_times)   # Reuses cross_prod!
    
    n_adj_all = -1 * (mean_diff_all - mean_sum_all) / 2
    m_adj_all = mean_diff_all + n_adj_all
    
    # Denominator (lighter loop - just lookups, no more circular mean computation)
    angle = np.angle(complex_signal)
    den = np.zeros((n_epochs, n_freq, n_ch_total, n_ch_total))
    
    total_pairs = (n_ch_total * (n_ch_total + 1)) // 2
    pbar = tqdm(total=total_pairs, desc="    accorr_opt (denominator)", 
                disable=not show_progress, leave=False)
    
    for i in range(n_ch_total):
        for j in range(i, n_ch_total):
            alpha1 = angle[:, :, i, :]
            alpha2 = angle[:, :, j, :]
            
            # Just lookup, no more computation!
            m_adj = m_adj_all[:, :, i, j, np.newaxis]
            n_adj = n_adj_all[:, :, i, j, np.newaxis]
            
            x_sin = np.sin(alpha1 - m_adj)
            y_sin = np.sin(alpha2 - n_adj)
            
            den_ij = 2 * np.sqrt(np.sum(x_sin**2, axis=2) * np.sum(y_sin**2, axis=2))
            den[:, :, i, j] = den_ij
            den[:, :, j, i] = den_ij
            
            pbar.update(1)
    
    pbar.close()
    
    den = np.where(den == 0, 1, den)
    con = num / den
    con = con.swapaxes(0, 1)  # n_freq x n_epoch x 2*n_ch x 2*n_ch
    
    if epochs_average:
        con = np.nanmean(con, axis=1)
    
    return con


if NUMBA_AVAILABLE:
    # TODO(@m2march): research why parallelization is not working
    @njit(parallel=False, cache=True)
    def _accorr_den_calc_precalc(n_epochs : int, n_freq : int, n_ch_total : int, 
                                 angle : np.array, m_adj_all : np.array, 
                                 n_adj_all : np.array):
        """
        Computes denominator for adjusted circular correlation using precomputed m_adj and n_adj.
        
        This helper function is JIT-compiled and parallelized with numba for performance.
        It computes the denominator values for all channel pairs using lookup tables of
        precomputed m_adj and n_adj values.
        
        Parameters
        ----------
        n_epochs : int
            Number of epochs
        n_freq : int
            Number of frequency bands
        n_ch_total : int
            Total number of channels
        angle : np.ndarray
            Phase angles with shape (n_epochs, n_freq, n_ch_total, n_times)
        m_adj_all : np.ndarray
            Precomputed m_adj values with shape (n_epochs, n_freq, n_ch_total, n_ch_total)
        n_adj_all : np.ndarray
            Precomputed n_adj values with shape (n_epochs, n_freq, n_ch_total, n_ch_total)
        
        Returns
        -------
        den : np.ndarray
            Denominator matrix with shape (n_epochs, n_freq, n_ch_total, n_ch_total)
        """
        den = np.zeros((n_epochs, n_freq, n_ch_total, n_ch_total))

        for i in range(den.shape[2]):
            for j in range(i, den.shape[3]):
                alpha1 = angle[:, :, i, :]
                alpha2 = angle[:, :, j, :]
                
                # Just lookup, no more computation!
                m_adj = m_adj_all[:, :, i, j]
                n_adj = n_adj_all[:, :, i, j]
                
                x = alpha1.copy() 
                for xi in range(x.shape[0]):
                    for xj in range(x.shape[1]):
                        for xk in range(x.shape[2]):
                            x[xi, xj, xk] -= m_adj[xi, xj]
                x_sin = np.sin(x)

                y = alpha2.copy() 
                for yi in range(y.shape[0]):
                    for yj in range(y.shape[1]):
                        for yk in range(y.shape[2]):
                            y[yi, yj, yk] -= n_adj[yi, yj]
                y_sin = np.sin(y)
                
                den_ij = 2 * np.sqrt(np.sum(x_sin**2, axis=2) * np.sum(y_sin**2, axis=2))
                den[:, :, i, j] = den_ij
                den[:, :, j, i] = den_ij

        return den


    def _accorr_hybrid_precompute_numba(
        complex_signal: NDArray[np.complexfloating], 
        epochs_average: bool = True, 
    ) -> NDArray[np.floating]:
        """
        Computes Adjusted Circular Correlation using numba for optimization.
        
        Notes
        -----
        This optimized version pre-compiles and parallelizes the main loops in 
        _accorr_hybrid_precompute using the numba library.

        See Also
        --------
        _accorr : Main function with full parameter and return value descriptions
        """
        n_epochs, n_freq, n_ch_total, n_times = complex_signal.shape
        transpose_axes = (0, 1, 3, 2)
        
        # Numerator (vectorized)
        z = complex_signal / np.abs(complex_signal)
        c, s = np.real(z), np.imag(z)
        
        cross_conj = _multiply_conjugate(c, s, transpose_axes)
        cross_prod = _multiply_product(c, s, transpose_axes)
        
        r_minus = np.abs(cross_conj)
        r_plus = np.abs(cross_prod)
        num = r_minus - r_plus
        
        # === OPTIMIZATION: Pre-compute m_adj and n_adj for ALL pairs ===
        # cross_conj[i,j] = sum(z_i * conj(z_j)) = sum(exp(i*(alpha_i - alpha_j)))
        # cross_prod[i,j] = sum(z_i * z_j) = sum(exp(i*(alpha_i + alpha_j)))
        mean_diff_all = np.angle(cross_conj / n_times)  # Reuses cross_conj!
        mean_sum_all = np.angle(cross_prod / n_times)   # Reuses cross_prod!
        
        n_adj_all = -1 * (mean_diff_all - mean_sum_all) / 2
        m_adj_all = mean_diff_all + n_adj_all
        
        # Denominator (lighter loop - just lookups, no more circular mean computation)
        angle = np.angle(complex_signal)
                
        den = _accorr_den_calc_precalc(n_epochs, n_freq, n_ch_total, angle, m_adj_all, n_adj_all)
        
        den = np.where(den == 0, 1, den)
        con = num / den
        con = con.swapaxes(0, 1)  # n_freq x n_epoch x 2*n_ch x 2*n_ch
        
        if epochs_average:
            con = np.nanmean(con, axis=1)
        
        return con


if TORCH_AVAILABLE:
    def _accorr_hybrid_precompute_torch_loop(
        complex_signal: NDArray[np.complexfloating], 
        epochs_average: bool = True, 
        show_progress: bool = True,
        device = 'cpu'
    ) -> NDArray[np.floating]:
        """
        Computes Adjusted Circular Correlation using pytorch for optimization.

        Parameters
        ----------
        device : str
            If 'cpu', computations are carried out in cpu
            If 'mps', computations are carried out using 
                      [Apple’s Metal Performance Shaders](https://huggingface.co/docs/accelerate/usage_guides/mps)
        
        Notes
        -----
        This version using pytorch numeric operation libraries for optimization.
        It also allows using special hardware (MPS) for optimization by pushing
        the precalculated vectors to the device and running the main loops over 
        them (see _accorr_hybrid_precompute for details on the precomputation)

        See Also
        --------
        _accorr : Main function with full parameter and return value descriptions
        """
        SUPPORTED_DEVICES = ['cpu', 'mps']
        if device not in SUPPORTED_DEVICES:
            raise ValueError(f'Unsupported device requested, must be one of {SUPPORTED_DEVICES}')
        
        if device == 'mps':
            if not torch.backends.mps.is_available():
                raise ValueError(f'MSP device requested, but not supported in this device')
            else:
                float_type = torch.float32
                complex_type = torch.complex64
        else:
            float_type = torch.float64
            complex_type = torch.complex128

        
        # Convert to torch tensors (use double precision to match numpy)
        complex_tensor = torch.from_numpy(complex_signal).to(device=device, dtype=complex_type)
        
        n_epochs, n_freq, n_ch_total, n_times = complex_tensor.shape
        
        # Numerator (vectorized)
        z = complex_tensor / torch.abs(complex_tensor)
        c, s = z.real, z.imag
        
        # Cross products using einsum
        formula = 'efit,efjt->efij'
        
        # _multiply_conjugate: (real × real.T + imag × imag.T) - i(real × imag.T - imag × real.T)
        cross_conj = (torch.einsum(formula, c, c) + torch.einsum(formula, s, s)) - 1j * \
                    (torch.einsum(formula, c, s) - torch.einsum(formula, s, c))
        
        # _multiply_product: (real × real.T - imag × imag.T) + i(real × imag.T + imag × real.T)
        cross_prod = (torch.einsum(formula, c, c) - torch.einsum(formula, s, s)) + 1j * \
                    (torch.einsum(formula, c, s) + torch.einsum(formula, s, c))
        
        r_minus = torch.abs(cross_conj)
        r_plus = torch.abs(cross_prod)
        num = r_minus - r_plus
        
        # Pre-compute m_adj and n_adj for ALL pairs
        mean_diff_all = torch.angle(cross_conj / n_times)
        mean_sum_all = torch.angle(cross_prod / n_times)
        
        n_adj_all = -0.5 * (mean_diff_all - mean_sum_all)
        m_adj_all = mean_diff_all + n_adj_all
        
        # Denominator - loop-based but on device
        angle = torch.angle(complex_tensor)
        den = torch.zeros((n_epochs, n_freq, n_ch_total, n_ch_total), device=device, dtype=float_type)
        
        total_pairs = (n_ch_total * (n_ch_total + 1)) // 2
        pbar = tqdm(total=total_pairs, desc="    accorr_torch (denominator)", 
                    disable=not show_progress, leave=False)
        
        for i in range(n_ch_total):
            for j in range(i, n_ch_total):
                alpha1 = angle[:, :, i, :]  # [e, f, t]
                alpha2 = angle[:, :, j, :]  # [e, f, t]
                
                # Lookup precomputed values
                m_adj = m_adj_all[:, :, i, j].unsqueeze(-1)  # [e, f, 1]
                n_adj = n_adj_all[:, :, i, j].unsqueeze(-1)  # [e, f, 1]
                
                x_sin = torch.sin(alpha1 - m_adj)
                y_sin = torch.sin(alpha2 - n_adj)
                
                den_ij = 2 * torch.sqrt(torch.sum(x_sin**2, dim=2) * torch.sum(y_sin**2, dim=2))
                den[:, :, i, j] = den_ij
                den[:, :, j, i] = den_ij
                
                pbar.update(1)
        
        pbar.close()
        
        # Avoid division by zero
        den = torch.where(den == 0, torch.ones_like(den), den)
        
        # Compute connectivity
        con = num / den
        con = con.permute(1, 0, 2, 3)  # [n_freq, n_epochs, n_ch, n_ch]
        
        if epochs_average:
            con = torch.nanmean(con, dim=1)
        
        # Convert back to numpy
        return con.cpu().numpy()

