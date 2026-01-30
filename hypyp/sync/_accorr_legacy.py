
def _accorr_hybrid(complex_signal: np.ndarray, epochs_average: bool = True, 
                   show_progress: bool = True) -> np.ndarray:
    """
    Computes Adjusted Circular Correlation using a hybrid approach.
    
    This function calculates the adjusted circular correlation coefficient between
    all channel pairs. It uses a vectorized computation for the numerator and an
    exact loop-based computation for the denominator.
    
    Parameters
    ----------
    complex_signal : np.ndarray
        Complex analytic signals with shape (n_epochs, n_freq, 2*n_channels, n_times)
        Note: This is the already reshaped signal from compute_sync.
        
    epochs_average : bool, optional
        If True, connectivity values are averaged across epochs (default)
        If False, epoch-by-epoch connectivity is preserved
        
    show_progress : bool, optional
        If True, display a progress bar during computation (default)
        If False, no progress bar is shown
    
    Returns
    -------
    con : np.ndarray
        Adjusted circular correlation matrix with shape:
        - If epochs_average=True: (n_freq, 2*n_channels, 2*n_channels)
        - If epochs_average=False: (n_freq, n_epochs, 2*n_channels, 2*n_channels)
    
    Notes
    -----
    The adjusted circular correlation is computed as:
    
    1. Numerator (vectorized): Uses the difference between the absolute values of
       the conjugate product and the direct product of normalized complex signals.
       
    2. Denominator (loop): For each channel pair, computes optimal phase centering
       parameters (m_adj, n_adj) that minimize the denominator, then calculates
       the normalization factor.
    
    This metric provides a more accurate measure of circular correlation by
    adjusting the phase centering for each channel pair individually, rather than
    using a global circular mean.
    
    References
    ----------
    Zimmermann, M., Schultz-Nielsen, K., Dumas, G., & Konvalinka, I. (2024).
    Arbitrary methodological decisions skew inter-brain synchronization estimates
    in hyperscanning-EEG studies. Imaging Neuroscience, 2.
    https://doi.org/10.1162/imag_a_00350
    """
    n_epochs = complex_signal.shape[0]
    n_freq = complex_signal.shape[1]
    n_ch_total = complex_signal.shape[2]
    
    transpose_axes = (0, 1, 3, 2)
    
    # Numerator (vectorized)
    z = complex_signal / np.abs(complex_signal)
    c, s = np.real(z), np.imag(z)
    
    cross_conj = _multiply_conjugate(c, s, transpose_axes=transpose_axes)
    r_minus = np.abs(cross_conj)
    
    cross_prod = _multiply_product(c, s, transpose_axes=transpose_axes)
    r_plus = np.abs(cross_prod)
    
    num = r_minus - r_plus
    
    # Denominator (loop)
    angle = np.angle(complex_signal)
    den = np.zeros((n_epochs, n_freq, n_ch_total, n_ch_total))
    
    total_pairs = (n_ch_total * (n_ch_total + 1)) // 2
    pbar = tqdm(total=total_pairs, desc="    accorr (denominator)", 
                disable=not show_progress, leave=False)
    
    for i in range(n_ch_total):
        for j in range(i, n_ch_total):
            alpha1 = angle[:, :, i, :]
            alpha2 = angle[:, :, j, :]
            
            phase_diff = alpha1 - alpha2
            phase_sum = alpha1 + alpha2
            
            mean_diff = np.angle(np.mean(np.exp(1j * phase_diff), axis=2, keepdims=True))
            mean_sum = np.angle(np.mean(np.exp(1j * phase_sum), axis=2, keepdims=True))
            
            n_adj = -1 * (mean_diff - mean_sum) / 2
            m_adj = mean_diff + n_adj
            
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


@njit(nopython=False, parallel=True, cache=True)
def _accorr_den_calc(n_epochs, n_freq, n_ch_total, angle):

    den = np.zeros((n_epochs, n_freq, n_ch_total, n_ch_total))
    for i in prange(den.shape[2]):
        for j in prange(i, den.shape[3]):
            alpha1 = angle[:, :, i, :]
            alpha2 = angle[:, :, j, :]
            
            phase_diff = alpha1 - alpha2
            phase_sum = alpha1 + alpha2

            def axis2_mean(m):
                return np.array([
                    m[i, j, :].mean()
                    for i in prange(m.shape[0])
                    for j in prange(m.shape[1])
                ]).reshape((m.shape[0], m.shape[1]))

            mean_diff = np.angle(axis2_mean(np.exp(1j * phase_diff)))
            mean_sum = np.angle(axis2_mean(np.exp(1j * phase_sum)))
            
            n_adj = -1 * (mean_diff - mean_sum) / 2
            m_adj = mean_diff + n_adj
            
            x = alpha1.copy() 
            for xi in prange(x.shape[0]):
                for xj in prange(x.shape[1]):
                    for xk in prange(x.shape[2]):
                        x[xi, xj, xk] -= m_adj[xi, xj]
            x_sin = np.sin(x)

            y = alpha2.copy() 
            for yi in prange(y.shape[0]):
                for yj in prange(y.shape[1]):
                    for yk in prange(y.shape[2]):
                        y[yi, yj, yk] -= n_adj[yi, yj]
            y_sin = np.sin(y)
            
            den_ij = 2 * np.sqrt(np.sum(x_sin**2, axis=2) * np.sum(y_sin**2, axis=2))
            den[:, :, i, j] = den_ij
            den[:, :, j, i] = den_ij

    return den

def _accorr_hybrid_numba(complex_signal: np.ndarray, epochs_average: bool = True, 
                   show_progress: bool = True) -> np.ndarray:
    """
    Computes Adjusted Circular Correlation using a hybrid approach.
    
    This function calculates the adjusted circular correlation coefficient between
    all channel pairs. It uses a vectorized computation for the numerator and an
    exact loop-based computation for the denominator.
    
    Parameters
    ----------
    complex_signal : np.ndarray
        Complex analytic signals with shape (n_epochs, n_freq, 2*n_channels, n_times)
        Note: This is the already reshaped signal from compute_sync.
        
    epochs_average : bool, optional
        If True, connectivity values are averaged across epochs (default)
        If False, epoch-by-epoch connectivity is preserved
        
    show_progress : bool, optional
        If True, display a progress bar during computation (default)
        If False, no progress bar is shown
    
    Returns
    -------
    con : np.ndarray
        Adjusted circular correlation matrix with shape:
        - If epochs_average=True: (n_freq, 2*n_channels, 2*n_channels)
        - If epochs_average=False: (n_freq, n_epochs, 2*n_channels, 2*n_channels)
    
    Notes
    -----
    The adjusted circular correlation is computed as:
    
    1. Numerator (vectorized): Uses the difference between the absolute values of
       the conjugate product and the direct product of normalized complex signals.
       
    2. Denominator (loop): For each channel pair, computes optimal phase centering
       parameters (m_adj, n_adj) that minimize the denominator, then calculates
       the normalization factor.
    
    This metric provides a more accurate measure of circular correlation by
    adjusting the phase centering for each channel pair individually, rather than
    using a global circular mean.
    
    References
    ----------
    Zimmermann, M., Schultz-Nielsen, K., Dumas, G., & Konvalinka, I. (2024).
    Arbitrary methodological decisions skew inter-brain synchronization estimates
    in hyperscanning-EEG studies. Imaging Neuroscience, 2.
    https://doi.org/10.1162/imag_a_00350
    """
    n_epochs = complex_signal.shape[0]
    n_freq = complex_signal.shape[1]
    n_ch_total = complex_signal.shape[2]
    
    transpose_axes = (0, 1, 3, 2)
    
    # Numerator (vectorized)
    z = complex_signal / np.abs(complex_signal)
    c, s = np.real(z), np.imag(z)
    
    cross_conj = _multiply_conjugate(c, s, transpose_axes=transpose_axes)
    r_minus = np.abs(cross_conj)
    
    cross_prod = _multiply_product(c, s, transpose_axes=transpose_axes)
    r_plus = np.abs(cross_prod)
    
    num = r_minus - r_plus
    
    # Denominator (loop)
    angle = np.angle(complex_signal)
    
    den = _accorr_den_calc(n_epochs, n_freq, n_ch_total, angle) 
    
    den = np.where(den == 0, 1, den)
    con = num / den
    con = con.swapaxes(0, 1)  # n_freq x n_epoch x 2*n_ch x 2*n_ch
    
    if epochs_average:
        con = np.nanmean(con, axis=1)
    
    return con


def _accorr_hybrid_vectorized(complex_signal: np.ndarray, epochs_average: bool = True, 
                             show_progress: bool = True) -> np.ndarray:
    """
    Computes Adjusted Circular Correlation using full vectorization.
    
    This function calculates the adjusted circular correlation coefficient between
    all channel pairs using fully vectorized operations, eliminating nested loops
    for significant performance improvements.
    
    Parameters
    ----------
    complex_signal : np.ndarray
        Complex analytic signals with shape (n_epochs, n_freq, 2*n_channels, n_times)
        Note: This is the already reshaped signal from compute_sync.
        
    epochs_average : bool, optional
        If True, connectivity values are averaged across epochs (default)
        If False, epoch-by-epoch connectivity is preserved
        
    show_progress : bool, optional
        If True, display a progress bar during computation (default)
        If False, no progress bar is shown
    
    Returns
    -------
    con : np.ndarray
        Adjusted circular correlation matrix with shape:
        - If epochs_average=True: (n_freq, 2*n_channels, 2*n_channels)
        - If epochs_average=False: (n_freq, n_epochs, 2*n_channels, 2*n_channels)
    
    Notes
    -----
    The adjusted circular correlation is computed using full vectorization:
    
    1. Numerator (vectorized): Uses the difference between the absolute values of
       the conjugate product and the direct product of normalized complex signals.
       
    2. Denominator (vectorized): Computes optimal phase centering parameters
       (m_adj, n_adj) for all channel pairs simultaneously using broadcasting,
       then calculates the normalization factor without loops.
    
    This fully vectorized approach provides significant speedup compared to the
    loop-based implementation while maintaining numerical accuracy.
    
    References
    ----------
    Zimmermann, M., Schultz-Nielsen, K., Dumas, G., & Konvalinka, I. (2024).
    Arbitrary methodological decisions skew inter-brain synchronization estimates
    in hyperscanning-EEG studies. Imaging Neuroscience, 2.
    https://doi.org/10.1162/imag_a_00350
    """
    n_epochs = complex_signal.shape[0]
    n_freq = complex_signal.shape[1]
    n_ch_total = complex_signal.shape[2]
    n_times = complex_signal.shape[3]
    
    transpose_axes = (0, 1, 3, 2)
    
    # Numerator (vectorized)
    z = complex_signal / np.abs(complex_signal)
    c, s = np.real(z), np.imag(z)
    
    cross_conj = _multiply_conjugate(c, s, transpose_axes=transpose_axes)
    r_minus = np.abs(cross_conj)
    
    cross_prod = _multiply_product(c, s, transpose_axes=transpose_axes)
    r_plus = np.abs(cross_prod)
    
    num = r_minus - r_plus
    
    # Denominator (fully vectorized)
    angle = np.angle(complex_signal)
    
    # Expand dimensions for broadcasting
    # angle shape: (n_epochs, n_freq, n_ch_total, n_times)
    alpha1_all = angle[:, :, :, None, :]      # (n_epochs, n_freq, n_ch_total, 1, n_times)
    alpha2_all = angle[:, :, None, :, :]      # (n_epochs, n_freq, 1, n_ch_total, n_times)
    
    # Compute phase differences and sums for all pairs
    phase_diff = alpha1_all - alpha2_all      # (n_epochs, n_freq, n_ch_total, n_ch_total, n_times)
    phase_sum = alpha1_all + alpha2_all       # (n_epochs, n_freq, n_ch_total, n_ch_total, n_times)
    
    mean_diff = np.angle(np.mean(np.exp(1j * phase_diff), axis=4, keepdims=True))
    mean_sum = np.angle(np.mean(np.exp(1j * phase_sum), axis=4, keepdims=True))
    
    # Compute optimal phase centering parameters
    n_adj = -1 * (mean_diff - mean_sum) / 2
    m_adj = mean_diff + n_adj
    
    # Compute sine deviations for all pairs
    x_sin = np.sin(alpha1_all - m_adj)        # (n_epochs, n_freq, n_ch_total, n_ch_total, n_times)
    y_sin = np.sin(alpha2_all - n_adj)        # (n_epochs, n_freq, n_ch_total, n_ch_total, n_times)
    
    # Sum of squared sines
    x_sin_sq_sum = np.sum(x_sin**2, axis=4)   # (n_epochs, n_freq, n_ch_total, n_ch_total)
    y_sin_sq_sum = np.sum(y_sin**2, axis=4)   # (n_epochs, n_freq, n_ch_total, n_ch_total)
    
    # Compute denominator
    den = 2 * np.sqrt(x_sin_sq_sum * y_sin_sq_sum)
    
    # Handle division by zero
    den = np.where(den == 0, 1, den)
    
    # Compute connectivity
    con = num / den
    con = con.swapaxes(0, 1)  # n_freq x n_epoch x 2*n_ch x 2*n_ch
    
    if epochs_average:
        con = np.nanmean(con, axis=1)
    
    return con


def _compute_pair_denominator(args):
    """Helper function for multiprocessing implementation."""
    i, j, angle = args
    n_epochs, n_freq = angle.shape[0], angle.shape[1]
    
    alpha1 = angle[:, :, i, :]
    alpha2 = angle[:, :, j, :]
    
    phase_diff = alpha1 - alpha2
    phase_sum = alpha1 + alpha2
    
    mean_diff = np.angle(np.mean(np.exp(1j * phase_diff), axis=2, keepdims=True))
    mean_sum = np.angle(np.mean(np.exp(1j * phase_sum), axis=2, keepdims=True))
    
    n_adj = -1 * (mean_diff - mean_sum) / 2
    m_adj = mean_diff + n_adj
    
    x_sin = np.sin(alpha1 - m_adj)
    y_sin = np.sin(alpha2 - n_adj)
    
    den_ij = 2 * np.sqrt(np.sum(x_sin**2, axis=2) * np.sum(y_sin**2, axis=2))
    
    return i, j, den_ij


def _accorr_hybrid_multiprocessing(complex_signal: np.ndarray, epochs_average: bool = True, 
                                   show_progress: bool = True, n_jobs: int = 4) -> np.ndarray:
    """
    Computes Adjusted Circular Correlation using multiprocessing parallelization.
    
    This implementation distributes channel pair computations across multiple
    processes for improved performance on multi-core systems.
    
    Parameters
    ----------
    complex_signal : np.ndarray
        Complex analytic signals with shape (n_epochs, n_freq, 2*n_channels, n_times)
        
    epochs_average : bool, optional
        If True, connectivity values are averaged across epochs (default)
        If False, epoch-by-epoch connectivity is preserved
        
    show_progress : bool, optional
        If True, display a progress bar during computation (default)
        
    n_jobs : int, optional
        Number of parallel processes to use (default: 4)
        
    Returns
    -------
    con : np.ndarray
        Adjusted circular correlation matrix with shape:
        - If epochs_average=True: (n_freq, 2*n_channels, 2*n_channels)
        - If epochs_average=False: (n_freq, n_epochs, 2*n_channels, 2*n_channels)
    """
    n_epochs = complex_signal.shape[0]
    n_freq = complex_signal.shape[1]
    n_ch_total = complex_signal.shape[2]
    
    transpose_axes = (0, 1, 3, 2)
    
    # Numerator (vectorized)
    z = complex_signal / np.abs(complex_signal)
    c, s = np.real(z), np.imag(z)
    
    cross_conj = _multiply_conjugate(c, s, transpose_axes=transpose_axes)
    r_minus = np.abs(cross_conj)
    
    cross_prod = _multiply_product(c, s, transpose_axes=transpose_axes)
    r_plus = np.abs(cross_prod)
    
    num = r_minus - r_plus
    
    # Denominator (multiprocessing)
    angle = np.angle(complex_signal)
    den = np.zeros((n_epochs, n_freq, n_ch_total, n_ch_total))
    
    # Prepare list of channel pairs
    pairs = [(i, j, angle) for i in range(n_ch_total) for j in range(i, n_ch_total)]
    total_pairs = len(pairs)
    
    pbar = tqdm(total=total_pairs, desc="    accorr (denominator)", 
                disable=not show_progress, leave=False)
    
    # Use 'fork' context on Unix systems (macOS, Linux) for better performance
    # On Windows, this will fall back to 'spawn'
    ctx = get_context('fork' if sys.platform != 'win32' else 'spawn')
    
    with ctx.Pool(processes=n_jobs) as pool:
        for i, j, den_ij in pool.imap_unordered(_compute_pair_denominator, pairs):
            den[:, :, i, j] = den_ij
            den[:, :, j, i] = den_ij
            pbar.update(1)
    
    pbar.close()
    
    den = np.where(den == 0, 1, den)
    con = num / den
    con = con.swapaxes(0, 1)
    
    if epochs_average:
        con = np.nanmean(con, axis=1)
    
    return con


def _accorr_hybrid_precompute_torch(
    complex_signal: NDArray[np.complexfloating], 
    epochs_average: bool = True, 
    show_progress: bool = True,
    device = 'cpu'
) -> NDArray[np.floating]:
    """
    PyTorch-optimized version of Adjusted Circular Correlation with precomputation.
    
    Uses PyTorch for GPU acceleration and efficient tensor operations. This version
    computes the entire denominator matrix without loops using advanced broadcasting.
    
    Parameters
    ----------
    complex_signal : np.ndarray
        Complex analytic signals with shape (n_epochs, n_freq, 2*n_channels, n_times)
        
    epochs_average : bool, optional
        If True, connectivity values are averaged across epochs (default)
        If False, epoch-by-epoch connectivity is preserved
        
    show_progress : bool, optional
        If True, display a progress bar during computation (default)
        If False, no progress bar is shown
    
    Returns
    -------
    con : np.ndarray
        Adjusted circular correlation matrix with shape:
        - If epochs_average=True: (n_freq, 2*n_channels, 2*n_channels)
        - If epochs_average=False: (n_freq, n_epochs, 2*n_channels, 2*n_channels)
    
    Notes
    -----
    This implementation uses PyTorch to:
    1. Leverage GPU acceleration if available
    2. Eliminate all Python loops using advanced tensor broadcasting
    3. Optimize memory access patterns for better cache utilization
    
    The denominator computation is fully vectorized by expanding dimensions and
    using broadcasting to compute all channel pairs simultaneously.
    """
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA is not available on this computer')
    elif device == 'mps':
        if not torch.backends.mps.is_available():
            raise ValueError('MPS is not available on this computer')
    
    complex_type = torch.complex64 if device == 'mps' else torch.complex128
    
    # Convert to torch tensors (use double precision to match numpy)
    complex_tensor = torch.from_numpy(complex_signal).to(device=device, dtype=complex_type)
    
    n_epochs, n_freq, n_ch_total, n_times = complex_tensor.shape

    
    # Numerator (vectorized)
    z = complex_tensor / torch.abs(complex_tensor)
    c, s = z.real, z.imag
    
    # Cross products using einsum - matching the numpy implementation formula: 'jilm,jimk->jilk'
    # where j=epoch, i=freq, l=ch1, m/k=time
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
    
    # Denominator - fully vectorized using broadcasting
    angle = torch.angle(complex_tensor)
    
    # For the denominator, we need to compute for each pair (i,j):
    # x_sin = sin(alpha_i - m_adj[i,j])
    # y_sin = sin(alpha_j - n_adj[i,j])
    # where alpha_i has shape [e, f, t] for channel i
    
    # Expand angle dimensions: [e, f, i, t] -> [e, f, i, 1, t] and [e, f, 1, j, t]
    angle_i = angle.unsqueeze(3)  # [e, f, i, 1, t]
    angle_j = angle.unsqueeze(2)  # [e, f, 1, j, t]
    
    # Expand m_adj and n_adj: [e, f, i, j] -> [e, f, i, j, 1]
    m_adj = m_adj_all.unsqueeze(-1)  # [e, f, i, j, 1]
    n_adj = n_adj_all.unsqueeze(-1)  # [e, f, i, j, 1]
    
    # Compute sin terms with proper broadcasting:
    # For each pair (i,j), subtract m_adj[i,j] from angle[i,:] and n_adj[i,j] from angle[j,:]
    x_sin = torch.sin(angle_i - m_adj)  # [e, f, i, j, t] - broadcasts [e,f,i,1,t] - [e,f,i,j,1]
    y_sin = torch.sin(angle_j - n_adj)  # [e, f, i, j, t] - broadcasts [e,f,1,j,t] - [e,f,i,j,1]
    
    # Sum over time dimension and compute denominator
    x_sin_sq_sum = torch.sum(x_sin**2, dim=-1)  # [e, f, i, j]
    y_sin_sq_sum = torch.sum(y_sin**2, dim=-1)  # [e, f, i, j]
    
    den = 2 * torch.sqrt(x_sin_sq_sum * y_sin_sq_sum)
    
    # Avoid division by zero
    den = torch.where(den == 0, torch.ones_like(den), den)
    
    # Compute connectivity
    con = num / den
    con = con.permute(1, 0, 2, 3)  # [n_freq, n_epochs, n_ch, n_ch]
    
    if epochs_average:
        con = torch.nanmean(con, dim=1)
    
    # Convert back to numpy
    return con.cpu().numpy()