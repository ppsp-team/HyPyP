import numpy as np

# helper function
def _multiply_conjugate(real: np.ndarray, imag: np.ndarray, transpose_axes: tuple) -> np.ndarray:
    """
    Computes the product of a complex array and its conjugate efficiently.
    
    This helper function performs matrix multiplication between complex arrays
    represented by their real and imaginary parts, collapsing the last dimension.
    
    Parameters
    ----------
    real : np.ndarray
        Real part of the complex array
        
    imag : np.ndarray
        Imaginary part of the complex array
        
    transpose_axes : tuple
        Axes to transpose for matrix multiplication
    
    Returns
    -------
    product : np.ndarray
        Product of the array and its complex conjugate
    
    Notes
    -----
    This function implements the formula:
    product = (real × real.T + imag × imag.T) - i(real × imag.T - imag × real.T)
    
    Using einsum for efficient computation without explicitly creating complex arrays.
    """

    formula = 'jilm,jimk->jilk'
    product = np.einsum(formula, real, real.transpose(transpose_axes)) + \
              np.einsum(formula, imag, imag.transpose(transpose_axes)) - 1j * \
              (np.einsum(formula, real, imag.transpose(transpose_axes)) - \
               np.einsum(formula, imag, real.transpose(transpose_axes)))

    return product


# helper function
def _multiply_conjugate_time(real: np.ndarray, imag: np.ndarray, transpose_axes: tuple) -> np.ndarray:
    """
    Computes the product of a complex array and its conjugate without collapsing time dimension.
    
    Similar to _multiply_conjugate, but preserves the time dimension, which is
    needed for certain connectivity metrics like wPLI.
    
    Parameters
    ----------
    real : np.ndarray
        Real part of the complex array
        
    imag : np.ndarray
        Imaginary part of the complex array
        
    transpose_axes : tuple
        Axes to transpose for matrix multiplication
    
    Returns
    -------
    product : np.ndarray
        Product of the array and its complex conjugate with time dimension preserved
    
    Notes
    -----
    This function uses a different einsum formula than _multiply_conjugate:
    'jilm,jimk->jilkm' instead of 'jilm,jimk->jilk'
    
    This preserves the time dimension (m) in the output, which is necessary for 
    computing metrics that require individual time point values rather than 
    time-averaged products.
    """
    formula = 'jilm,jimk->jilkm'
    product = np.einsum(formula, real, real.transpose(transpose_axes)) + \
              np.einsum(formula, imag, imag.transpose(transpose_axes)) - 1j * \
              (np.einsum(formula, real, imag.transpose(transpose_axes)) - \
               np.einsum(formula, imag, real.transpose(transpose_axes)))
    
    return product


# helper function
def _multiply_product(real: np.ndarray, imag: np.ndarray, transpose_axes: tuple) -> np.ndarray:
    """
    Computes the product of two complex arrays (not conjugate) efficiently.
    
    This helper function performs matrix multiplication between complex arrays
    represented by their real and imaginary parts, collapsing the last dimension.
    Unlike _multiply_conjugate, this computes z1 * z2 instead of z1 * conj(z2).
    
    Parameters
    ----------
    real : np.ndarray
        Real part of the complex array
        
    imag : np.ndarray
        Imaginary part of the complex array
        
    transpose_axes : tuple
        Axes to transpose for matrix multiplication
    
    Returns
    -------
    product : np.ndarray
        Product of the array with itself (non-conjugate)
    
    Notes
    -----
    This function implements the formula for z1 * z2:
    product = (real × real.T - imag × imag.T) + i(real × imag.T + imag × real.T)
    
    Using einsum for efficient computation without explicitly creating complex arrays.
    This is used in the adjusted circular correlation (accorr) metric.
    """
    formula = 'jilm,jimk->jilk'
    product = np.einsum(formula, real, real.transpose(transpose_axes)) - \
              np.einsum(formula, imag, imag.transpose(transpose_axes)) + 1j * \
              (np.einsum(formula, real, imag.transpose(transpose_axes)) + \
               np.einsum(formula, imag, real.transpose(transpose_axes)))

    return product
