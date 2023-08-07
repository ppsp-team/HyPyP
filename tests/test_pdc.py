import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal
from scipy.fftpack import fft


class pdc_test(unittest.TestCase):

    def test_function(self):
        # three sources  a <- b <- c
        # check connectivity is zero where expected
        var_coeffs_init = np.array([[0, 0.9, 0], [0, 0, 0.9], [0, 0, 0]])
        m, mp = var_coeffs_init.shape
        n_fft = 10
        p = mp // m
        var_coeffs  = np.reshape(var_coeffs_init, (m, m, p), 'c')
        cov_mat = np.eye(3)
        a = fft(np.dstack([np.eye(m), -var_coeffs]), n_fft * 2 - 1)[:, :, :n_fft]
        pdc_result = np.abs(a / np.sqrt(np.sum(a.conj() * a, axis=0, keepdims=True)))
        assert_array_equal(np.isclose(np.sum(np.abs(a), 2), 0), np.isclose(var_coeffs_init + cov_mat, 0))
        self.assertFalse(np.all(np.sum(np.abs(a), 2) == np.sum(np.abs(np.transpose(a, axes=[1, 0, 2])), 2)))
        assert_array_equal(np.isclose(np.sum(np.abs(pdc_result), 2), 0), np.isclose(var_coeffs_init + cov_mat, 0))
        self.assertFalse(np.all(np.sum(pdc_result, 2) == np.sum(np.transpose(pdc_result, axes=[1, 0, 2]), 2)))
        self.assertEqual(np.sum(pdc_result, 2)[0, 0], n_fft)
        self.assertEqual(np.sum(pdc_result, 2)[1, 1], np.sum(pdc_result, 2)[2, 2])
        self.assertEqual(np.sum(pdc_result, 2)[0, 1], np.sum(pdc_result, 2)[1, 2])

if __name__ == '__main__':
    unittest.main()