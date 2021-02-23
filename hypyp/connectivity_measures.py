import numpy as np
import scipy as sp
from scipy.fftpack import fft
from functools import partial
import os
import mne
import math
from scipy.stats import kurtosis

"""Turn seed into a np.random.RandomState instance.
If seed is None, return the RandomState singleton used by np.random.
If seed is an int, return a new RandomState instance seeded with seed.
If seed is already a RandomState instance, return it.
Otherwise raise ValueError.
"""
def check_random_state(seed):
  if seed is None or seed is np.random:
    return np.random.mtrand._rand
  if isinstance(seed, (int, np.integer)):
    return np.random.RandomState(seed)
  if isinstance(seed, np.random.RandomState):
    return seed
  raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)
  

  
"""Cache the return value of a method.
This class is meant to be used as a decorator of methods. The return value
from a given method invocation will be cached on the instance whose method
was invoked. All arguments passed to a method decorated with memoize must
be hashable.
If a memoized method is invoked directly on its class the result will not
be cached. Instead the method will be invoked like a static method
"""

class memoize:

  def __init__(self, func):
    self.func = func

  #noinspection PyUnusedLocal
  def __get__(self, obj, objtype=None):
    if obj is None:
      return self.func
    return partial(self, obj)

  def __call__(self, *args, **kw):
    obj = args[0]
    try:
      cache = obj.__cache
    except AttributeError:
      cache = obj.__cache = {}
    key = (self.func, args[1:], frozenset(kw.items()))
    try:
      res = cache[key]
    except KeyError:
      res = cache[key] = self.func(*args, **kw)
    return res

#This function change the input to be: 1) a numpy array, 2) 3D -> compatiblity of data
def atleast_3d(x):
  x = np.asarray(x)
  if x.ndim >= 3:
    return x
  elif x.ndim == 2:
    return x[np.newaxis, ...]
  else:
    return x[np.newaxis, np.newaxis, :]

"""Concatenate trials along time axis.
    Parameters
    ----------
    x3d : array, shape (t, m, n)
        Segmented input data with t trials, m signals, and n samples.
    Returns
    -------
    x2d : array, shape (m, t * n)
        Trials are concatenated along the second axis.
    See also
    --------
    cut_segments : Cut segments from continuous data.
    Examples
    --------
    >>> x = np.random.randn(6, 4, 150)
    >>> y = cat_trials(x)
    >>> y.shape
    (4, 900)
"""
def cat_trials(x3d):
  x3d = atleast_3d(x3d)
  t = x3d.shape[0]
  return np.concatenate(np.split(x3d, t, 0), axis=2).squeeze(0)


"""Segment-wise dot product.
    This function calculates the dot product of x2d with each trial of x3d.
    Parameters
    ----------
    x2d : array, shape (p, m)
        Input argument.
    x3d : array, shape (t, m, n)
        Segmented input data with t trials, m signals, and n samples. The dot
        product with x2d is calculated for each trial.
    Returns
    -------
    out : array, shape (t, p, n)
        Dot product of x2d with each trial of x3d.
    Examples
    --------
    >>> x = np.random.randn(6, 40, 150)
    >>> a = np.ones((7, 40))
    >>> y = dot_special(a, x)
    >>> y.shape
    (6, 7, 150)
"""
def dot_special(x2d, x3d):
  x3d = atleast_3d(x3d)
  x2d = np.atleast_2d(x2d)
  return np.concatenate([x2d.dot(x3d[i, ...])[np.newaxis, ...] for i in range(x3d.shape[0])])


"""Compute autocovariance matrix at lag l.
    This function calculates the autocovariance matrix of `x` at lag `l`.
    Parameters
    ----------
    x : array, shape (n_trials, n_channels, n_samples)
        Signal data (2D or 3D for multiple trials)
    l : int
        Lag
    Returns
    -------
    c : ndarray, shape = [nchannels, n_channels]
        Autocovariance matrix of `x` at lag `l`.
"""
def acm(x, l):
  x = atleast_3d(x)
  if l > x.shape[2]-1:
    raise AttributeError("lag exceeds data length")

  if l == 0:
    a, b = x, x
  else:
    a = x[:, :, l:]
    b = x[:, :, 0:-l]

  c = np.zeros((x.shape[1], x.shape[1]))
  for t in range(x.shape[0]):
    c += a[t, :, :].dot(b[t, :, :].T) / a.shape[2]
  c /= x.shape[0]

  return c.T


""" Multi-trial cross-validation schema
    Use one trial for testing, all others for training.
    Parameters
    ----------
    num_trials : int
        Total number of trials
    skipstep : int
        only use every `skipstep` trial for testing
    Returns
    -------
    gen : generator object
        the generator returns tuples (trainset, testset)
"""
def multitrial(num_trials, skipstep=1):

  for t in range(0, num_trials, skipstep):
    testset = [t]
    trainset = [i for i in range(testset[0])] + [i for i in range(testset[-1] + 1, num_trials)]
    trainset = sort([t % num_trials for t in trainset])
    yield trainset, testset


"""run loops in parallel, if joblib is available.
    Parameters
    ----------
    func : function
        function to be executed in parallel
    n_jobs : int | None
        Number of jobs. If set to None, do not attempt to use joblib.
    verbose : int
        verbosity level
    Notes
    -----
    Execution of the main script must be guarded with `if __name__ == '__main__':` when using parallelization.
"""
def parallel_loop(func, n_jobs=1, verbose=1):

  if n_jobs:
    try:
      from joblib import Parallel, delayed
    except ImportError:
      try:
        from sklearn.externals.joblib import Parallel, delayed
      except ImportError:
        n_jobs = None

  if not n_jobs:
    if verbose:
      print('running ', func, ' serially')
    par = lambda x: list(x)
  else:
    if verbose:
      print('running ', func, ' in parallel')
    func = delayed(func)
    par = Parallel(n_jobs=n_jobs, verbose=verbose)

  return par, func

class logger:
  @staticmethod
  def info(*args, **kwargs):
    pass

"""Run the (extended) Infomax ICA decomposition on raw data
    based on the publications of Bell & Sejnowski 1995 (Infomax)
    and Lee, Girolami & Sejnowski, 1999 (extended Infomax)
    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        The data to unmix.
    w_init : np.ndarray, shape (n_features, n_features)
        The initialized unmixing matrix. Defaults to None. If None, the
        identity matrix is used.
    l_rate : float
        This quantity indicates the relative size of the change in weights.
        Note. Smaller learining rates will slow down the procedure.
        Defaults to 0.010d / alog(n_features ^ 2.0)
    block : int
        The block size of randomly chosen data segment.
        Defaults to floor(sqrt(n_times / 3d))
    w_change : float
        The change at which to stop iteration. Defaults to 1e-12.
    anneal_deg : float
        The angle at which (in degree) the learning rate will be reduced.
        Defaults to 60.0
    anneal_step : float
        The factor by which the learning rate will be reduced once
        ``anneal_deg`` is exceeded:
            l_rate *= anneal_step
        Defaults to 0.9
    extended : bool
        Wheather to use the extended infomax algorithm or not. Defaults to
        True.
    n_subgauss : int
        The number of subgaussian components. Only considered for extended
        Infomax.
    kurt_size : int
        The window size for kurtosis estimation. Only considered for extended
        Infomax.
    ext_blocks : int
        The number of blocks after which to recompute Kurtosis.
        Only considered for extended Infomax.
    max_iter : int
        The maximum number of iterations. Defaults to 200.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    Returns
    -------
    unmixing_matrix : np.ndarray of float, shape (n_features, n_features)
        The linear unmixing operator.
"""
def infomax(data, weights=None, l_rate=None, block=None, w_change=1e-12,
            anneal_deg=60., anneal_step=0.9, extended=False, n_subgauss=1,
            kurt_size=6000, ext_blocks=1, max_iter=200,
            random_state=None, verbose=None):

  rng = check_random_state(random_state)

  # define some default parameter
  max_weight = 1e8
  restart_fac = 0.9
  min_l_rate = 1e-10
  blowup = 1e4
  blowup_fac = 0.5
  n_small_angle = 20
  degconst = 180.0 / np.pi

  # for extended Infomax
  extmomentum = 0.5
  signsbias = 0.02
  signcount_threshold = 25
  signcount_step = 2
  if ext_blocks > 0:  # allow not to recompute kurtosis
    n_subgauss = 1  # but initialize n_subgauss to 1 if you recompute

  # check data shape
  n_samples, n_features = data.shape
  n_features_square = n_features ** 2

  # check input parameter
  # heuristic default - may need adjustment for
  # large or tiny data sets
  if l_rate is None:
    l_rate = 0.01 / math.log(n_features ** 2.0)

  if block is None:
    block = int(math.floor(math.sqrt(n_samples / 3.0)))

  logger.info('computing%sInfomax ICA' % ' Extended ' if extended is True
                else ' ')

  # collect parameter
  nblock = n_samples // block
  lastt = (nblock - 1) * block + 1

  # initialize training
  if weights is None:
  # initialize weights as identity matrix
    weights = np.identity(n_features, dtype=np.float64)

  BI = block * np.identity(n_features, dtype=np.float64)
  bias = np.zeros((n_features, 1), dtype=np.float64)
  onesrow = np.ones((1, block), dtype=np.float64)
  startweights = weights.copy()
  oldweights = startweights.copy()
  step = 0
  count_small_angle = 0
  wts_blowup = False
  blockno = 0
  signcount = 0

  # for extended Infomax
  if extended is True:
    signs = np.identity(n_features)
    signs.flat[slice(0, n_features * n_subgauss, n_features)]
    kurt_size = min(kurt_size, n_samples)
    old_kurt = np.zeros(n_features, dtype=np.float64)
    oldsigns = np.zeros((n_features, n_features))

  # trainings loop
  olddelta, oldchange = 1., 0.
  while step < max_iter:
    # shuffle data at each step
    permute = list(range(n_samples))
    rng.shuffle(permute)

    # ICA training block
    # loop across block samples
    for t in range(0, lastt, block):
      u = np.dot(data[permute[t:t + block], :], weights)
      u += np.dot(bias, onesrow).T

      if extended is True:
        # extended ICA update
        y = np.tanh(u)
        weights += l_rate * np.dot(weights, BI - np.dot(np.dot(u.T, y), signs) - np.dot(u.T, u))
        bias += l_rate * np.reshape(np.sum(y, axis=0, dtype=np.float64) * -2.0, (n_features, 1))

      else:
        # logistic ICA weights update
        y = 1.0 / (1.0 + np.exp(-u))
        weights += l_rate * np.dot(weights, BI + np.dot(u.T, (1.0 - 2.0 * y)))
        bias += l_rate * np.reshape(np.sum((1.0 - 2.0 * y), axis=0,
                                            dtype=np.float64), (n_features, 1))

      # check change limit
      max_weight_val = np.max(np.abs(weights))
      if max_weight_val > max_weight:
        wts_blowup = True

      blockno += 1
      if wts_blowup:
        break

      # ICA kurtosis estimation
      if extended is True:
        n = np.fix(blockno / ext_blocks)
        if np.abs(n) * ext_blocks == blockno:
          if kurt_size < n_samples:
            rp = np.floor(rng.uniform(0, 1, kurt_size) * (n_samples - 1))
            tpartact = np.dot(data[rp.astype(int), :], weights).T
          else:
            tpartact = np.dot(data, weights).T

          # estimate kurtosis
          kurt = kurtosis(tpartact, axis=1, fisher=True)

          if extmomentum != 0:
            kurt = (extmomentum * old_kurt + (1.0 - extmomentum) * kurt)
            old_kurt = kurt

          # estimate weighted signs
          signs.flat[::n_features + 1] = ((kurt + signsbias) /
                                          np.abs(kurt + signsbias))

          ndiff = ((signs.flat[::n_features + 1] - oldsigns.flat[::n_features + 1]) != 0).sum()
                    
          if ndiff == 0:
            signcount += 1
          else:
            signcount = 0
          oldsigns = signs

          if signcount >= signcount_threshold:
            ext_blocks = np.fix(ext_blocks * signcount_step)
            signcount = 0

    # here we continue after the for
    # loop over the ICA training blocks
    # if weights in bounds:
    if not wts_blowup:
      oldwtchange = weights - oldweights
      step += 1
      angledelta = 0.0
      delta = oldwtchange.reshape(1, n_features_square)
      change = np.sum(delta * delta, dtype=np.float64)
      if step > 1:
        angledelta = math.acos(np.sum(delta * olddelta) /
                                       math.sqrt(change * oldchange))
        angledelta *= degconst

      # anneal learning rate
      oldweights = weights.copy()
      if angledelta > anneal_deg:
        l_rate *= anneal_step    # anneal learning rate
        # accumulate angledelta until anneal_deg reached l_rates
        olddelta = delta
        oldchange = change
        count_small_angle = 0  # reset count when angle delta is large
      else:
        if step == 1:  # on first step only
            olddelta = delta  # initialize
            oldchange = change
        count_small_angle += 1
        if count_small_angle > n_small_angle:
          max_iter = step

      # apply stopping rule
      if step > 2 and change < w_change:
        step = max_iter
      elif change > blowup:
        l_rate *= blowup_fac

    # restart if weights blow up
    # (for lowering l_rate)
    else:
      step = 0  # start again
      wts_blowup = 0  # re-initialize variables
      blockno = 1
      l_rate *= restart_fac  # with lower learning rate
      weights = startweights.copy()
      oldweights = startweights.copy()
      olddelta = np.zeros((1, n_features_square), dtype=np.float64)
      bias = np.zeros((n_features, 1), dtype=np.float64)

      # for extended Infomax
      if extended:
        signs = np.identity(n_features)
        signs.flat[slice(0, n_features * n_subgauss, n_features)]
        oldsigns = np.zeros((n_features, n_features))

      if l_rate > min_l_rate:
        if verbose:
          logger.info('... lowering learning rate to %g'
                                '\n... re-starting...' % l_rate)
      else:
        raise ValueError('Error in Infomax ICA: unmixing_matrix matrix'
                                 'might not be invertible!')

  # prepare return values
  return weights.T

"""Represents a vector autoregressive (VAR) model.
    .. warning:: `VARBase` is an abstract class that defines the interface for
    VAR model implementations. Several methods must be implemented by derived
    classes.
    Parameters
    ----------
    model_order : int
        Autoregressive model order.
    n_jobs : int | None, optional
        Number of jobs to run in parallel for various tasks (e.g. whiteness
        testing). If set to None, joblib is not used at all. Note that the main
        script must be guarded with `if __name__ == '__main__':` when using
        parallelization.
    verbose : bool | None, optional
        Whether to print information to stdout. The default is None, which
        means the verbosity setting from the global configuration is used.
    Notes
    -----
    *b* is of shape [m, m*p], with sub matrices arranged as follows:
    +------+------+------+------+
    | b_00 | b_01 | ...  | b_0m |
    +------+------+------+------+
    | b_10 | b_11 | ...  | b_1m |
    +------+------+------+------+
    | ...  | ...  | ...  | ...  |
    +------+------+------+------+
    | b_m0 | b_m1 | ...  | b_mm |
    +------+------+------+------+
    Each sub matrix b_ij is a column vector of length p that contains the
    filter coefficients from channel j (source) to channel i (sink).
"""
class VARBase(object):

  def __init__(self, model_order, n_jobs=1, verbose=None):
    self.p = model_order
    self.coef = None
    self.residuals = None
    self.rescov = None
    self.n_jobs = n_jobs

  def copy(self):
    #Create a copy of the VAR model.
    other = self.__class__(self.p)
    other.coef = self.coef.copy()
    other.residuals = self.residuals.copy()
    other.rescov = self.rescov.copy()
    return other

  def fit(self, data):
    raise NotImplementedError('method fit() is not implemented in ' +
                                  str(self))

  def optimize(self, data):
    raise NotImplementedError('method optimize() is not implemented in ' +
                                  str(self))

  def from_yw(self, acms):
  #Determine VAR model from autocorrelation matrices by solving the
  #Yule-Walker equations.
  #Parameters
  #acms : array, shape (n_lags, n_channels, n_channels)
  #acms[l] contains the autocorrelation matrix at lag l. The highest
  #lag must equal the model order.
  # Returns :self : :class:`VAR' The :class:`VAR` object to facilitate method chaining 

    if len(acms) != self.p + 1:
      raise ValueError("Number of autocorrelation matrices ({}) does not"
                             " match model order ({}) + 1.".format(len(acms),
                                                                   self.p))

    n_channels = acms[0].shape[0]

    acm = lambda l: acms[l] if l >= 0 else acms[-l].T

    r = np.concatenate(acms[1:], 0)

    rr = np.array([[acm(m-k) for k in range(self.p)]
                      for m in range(self.p)])
    rr = np.concatenate(np.concatenate(rr, -2), -1)

    c = sp.linalg.solve(rr, r)

    # calculate residual covariance
    r = acm(0)
    for k in range(self.p):
      bs = k * n_channels
      r -= np.dot(c[bs:bs + n_channels, :].T, acm(k + 1))

      self.coef = np.concatenate([c[m::n_channels, :]
                                    for m in range(n_channels)]).T
      self.rescov = r
      return self

  def simulate(self, l, noisefunc=None, random_state=None):
        #Simulate vector autoregressive (VAR) model.
        #This function generates data from the VAR model.
        #Parameters
       # ----------
       # l : int or [int, int]
            #Number of samples to generate. Can be a tuple or list, where l[0]
            #is the number of samples and l[1] is the number of trials.
        #noisefunc : func, optional
            #This function is used to create the generating noise process. If
            #set to None, Gaussian white noise with zero mean and unit variance
            #is used.
        #Returns
        #-------
        #data : array, shape (n_trials, n_samples, n_channels)
            #Generated data.
    m, n = np.shape(self.coef)
    p = n // m

    try:
      l, t = l
    except TypeError:
      t = 1

    if noisefunc is None:
      rng = check_random_state(random_state)
      noisefunc = lambda: rng.normal(size=(1, m))

    n = l + 10 * p

    y = np.zeros((n, m, t))
    res = np.zeros((n, m, t))

    for s in range(t):
      for i in range(p):
          e = noisefunc()
          res[i, :, s] = e
          y[i, :, s] = e
      for i in range(p, n):
          e = noisefunc()
          res[i, :, s] = e
          y[i, :, s] = e
          for k in range(1, p + 1):
            y[i, :, s] += self.coef[:, (k - 1)::p].dot(y[i - k, :, s])

    self.residuals = res[10 * p:, :, :].T
    self.rescov = sp.cov(cat_trials(self.residuals).T, rowvar=False)

    return y[10 * p:, :, :].transpose([2, 1, 0])

  def predict(self, data):
  #Predict samples on actual data.
  #The result of this function is used for calculating the residuals.
  #Parameters
  #----------
  # data : array, shape (trials, channels, samples) or (channels, samples)
  #Epoched or continuous data set.
  #Returns
  #-------
  #predicted : array, shape `data`.shape
  #Data as predicted by the VAR model.
  #Notes
  #-----
  #Residuals are obtained by r = x - var.predict(x)
    data = atleast_3d(data)
    t, m, l = data.shape

    p = int(np.shape(self.coef)[1] / m)

    y = np.zeros(data.shape)
    if t > l - p:  # which takes less loop iterations
      for k in range(1, p + 1):
        bp = self.coef[:, (k - 1)::p]
        for n in range(p, l):
          y[:, :, n] += np.dot(data[:, :, n - k], bp.T)
    else:
      for k in range(1, p + 1):
        bp = self.coef[:, (k - 1)::p]
        for s in range(t):
          y[s, :, p:] += np.dot(bp, data[s, :, (p - k):(l - k)])

    return y

  def is_stable(self):
  #Test if VAR model is stable.
  #This function tests stability of the VAR model as described in [1]_.
  #Returns
  #-------
  #out : bool
  #True if the model is stable.
  #References
  #----------
  #.. [1] H. Lütkepohl, "New Introduction to Multiple Time Series Analysis", 2005, Springer, Berlin, Germany.

    m, mp = self.coef.shape
    p = mp // m
    assert(mp == m * p)  # TODO: replace with raise?

    top_block = []
    for i in range(p):
      top_block.append(self.coef[:, i::p])
    top_block = np.hstack(top_block)

    im = np.eye(m)
    eye_block = im
    for i in range(p - 2):
      eye_block = sp.linalg.block_diag(im, eye_block)
    eye_block = np.hstack([eye_block, np.zeros((m * (p - 1), m))])

    tmp = np.vstack([top_block, eye_block])

    return np.all(np.abs(np.linalg.eig(tmp)[0]) < 1)

  def test_whiteness(self, h, repeats=100, get_q=False, random_state=None):

    return test_whiteness(self.residuals, h=h, p=self.p, repeats=repeats,
                              get_q=get_q, n_jobs=self.n_jobs,
                              verbose=self.verbose, random_state=random_state)

  def _construct_eqns(self, data):
  #Construct VAR equation system.
    return _construct_var_eqns(data, self.p)


def _construct_var_eqns(data, p, delta=None):
#Construct VAR equation system (optionally with RLS constraint).

  t, m, l = np.shape(data)
  n = (l - p) * t  # number of linear relations
  rows = n if delta is None else n + m * p

  # Construct matrix x (predictor variables)
  x = np.zeros((rows, m * p))
  for i in range(m):
    for k in range(1, p + 1):
      x[:n, i * p + k - 1] = np.reshape(data[:, i, p - k:-k].T, n)
  if delta is not None:
    np.fill_diagonal(x[n:, :], delta)

  # Construct vectors yi (response variables for each channel i)
  y = np.zeros((rows, m))
  for i in range(m):
    y[:n, i] = np.reshape(data[:, i, p:].T, n)

  return x, y


def test_whiteness(data, h, p=0, repeats=100, get_q=False, n_jobs=1,
                   verbose=0, random_state=None):
    #Test if signals are white (serially uncorrelated up to a lag of h).
    #This function calculates the Li-McLeod Portmanteau test statistic Q to test
    #against the null hypothesis H0 (the residuals are white) [1]_.
    #Surrogate data for H0 is created by sampling from random permutations of
    #the residuals.
    #Usually, the returned p-value is compared against a pre-defined type I
    #error level of alpha=0.05 or alpha=0.01. If p<=alpha, the hypothesis of
    #white residuals is rejected, which indicates that the VAR model does not
    #adequately describe the data.
    #Parameters
    #----------
    #data : array, shape (trials, channels, samples) or (channels, samples)
        #Epoched or continuous data set.
    #h : int
        #Maximum lag that is included in the test statistic.
    #p : int, optional
        #Model order (if `data` are the residuals resulting from fitting a VAR model).
    #repeats : int, optional
        #Number of samples to create under the null hypothesis.
    #get_q : bool, optional
        #Return Q statistic along with *p*-value
   # n_jobs : int | None, optional
        #Number of jobs to run in parallel. If set to None, joblib is not used at all. 
    #verbose : int
        #Verbosity level passed to joblib.
    #Returns
    #-------
    #pr : float
        #Probability of observing a more extreme value of Q under the assumption that H0 is true.
    #q0 : list of float, optional (`get_q`)
        #Individual surrogate estimates that were used for estimating the
        #distribution of Q under H0.
    #q : float, optional (`get_q`)
        #Value of the Q statistic of the residuals.
    #Notes
    #-----
    #According to [2]_, h must satisfy h = O(n^0.5), where n is the length (time
    #samples) of the residuals.
    #References
    #[1] H. Lütkepohl, "New Introduction to Multiple Time Series Analysis",
           #2005, Springer, Berlin, Germany.
    #[2] J.R.M. Hosking, "The Multivariate Portmanteau Statistic", 1980, J. Am. Statist. Assoc.

  res = data[:, :, p:]
  t, m, n = res.shape
  nt = (n - p) * t

  q0 = _calc_q_h0(repeats, res, h, nt, n_jobs, verbose,
                    random_state=random_state)[:, 2, -1]
  q = _calc_q_statistic(res, h, nt)[2, -1]

  # probability of observing a result more extreme than q
  # under the null-hypothesis
  pr = np.sum(q0 >= q) / repeats

  if get_q:
    return pr, q0, q
  else:
    return pr


def _calc_q_statistic(x, h, nt):
    #Calculate Portmanteau statistics up to a lag of h.

  t, m, n = x.shape

    # covariance matrix of x
  c0 = acm(x, 0)

  # LU factorization of covariance matrix
  c0f = sp.linalg.lu_factor(c0, overwrite_a=False, check_finite=True)

  q = np.zeros((3, h + 1))
  for l in range(1, h + 1):
    cl = acm(x, l)

    # calculate tr(cl' * c0^-1 * cl * c0^-1)
    a = sp.linalg.lu_solve(c0f, cl)
    b = sp.linalg.lu_solve(c0f, cl.T)
    tmp = a.dot(b).trace()

    # Box-Pierce
    q[0, l] = tmp

    # Ljung-Box
    q[1, l] = tmp / (nt - l)

    # Li-McLeod
    q[2, l] = tmp

  q *= nt
  q[1, :] *= (nt + 2)

  q = np.cumsum(q, axis=1)

  for l in range(1, h+1):
    q[2, l] = q[0, l] + m * m * l * (l + 1) / (2 * nt)

  return q


def _calc_q_h0(n, x, h, nt, n_jobs=1, verbose=0, random_state=None):
    #Calculate q under the null hypothesis of whiteness.

  rng = check_random_state(random_state)
  par, func = parallel_loop(_calc_q_statistic, n_jobs, verbose)
  q = par(func(rng.permutation(x.T).T, h, nt) for _ in range(n))
    
  return np.array(q)
  
"""Builtin VAR implementation.
    This class provides least squares VAR model fitting with optional ridge
    regression.
    
    Parameters    
    ----------
    model_order : int
        Autoregressive model order.
    delta : float, optional
        Ridge penalty parameter.
    xvschema : func, optional
        Function that creates training and test sets for cross-validation. The
        function takes two parameters: the current cross-validation run (int)
        and the number of trials (int). It returns a tuple of two arrays: the
        training set and the testing set.
    n_jobs : int | None, optional
        Number of jobs to run in parallel for various tasks (e.g. whiteness
        testing). If set to None, joblib is not used at all. Note that the main
        script must be guarded with `if __name__ == '__main__':` when using
        parallelization.
    verbose : bool | None, optional
        Whether to print information to stdout. The default is None, which
        means the verbosity setting from the global configuration is used.
"""
class VAR(VARBase):
  def __init__(self, model_order, delta=0, xvschema=multitrial, n_jobs=1,verbose=None):
    super(VAR, self).__init__(model_order=model_order, n_jobs=n_jobs, verbose = verbose)
    self.delta = delta
    self.xvschema = xvschema

  def fit(self, data):
    
    #Fit VAR model to data.
        
    #Parameters
    #----------
    #data : array, shape (trials, channels, samples) or (channels, samples)
    #Epoched or continuous data set.
            
    #Returns
    #-------
    #self : :class:`VAR`
    #The :class:`VAR` object to facilitate method chaining

    data = atleast_3d(data)

    if self.delta == 0 or self.delta is None:
      # ordinary least squares
      x, y = self._construct_eqns(data)
    else:
      # regularized least squares (ridge regression)
      x, y = self._construct_eqns_rls(data)

    b, res, rank, s = sp.linalg.lstsq(x, y)

    self.coef = b.transpose()

    self.residuals = data - self.predict(data)
    self.rescov = sp.cov(cat_trials(self.residuals[:, :, self.p:]))

    return self

  def optimize_order(self, data, min_p=1, max_p=None):
        #Determine optimal model order by minimizing the mean squared
        #generalization error.
        #Parameters
       # ----------
        #data : array, shape (n_trials, n_channels, n_samples)
           # Epoched data set on which to optimize the model order. At least two
           # trials are required.
       # min_p : int
           # Minimal model order to check.
        #max_p : int
           # Maximum model order to check
    data = np.asarray(data)
    if data.shape[0] < 2:
      raise ValueError("At least two trials are required.")

    msge, prange = [], []

    par, func = parallel_loop(_get_msge_with_gradient, n_jobs=self.n_jobs,
                                  verbose=self.verbose)
    if self.n_jobs is None:
      npar = 1
    elif self.n_jobs < 0:
      npar = 4  # is this a sane default?
    else:
      npar = self.n_jobs

    p = min_p
    while True:
      result = par(func(data, self.delta, self.xvschema, 1, p_) for p_ in range(p, p + npar))
      j, k = zip(*result)
      prange.extend(range(p, p + npar))
      msge.extend(j)
      p += npar
      if max_p is None:
        if len(msge) >= 2 and msge[-1] > msge[-2]:
          break
        else:
          if prange[-1] >= max_p:
            i = prange.index(max_p) + 1
            prange = prange[:i]
            msge = msge[:i]
            break
    self.p = prange[np.argmin(msge)]
    return zip(prange, msge)

  def optimize_delta_bisection(self, data, skipstep=1, verbose=None):
  #Find optimal ridge penalty with bisection search.
        
  #Parameters
  #----------
  #data : array, shape (n_trials, n_channels, n_samples)
  #Epoched data set. At least two trials are required.
  #skipstep : int, optional
  #Speed up calculation by skipping samples during cost function calculation.
  #Returns self : :class:`VAR`

    data = atleast_3d(data)
    if data.shape[0] < 2:
      raise ValueError("At least two trials are required.")

    if verbose is None:
      verbose = config.getboolean('scot', 'verbose')

    maxsteps = 10
    maxdelta = 1e50

    a = -10
    b = 10

    trform = lambda x: np.sqrt(np.exp(x))

    msge = _get_msge_with_gradient_func(data.shape, self.p)

    ja, ka = msge(data, trform(a), self.xvschema, skipstep, self.p)
    jb, kb = msge(data, trform(b), self.xvschema, skipstep, self.p)

    # before starting the real bisection, assure the interval contains 0
    while np.sign(ka) == np.sign(kb):
      if verbose:
        print('Bisection initial interval (%f,%f) does not contain 0. '
                      'New interval: (%f,%f)' % (a, b, a * 2, b * 2))
      a *= 2
      b *= 2
      ja, ka = msge(data, trform(a), self.xvschema, skipstep, self.p)
      jb, kb = msge(data, trform(b), self.xvschema, skipstep, self.p)

      if trform(b) >= maxdelta:
        if verbose:
          print('Bisection: could not find initial interval.')
          print(' ********* Delta set to zero! ************ ')
        return 0

    nsteps = 0

    while nsteps < maxsteps:
      c = (a + b) / 2
      j, k = msge(data, trform(c), self.xvschema, skipstep, self.p)
      if np.sign(k) == np.sign(ka):
        a, ka = c, k
      else:
        b, kb = c, k

      nsteps += 1
      tmp = trform([a, b, a + (b - a) * np.abs(ka) / np.abs(kb - ka)])
      if verbose:
        print('%d Bisection Interval: %f - %f, (projected: %f)' %
                      (nsteps, tmp[0], tmp[1], tmp[2]))

    self.delta = trform(a + (b - a) * np.abs(ka) / np.abs(kb - ka))
    if verbose:
      print('Final point: %f' % self.delta)
    return self
        
  optimize = optimize_delta_bisection

  def _construct_eqns_rls(self, data):
  #Construct VAR equation system with RLS constraint
    return _construct_var_eqns(data, self.p, self.delta)


def _msge_with_gradient_underdetermined(data, delta, xvschema, skipstep, p):
    #Calculate mean squared generalization error and its gradient for
    #underdetermined equation system.
 
  t, m, l = data.shape
  d = None
  j, k = 0, 0
  nt = np.ceil(t / skipstep)
  for trainset, testset in xvschema(t, skipstep):

    a, b = _construct_var_eqns(atleast_3d(data[trainset, :, :]), p)
    c, d = _construct_var_eqns(atleast_3d(data[testset, :, :]), p)

    e = sp.linalg.inv(np.eye(a.shape[0]) * delta ** 2 + a.dot(a.T))

    cc = c.transpose().dot(c)

    be = b.transpose().dot(e)
    bee = be.dot(e)
    bea = be.dot(a)
    beea = bee.dot(a)
    beacc = bea.dot(cc)
    dc = d.transpose().dot(c)

    j += np.sum(beacc * bea - 2 * bea * dc) + np.sum(d ** 2)
    k += np.sum(beea * dc - beacc * beea) * 4 * delta

  return j / (nt * d.size), k / (nt * d.size)


def _msge_with_gradient_overdetermined(data, delta, xvschema, skipstep, p):
    #Calculate mean squared generalization error and its gradient for
    #overdetermined equation system.
  t, m, l = data.shape
  d = None
  l, k = 0, 0
  nt = np.ceil(t / skipstep)
  for trainset, testset in xvschema(t, skipstep):

    a, b = _construct_var_eqns(atleast_3d(data[trainset, :, :]), p)
    c, d = _construct_var_eqns(atleast_3d(data[testset, :, :]), p)

    e = sp.linalg.inv(np.eye(a.shape[1]) * delta ** 2 + a.T.dot(a))

    ba = b.transpose().dot(a)
    dc = d.transpose().dot(c)
    bae = ba.dot(e)
    baee = bae.dot(e)
    baecc = bae.dot(c.transpose().dot(c))

    l += np.sum(baecc * bae - 2 * bae * dc) + np.sum(d ** 2)
    k += np.sum(baee * dc - baecc * baee) * 4 * delta

  return l / (nt * d.size), k / (nt * d.size)


def _get_msge_with_gradient_func(shape, p):
    #Select which function to use for MSGE calculation (over- or
    #underdetermined).
  t, m, l = shape

  n = (l - p) * t
  underdetermined = n < m * p

  if underdetermined:
    return _msge_with_gradient_underdetermined
  else:
    return _msge_with_gradient_overdetermined


def _get_msge_with_gradient(data, delta, xvschema, skipstep, p):
    #Calculate mean squared generalization error and its gradient,
    #automatically selecting the best function.
  t, m, l = data.shape

  n = (l - p) * t
  underdetermined = n < m * p

  if underdetermined:
    return _msge_with_gradient_underdetermined(data, delta, xvschema,
                                                   skipstep, p)
  else:
    return _msge_with_gradient_overdetermined(data, delta, xvschema, skipstep, p)
    
def generate():
  def wrapper_infomax(data, random_state=None):
    u = infomax(cat_trials(data).T, extended=True,
                    random_state=random_state).T
    m = sp.linalg.pinv(u)
    return m, u
  return {'ica': wrapper_infomax, 'var': VAR}  

""" Performs joint VAR model fitting and ICA source separation.
    
    This function implements the MVARICA procedure [1]_.
    
    Parameters
    ----------
    x : array-like, shape = [n_trials, n_channels, n_samples] or [n_channels, n_samples]
        data set
    var : :class:`~scot.var.VARBase`-like object
        Vector autoregressive model (VAR) object that is used for model fitting.
    cl : list of valid dict keys, optional
        Class labels associated with each trial.
    reducedim : {int, float, 'no_pca', None}, optional
        A number of less than 1 is interpreted as the fraction of variance that should remain in the data. All
        components that describe in total less than `1-reducedim` of the variance are removed by the PCA step.
        An integer number of 1 or greater is interpreted as the number of components to keep after applying PCA.
        If set to None, all PCA components are retained. If set to 'no_pca', the PCA step is skipped.
    optimize_var : bool, optional
        Whether to call automatic optimization of the VAR fitting routine.
    backend : dict-like, optional
        Specify backend to use. When set to None the backend configured in config.backend is used.
    varfit : string
        Determines how to calculate the residuals for source decomposition.
        'ensemble' (default) fits one model to the whole data set,
        'class' fits a new model for each class, and
        'trial' fits a new model for each individual trial.
        
    Returns
    -------
    result : class
        A class with the following attributes is returned:
            
        +---------------+----------------------------------------------------------+
        | mixing        | Source mixing matrix                                     |
        +---------------+----------------------------------------------------------+
        | unmixing      | Source unmixing matrix                                   |
        +---------------+----------------------------------------------------------+
        | residuals     | Residuals of the VAR model(s) in source space            |
        +---------------+----------------------------------------------------------+
        | var_residuals | Residuals of the VAR model(s) in EEG space (before ICA)  |
        +---------------+----------------------------------------------------------+
        | c             | Noise covariance of the VAR model(s) in source space     |
        +---------------+----------------------------------------------------------+
        | b             | VAR model coefficients (source space)                    |
        +---------------+----------------------------------------------------------+
        | a             | VAR model coefficients (EEG space)                       |
        +---------------+----------------------------------------------------------+
        
    Notes
    -----
    MVARICA is performed with the following steps:        
    1. Optional dimensionality reduction with PCA
    2. Fitting a VAR model tho the data
    3. Decomposing the VAR model residuals with ICA
    4. Correcting the VAR coefficients
        
    References
    ----------
    .. [1] G. Gomez-Herrero et al. "Measuring directional coupling between EEG sources", NeuroImage, 2008
"""

def mvarica(x, var, optimize_var=False, backend=None, varfit='ensemble', random_state=None):

  x = atleast_3d(x)
  t, m, l = np.shape(x)

  if backend is None:
    backend = generate()

  c = np.eye(m)
  d = np.eye(m)
  xpca = x

  if optimize_var:
    var.optimize(xpca)

  if varfit == 'trial':
    r = np.zeros(xpca.shape)
    for i in range(t):
      # fit MVAR model
      a = var.fit(xpca[i, :, :])
      # residuals
      r[i, :, :] = xpca[i, :, :] - var.predict(xpca[i, :, :])[0, :, :]
  elif varfit == 'class':
    r = np.zeros(xpca.shape)
    for i in np.unique(cl):
      mask = cl == i
      a = var.fit(xpca[mask, :, :])
      r[mask, :, :] = xpca[mask, :, :] - var.predict(xpca[mask, :, :])
  elif varfit == 'ensemble':
    # fit MVAR model
    a = var.fit(xpca)
    # residuals
    r = xpca - var.predict(xpca)
  else:
    raise ValueError('unknown VAR fitting mode: {}'.format(varfit))

  # run on residuals ICA to estimate volume conduction    
  mx, ux = backend['ica'](cat_trials(r), random_state=random_state)

  # driving process
  e = dot_special(ux.T, r)

  # correct AR coefficients
  b = a.copy()
  for k in range(0, a.p):
    b.coef[:, k::a.p] = mx.dot(a.coef[:, k::a.p].transpose()).dot(ux).transpose()

  # correct (un)mixing matrix estimatees
  mx = mx.dot(d)
  ux = c.dot(ux)

  class Result:
    unmixing = ux
    mixing = mx
    residuals = e
    var_residuals = r
    c = np.cov(cat_trials(e).T, rowvar=False)

  Result.b = b
  Result.a = a
  Result.xpca = xpca
        
  return Result
  
'''
Calculate connectivity measures.
    Parameters
    ----------
    measure_names : str or list of str
        Name(s) of the connectivity measure(s) to calculate. See
        :class:`Connectivity` for supported measures.
    b : array, shape (n_channels, n_channels * model_order)
        VAR model coefficients. See :ref:`var-model-coefficients` for details
        about the arrangement of coefficients.
    c : array, shape (n_channels, n_channels), optional
        Covariance matrix of the driving noise process. Identity matrix is used
        if set to None (default).
    nfft : int, optional
        Number of frequency bins to calculate. Note that these points cover the
        range between 0 and half the sampling rate.
    Returns
    -------
    result : array, shape (n_channels, n_channels, `nfft`)
        An array of shape (m, m, nfft) is returned if measures is a string. If
        measures is a list of strings, a dictionary is returned, where each key
        is the name of the measure, and the corresponding values are arrays of
        shape (m, m, nfft).
    Notes
    -----
    When using this function, it is more efficient to get several measures at
    once than calling the function multiple times.
'''

def connectivity(measure_names, b, c=None, nfft=512):
  con = Connectivity(b, c, nfft)
  try:
    return getattr(con, measure_names)()
  except TypeError:
    return dict((m, getattr(con, m)()) for m in measure_names)

'''
Calculation of connectivity measures.
    This class calculates various spectral connectivity measures from a vector
    autoregressive (VAR) model.
    Parameters
    ----------
    b : array, shape (n_channels, n_channels * model_order)
        VAR model coefficients. See :ref:`var-model-coefficients` for details
        about the arrangement of coefficients.
    c : array, shape (n_channels, n_channels), optional
        Covariance matrix of the driving noise process. Identity matrix is used
        if set to None (default).
    nfft : int, optional
        Number of frequency bins to calculate. Note that these points cover the
        range between 0 and half the sampling rate.
    Methods
    -------
    :func:`A`
        Spectral representation of the VAR coefficients.
    :func:`H`
        Transfer function (turns the innovation process into the VAR process).
    :func:`S`
        Cross-spectral density.
    :func:`logS`
        Logarithm of the cross-spectral density (S).
    :func:`G`
        Inverse cross-spectral density.
    :func:`logG`
        Logarithm of the inverse cross-spectral density.
    :func:`PHI`
        Phase angle.
    :func:`COH`
        Coherence.
    :func:`pCOH`
        Partial coherence.
    :func:`PDC`
        Partial directed coherence.
    :func:`ffPDC`
        Full frequency partial directed coherence.
    :func:`PDCF`
        PDC factor.
    :func:`GPDC`
        Generalized partial directed coherence.
    :func:`DTF`
        Directed transfer function.
    :func:`ffDTF`
        Full frequency directed transfer function.
    :func:`dDTF`
        Direct directed transfer function.
    :func:`GDTF`
        Generalized directed transfer function.
    Notes
    -----
    Connectivity measures are returned by member functions that take no
    arguments and return a matrix of shape (m, m, nfft). The first dimension is
    the sink, the second dimension is the source, and the third dimension is
    the frequency.
'''



class Connectivity:

  def __init__(self, b, c=None, nfft=512):
    b = np.asarray(b)
    m, mp = b.shape
    p = mp // m
    if m * p != mp:
      raise AttributeError('Second dimension of b must be an integer '
                                 'multiple of the first dimension.')
    if c is None:
      self.c = np.identity(b.shape[0])
    else:
      self.c = np.atleast_2d(c)

    self.b = np.reshape(b, (m, m, p), 'c')
    self.m = m
    self.p = p
    self.nfft = nfft

  @memoize
  def Cinv(self):
    try:
      return np.linalg.inv(self.c)
    except np.linalg.linalg.LinAlgError:
      print('Warning: non-invertible noise covariance matrix c.')
      return np.eye(self.c.shape[0])

  @memoize
  def A(self):
    return fft(np.dstack([np.eye(self.m), -self.b]),
                   self.nfft * 2 - 1)[:, :, :self.nfft]

  @memoize
  def H(self):
    return _inv3(self.A())

  @memoize
  def S(self):
    if self.c is None:
      raise RuntimeError('Cross-spectral density requires noise '
                               'covariance matrix c.')
    H = self.H()
        
    S = np.empty(H.shape, dtype=H.dtype)
    for f in range(H.shape[2]):
      S[:, :, f] = H[:, :, f].dot(self.c).dot(H[:, :, f].conj().T)
    return S

  @memoize
  def logS(self):
    return np.log10(np.abs(self.S()))

  @memoize
  def absS(self):
    return np.abs(self.S())

  @memoize
  def G(self):
    if self.c is None:
      raise RuntimeError('Inverse cross spectral density requires '
                               'invertible noise covariance matrix c.')
    A = self.A()
        
    G = np.einsum('ji..., jk... ->ik...', A.conj(), self.Cinv())
    G = np.einsum('ij..., jk... ->ik...', G, A)
    return G

  @memoize
  def logG(self):
    return np.log10(np.abs(self.G()))

  @memoize
  def COH(self):
    S = self.S()
    return S / np.sqrt(np.einsum('ii..., jj... ->ij...', S, S.conj()))

  @memoize
  def PHI(self):
    return np.angle(self.S())

  @memoize
  def pCOH(self):
    G = self.G()
    return G / np.sqrt(np.einsum('ii..., jj... ->ij...', G, G))

  @memoize
  def PDC(self):
    A = self.A()
    return np.abs(A / np.sqrt(np.sum(A.conj() * A, axis=0, keepdims=True)))

  @memoize
  def sPDC(self):
    return self.PDC()**2

  @memoize
  def ffPDC(self):
    A = self.A()
    return np.abs(A * self.nfft / np.sqrt(np.sum(A.conj() * A, axis=(0, 2),
                                                     keepdims=True)))

  @memoize
  def PDCF(self):
    A = self.A()
    return np.abs(A / np.sqrt(np.einsum('aj..., ab..., bj... ->j...',
                                            A.conj(), self.Cinv(), A)))

  @memoize
  def GPDC(self):
    A = self.A()
    tmp = A / np.sqrt(np.einsum('aj..., a..., aj..., ii... ->ij...',
                                    A.conj(), 1 / np.diag(self.c), A, self.c))
    return np.abs(tmp)

  @memoize
  def DTF(self):
    H = self.H()
    return np.abs(H / np.sqrt(np.sum(H * H.conj(), axis=1, keepdims=True)))

  @memoize
  def ffDTF(self):
    H = self.H()
    return np.abs(H * self.nfft / np.sqrt(np.sum(H * H.conj(), axis=(1, 2),
                                                     keepdims=True)))

  @memoize
  def dDTF(self):
    return np.abs(self.pCOH()) * self.ffDTF()

  @memoize
  def GDTF(self):
    H = self.H()
    tmp = H / np.sqrt(np.einsum('ia..., aa..., ia..., j... ->ij...',
                                    H.conj(), self.c, H,
                                    1 / self.c.diagonal()))
    return np.abs(tmp)


def _inv3(x):
    identity = np.eye(x.shape[0])
    return np.array([sp.linalg.solve(a, identity) for a in x.T]).T

def connectivity_measure(input_data, model_order, measure, c_, nfft_, optimize_var_=False, backend_=None, varfit_='ensemble', random_state_=None):
  aux_VAR = VAR(model_order = model_order)
  final_model = mvarica(input_data, aux_VAR, optimize_var=False, backend=None, varfit='ensemble', random_state=None)
  Coef_array = final_model.b.coef
  connectivity_measure = Connectivity(Coef_array, c = c_, nfft= nfft_)
  measure = measure.upper()
  connectivity_dict = {}
  if measure == "PDC" :
    pdc = connectivity_measure.PDC()
    return pdc
  if measure == "A" :
    A = connectivity_measure.A()
    return A
  if measure == "H" :
    H = connectivity_measure.H()
    return H
  if measure == "S" :
    S = connectivity_measure.S()
    return S
  if measure == "G" :
    G = connectivity_measure.G()
    return G
  if measure == "logG" :
    logG = connectivity_measure.logG()
    return logG
  if measure == "PHI" :
    PHI = connectivity_measure.PHI()
    return PHI
  if measure == "COH" :
    COH = connectivity_measure.COH()
    return COH
  if measure == "pCOH" :
    pCOH = connectivity_measure.pCOH()
    return pCOH
  if measure == "ffPDC" :
    ffPDC = connectivity_measure.ffPDC()
    return ffPDC
  if measure == "PDCF" :
    PDCF = connectivity_measure.PDCF()
    return PDCF
  if measure == "pCOH" :
    pCOH = connectivity_measure.pCOH()
    return pCOH
  if measure == "GPDC" :
    GPDC = connectivity_measure.GPDC()
    return GPDC
  if measure == "DTF" :
    DTF = connectivity_measure.DTF()
    return DTF
  if measure == "ffDTF" :
    ffDTF = connectivity_measure.ffDTF()
    return ffDTF
  if measure == "dDTF" :
    dDTF = connectivity_measure.dDTF()
    return dDTF
  if measure == "GDTF":
    GDTF = connectivity_measure.GDTF()
    return GDTF      
