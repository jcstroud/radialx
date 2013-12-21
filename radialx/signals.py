import math
import operator

import numpy
from scipy import special, polyfit, optimize

from itertools import izip, product

floor = math.floor
ceil = math.ceil

DEBUG = True

class SignalError(Exception): pass

def hilo(atup, op=operator.lt):
  if atup is None:
    atup = None, None
  else:
    hi, lo = atup
    if op(hi, lo):
      atup = lo, hi
  return atup

def lohi(atup):
  return hilo(atup, op=operator.gt)

def chauvenet_keeplow(values):
  values = numpy.array(values)
  keeplow = retainerize(low_pass(values.mean()))
  return filtainer(values, filters=[chauvenet],
                           retainers=[keeplow])

def chauvenet_applied(values):
  return filtainer(values, filters=[chauvenet])

def retainerize(a_filter):
  def _f(ary):
    return numpy.logical_not(a_filter(ary))
  return _f

def intersect2(m1, b1, m2, b2):
  """
  Takes the two slopes, I{m1} and I{m2}, and the two
  Y-intercepts, I{b1} and I{b2}, of two lines and returns
  the point at which the intersect as an (x, y) tuple.
  """
  delta = m1 - m2
  x = (b2 - b1) / delta
  y = ((b2 * m1) - (b1 * m2)) / delta
  return x, y

def intersect2ary(ary1, ary2):
  """
  Takes the two 2-d arrays I{ary1} and I{ary2}
  and finds their intersection. The arrays are organized
  [X, Y], where X is a 1-d array for the abscissa
  and Y is the corresponding 1-d array for the ordinate.
  """
  X1, Y1 = ary1
  X2, Y2 = ary2
  m, c = polyfit(X1, Y1, 1)
  n, d = polyfit(X2, Y2, 1)
  return intersect2(m, c, n, d)


def neighborhoods(ary, idx, radius):
  """
  Returns as a `list` of numpy fancy indices all of the neighborhoods
  of `radius` of the elements indexed by the numpy fancy indices `idx`.
  These neigborhoods exclude the original elements, turning
  out to be ``1 + radius**2`` blocks of the dimensionality of
  `ary` with holes of one element in the middle.
  The exception is at the edges, where the neighborhoods do not
  extend pass the array `ary`.

  >>> array = numpy.array
  >>> b = numpy.arange(25).reshape(5, 5)
  >>> z = (array([1,3]), array([1,3]))
  >>> neighborhoods(b, z, 1)
    [(array([0, 0, 0, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 2, 0, 1, 2])),
     (array([2, 2, 2, 3, 3, 4, 4, 4]), array([2, 3, 4, 2, 4, 2, 3, 4]))]
  >>> c = numpy.arange(1000).reshape(10, 10, 10)
  >>> x = (array([5, 0]), array([5, 2]), array([9,8]))
  >>> [c[n].sum() for n in neighborhoods(c, x, 1)]
    [9494, 1376]
  >>> [c[n].shape for n in neighborhoods(c, x, 1)]
    [(17,), (17,)]
  >>> y = (array([5, 1]), array([5, 2]), array([7, 8]))
  >>> [c[n].sum() for n in neighborhoods(c, y, 1)]
    [14482, 3328]
  >>> [c[n].shape for n in neighborhoods(c, y, 1)]
    [(26,), (26,)]
  """
  nn = []
  for x in izip(*idx):
    lo = [max([i - radius, 0]) for i in x]
    hi = [min([i + radius, s - 1]) for (i, s) in izip(x, ary.shape)]
    n = [range(*(p, q+1)) for (p, q) in izip(lo, hi)]
    m = [p for p in product(*n) if p != x]
    m = [numpy.array(p) for p in zip(*m)]
    nn.append(tuple(m))
  return nn


def apply_nearest_neighbors(ary, idx, f):
  """
  Returns as a `list` the function `f` applied to the
  nearest neighbors in the array `ary` of the elements
  indexed by the numpy fancy indices `idx`.

  >>> array = numpy.array
  >>> b = numpy.arange(25).reshape(5, 5)
  >>> z = (array([1,3]), array([1,3]))
  >>> apply_nearest_neighbors(b, z, numpy.mean)
    [6.0, 18.0]
  """
  nn = neighborhoods(ary, idx, 1)
  ixx = izip(*idx)
  return [f(ary, n, i) for n, i in izip(nn, ixx)]


def dezinger(ary, fltr, filler):
  """
  Returns a new array that is the dezingered `ary`.
  Parameter `fltr` returns a mask of the shape of `ary`
  that is False where zingers are.  E.g. `chauvenet()`.
  Parameter `filler` takes the `ary` array, a set of numpy
  fancy indices, and a single fancy index (as a `tuple` of `int`s)
  as positional arguments and produces a 1-d array
  of filler values for the indices.

  >>> f = lambda a, n, i: min(a[n].mean(), a[i])
  >>> b = numpy.arange(16, dtype=float).reshape(4, 4)
  >>> b[([1,3],[1,3])] = [42, -42]
  >>> dezinger(b, chauvenet, f)
    array([[  0.,   1.,   2.,   3.],
           [  4.,   5.,   6.,   7.],
           [  8.,   9.,  10.,  11.],
           [ 12.,  13.,  14., -42.]])
  """
  mask = fltr(ary)
  not_mask = numpy.logical_not(mask)
  idxs = numpy.where(not_mask)
  values = apply_nearest_neighbors(ary, idxs, filler)
  bry = ary.copy()
  bry[idxs] = values
  return bry

def chauvenet(values, criterion=0.5):
  """
  Uses Chauvenet's `criterion` (default 0.5)
  for one round of rejection.
  Returns a mask that is `True` for data that is
  not rejected.
  """
  n = values.size
  if n < 7:
    return numpy.ones(values.shape, dtype=numpy.bool)
  av = numpy.mean(values)
  stdev = numpy.std(values)
  distance = abs(special.ndtri(criterion/n) * stdev)
  lo = av - distance
  hi = av + distance
  mask = (lo <= values) & (values <= hi)
  return mask
  
def reduce_funcs(values, funcs, op):
  result = None
  for i, f in enumerate(funcs):
    if result is None:
      result = f(values)
    else:
      mask = f(values)
      result = op(result, mask)
  if result is None:
    raise SignalError, "No functions supplied."
  return result

def filtainer(values, filters=None, retainers=None, subset=None):
  """
  Returns a 2-tuple with the the first item being the
  indexes of accepted elements and the second item being the
  rejected elements from the array `values`.

  Either `filters` or `retainers` or both must be supplied
  or a `SignalError` will be raised.

  The `subset` parameter, if supplied, are numpy fancy indices
  that index a subset of `values`. This subset is used in
  filtainering and those elements not in the subset are included
  in the returned rejections.

  Filters are applied with logical *and* while retainers are
  applied with logical or. The items that pass both filtering
  and retaining (logical and) are accepted.

  >>> b = numpy.arange(10).reshape(5,2)
  >>> b[0,0] = -42
  >>> accepts, rejects = filtainer(b, filters=[lambda x: x != 3])
  >>> # keep the union of elements that are even or less than 7
  >>> # but that also pass chauvenet
  >>> fs = [chauvenet]
  >>> rs = [low_pass(7), lambda x: numpy.logical_not(x % 2)]
  >>> accepts, rejects = filtainer(b, filters=fs,
  ...                                 retainers=rs,
  ...                                 subset=accepts)
  >>> b[accepts]
      array([1, 2, 3, 4, 5, 6, 7, 8])
  >>> b[rejects]
      array([-42,   3,   9])
  """
  _f = None
  if subset is not None:
    def _f(ary):
      a = numpy.zeros_like(values, dtype=bool)
      a[subset] = True
      return a
    if filters is None:
      filters = [_f]
    else:
      filters = [_f] + filters
  if (filters is None) and (retainers is None):
    raise SignalError, "Filters and retainers can't both be None."
  elif filters is None:
    accepts = reduce_funcs(values, retainers, numpy.logical_or)
  elif retainers is None:
    accepts = reduce_funcs(values, filters, numpy.logical_and)
  else:
    accepts = numpy.logical_and(
                 reduce_funcs(values, filters, numpy.logical_and),
                 reduce_funcs(values, retainers, numpy.logical_or))
  rejects = numpy.logical_not(accepts)
  accept_idxs = accepts.nonzero()
  reject_idxs = rejects.nonzero()
  return accept_idxs, reject_idxs

def smooth(x, window_len=11, window='hanning'):
    """
    smooth the data using a window with requested size.

    ripped from the numpy cookbook

    This method is based on the convolution of a scaled window
    with the signal. The signal is prepared by introducing reflected
    copies of the signal (with the window size) in both ends so that
    transient parts are minimized in the begining and end part
    of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window;
                    should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming',
                'bartlett', 'blackman'
                flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman,
    numpy.convolve, scipy.signal.lfilter, numpy.kaiser
    """
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
        msg = "Input vector needs to be bigger than window size."
        raise ValueError, msg
    if window_len<3:
        return x

    acceptable = ['flat', 'hanning', 'hamming',
                  'bartlett', 'blackman', 'kaiser']
    if not window in acceptable:
        raise ValueError, "Window is one of %s" % ",".join(acceptable)

    s=numpy.r_[2*x[0] - x[window_len:1:-1], x,
               2*x[-1] - x[-1:-window_len:-1]]
    if window == 'flat':
        w = ones(window_len,'d')
    else:
        window_function = getattr(numpy, window)
        w = window_function(window_len)

    y=numpy.convolve(w/w.sum(), s, mode='same')
    return y[window_len-1:-window_len+1]


def high_pass(minimum, inclusive=True):
  if inclusive:
    def _f(values):
      return values >= minimum
  else:
    def _f(values):
      return values > minimum
  return _f

def low_pass(maximum, inclusive=True):
  if inclusive:
    def _f(values):
      return values <= maximum
  else:
    def _f(values):
      return values < maximum
  return _f

def interpolate2(idx, ary):
  idx1 = int(idx)
  idx2 = idx1 + 1
  v1 = ary[idx1]
  v2 = ary[idx2]
  return ((v2 - v1) / (idx2 - idx1)) + v1
 
def bound(v, lower, upper):
  upper, lower = hilo((upper, lower))
  v = min(upper, v)
  return max(lower, v)

def make_gauss(height, center, width):
  """
  This appears to need a `noiseit()` function.
  """
  bottom = 2 * width**2
  def _g(x, noise_scale=None):
    ary = height * numpy.exp((-(x - center)**2)/bottom)
    if noise_scale is not None:
      ary = noiseit(ary, noise_scale)
    return ary
  return _g

def make_gausses(acw_ary):
  """
  The `acw_ary` parameter is a shape (n, 3) array where
  each of the n rows is an a, c, and w.
  """
  ff = [make_gauss(*acw) for acw in acw_ary]
  def _g(X, noise_scale=None):
    G = numpy.zeros(X.shape, dtype=float)
    for f in ff:
      G += f(X)
    return G
  return _g

def gaussian_deconvolution(sample, X, Y, guesses=None):
  """
  The `sample` parameter is a `dict` that should have
  the keys ``min_sigma``, ``max_sigma``, ``minimization_type``,
  ``disp``, and ``max_steps``. The `sample` parameter can
  also have an ``n`` key, that is used to specify how many peaks to
  find if the parameter `guesses` is `None`.
  """
  inf = numpy.inf
  span_lo = X[0]
  span_hi = X[-1]
  span = span_hi - span_lo
  m = Y.max() - Y.min()
  M = 2 * m

  if guesses is None:
    n = sample['n']
    fn = float(n)
    a = m / fn
    step = span / fn
    s = step / 2
    cs = (numpy.arange(n) * step) + X[0] + s
    guesses = []
    for c in cs:
      guesses.extend((a, c, s))
    guesses = numpy.array(guesses)
  else:
    n = len(guesses) / 3

  # print "=" * 70
  # print "guesses"
  # print guesses
  # print "=" * 70

  infs = numpy.empty(X.shape)
  infs.fill(inf)

  pw_min = sample['min_sigma']
  pw_max = sample['max_sigma']

  # print guesses

  def _residuals(params):
    err = Y.copy()
    for i in xrange(n):
      j = i * 3
      maximum, expect, sigma = params[j:j+3]
      if ((maximum <= 0) or (maximum > M) or
          (expect <= span_lo) or (expect >= span_hi) or
          (sigma > pw_max) or (sigma <= pw_min)):
        err = infs.copy()
        break
      err = err - (maximum * numpy.exp((-((X-expect)/sigma)**2)/2))
    return err

  #####################################################################
  # move to config
  #####################################################################
  # min_type should be 'leastsq' or 'fmin'
  min_type = sample['minimization_type']
  disp = sample.get('disp', False)
  max_steps = int(sample['max_steps'])
  #####################################################################
  if min_type == 'leastsq':
    # MUST DO ERROR CHECKING ON plsq!!!
    plsq = optimize.leastsq(_residuals, guesses, maxfev=max_steps)
    k = plsq[0]
  elif min_type:
    f = getattr(optimize, min_type)
    # print 'doing fmin'
    def _f(params):
      return _residuals(params).sum()
    aaa = f(_f, guesses, full_output=True,
                         disp=disp,
                         maxfun=max_steps)
    print aaa
    # if not (aaa[-1] and numpy.isfinite(aaa[0][0])):
    if numpy.isfinite(aaa[0][0]):
      k = aaa[0]
    else:
      # print aaa
      k = None

  # k is (amplitude, center, width)
  return k

def binopt(data, min_bins=7, max_bins=None, resolution=100):
  """
  Optimize the number of bins for `data` using `N_0`
  as an initial guess using the method from
  Shimazaki and Shinomoto, Neural Comput 19 1503-1527, 2007.
  guess is returned.

  :parameters:
     data : sequence
        observations as a 1-d sequence such as
        a `numpy.ndarray` of shape ``(1,)``

  :keywords:
     min_bins : int
        minimum number of bins over which to
        calculate the optimization (default ``7``)
     max_bins : int
        maximum number of bins over which to
        calculate the optimization, if ``None``
        then the max_bins will correspond to the number
        of bins to give an average of at least 3 observations per bin
        (default ``None``)
     resolution : float, int
        how finely to sample the bin widths
        a bigger resolution is more fine (default ``100``)

  :return: A `dict` with the following keys:
     :bin_width: optimal bin width (`float`)
     :bins: number of bins for the optimal hitogram
     :hist: the optimized histogram (`numpy.ndarray`)
     :edges: the edges of the histogram (`numpy.ndarray`)
     :sizes: a 1-d array of the sizes used to find the
             bin size optimium (`numpy.ndarray`)
     :costs: cost estimates of the Mean Integrated
             Squared Error (MISE) corresponding to the bin sizes
     :best_score: cost estimate corresponding to the returned,
                  optimal, histogram
  """
  data = numpy.array(data)

  if max_bins is None:
    max_bins = int(data.size / 3.0)

  dmax = float(data.max())
  dmin = float(data.min())
  diff = dmax - dmin

  # This is good code because I like what it does.
  # It doesn't belong here, but I'm leaving it here
  # until I figure out what to do with it
  def get_limits_old(delta, dmin, dmax, parts, margin):
    """
    The keyword arguments `parts` and `margin` are for
    cost function smoothing. If no smoothing is desired,
    set either to ``None``.

    The `parts` and `margin` arguments
    are used for cost function smoothing. These two arguments
    work together to specify `parts` different binning
    schemes for the histogram where the limits of the
    histogram are extended up to `margin` number of bins
    on either side of the not-extended limits. For example
    if the max of the data is 10, the min is 0, and 
    if the number of bins is 10, then
    the data would be binned from 0 to 10 with steps of 1.
    To smooth the cost function using 2 partitions and a
    margin of 1, there would be two sets of limits:
    -1 to 11 and 0 to 10. Each set of limits would be result
    in a different histogram of 10 bins. The individual costs
    for each set would be calculated and the average would
    be used for the optimization.
    """
    margin_width = delta * margin
    low_start = dmin - margin_width
    high_end = dmax + margin_width
    step_size = margin_width / (parts - 1)
    # adjusting the range limits with half steps hedges against
    # floating point imprecision when calculating the ranges
    half_step = step_size / 2
    low_stop = dmin + half_step
    high_stop = dmax - half_step
    starts = numpy.arange(low_start, low_stop, step_size)
    ends = numpy.arange(high_end, high_stop, -step_size)
    return numpy.array([starts, ends]).T

  def make_hist(data, delta, dmin, dmax, diff):
    bins_f = diff / delta
    # hedge for floating point imprecision
    bins_i = round(bins_f)
    if (bins_f - bins_i) < 1e-15:
      N = int(bins_i)
    else:
      N = int(math.ceil(bins_f))
    width = N * delta
    xtra = (width - diff) / 2.0
    start = dmin - xtra
    edges = (numpy.arange(N+1) * delta) + start
    return numpy.histogram(data, bins=edges)

  def _optf(delta):
    hist, edges = make_hist(data, delta, dmin, dmax, diff)
    k = hist.mean()
    v = hist.var()
    return ((2*k) - v) / (delta**2)

  denominators = numpy.arange(min_bins, max_bins, 1.0/resolution)
  deltas = (dmax - dmin) / denominators
  print "deltas:", deltas
  mises = numpy.array([_optf(d) for d in deltas])

  best_idx = mises.argmin()
  delta_f = deltas[best_idx]
  best_score = mises[best_idx]

  hist, edges = make_hist(data, delta_f, dmin, dmax, diff)
  results = { 'bin_size' : delta_f, 'best_score' : best_score,
              'hist' : hist, 'edges' : edges, 'bins' : len(hist),
              'sizes' : deltas, 'costs' : mises }

  return results
