#! /usr/bin/env python

import numpy

class DistMatError(Exception): pass
class StrictError(DistMatError): pass

class DistMat(object):
  """
  A crazy class that caches instances of itself
  and returns 
  """
  __instances = {}
  def __new__(cls, shape):
    rows, cols = shape
    shape = rows, cols
    if shape in DistMat.__instances:
      dm = DistMat.__instances[shape]
    else:
      i = (rows-1)/2
      j = (cols-1)/2
      ridxs, cidxs = numpy.indices(shape)
      dists = numpy.sqrt((ridxs-i)**2 + (cidxs-j)**2)
      dm = object.__new__(cls)
      dm.center = (i, j)
      dm.dists = dists
      dm.shape = dm.dists.shape
      DistMat.__instances[shape] = dm
    d = object.__new__(cls)
    d.center = dm.center
    d.dists = dm.dists.copy()
    d.shape = dm.shape
    return d
  def __repr__(self):
    params = (self.__class__.__name__, self.center, self.dists)
    rstr = "<\n%s @ %s\n%s\n>" % params
    return rstr
  def __getitem__(self, i):
    return self.dists[i]
  def reset(self):
    d = self.__class__(self.shape)
    self.center = d.center
    self.dists = d.dists
    self.shape = d.shape
  def move_rows(self, rows):
    if rows > self.shape[0]:
      msg = "Movement (%s) more than rows (%s)" % rows, self.shape[0]
      raise DistMatError, msg
    elif rows == 0:
      return
    self.center = self.center[0] + rows, self.center[1]
    new_dists = numpy.empty_like(self.dists)
    if rows > 0:
      new_dists[rows:] = self.dists[:-rows]
      new_dists[:rows].fill(numpy.NaN)
    elif rows < 0:
      new_dists[:rows] = self.dists[-rows:]
      new_dists[rows:].fill(numpy.NaN)
    self.dists = new_dists
  def move_cols(self, cols):
    if cols > self.shape[1]:
      msg = "Movement (%s) more than cols (%s)" % cols, self.shape[0]
      raise DistMatError, msg
    elif cols == 0:
      return
    self.center = self.center[0], self.center[1] + cols
    dists = self.dists
    new_dists = numpy.empty_like(self.dists)
    if cols > 0:
      new_dists[:,cols:] = dists[:,:-cols]
      new_dists[:,:cols].fill(numpy.NaN)
    elif cols < 0:
      new_dists[:,:cols] = dists[:,-cols:]
      new_dists[:,cols:].fill(numpy.NaN)
    self.dists = new_dists
  def recenter(self, center):
    i, j = center
    rows = i - self.center[0]
    cols = j - self.center[1]
    if rows:
      self.move_rows(rows)
    if cols:
      self.move_cols(cols)
  def bins(self, disc, nbins, bin=None, strict=False, cache={}):

    rout = disc['rout']
    rin = disc['rin']

    # this is used to calculate the radius
    k, h = disc['center']
    # the actual center is the center of the pixel
    hc = h + 0.5
    kc = k + 0.5

    # this is used to clip the array
    sk, sh = self.center
    # the actual center is the center of the pixel
    shc = sh + 0.5
    skc = sk + 0.5

    height, width = self.shape

    radius = min(hc, width - hc, kc, height - kc)
    if rout is None:
      r = radius
    else:
      if strict and (rout > radius):
        tmplt = "Outer radius (%s) too big for given center %s."
        msg = tmplt % (rout, (k, h))
        raise StrictError, msg
      r = min([float(rout), radius])

    if rin is None:
      rin = 0.0
    elif 0 <= rin < rout:
      rin = float(rin)
    else:
      raise DistMatError, "Bad value for inner r: %s", rin

    # for cutting out the dm
    dm_min_c = int(shc - r)
    dm_min_r = int(skc - r)
    dm_max_c = int(shc + r)
    dm_max_r = int(skc + r)

    # for the returned index
    min_col = int(hc - r)
    min_row = int(kc - r)
    max_col = int(hc + r)
    max_row = int(kc + r)

    _height, _width = self.shape
    binned_key = (nbins, _height, _width,
                  dm_min_c, dm_min_r, dm_max_c, dm_max_r)
    where_key = (nbins, _height, _width,
                 dm_min_c, dm_min_r, dm_max_c, dm_max_r, bin)
    if binned_key in cache:
      binned = cache[binned_key]
    else:
      dm = self.__class__(self.shape)
      pxperbin = float(rout - rin)/nbins
      bin_offset = rin/pxperbin
      bins = dm.dists[dm_min_r:dm_max_r,dm_min_c:dm_max_c]/pxperbin
      bins -= bin_offset
      bins = numpy.floor(bins)
      binned = bins.astype(numpy.int)
      cache[binned_key] = binned

    idx = (slice(min_row, max_row), slice(min_col, max_col))

    if bin is None:
      ary = binned
    else:
      if where_key in cache:
        ary = cache[where_key]
      else:
        ary = numpy.argwhere(binned==bin).transpose()
        cache[where_key] = ary

    return ary, idx
