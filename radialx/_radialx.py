#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re
import math
import string
import operator
import logging
import time
import io
import bz2
import json
import base64

from itertools import izip, count, product
from tkMessageBox import askyesno, showwarning

import numpy
import yaml
import pyfscache
import pygmyplot

from PIL import Image
from scipy import optimize
from configobj import ConfigObj

import distmat
import signals

from ordereddict import OrderedDict

class RadialXError(Exception): pass
class ADXVError(RadialXError): pass
class ShapeError(RadialXError): pass
class ConfigError(RadialXError): pass
class GroupError(RadialXError): pass
class ResolutionError(RadialXError): pass
class ParameterError(RadialXError): pass
class OptimizationError(RadialXError): pass
class DummyError(Exception): pass

DEBUG = False

GOODBYTES = string.printable[:-3]

BYTE_ORDER = {'little_endian': '<',
              'big_endian':    '>'}

DATA_TYPES = {'short':              'h',
              'unsigned_short':     'H',
              'int':                'i',
              'unsigned_int':       'I',
              'long':               'l',
              'unsigned_long':      'L',
              'long_long':          'q',
              'unsigned_long_long': 'Q',
              'float':              'f',
              'double':             'd'}

TYPE_SIZES = {'short':              2,
              'unsigned_short':     2,
              'int':                4,
              'unsigned_int':       4,
              'long':               4,
              'unsigned_long':      4,
              'long_long':          8,
              'unsigned_long_long': 8,
              'float':              4,
              'double':             8}


GROUP_START = "("
GROUP_END = ")"
GROUP_SLICE = slice(len(GROUP_START), -len(GROUP_END))

MAXB = 10
MAXLIK = 1e9

def load_config(cfg_file):
  if cfg_file.endswith(('.cfg', '.ini')):
    cfg = ConfigObj(cfg_file, unrepr=True)
  elif cfg_file.endswith(('.yml', '.yaml')):
    cfg = yaml.load(open(cfg_file))
  else:
    msg = "Unrecognized configuration file type: '%s'" % cfg_file
    raise ConfigError, msg
  return cfg


def count_function():
  c = count()
  def _c(*args, **kwargs):
    return c.next()
  return _c


def donothing(msg, frac):
  """
  This is a dummy function for progress updating.

  @param msg: a string with the message for the progress
  @param frac: fraction done: a number between 0 and 1 inclusive
  """
  pass

def print_it(msg, frac, f=None):
  """
  This is a simple progress updater that writes the
  progress message and the percent done to the
  file I{f} if provided. If I{f} isn't provided,
  output goes to stdout.

  @param msg: a string with the message for the progress
  @param frac: fraction done: a number between 0 and 1 inclusive
  """
  if f is None:
    f = sys.stdout
  f.write("%s (%6.2f%%)\n" % (msg, frac*100))

def derivative(y, order=1):
  '''
  Calculates the order-th derivative of the values in I{y}.
  '''
  for der in xrange(order):
    y = (y[1:]-y[:-1])
    #one more value is added because the length
    y = numpy.append(y,y.mean())
  return y

def scale_pt(pt, f):
  """
  Scales the (x,y) point I{pt} by the factor I{f}.
  """
  return pt[0]*f, pt[1]*f

def flip_pt(pt):
  """
  Returns the (x,y) point I{pt} as (y,x).
  """
  return pt[1], pt[0]

def threesides2alpha(a, b, c):
  s = (a + b + c) / 2.0
  return math.acos(1 - (2 * (s-b) * (s-c) / (b*c)))

def guess_type(aval, converters):
  """
  Give it a list of callable I{converters} and
  each will get called successively with the argument
  I{aval} until one does not raise a I{ValueError}. The
  return value of that call will be returned.
  """
  success = False
  for converter in converters:
    try:
      converted = converter(aval)
    except ValueError:
      continue
    else:
      success = True
      break
  if success:
    return converted
  else:
    raise ValueError, "Couldn't convert %s." % aval

def read_header(adxv_file):
  """
  Reads an adxv .img binary file header and returns
  it as a dict, with primitive types I{int}, I{float},
  and I{str} inferred.
  """
  logging.info("Processing '%s'.", adxv_file)
  endlineRE = re.compile(r';\s*?\n')
  adxv_file.seek(0)
  header = []
  while True:
    achar = adxv_file.read(1)
    if achar == "{":
      break
  while True:
    achar = adxv_file.read(1)
    if achar == "}":
      break
    if achar in GOODBYTES:
      header.append(achar)
  header = "".join(header)
  header_indent = "".join(("    " + h + "\n") for h in
                                               header.splitlines())
  logging.info("HEADER:\n%s", header_indent)
  header = endlineRE.split(header)
  header = [s.split("=", 1) for s in header if "=" in s]
  header = [(t.strip() for t in s) for s in header]
  hod = OrderedDict()
  for k, v in header:
    if k in hod:
      if hod[k] != v:
        tplt = "Contradicting values for '%s' in '%s' header."
        logging.warning(tplt, k, adxv_file.name)
    hod[k] = v
  header = hod
  if not header:
    raise ADXVError, 'Improper header in adxv file.'
  converters = [int, float, str]
  for key, value in header.items():
    header[key] = guess_type(value, converters)
  return header

def read_data(adxv_file, header):
  """
  Helper function to read a binary adxv .img file
  starting from the first data byte. Parameters describing
  the layout of the data are stored in I{header}. See
  the adxv documentation for details.
  """
  adxv_file.seek(header['HEADER_BYTES'])
  byte_order = BYTE_ORDER[header['BYTE_ORDER']]
  data_type = DATA_TYPES[header['TYPE']]
  dtype = byte_order + data_type
  ishape = [header['SIZE'+str(d+1)] for d in xrange(header['DIM'])]
  shape = tuple(ishape[::-1])
  ary = numpy.fromfile(file=adxv_file, dtype=dtype)
  ary = ary.reshape(*shape)
  return ary

def integrate(data, nbins, disc, minI=None,
                                 progress=None,
                                 cache={}):
  """
  Integrate and average I{data} within the I{disc} divided into
  I{nbins} concentric and equivalently wide bins. Ignore any
  data less than I{minI}. The I{progress} argument should be a
  function that takes two arguments: astring and a number between
  zero and 1.

  The I{data} is a 2-D numpy array, fast in rows.

  The I{disc} argument is a mapping object with keys:
    - center - (x,y) tuple C-indexed so the first pixel is 0, not 1
    - rout - outer radius in pixels
    - rin - inner radius in pixels

  See the documentation for L{donothing} for the parameter
  list of the function argument I{progress}.
  """
  dm = distmat.DistMat(data.shape)
  if progress is None:
    progress = donothing
  progress("Binning distances.", 0.0)
  bins, slc = dm.bins(disc, nbins, strict=True, cache=cache)
  all_values = []
  totals = []
  sums = []
  means = []
  stds = []
  rjcts = []
  ns = []
  hpass = [signals.high_pass(minI)]
  chauv = [signals.chauvenet]
  filtainer = signals.filtainer
  for bin in xrange(nbins):
    msg = "Averaging bin %d of %d." % (bin+1, nbins)
    frac = float(bin) / nbins
    progress(msg, frac)
    idxs, slc = dm.bins(disc, nbins, bin=bin, strict=True, cache=cache)
    bin_values = data[slc][idxs[0], idxs[1]]
    # must break into two steps so that the low values do not
    # influence the statistics for Chauvenet
    acpt_idxs, rjct_idxs = filtainer(bin_values, filters=hpass)
    acpt_idxs, rjct_idxs = filtainer(bin_values,
                                     filters=chauv,
                                     subset=acpt_idxs)
    bin_values = bin_values[acpt_idxs]
    all_values.append(bin_values)
    sums.append(numpy.sum(bin_values))
    means.append(numpy.mean(bin_values))
    stds.append(numpy.std(bin_values))
    ns.append(numpy.size(bin_values))
    if rjct_idxs[0].size == 0:
      rjcts.append(numpy.array([], dtype=int))
    else:
      rjcts.append(idxs[:, rjct_idxs])
  sigma = numpy.std(numpy.concatenate(all_values))
  progress(msg, 1.0)
  sums = numpy.array(sums)
  means = numpy.array(means)
  stds = numpy.array(stds)
  ns = numpy.array(ns)
  return {
           'sums': sums,
           'means': means,
           'stds': stds,
           'rejects': rjcts,
           'counts': ns,
           'sigma': sigma,
           'slice': slc,
         }

def scale_min_max(data, limits, progress=None):
  """
  Scaling of intensities in I{data} to the minimum and
  maximum values in the 2-tuple I{limits}. If the
  maximum is less than the minimum, the resulting data
  will be intensity-inverted in the scale.

  See the documentation for L{donothing} for the parameter
  list of the function argument I{progress}.
  """
  msg = "Scaling data to min/max."
  lowest, highest = limits
  if lowest == highest:
    raise RadialXError, "lowest equals highest: %s" % limits
  if progress is None:
    progress = donothing
  limits_range = float(highest - lowest)
  data_max = numpy.max(data)
  data_min = numpy.min(data)
  data_range = float(data_max - data_min)
  multiplier = limits_range/data_range
  progress(msg, 0.00)
  ary = data.astype(float)
  progress(msg, 0.10)
  ary -= data_min
  progress(msg, 0.33)
  ary *= multiplier
  progress(msg, 0.67)
  ary += lowest
  progress(msg, 1.00)
  return ary

def rebin(data, binsize, strict=False, progress=donothing):
  """
  This is the U{scipy cookbook <www.scipy.org/Cookbook/Rebinning>}
  approved way to rebin, which is very clever.
  I'm modifying it here for 2-d data removing the need for
  an I{eval()} and to do some user-friendly error checking.
  The data will be rebinned along both axes by a factor of
  I{binsize}, so these axes should be an integer multiple
  of I{binsize} in length, otherwise a L{RadialXError}
  is raised. If `strict` is `False`, then the data will be expanded
  until it's dimensions are evenly divisible by the binsize,
  otherwise a `RadialXError` is raised.

  See the documentation for L{donothing} for the parameter
  list of the function argument I{progress}.
  """
  # for debugging
  old_data = data
  msg = "Rebinning data."
  if binsize == 1:
    return data.copy()
  progress(msg, 0.0)
  rows, cols = data.shape
  new_rows, rem_rows = divmod(rows, binsize)
  new_cols, rem_cols = divmod(cols, binsize)
  if (rem_rows != 0) or (rem_cols != 0):
    if strict:
      tmplt = "Data dimensions (%s) not divisible by binsize (%s)."
      msg = tmplt % (data.shape, binsize)
      raise RadialXError, tmplt
    else:
      rstart, rr = divmod(rem_rows, 2)
      rend = rows - (rstart + rr)
      cstart, cr = divmod(rem_cols, 2)
      cend = cols - (cstart + cr)
      new_data = data[rstart:rend, cstart:cend]
      rows, cols = new_data.shape
      new_rows, rem_rows = divmod(rows, binsize)
      new_cols, rem_cols = divmod(cols, binsize)
      assert rem_rows == 0
      assert rem_cols == 0
  else:
    new_data = data.copy()
  rebinned = new_data.reshape(new_rows, binsize, new_cols, binsize)
  rebinned = rebinned.mean(1)
  progress(msg, 0.67)
  rebinned = rebinned.mean(2)
  progress(msg, 1.0)
  return rebinned

def downsample_1d(data, newsize):
  """
  Apply numpy style cleverness to downsampling the 1D I{data}
  to I{newsize}.  ND data will be resampled only along
  the last dimension. Extra data will be truncated.
  """
  newshape = list(data.shape)
  dlen = newshape[-1]
  binsize = dlen // newsize
  plen = binsize * newsize
  part = data[...,:plen]
  newshape[-1:] = [newsize, binsize]
  part = part.reshape(newshape)
  sampled = part.mean(axis=-1)
  return sampled

def res2rad(d, wl, L):
  """
  Returns the radius meters that reflections
  of resolution I{d} will fall given the
  wavelength I{wl} and the detector distance
  I{L} of the experiment, all given in
  meters. Two-theta is assumed 0.

  If I{d} and I{wl} have the same units, then the
  returned radius will take the units of I{L}.

  @param d: resolution
  @param L: detector distance
  @param wl: wavelength
  """
  return  L*(math.tan(2.0*math.asin(wl/(2.0*d))))

def res2radpx(res, experiment):
  """
  Returns the radius in pixels that reflections
  of resolution I{res} given in Angstroms will fall
  given the experimental conditions passed in I{experiment}.
  The I{experiment} should be a mapping with the minimal key,
  value pairs:
     - I{WAVELENGTH} : wavelength in Angstroms
     - I{DISTANCE} : distance to detector in mm
     - I{PIXEL_SIZE} : size of pixels in mm
  """
  d = float(res)
  wl = float(experiment['WAVELENGTH'])
  L = float(experiment['DISTANCE'])
  ps = float(experiment['PIXEL_SIZE'])
  R = res2rad(d, wl, L)
  return int(round(R / ps))

vec_res2radpx = numpy.vectorize(res2radpx)

def res2radfpx(res, experiment):
  """
  Returns the radius in pixels that reflections
  of resolution I{res} given in Angstroms will fall
  given the experimental conditions passed in I{experiment}.
  The I{experiment} should be a mapping with the minimal key,
  value pairs:
     - I{WAVELENGTH} : wavelength in Angstroms
     - I{DISTANCE} : distance to detector in mm
     - I{PIXEL_SIZE} : size of pixels in mm
  """
  d = float(res)
  wl = float(experiment['WAVELENGTH'])
  L = float(experiment['DISTANCE'])
  ps = float(experiment['PIXEL_SIZE'])
  R = res2rad(d, wl, L)
  return R / ps

vec_res2radfpx = numpy.vectorize(res2radfpx)

def res2rho(res, experiment=None):
  return 1.0 / (2.0 * res)

vec_res2rho = numpy.vectorize(res2rho)

def res2theta(res, experiment):
  """
  Returns the angle theta in radians that reflections
  of resolution I{res} given in Angstroms will fall
  given the experimental conditions passed in I{experiment}.
  The I{experiment} should be a mapping with the minimal key,
  value pairs:
     - I{WAVELENGTH} : wavelength in Angstroms
  """
  d = float(res)
  wl = float(experiment['WAVELENGTH'])
  return math.asin(wl / (2.0 * d))

def res2twotheta(res, experiment):
  """
  Returns the angle 2*theta in radians that reflections
  of resolution I{res} given in Angstroms will fall
  given the experimental conditions passed in I{experiment}.
  The I{experiment} should be a mapping with the minimal key,
  value pairs:
     - I{WAVELENGTH} : wavelength in Angstroms
  """
  return 2 * res2theta(res, experiment)

vec_res2twotheta = numpy.vectorize(res2twotheta)

def rho2res(rho, experiment=None):
  return 1 / (2.0 * rho)

vec_rho2res = numpy.vectorize(rho2res)

def rho2twotheta(rho, experiment):
  """
  Returns the angle 2*theta in radians from `rho`
  which is :math:`\dfrac{\sin \theta}{\lambda}`
  The I{experiment} should be a mapping with the minimal key,
  value pairs:
     - I{WAVELENGTH} : wavelength in Angstroms
  """
  wl = experiment['WAVELENGTH']
  return 2 * math.asin(wl * rho)

vec_rho2twotheta = numpy.vectorize(rho2twotheta)

def res2tantwotheta(res, experiment):
  """
  Returns the tangent of 2 * theta that corresponds to the
  resolution `res`, given the experimental conditions passed in
  `experiment`. The `experiment` s hould be a mapping with the
  minimal key, value pairs:
     - `WAVELENGTH` : wavelength in Angstroms
  """
  wl = float(experiment['WAVELENGTH'])
  return math.tan(2 * math.asin(wl / (2.0 * res)))

def radpx2tantwotheta(px, experiment):
  """
  Returns the tangent of 2 * theta that corresponds to the
  resolution `res`, given the experimental conditions passed in
  `experiment`. The `experiment` s hould be a mapping with the
  minimal key, value pairs:
     - `DISTANCE` : detector distance in mm
     - `PIXEL_SIZE` : size of pixels in mm
  """
  if numpy.allclose(px, 0):
    return 0.0
  else:
    L = float(experiment['DISTANCE'])
    ps = float(experiment['PIXEL_SIZE'])
    return px * ps / L

def theta2res(theta, experiment):
  """
  Returns the resolution in Angstroms that reflections
  at theta (given in radians) will fall
  given the experimental conditions passed in I{experiment}.
  The I{experiment} should be a mapping with the minimal key,
  value pairs:
     - I{WAVELENGTH} : wavelength in Angstroms
  """
  wl = float(experiment['WAVELENGTH'])
  return wl / (2.0 * math.sin(theta))

def twotheta2rho(twotheta, experiment):
  """
  Returns the :math:`\rho` (:math:`\dfrac{\sin \theta}{\lambda}`)
  of reflections at 2*theta (given in radians)
  given the experimental conditions passed in I{experiment}.
  The I{experiment} should be a mapping with the minimal key,
  value pairs:
     - I{WAVELENGTH} : wavelength in Angstroms
  """
  wl = float(experiment['WAVELENGTH'])
  return math.sin(twotheta / 2.0) / wl

vec_twotheta2rho = numpy.vectorize(twotheta2rho)

def twotheta2res(twotheta, experiment):
  """
  Returns the resolution in Angstroms that reflections
  at 2*theta (given in radians) will fall
  given the experimental conditions passed in I{experiment}.
  The I{experiment} should be a mapping with the minimal key,
  value pairs:
     - I{WAVELENGTH} : wavelength in Angstroms
  """
  theta = twotheta / 2.0
  return theta2res(theta, experiment)

def twotheta2radpx(twotheta, experiment):
  """
  Returns the radius in pixels that reflections
  at two-theta angle I{twotheta} given in radians will fall
  given the experimental conditions passed in I{experiment}.
  The I{experiment} should be a mapping with the minimal key,
  value pairs:
     - I{DISTANCE} : distance to detector in mm
     - I{PIXEL_SIZE} : size of pixels in mm
  """
  twotheta = float(twotheta)
  L = float(experiment['DISTANCE'])
  ps = float(experiment['PIXEL_SIZE'])
  return L * math.tan(twotheta) / ps

vec_twotheta2radpx = numpy.vectorize(twotheta2radpx,
                                     otypes=[numpy.float])

def twothetadegs2radpx(twotheta, experiment):
  """
  Returns the radius in pixels that reflections
  at two-theta angle I{twotheta} given in degrees will fall
  given the experimental conditions passed in I{experiment}.
  The I{experiment} should be a mapping with the minimal key,
  value pairs:
     - I{DISTANCE} : distance to detector in mm
     - I{PIXEL_SIZE} : size of pixels in mm
  """
  twotheta = math.radians(float(twotheta))
  return twotheta2radpx(twotheta, experiment)

vec_twothetadegs2radpx = numpy.vectorize(twothetadegs2radpx)

def radpx2res(px, experiment):
  if px == 0:
    return sys.float_info.max
  else:
    wl = float(experiment['WAVELENGTH'])
    L = float(experiment['DISTANCE'])
    ps = float(experiment['PIXEL_SIZE'])
    R = px * ps
    return wl/(2.0*math.sin(math.atan(R/L)/2.0))

vec_radpx2res = numpy.vectorize(radpx2res)

# rho is sin(theta)/lambda
def radpx2rho(px, experiment):
  if px == 0:
    return 0.0
  else:
    wl = float(experiment['WAVELENGTH'])
    L = float(experiment['DISTANCE'])
    ps = float(experiment['PIXEL_SIZE'])
    R = px * ps
    return math.sin(math.atan(R/L)/2.0)/wl

vec_radpx2rho = numpy.vectorize(radpx2rho)


def radpx2twotheta(px, experiment):
  """
  Returns the angle twotheta (2*theta) in radians
  of the reflection plane to which
  reflections at the radius I{px} in pixels
  correspond.
  """
  L = float(experiment['DISTANCE'])
  ps = float(experiment['PIXEL_SIZE'])
  return math.atan(px*ps/L)

vec_radpx2twotheta = numpy.vectorize(radpx2twotheta)

def radpx2twothetadegs(px, experiment):
  """
  Returns the angle twotheta (2*theta!) in radians
  of the reflection plane to which
  reflections at the radius I{px} in pixels
  correspond.
  """
  return math.degrees(radpx2twotheta(px, experiment))

vec_radpx2twothetadegs = numpy.vectorize(radpx2twothetadegs)

def radpx2theta(px, experiment):
  """
  Returns the angle theta in radians
  of the reflection plane to which
  reflections at the radius I{px} in pixels
  correspond.
  """
  L = float(experiment['DISTANCE'])
  ps = float(experiment['PIXEL_SIZE'])
  return math.atan(px*ps/L)/2.0

vec_radpx2theta = numpy.vectorize(radpx2theta)

def radpx2thetadegs(px, experiment):
  return math.degrees(radpx2theta(px, experiment))

def radpx2dstar(px, experiment):
  wl = float(experiment['WAVELENGTH'])
  sintheta = math.sin(radpx2theta(px, experiment))
  return 2 * (sintheta / wl)

def radpx2dstarsq(px, experiment):
  wl = float(experiment['WAVELENGTH'])
  sintheta = math.sin(radpx2theta(px, experiment))
  return (2 * (sintheta / wl))**2

def twothetadegs2res(twotheta, experiment):
  theta = math.radians(twotheta) / 2
  return float(experiment['WAVELENGTH']) / (2 * math.sin(theta))

vec_twothetadegs2res = numpy.vectorize(twothetadegs2res)

def twothetadegs2rho(twotheta, experiment):
  theta = math.radians(twotheta) / 2
  wl = float(experiment['WAVELENGTH'])
  return math.sin(theta) / wl

vec_twothetadegs2rho = numpy.vectorize(twothetadegs2rho)


def bin_area(limits, experiment):
  """
  Returns the area of a bin with the resolution `limits` in Angstroms
  (a length-2 sequence) on the detector in square milimeters,
  given the experimental conditions passed in  `experiment`.
  The `experiment` should be a mapping with the minimal key,
  value pairs:
     - `WAVELENGTH` : wavelength in Angstroms
     - `DISTANCE` : detector distance in mm
  """
  wl = experiment['WAVELENGTH']
  L = experiment['DISTANCE']
  limits = signals.hilo(limits)
  rad_in = L * (math.tan(2.0 * math.asin(wl / (2.0 * limits[0]))))
  rad_out = L * (math.tan(2.0 * math.asin(wl / (2.0 * limits[1]))))
  a_in = 2 * math.pi * (rad_in**2)
  a_out = 2 * math.pi * (rad_out**2)
  return a_out - a_in

def make_borders_rho(n_bins, d_max, d_min, experiment=None):
  """
  Returns borders in :math:`\rho`
  (:math:`\dfrac{\sin \theta}{\lambda}`) and linear in :math:`\rho`.
  The borders are for `n_bins` bins, ranging in
  d-spacing resolution from `d_max` to `d_min`. Remember
  that `d_min` is higher resolution (higher :math:`\rho`)
  than `d_max`. The d-spacings are expressed in Angstroms.
  The `experiment` keyword argument is not used, but only exists
  for homology with other make_borders functions.

  The returned borders are 1 greater in length
  than the number `n_bins`.
  """
  d_max, d_min = signals.hilo((d_max, d_min))
  rho_min = 1.0 / (2.0 * d_max)
  rho_max = 1.0 / (2.0 * d_min)
  return numpy.linspace(rho_min, rho_max, num=(n_bins + 1))

def make_borders_twotheta(n_bins, d_max, d_min, experiment):
  """
  Returns borders in :math:`\rho`
  (:math:`\dfrac{\sin \theta}{\lambda}`) but linear in :math:`2\theta`
  The borders are for `n_bins` bins, ranging in
  d-spacing resolution from `d_max` to `d_min`. Remember
  that `d_min` is higher resolution (higher :math:`\rho`)
  than `d_max`. The d-spacings are expressed in Angstroms.
  The `experiment` should be a mapping with the minimal key,
  value pairs:
     - `WAVELENGTH` : wavelength in Angstroms

  The returned borders are 1 greater in length
  than the number `n_bins`.
  """
  d_max, d_min = signals.hilo((d_max, d_min))
  wl = float(experiment['WAVELENGTH'])
  tth_max = 2 * math.asin(wl / (2.0 * d_max))
  tth_min = 2 * math.asin(wl / (2.0 * d_min))
  tths = numpy.linspace(tth_max, tth_min, num=(n_bins + 1))
  return vec_twotheta2rho(tths, experiment)


LOCS_DICT = { 'd-star': radpx2dstar,
              'd-star squared': radpx2dstarsq,
              'pixel': count_function(),
              'resolution': radpx2res,
              'two-theta degs' : radpx2twothetadegs,
              'theta degs' : radpx2thetadegs,
              'tan two-theta' : radpx2tantwotheta }

XLABS_DICT = {'resolution': radpx2res,
              'two-theta radians': radpx2twotheta,
              'two-theta degrees': radpx2twothetadegs}

def spiraller(center, radius, step=1):
  """
  A generator that yields x,y pairs starting from `center`
  and proceding outwards. The `center` should be integers
  and so should the step.
  """
  if step < 1:
    step = 1
  yield center
  rad_sq = radius**2
  h, k = center
  x, y = center
  distance = 1
  while True:
    stop = True
    for move in [1, -1]:
      for i in xrange(distance):
        x += step*move
        if ((x-h)**2 + (y-k)**2) < rad_sq:
          stop = False
          yield (x, y)
      for j in xrange(distance):
        y += step*move
        if ((x-h)**2 + (y-k)**2) < rad_sq:
          stop = False
          yield (x, y)
      distance += 1
    if stop:
      break

def open_file_osx(filename):
  os.system ('open %s' % filename)
  proceed = askyesno("File opened...", "Proceed?")
  if not proceed:
    sys.exit(0)

def draw_circle(data, center, radius, value):
  for idx in spiraller(center, radius):
    data[idx] = value

def save_img(im, fileout, wrapup=None):
  im.save(fileout)
  if wrapup is not None:
    wrapup(fileout)

def make_img(data, minmax=(254, 1), progress=None):
  imdata = scale_min_max(data, minmax, progress=progress)
  imdata = numpy.array(imdata, dtype='u1')
  mode = 'L'
  size = (imdata.shape[1], imdata.shape[0])
  im = Image.frombuffer(mode, size, imdata, 'raw', mode, 0, 1)
  return im

def distance(p1, p2):
  """
  Returns the distance between the two (x,y) points
  I{p1} and I{p2}.
  """
  return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def recenter(general, config, image,
             plt=None, job=None, task=None, cache={}):
  """
  Uses a radial centering algorithm to return a dictionary
  keyed by (x,y) tuples describing the centers of the
  various calculations and valued by the results of those
  calculations. The returned results reflect image space
  (column, row) rather than array space (row, column).

  The algorithm breaks the specified disc in the
  image down by the number of bins specified, each with equal
  width in pixels, and calculates the average pixel value for
  each bin after discarding outliers and shadow pixels (where
  the pixel value is below the minimum specified). The values
  of the returned dictionary are maximum of the bins for
  the disc centered at the key. The relative magnitudes of these
  values roughtly reflect the probability that the center of the
  image is at the the pixel represented in the key.

  The I{header} parameter is a valid adxv binary .img file header.

  If the I{plt} argument is supplied, it should take as a
  parameter a sequence of values to plot. The I{progress}
  parameter, if supplied should take the parameter list described
  in L{donothing}.

  I{config} is a mapping object with the following spec:
    - start_center: None or tuple, 2-tuple (x,y) of the start center
    - bin_factor: int, re-binning factor to reduce image size
    - num_bins: int, number of resolution bins in the disc
    - recenter_radius: int, radius of circle to explore, in pixels
    - mark_center: bool, whether to check image by marking the center
    - mark_radius: int, how big to make the mark if checking
    - mark_tif: name of the file written the marked image

  I{image} is a mapping object with the following spec:
    - filename: str, name of the adxv binary .img file
    - center: tuple, 2-tuple (x,y) of the image center
    - disc: tuple, 2-tuple with the inner and outer radii of the disc
    - disc_units: str, "A" for Angstroms or "px" for pixels
    - min_i: float or int, minimum cutoff value for pixels
  """

  start_center = config['start_center']
  bin_factor = config['bin_factor']
  num_bins = config['num_bins']
  recenter_radius = config['recenter_radius']
  mark_center = config['mark_center']

  # the data is organized by row (y), so fix the input
  if start_center is None:
    start_center = flip_pt(image['center'])
  else:
    start_center = flip_pt(start_center)

  datadir = general['img_dir']
  image['filepath'] = os.path.join(datadir, image['filename'])
  adxv_file = open(image['filepath'], "rb")
  header = read_header(adxv_file)

  rin, rout = config['disc']
  unit = config['disc_units']
  if unit == "A":
    rin = res2radpx(rin, header)
    rout = res2radpx(rout, header)
  if rin > rout:
    rout, rin = rin, rout

  # data is organized (row, column) unlike images (column, row)
  data = read_data(adxv_file, header)
  shape = data.shape
  criterion = config['criterion']
  def _f(a):
    return signals.chauvenet(a, criterion=criterion)
  def _g(a, n, i):
    return 0.0
  if criterion is None:
    dezingered = data.copy()
  else:
    dezingered = signals.dezinger(data, _f, _g)
  rebinned = rebin(dezingered, bin_factor, progress=task)

  if mark_center:
    fill = numpy.max(rebinned)
    cx, cy = start_center
    c_center = (cx/bin_factor, cy/bin_factor)
    c_radius = config['mark_radius'] / bin_factor
    draw_circle(rebinned, c_center, c_radius, fill)
    im = make_img(rebinned, progress=task)
    save_img(im, config['mark_tif'], wrapup=open_file_osx)

  stdevs = []
  maxes = []
  coords = []
  cx, cy = start_center
  _start_center = (cx/bin_factor, cy/bin_factor)
  recenter_radius = recenter_radius/bin_factor
  sp = spiraller(_start_center, recenter_radius)
  centers = numpy.array(sorted(sp))
  num_centers = centers.shape[0]
  start_x = centers[:,0].min()
  end_x = centers[:,0].max()
  span_x = 1 + end_x - start_x
  start_y = centers[:,1].min()
  end_y = centers[:,1].max()
  span_y = 1 + end_y - start_y
  ayes = numpy.arange(span_x)
  jays = numpy.arange(span_y)
  n = 0
  for i, j in product(ayes, jays):
    x = i + start_x
    y = j + start_y
    center = (x, y)
    if numpy.logical_and.reduce(centers == center, axis=1).any():
      n += 1
    else:
      continue
    frac = float(n)/num_centers
    tmplt = 'Doing Center %d of %d: %s'
    # the center is already flipped, so flip back for reporting
    msg = tmplt % (n, num_centers, flip_pt(center))
    job(msg, frac)
    disc = {'center':center,
            'rin':rin/bin_factor,
            'rout':rout/bin_factor}
    idict = integrate(rebinned, num_bins, disc,
                      minI=image['min_i'],
                      progress=task,
                      cache=cache)
    avs = idict['means']
    sters = idict['stds']
    ns = idict['counts']
    if plt is not None:
      plt.plot(avs)
    mean_err = numpy.mean(sters)
    stdevs.append((center, mean_err))
    max_av = numpy.max(avs)
    maxes.append(max_av)
    coords.append((i, j))
  job(msg, 1)

  exes = (ayes + start_x) * bin_factor
  whys = (jays + start_y) * bin_factor
  
  vals = numpy.array(maxes)
  coords = numpy.array(coords)
  max_xy = bin_factor * (coords[vals.argmax()] + [start_x, start_y])

  # flip everything back to image space so function is black box
  coords = numpy.array([coords[:,1], coords[:,0]]).T
  max_xy = flip_pt(max_xy)
  exes, whys = whys, exes

  result = {"coords" : coords,
            "vals" : vals,
            "exes" : exes,
            "whys" : whys,
            "max_xy" : max_xy}

  return result

def load_spectra(config, spectra):
  # this will pack up spectra as if they were images
  sm_fac = int(config.get('smoothing_factor', 20))
  min_w = int(config.get('min_window', 7))
  max_w = int(config.get('max_window', 13))
  roi_low, roi_high = signals.hilo(config['roi'])
  # these have to be something to make them like an experiment
  expt_defs = {'DISTANCE':250, 'PIXEL_SIZE':0.1}
  ms = OrderedDict()
  for k, spec in spectra:
    expt = {'WAVELENGTH':spec['wavelength']}
    expt.update(expt_defs)
    key = (spec['filepath'],)
    spectrum = yaml.load(open(spec['filepath']))
    if isinstance(spectrum, dict):
      spectrum = spectrum['pattern']
    spectrum = numpy.transpose(spectrum)
    bin_middles_2th = spectrum[0,1:-1]
    bin_borders_2th = (spectrum[0,1:] + spectrum[0,:-1]) / 2
    bin_middles_px = vec_twothetadegs2radpx(bin_middles_2th, expt)
    bin_borders_px = vec_twothetadegs2radpx(bin_borders_2th, expt)
    bin_borders_res = vec_twothetadegs2res(bin_borders_2th, expt)
    bin_borders_rho = vec_twothetadegs2rho(bin_borders_2th, expt)
    bins = len(bin_middles_px)
    roi_low_px = res2radfpx(roi_low, expt)
    roi_high_px = res2radfpx(roi_high, expt)
    # this should be the intersection
    bois = numpy.where((roi_low_px < bin_borders_px[1:]) &
                       (roi_high_px > bin_borders_px[:-1]))[0]
    window_len, _r = divmod(bins, sm_fac)
    window_len = min(max(window_len, min_w), max_w)
    whys = spectrum[1,1:-1]
    if config['smooth_spectra']:
      winlen = int(whys.size *
                   (float(window_len) / config['roi_bins']))
      whys = signals.smooth(whys, window_len=winlen)
    bin_borders = zip(bin_borders_res[:-1], bin_borders_res[1:])
    bin_areas = [bin_area(bb, expt) for bb in bin_borders]
    bin_areas = numpy.array(bin_areas)
    meas = OrderedDict()
    meas['image_key'] = k
    meas['image'] = spec
    meas['experiment'] = expt
    meas['spectrum'] = spectrum
    meas['sum Is'] = whys
    # meas['average Is'] = whys
    meas['std errors'] = None
    meas['sigma'] = meas['sum Is'].std()
    # meas['Is over sigma'] = meas['average Is'] / meas['sigma']
    meas['counts'] = None
    meas['bins'] = bins
    meas['bin middles px'] = bin_middles_px
    meas['bin borders px'] = bin_borders_px
    meas['bin borders res'] = bin_borders_res
    meas['bin borders rho'] = bin_borders_rho
    meas['bin areas'] = bin_areas
    meas['roi high'] = roi_high
    meas['roi low'] = roi_low
    meas['bins of interest'] = bois
    meas['window length'] = window_len
    meas['alpha'] = None
    ms[k] = meas
  return ms

def get_legend_handles_labels(plt):
  legend = plt.axes.get_legend()
  if legend is None:
    handles, labels = [], []
  else:
    handles = legend.get_lines()
    labels = [t.get_text() for t in legend.get_texts()]
  return handles, labels


def expand(group, groups, added=None):
  """
  Expands nested groups as specified by tokens
  GROUP_START, GROUP_END, and GROUP_SLICE.

  Note to self: This function needs an example.
  """
  if added is None:
    added = []
  grouped = []
  if isinstance(group, basestring):
    if group not in added:
      added.append(group)
      if group.startswith(GROUP_START) and group.endswith(GROUP_END):
        try:
          group = groups[group[GROUP_SLICE]]
        except KeyError, e:
          raise GroupError, "No such group: '%s'." % group
        expanded = expand(group, groups, added)
        grouped.extend(expanded)
      else:
        grouped.append(group)
  else:
    for gstring in group:
      expanded = expand(gstring, groups, added)
      grouped.extend(expanded)
  return grouped

def get_data(cfg, dirkey, groupkey, datadb, mode):
  group = cfg[mode][groupkey]
  keyed = []
  if group:
    datadir = cfg['general'][dirkey]
    if "groups" in cfg:
      groups = cfg['groups'][groupkey]
      keys = expand(group, groups)
    else:
      keys = group[:]
    for k in keys:
      datum = datadb[k]
      filepath = os.path.join(datadir, datum['filename'])
      datum['filepath'] = filepath
      keyed.append((k, datum))
  return keyed

def get_images(cfg, images, mode):
  dirkey = 'img_dir'
  groupkey = 'images'
  return get_data(cfg, dirkey, groupkey, images, mode)

def get_spectra(cfg, spectra, mode):
  dirkey = 'spec_dir'
  groupkey = 'spectra'
  return get_data(cfg, dirkey, groupkey, spectra, mode)

def stretch(values, bois, min_val=None):
  if min_val is None:
    min_val = numpy.min(values)
  if bois is None:
    chosen = values
  else:
    chosen = numpy.array([values[b] for b in bois])
  delta = numpy.max(chosen) - min_val
  if delta == 0:
    new_vals = values
  else:
    new_vals = (values - min_val) / delta
  return new_vals

NORM_FUNCS = {None:None, 'max':numpy.max, 'mean':numpy.mean,
              'stretch':stretch}

def plot_image(plt, measurement, config, pltcfg, plot_number):
  """
  :param plt: the plot instance that works like pygmyplot.MyXYPlot
  :param measurement: a measurement dictionary with the following
                      structure:

         - image_key: key from the database
         - image: entry describing image in the image database
         - experiment: union of information reconciled between
                       the image header and the image database
         - average Is: the measured average intensities for the bins
         - std errors: standard errors corresponding to average Is
         - Is over sigma: {average Is} / sigma
         - counts: number of pixels for the bins
         - sigma: std error for all pixels after Chauvenet filtering
         - bins: number of bins (same as length of sum Is, etc)
         - bin middles px: middle of the bins as measured in pixels
         - bin borders px:  borders of the bins as measured in pixels
                            so there are bins+1 borders in the array
         - roi high: upper limit of the resolution range
                     of interest in Angstroms
         - roi low: lower limit of the resolution range
                    of interest in Angstroms
         - bins of interest: indices of those bins within the
                             resolution range of interest (roi)
         - window length: length of window for smoothing and
                          sharpness calculation
         - alpha: angle of sharpness of peak within roi (radians)

  :param config: configuration for the averaging procedure
  :param pltcfg: the plot configuration
  :param plot_number: how many spectra have already been plotted
  """

  norm_func = NORM_FUNCS[config['norm_type']]

  legend_loc = pygmyplot.LEGEND_LOCS.get(config['legend'], None)

  experiment = measurement['experiment']

  image = measurement['image']
  nickname = image['nickname']

  if 'scaled to' in measurement:
    bin_middles_px = measurement['scaled bmpx']
    # i_over_sigs = measurement['scaled']
    if 'bg corrected' in measurement:
      sums = measurement['bg corrected']
    else:
      sums = measurement['scaled']
    bois = measurement['scaled bois']
    bins = measurement['scaled bins']
  else:
    bin_middles_px = measurement["bin middles px"]
    # i_over_sigs = measurement["Is over sigma"]
    sums = measurement['sum Is']
    bois = measurement["bins of interest"]
    bins = measurement["bins"]

  window_len = measurement["window length"]

  do_smoothed = pltcfg.get('smoothed', True)
  x_scaling = pltcfg.get('x_scaling', 'pixel')
  xlab_units = pltcfg.get('x_labels', 'resolution')
  xlab_fmt = "%" + pltcfg.get("x_label_format", "5.2f")
  xlab_sp = pltcfg.get("x_label_spacing")
  expl_x_labs = pltcfg.get("explicit_x_labels")
  line_width = pltcfg.get('line_width', 1)
  tick_length = pltcfg.get('tick_length', 7)
  tick_width = pltcfg.get('tick_length', 1.5)

  num_xticks = pltcfg.get('num_xticks', 0)
  if (num_xticks is None) or (num_xticks == "None"):
    num_xticks = 0

  do_xlabels = ((num_xticks >= 0) or
                (expl_x_labs is not None) and
                (plot_number == 0))

  textprop = pygmyplot.fontprop(size=pltcfg['text_size'],
                                weight=pltcfg['text_weight'])
  titleprop = pygmyplot.fontprop(size=pltcfg['title_size'],
                              weight=pltcfg['title_weight'])
  legendprop = pygmyplot.fontprop(size=pltcfg['legend_size'],
                                  weight=pltcfg['legend_weight'])
  labelprop = pygmyplot.fontprop(size=pltcfg['label_size'],
                                 weight=pltcfg['label_weight'])

  hndls, lbls = get_legend_handles_labels(plt)

  locs_func = LOCS_DICT[x_scaling]
  tick_locs = [locs_func(px, experiment) for px in bin_middles_px]
  tick_locs = numpy.array(tick_locs)

  smoothed = signals.smooth(sums, window_len=window_len)
  if pltcfg['smoothed']:
    whys = smoothed
  else:
    whys = sums
  fctr = 1.0
  if config['stretch']:
    whys = stretch(whys, bois)
  if config['scaled_to']:
    whys = measurement['scaled']
  elif config['do_norm'] and (norm_func is not None):
    fctr = norm_func([whys[b] for b in bois])
    whys = whys/fctr

  if do_xlabels:
    do_xlabels = False
    xlab_func = XLABS_DICT[xlab_units]
    tick_labs = [xlab_func(px, experiment) for px in bin_middles_px]

    xlabels = []
    if num_xticks > 0:
      for b, bc in enumerate(tick_labs):
        bins_per_lab = bins/num_xticks
        if bins_per_lab < 1:
          bins_per_lab = 1
        if b % bins_per_lab == 0:
          xlabels.append((xlab_fmt % bc).strip())

    if xlab_sp > 0:
      def _nudge(alab):
        slab = iter(str(alab))
        new_lab = []
        digs = 0
        for achar in slab:
          if achar in string.digits:
            digs += 1
          if digs > xlab_sp:
            break
          else:
            new_lab.append(achar)
        for achar in slab:
          if achar in string.digits:
            if int(achar) > 4:
              if new_lab[-1] == '.':
                new_lab[-2] = str(int(new_lab[-2]) + 1)
              else:
                new_lab[-1] = str(int(new_lab[-1]) + 1)
            break
        return float("".join(new_lab))
      xlabels = [_nudge(alab) for alab in xlabels]

    if expl_x_labs is not None:
      expl_x_labs = [float(v) for v in expl_x_labs]
      xlabels += expl_x_labs

    xlabels = list(set(xlabels))
    revloc_dict = { 'resolution': res2radfpx,
                    'two-theta radians': twotheta2radpx,
                    'two-theta degrees': twothetadegs2radpx }
    revloc_func = revloc_dict[xlab_units]
    xtick_locs_px = [revloc_func(xlb, experiment) for xlb in xlabels]
    xticks = [locs_func(px, experiment) for px in xtick_locs_px]
    xlabels = [(xlab_fmt % lb).strip() for lb in xlabels]
    if pltcfg['clean_x_labels']:
      xlabels = [(lb.rstrip("0") if "." in lb else lb)
                                        for lb in xlabels]
      xlabels = [lb.rstrip(".") for lb in xlabels]

  line = plt.plot(tick_locs, whys, linewidth=pltcfg['line_width'])
  ticklines = plt.axes.get_xticklines() + plt.axes.get_yticklines()
  for tickline in ticklines:
    tickline.set_markersize(float(pltcfg['tick_length']))
    tickline.set_markeredgewidth(float(pltcfg['tick_width']))
  plt.axes.set_xticklabels(xlabels, fontproperties=textprop)
  plt.axes.set_xticks(xticks)

  x_limits = pltcfg.get("x_limits")
  if x_limits is None:
    x_loc_min, x_loc_max = (min(tick_locs), max(tick_locs))
    x_loc_limits = pltcfg.get("x_loc_limits")
    if x_loc_limits is not None:
      x_loc_limits_min, x_loc_limits_max = x_loc_limits
      x_loc_min = min(x_loc_min, x_loc_limits_min)
      x_loc_max = max(x_loc_max, x_loc_limits_max)
    x_loc_limits = x_loc_min, x_loc_max
  else:
    x_min, x_max = x_limits
    x_loc_min = locs_func(res2radfpx(x_min, experiment), experiment)
    x_loc_max = locs_func(res2radfpx(x_max, experiment), experiment)
    x_loc_limits = signals.lohi((x_loc_min, x_loc_max))
  pltcfg['x_loc_limits'] = x_loc_limits
  if legend_loc is not None:
    hndls.append(line.marker_line.line2d)
    lbls.append(nickname)
    plt.axes.legend(hndls, lbls, loc=legend_loc, prop=legendprop)
  for tick in plt.axes.get_yaxis().get_major_ticks():
    tick.label1.set_fontproperties(textprop)
  plt.canvas.draw()

def plot_adjust(plt, pltcfg):
  adjustments = {}
  for adj in ["left", "bottom", "right", "top", "wspace", "hspace"]:
    adjustments[adj] = pltcfg[adj]

  plt.figure.subplots_adjust(**adjustments)

def plot_images(plt, measurements, config, pltcfg, start):

  # textprop = pygmyplot.fontprop(size=pltcfg['text_size'],
  #                               weight=pltcfg['text_weight'])
  titleprop = pygmyplot.fontprop(size=pltcfg['title_size'],
                              weight=pltcfg['title_weight'])
  # legendprop = pygmyplot.fontprop(size=pltcfg['legend_size'],
  #                                 weight=pltcfg['legend_weight'])
  labelprop = pygmyplot.fontprop(size=pltcfg['label_size'],
                                 weight=pltcfg['label_weight'])
  y_limits = signals.lohi(pltcfg['y_limits'])

  pn = start
  for (i, image_key) in enumerate(measurements):
    measurement = measurements[image_key]
    pn += 1
    plot_image(plt, measurement, config, pltcfg, pn)

  plot_adjust(plt, pltcfg)

  if config['plot_title'] is not None:
    plt.axes.set_title(config['plot_title'],
                       fontproperties=titleprop)
  plt.axes.set_xlabel(pltcfg['x_title'], fontproperties=labelprop)
  plt.axes.set_ylabel(pltcfg['y_title'], fontproperties=labelprop)
  if len(measurements) > 1:
    plt.axes.set_ylim(y_limits)
  plt.axes.set_xlim(pltcfg.get('x_loc_limits', pltcfg['x_limits']))
  plt.canvas.draw()
  return pn


def sharpness(config, measurement):
  image = measurement['image']
  bins = measurement['bins']
  last_idx = bins - 1
  bois = measurement['bins of interest']
  window_len = measurement['window length']
  # i_over_sigs = measurement['Is over sigma']
  sums = measurement['sum Is']
  bin_middles_px = measurement['bin middles px']
  header = measurement['experiment']
  sharpness_roi = config['sharpness_roi']

  if sharpness_roi is None:
    min_val = None
  else:
    bin_borders_res = measurement['bin borders res']
    bin_borders = zip(bin_borders_res[:-1], bin_borders_res[1:])
    roi_low, roi_high = signals.hilo(sharpness_roi)
    sh_bois = get_bins_of_interest(bin_borders, roi_low, roi_high)
    # min_val = min(i_over_sigs[b] for b in sh_bois)
    min_val = min(sums[b] for b in sh_bois)

  # maximum, boi = max((i_over_sigs[b], b) for b in bois)
  maximum, boi = max((sums[b], b) for b in bois)
  # normed = i_over_sigs / maximum
  # normed = stretch(i_over_sigs, bois, min_val=min_val)
  normed = stretch(sums, bois, min_val=min_val)
  normed_smoothed = signals.smooth(normed, window_len=window_len)
  go_back = 1
  extend_peak = int(max(config.get('extend_peak', 1), 1))
  av_n = go_back + extend_peak
  scnd = derivative(normed_smoothed, 2)
  if scnd[boi] < 0:
    op = operator.gt
  elif scnd[boi] > 0:
    op = operator.lt
  else:
    op = lambda x,y: True
  hi_med = None
  lo_med = None
  for delta in count(1):
    lo_idx = boi - delta
    hi_idx = boi + delta
    if lo_med is None:
      if lo_idx >= av_n:
        if op(scnd[lo_idx], 0):
            lo_med = lo_idx
      else:
        lo_med = extend_peak
    if hi_med is None:
      if hi_idx <= (last_idx - extend_peak):
        if op(scnd[hi_idx], 0):
            hi_med = hi_idx
      else:
        hi_med = (last_idx - extend_peak)
    if (lo_med is not None) and (hi_med is not None):
      break
  boi_i = normed[boi]
  lo_idxs = numpy.arange(lo_med + go_back,
                         lo_med - av_n + go_back, -1)
  hi_idxs = numpy.arange(hi_med - go_back,
                         hi_med + av_n - go_back)
  lo_i = normed[lo_idxs].mean()
  hi_i = normed[hi_idxs].mean()
  boi_px = bin_middles_px[boi]
  lo_px = bin_middles_px[lo_idxs].mean()
  hi_px = bin_middles_px[hi_idxs].mean()

  boi_res = radpx2dstar(boi_px, header)
  lo_res = radpx2dstar(lo_px, header)
  hi_res = radpx2dstar(hi_px, header)

  h3 = math.sqrt((hi_res - lo_res)**2 + (hi_i - lo_i)**2)
  lo_h = math.sqrt((boi_res - lo_res)**2 + (boi_i - lo_i)**2)
  hi_h = math.sqrt((hi_res - boi_res)**2 + (hi_i - boi_i)**2)
  alpha = threesides2alpha(h3, lo_h, hi_h)
  atup = (image['nickname'], math.degrees(alpha))
  logging.info('%s has alpha of %s' % atup)
  return alpha

def sharpnesses(config, measurements):
  if config.get("do_sharpness", False):
    for measurement in measurements:
      measurement['alpha'] = sharpness(config, measurement)
  else:
    for measurement in measurements:
      measurement['alpha'] = None

def get_bins_of_interest(bin_borders, roi_low, roi_high):
  if roi_low is None or roi_high is None:
    bois = None
  else:
    bois = []
    for res, (lower, higher) in enumerate(bin_borders):
      # these are resolutions, hence comparison backwards
      if ((roi_high <= lower <= roi_low) or
          (roi_high <= higher <= roi_low)):
        bois.append(res)
  return bois

def averages(config, images, pltcfg=None,
                             plt=None,
                             job=None,
                             task=None,
                             cache={}):
  """
  The I{job} and I{task} parameters are progress functions
  (see L{donothing} for the expected parameter list). The I{task}
  function will be called more often than the I{job} function.
  """
  res_low, res_high = signals.hilo(config['limits'])
  bins = config['img_bins']
  last_idx = bins - 1
  roi_low, roi_high = signals.hilo(config['roi'])
  norm_func = NORM_FUNCS[config['norm_type']]

  if job is None:
    job = donothing
  if task is None:
    task = donothing
  # for legend if there will be one
  measurements = OrderedDict()
  num_files = len(images)
  for filenum, (image_key, image) in enumerate(images):
    image_file = image['filepath']
    minI = image['min_i']
    params = (filenum+1, num_files, image_file)
    msg = "Averaging file %s of %s: %s." % params
    frac = float(filenum)/float(num_files)
    job(msg, frac)
    center = flip_pt(image['center'])
    adxv_file = open(image_file, "rb")
    header = read_header(adxv_file)
    wl = image.get('wavelength')
    if wl is None:
      wl = header['WAVELENGTH']
      image['wavelength'] = wl
    else:
      header['WAVELENGTH'] = wl
    rin = res2radpx(res_low, header)
    rout = res2radpx(res_high, header)

    pxperbin = float(rout - rin)/bins
    bin_borders_px = [(rin+(pxperbin*b)) for b in xrange(bins+1)]
    bin_borders_px = numpy.array(bin_borders_px)
    bin_borders = [radpx2res(px, header) for px in bin_borders_px]
    bin_borders = zip(bin_borders[:-1], bin_borders[1:])
    bin_middles_px = [(rin+(pxperbin*(b+0.5))) for b in xrange(bins)]
    bin_middles_px = numpy.array(bin_middles_px)

    bois = get_bins_of_interest(bin_borders, roi_low, roi_high)

    disc = {'center':center, 'rout':rout, 'rin':rin}

    keytup = (image_key, image, bins, disc, minI)
    cache_key = pyfscache.make_digest(keytup)
    if cache_key in cache:
      idict = cache[cache_key]
    else:
      data = read_data(adxv_file, header)
      idict = integrate(data, bins, disc, minI=minI,
                        progress=task,
                        cache=cache)
      cache[cache_key] = idict
    avs = idict['means']
    sums = idict['sums']
    sters = idict['stds']
    counts = idict['counts']
    sigma = idict['sigma']
    i_over_sigs = (avs / sigma)

    sm_fac = int(config.get('smoothing_factor', 20))
    min_w = int(config.get('min_window', 7))
    max_w = int(config.get('max_window', 13))
    window_len, _r = divmod(bins, sm_fac)
    window_len = min(max(window_len, min_w), max_w)
    if bois is not None:
      if not bois:
        msg = "No bins within %s-%s range." % roi_low, roi_high
        raise RadialXError, msg
    bin_borders_rho = vec_radpx2rho(bin_borders_px, header)
    # just unpack the bin_borders to make bin_borders_res
    bin_borders_res = [lo for (lo, hi) in bin_borders]
    bin_borders_res.append(bin_borders[-1][-1])
    bin_borders_res = numpy.array(bin_borders_res)

    bin_areas = [bin_area(bb, header) for bb in bin_borders]
    bin_areas = numpy.array(bin_areas)

    measurements[image_key] = {
         # image_key: key from the database
         "image_key" : image_key,
         # image: entry describing image in the image database
         "image" : image,
         # experiment: union of information reconciled between
         #             the image header and the image database
         "experiment" : header,
         # average Is: the measured average intensities for the bins
         "average Is" : avs,
         # sum Is: the sums of the intensities for the bins
         "sum Is": sums,
         # std errors: standard errors corresponding to average Is
         "std errors" : sters,
         # Is over sigma: {average Is} / sigma
         "Is over sigma" : i_over_sigs,
         # counts: number of pixels for the bins
         "counts" : counts,
         # sigma: std error for all pixels after Chauvenet filtering
         "sigma" : sigma,
         # bins: number of bins (same as length of average Is, etc)
         "bins" : bins,
         # bin middles px: middle of the bins as measured in pixels
         "bin middles px" : bin_middles_px,
         # bin borders px:  borders of the bins as measured in pixels
         #                  so there are bins+1 borders in the array
         "bin borders px" : bin_borders_px,
         # bin borders rho: borders of the bins as measured in
         #                  sin(theta) / lambda
         "bin borders rho" : bin_borders_rho,
         # bin borders res: borders of the bins as measured in
         #                  sin(theta) / lambda
         "bin borders res" : bin_borders_res,
         # bin areas: areas of the bins in square mm
         "bin areas": bin_areas,
         # roi high: upper limit of the resolution range
         #           of interest in Angstroms
         "roi high" : roi_high,
         # roi low: lower limit of the resolution range
         #          of interest in Angstroms
         "roi low" : roi_low,
         # bins of interest: indices of those bins within the
         #                   resolution range of interest (roi)
         "bins of interest" : bois,
         # window length: length of window for smoothing and
         #                sharpness calculation
         "window length" : window_len
         }
  return measurements


def make_integrator(measurement, scaled=False):
  """
  Resolution limits for the integrator should be
  in Angstroms.
  """
  if scaled:
    C = measurement['scaled']
  else:
    # C = measurement['average Is']
    C = measurement['sum Is']
  experiment = measurement['experiment']
  bin_borders_rho = measurement['bin borders rho']
  delta_rho_bins = bin_borders_rho[1:] - bin_borders_rho[:-1]
  def _ig(res_low, res_high, exact=False):
    """
    Exact means the res_low and res_high are expected
    to fall right on the bin borders. Because of
    floating point, they may not, but this case is
    taken into consideration.
    """
    a = res2rho(res_low, experiment)
    b = res2rho(res_high, experiment)
    bin_m = bin_borders_rho.searchsorted(a)
    bin_a = bin_m - 1
    bin_n = bin_borders_rho.searchsorted(b) - 1
    if bin_n == bin_a:
      try:
        if exact:
          v = C[bin_a]
        else:
          v = C[bin_a] * (b - a)
      # except IndexError:
      except DummyError:
        msg = "Resolution limits may not be valid for spectra."
        raise ResolutionError(msg)
    else:
      if exact:
        v = C[bin_a:bin_n].sum()
      else:
        frac_a = (bin_borders_rho[bin_m] - a)
        frac_b = (b - bin_borders_rho[bin_n])
        if frac_a < 0 or frac_b < 0:
          msg = "Fraction less than 0. a:%s, b:%s" % (frac_a, frac_b)
          raise Exception, msg
        try:
          i_a = frac_a * C[bin_a]
          i_b = frac_b * C[bin_n]
        except IndexError, e:
          tplt = "ROI exceeds resolution limits of %s."
          raise ConfigError(tplt % measurement['image_key'])
        i_mn = C[bin_m:bin_n] * delta_rho_bins[bin_m:bin_n]
        v = i_a + i_mn.sum() + i_b
    return v 
  return _ig

def bin_for_scaling(m, N, res_min, res_max, norm=False):
  experiment = m['experiment']
  _ig = make_integrator(m)
  bbres_bbrhos = get_bbres_bbrhos_from_N(N, res_min, res_max)
  bbres = bbres_bbrhos[:,0,:]
  bbrhos = bbres_bbrhos[:,1,:]
  scaled = numpy.array([_ig(p, q) for (p, q) in bbres])
  if norm:
    scaled = scaled / scaled.sum()
  bbpx = vec_res2radfpx(bbres, experiment)
  bpx = numpy.hstack([bbpx[0], bbpx[1,-1:]])
  bmpx = (bbpx[:,0] + bbpx[:,1]) / 2.0
  if norm:
    m['normalized'] = (res_min, res_max)
    m['scaled to'] = m['image_key']
  # 'average Is' & 'Is over sigma'
  m['scaled'] = scaled
  # 'bins of interest'
  m['scaled bois'] = numpy.arange(N)
  # 'bins'
  m['scaled bins'] = N
  # 'bin borders px' (unfortunate variable name of bpx)
  m['scaled bbpx'] = bpx
  # 'bin middles px'
  m['scaled bmpx'] = bmpx
  # 'roi high'
  m['scaled reshi'] = res_max
  # 'roi low'
  m['scaled reslo'] = res_min
  # more unfortunate variable names
  m['scaled gbbrho'] = bbrhos
  m['scaled gbbres'] = bbres
  bmrho = vec_radpx2rho(m['scaled bmpx'], experiment)
  bm2th = vec_radpx2twothetadegs(m['scaled bmpx'], experiment)
  m['scaled bmrho'] = bmrho
  m['scaled bm2th'] = bm2th
  return m

def format_spectrum(model_name, twothetas, intensities, degrees):
  """
  Returns a yaml string with the following format::

     model : model.pdb
     pattern :
       # [   2-theta, intensity ]
       - [  0.000000, 0.0074386 ]
       - [  0.010179, 0.0075088 ]
       ...

  assuming the value of `model_name` is ``model.pdb``, the value of
  `twothetas` is ``[0.000000, 0.010179, ...]`` (in degrees), and the
  value of `intensities is ``[0.0074386, 0.0075088, ...]``.

  The output for the 2-theta values is in degrees.
  If `twothetas` are in degrees, then `degrees` should be ``True``;
  if `twothetas` are in radians, then `degrees` should be ``False``.
  """
  spectrum = []
  spectrum.append('model : "%s"' % model_name)
  spectrum.append('pattern :')
  spectrum.append('  # [   2-theta, intensity ]')
  if not degrees:
    twothetas = numpy.degrees(twothetas)
  for t, v in zip(twothetas, intensities):
    spectrum.append('  - [ %9.6f, %9.7f ]' % (t, v))
  return "\n".join(spectrum)

def write_spectrum(spectrum_file, model_name,
                                  twothetas,
                                  intensities,
                                  degrees):
  """
  Creates a yaml file of the name in `spectrum_file` with the
  following format::

     model : model.pdb
     pattern :
       # [   2-theta, intensity ]
       - [  0.000000, 0.0074386 ]
       - [  0.010179, 0.0075088 ]
       ...

  assuming the value of `model_name` is ``model.pdb``, the value of
  `twothetas` is ``[0.000000, 0.010179, ...]`` (degrees), and the
  value of `intensities is ``[0.0074386, 0.0075088, ...]``.

  The output for the 2-theta values is in degrees.
  If `twothetas` are in degrees, then `degrees` should be ``True``;
  if `twothetas` are in radians, then `degrees` should be ``False``.
  """
  spectrum = format_spectrum(model_name, twothetas, intensities, degrees)
  with open(spectrum_file, 'w') as specfile:
    specfile.write(spectrum)
    specfile.write("\n")

def bin_equivalent(m):
  """
  The scaled binning is *equivalent* to the integration binning.
  """
  experiment = m['experiment']
  brhos = vec_radpx2rho(m['bin borders px'], experiment)
  bbrhos = numpy.array([brhos[:-1], brhos[1:]]).T
  bmrho = vec_radpx2rho(m['bin middles px'], experiment)
  bm2th = vec_radpx2twothetadegs(m['bin middles px'], experiment)
  # m['scaled'] = m['average Is']
  m['scaled'] = m['sum Is']
  m['scaled bois'] = m['bins of interest']
  m['scaled bins'] = m['bins']
  m['scaled bbpx'] = m['bin borders px']
  m['scaled bmpx'] = m['bin middles px']
  m['scaled bmrho'] = bmrho
  m['scaled bm2th'] = bm2th
  m['scaled reshi'] = m['roi high']
  m['scaled reslo'] = m['roi low']
  m['scaled gbbrho'] = bbrhos
  m['bin middles rho'] = bmrho
  m['bin middles two-theta'] = bm2th
  return m
  
def get_bbres_bbrhos_from_N(N, res_min, res_max):
  """
  Returns the fancy indices for the bin borders in resolution (bbres)
  and in rho (bbrhos), running from res_min to res max, inclusinve.
  """
  rho_min = res2rho(res_min)
  rho_max = res2rho(res_max)
  rho_min, rho_max = signals.lohi((rho_min, rho_max))
  step = (rho_max - rho_min) / N
  first = rho_min
  last = rho_max + (step / N)
  brhos = numpy.arange(first, last, step)
  bbrhos = numpy.array([brhos[:-1], brhos[1:]]).T
  bres = numpy.array([rho2res(rho) for rho in brhos])
  # bin borders in res
  bbres = numpy.array([bres[:-1], bres[1:]]).T
  bbres_bbrhos = numpy.array([bbres, bbrhos]).swapaxes(0, 1)
  return bbres_bbrhos

def get_bois(C, res_min, res_max):
  bpxs = C['bin borders px']
  experiment = C['experiment']
  px_min = res2radfpx(res_min, experiment)
  px_max = res2radfpx(res_max, experiment)
  # this should be the intersection
  bois = numpy.where((px_min < bpxs[1:]) &
                     (px_max > bpxs[:-1]))
  return bois

def get_bbres_bbrhos_from_C(C, res_min, res_max):
  """
  Returns the fancy indexes of the bins that run from
  res_min to res_max, inclusive
  """
  bpxs = C['bin borders px']
  experiment = C['experiment']
  bois = get_bois(C, res_min, res_max)
  bpxois = bpxs[bois]
  bres = vec_radpx2res(bpxois, experiment)
  brhos = vec_radpx2rho(bpxois, experiment)
  bbres = numpy.array([bres[1:], bres[:-1]]).T
  bbrhos = numpy.array([brhos[1:], brhos[:-1]]).T
  bbres_bbrhos = numpy.array([bbres, bbrhos]).swapaxes(0,1)
  return bbres_bbrhos


# def make_rsqer(C, T, params, buoyancy=1):
#   """
#   `buoyancy` is buoyancy of C
#   """
#   # no background estimation
#   # I \alpha \exp \left \{ -2B\varrho^{2} \right \} + \beta
#   bbrhos = T['scaled gbbrho']
#   exp = math.exp
#   def _r(args):
#     if len(args) == 2:
#       alpha, beta = args
#       B = params[3]
#     else:
#       alpha, beta, B = args
#     if alpha <= 0:
#       rs = numpy.zeros(bbrhos.size) + MAXLIK
#     else:
#       def _cor(ii, lo, hi):
#         avrho = (lo + hi) / 2
#         return ii * alpha * exp(-2.0 * B * avrho**2) + beta
#       rs = []
#       iii = izip(bbrhos, C['scaled'], T['scaled'])
#       for (lo_rho, hi_rho), iiC, iiT in iii:
#         cor_iiC = _cor(iiC, lo_rho, hi_rho)
#         if cor_iiC > iiT:
#           dif = buoyancy * (cor_iiC - iiT)
#         else:
#           dif = cor_iiC - iiT
#         rs.append(dif)
#     return numpy.array(rs)
#   return _r


def make_rsqer_bg(C, T, params, buoyancy=1):
  # background estimation
  # \left \{ I_{\varrho} - \left (m\varrho + b \right ) \right \}
  # \alpha \exp \left \{ -2B\varrho^{2} \right \}
  bbrhos = T['scaled gbbrho']
  exp = math.exp
  def _r(args):
    if len(args) == 3:
      alpha, m, b = args
      B = params[3]
    elif len(args) == 4:
      alpha, m, b, B = args
    else:
      msg = "Number of args should be 3 or 4: %s" % args
      raise ParameterError(msg)
    if alpha <= 0:
      rs = numpy.zeros(bbrhos.size) + MAXLIK
    else:
      def _cor(ii, lo, hi):
        avrho = (lo + hi) / 2
        return (alpha * (ii - ((m * avrho) + b)) *
                exp(-2.0 * B * avrho**2))
      rs = []
      iii = izip(bbrhos, C['scaled'], T['scaled'])
      for (lo_rho, hi_rho), iiC, iiT in iii:
        cor_iiC = _cor(iiC, lo_rho, hi_rho)
        if (cor_iiC > iiT) and (buoyancy is not None):
          dif = buoyancy * (cor_iiC - iiT)
        else:
          dif = cor_iiC - iiT
        rs.append(dif)
    return numpy.array(rs)
  return _r

def scale(C, T, N, res_min, res_max, params, buoyancy):
  """
  Spectrum `C` will be scaled to `T`.
  """
  if buoyancy is None:
    if "normalized" not in T:
      bin_for_scaling(T, N, res_min, res_max, norm=True)
    bin_for_scaling(C, N, res_min, res_max)
    buoyancy = 1
  else:
    if "scaled to" not in T:
      bin_equivalent(T)
      T['scaled to'] = T['image_key']
    bin_equivalent(C)
  f_rsqr = make_rsqer_bg(C, T, params, buoyancy)
  if params is None:
    # [alpha, m, b B]
    guesses = [1.0, 0.0, 0.0, 0.0]
    aaa = optimize.leastsq(f_rsqr, guesses)
    if 1 <= aaa[-1] <= 4:
      alpha, m, b, B = aaa[0]
    else:
      msg = "Optimization failed: %s" % aaa
      raise OptimizationError(msg)
  else:
    # [alpha, m, b]
    guesses = [1.0, 0.0, 0.0]
    aaa = optimize.leastsq(f_rsqr, guesses)
    if 1 <= aaa[-1] <= 4:
      alpha, m, b = aaa[0]
      B = params[3]
    else:
      raise Exception, aaa
  exp = math.exp
  # no background estimation
  # def _cor(ii, ab):
  #   avrho = ab.mean()
  #   return ii * alpha * exp(-2.0 * B * avrho**2) + beta
  # background estimation
  def _cor(ii, ab):
    avrho = ab.mean()
    return (alpha * (ii - ((m * avrho) + b)) *
            exp(-2.0 * B * avrho**2))
  avIs = C['scaled']
  sums = C['scaled']
  bbrhos = C['scaled gbbrho']
  # scaled = [_cor(ii, ab) for (ii, ab) in izip(avIs, bbrhos)]
  scaled = [_cor(ii, ab) for (ii, ab) in izip(sums, bbrhos)]
  scaled = numpy.array(scaled)
  rsq = ((scaled - T['scaled'])**2).sum()
  worst = ((T['scaled'].mean() - T['scaled'])**2).sum()
  score = rsq / worst
  C['scaled score'] = score
  C['scaled worst'] = worst
  C['scaled'] = scaled
  C['scaled to'] = T['image_key']
  C['scaled Rsq'] = rsq
  # no background estimation
  # C['scaled params'] = {'alpha':alpha, 'beta':beta, 'B':B}
  # background estimation
  C['scaled params'] = {'alpha':alpha, 'm': m, 'b':b, 'B':B}
  return C

def serialize_array(ary, name="ary"):
  content = io.BytesIO()
  numpy.savez_compressed(content, **{name: ary})
  content.seek(0)
  s = content.read()
  content.close()
  return s

def deserialize_array(s, name="ary"):
  content = io.BytesIO()
  content.write(s)
  content.seek(0)
  d = numpy.load(content)
  ary = d[name]
  content.close()
  return ary

def scale_several(config, Cs, T, general, outlet, table):
  """
  The `equiv` flag means that all have equivalent binning.

  `T` is the reference spectrum, typically experimental.

  This function modifies `Cs`.
  """
  res_min, res_max = config['roi']
  params = config.get('scale_parameters', None)
  save_score = config['save_score']
  buoyancy = config.get('buoyancy')
  N = config['roi_bins']
  if res_min is None:
    res_min = T['roi_low']
  if res_max is None:
    res_min = T['roi_high']
  hline = "=" * 60
  outlet(hline)
  outlet(hline)
  outlet("  Scaled To: %s" % T['image_key'])
  outlet(hline)
  outlet(hline)
  if save_score:
    if len(Cs) > 1:
      showwarning('Scaling multiple patterns.',
                  'More than one pattern to scale.\n\n' +
                  'Scores will not be saved.',
                  parent=outlet.parent)
      config['save_score'] = False
  for C in Cs:
    scaled = scale(C, T, N, res_min, res_max, params, buoyancy)
    outlet("== %s" % C['image_key'])
    outlet("==      score: %09.7f" % C['scaled score'])
    outlet("==      R**2: %09.7f" % C['scaled Rsq'])
    outlet("==      worst: %09.7f" % C['scaled worst'])
    tmplt = "==      alpha: %(alpha)f, m: %(m)f, b: %(b)f, B: %(B)f"
    outlet(tmplt % C['scaled params'])
    outlet(hline)
    if save_score:
      if 'image' not in T:
        msg = "Reference for 'scaled_to' should be an image."
        raise RadialXError(msg)
      model = config['pdb_model']
      image_key = T['image_key']
      score = C['scaled score']
      Rsq = C['scaled Rsq']
      alpha = C['scaled params']['alpha']
      m = C['scaled params']['m']
      b = C['scaled params']['b']
      B = C['scaled params']['B']
      json_image = json.dumps(T['image'])
      if 'spectrum' in C:
        spectrum = C['spectrum']
      else:
        spectrum = None
        tplt = "Scaled spectrum (%s) may not be from a model."
        logging.warning(tplt, C['image_key'])
      if spectrum is not None:
        spectrum = numpy.transpose(spectrum)
        spectrum = serialize_array(spectrum)
        spectrum = base64.b64encode(spectrum)
      if table is not None:
        pdb = base64.b64encode((bz2.compress(open(model).read())))
        json_config = json.dumps(config)
        json_general = json.dumps(general)
        table.insert(model=model, image_key=image_key, score=score,
                     Rsq=Rsq, alpha=alpha, m=m, b=b, Bfac=B,
                     image=json_image, pdb=pdb, spectrum=spectrum,
                     config=json_config, general=json_general)
        table.commit()
  if config['background_correction']:
    if len(Cs) > 1:
      msg = "Background correction can not be used on >1 pattern."
      raise ConfigError(msg)
    else:
      T['bg corrected'] = bg_correct(T, alpha, B)
      C['bg corrected'] = bg_correct(C, alpha, B)
  return Cs

def bg_correct(S, alpha, B):
  """
  Keeping this as a function just for the comments.
  
  Scaling

  .. :math:

    I_{\varrho}^{\circ} =
    \left \{ I_{\varrho} - \left (m\varrho + b \right ) \right \}
    \alpha \exp \left \{ -2B\varrho^{2} \right \}

  Background Correction

  .. :math:

    I_{\varrho}^{\circ}
    \alpha^{-1} \exp \left \{ 2B\varrho^{2} \right \} =
    I_{\varrho} - \left (m\varrho + b \right ) 
  """
  return S['scaled'] * numpy.exp(2 * B * S['scaled bmrho']) / alpha
