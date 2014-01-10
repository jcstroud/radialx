#! /usr/bin/env python

DEBUG = False
 
import sys
import math
import logging

import Tkinter as TK

if DEBUG:
  logging.basicConfig(level=logging.DEBUG)

logging.debug('importing pygmyplot')
import pygmyplot
logging.debug('importing yaml')
import yaml
logging.debug('importing numpy')
import numpy
logging.debug('importing phyles')
import phyles

logging.debug('importing _radialx')
import _radialx

logging.debug('importing cctbx')
import cctbx

logging.debug('importing iotbx')
from iotbx import mtz, pdb

logging.debug('importing flex')
from scitbx.array_family import flex

from _version import __version__

class PowderError(Exception):
  pass
class GUIError(PowderError):
  pass
class VersionError(PowderError):
  pass

class Binner(object):
  def __init__(self, miller, borders, experiment):
    """
    The `borders` should be in :math:`\rho`
    (:math:`\dfrac{\sin \theta}{\lambda}`).
    The `miller` should be a :class:`cctbx.miller.array`.
    The `experiment` should be a mapping with minimal key-value pairs:
      - I{WAVELENGTH} : wavelength in Angstroms
      - I{DISTANCE} : distance to detector in mm
    """
    borders = numpy.array(borders)
    borders.sort()
    self.borders = borders
    self.miller = miller
    self.experiment = experiment

    borders_res = vec_rho2res(borders)
    bin_borders_rho = zip(borders[:-1], borders[1:])
    bin_borders_rho = numpy.array(bin_borders_rho)
    bin_borders_res = zip(borders_res[:-1], borders_res[1:])
    bin_borders_res = numpy.array(bin_borders_res)
    self.bin_borders_rho = bin_borders_rho
    self.bin_borders_res = bin_borders_res
    self.n_bins_used = len(bin_borders_rho)
    d_spc = miller.d_spacings().data()
    selections = []
    for (d_max, d_min) in bin_borders_res:
       sln = (d_spc <= d_max) & (d_spc >= d_min)
       selections.append(sln)
    self.selections = selections
    counts = [sel.as_numpy_array().sum() for sel in self.selections]
    self.counts = numpy.array(counts)
    self.centers = (borders[:-1] + borders[1:]) / 2
    self.centers_res = vec_rho2res(self.centers)
    self.centers_2theta = vec_rho2twotheta(self.centers, experiment)
  def bin_selection(self, i):
    """
    Return the selection for bin `i`.
    """
    return self.selections[i]
  def bin_d_range(self, i):
    """
    Return the max, min d_spacings for bin `i`.
    """
    return self.bin_borders_res[i]
  def bin_rho_range(self, i):
    """
    Return the min and max :math:`\rho` for bin `i`.
    """
    return self.bin_borders_rho[i]
  def bin(self, i):
    """
    Returns bin `i` selected from `self.miller`
    """
    return self.miller.select(self.selections[i])
  def bin_data(self, i):
    """
    Returns the data of bin `i` selected from `self.miller`.
    """
    return self.miller.select(self.selections[i]).data()
  def data(self):
    """
    Returns th data of each bin selected from `self.miller` as a
    :class:`numpy.ndarray`.
    """
    return [self.bin_data(i) for i in self.range_used()]
  def count(self, i):
    """
    Return the number of reflections in bin `i`.
    """
    return self.counts[i]
  def range_used(self):
    """
    Return an `xrange` generator for the bins used in `self`.
    """
    return xrange(self.n_bins_used)
  def binned(self, miller):
    """
    Given a :class:`cctbx.miller.array`, `miller`, return the
    :class:`Binner` for it.
    """
    return self.__class__(self.borders, miller)
  def bin_area(self, i):
    """
    Return the area on the detector for bin `i`.
    """
    limits = self.bin_d_range(i)
    return bin_area(limits, self.experiment)
  def mean_bin_data(self, i):
    """
    Return the mean of the data of bin `i`.
    """
    return flex.mean(self.bin_data(i))
  def mean_data(self):
    """
    Return the mean of the data of each bin.
    """
    return [self.mean_bin_data(i) for i in self.range_used()]
  def area_normalized_summed_bin_data(self, i):
    """
    Return the sum of the data of bin `i` divided by the bin area.
    """
    tot = flex.sum(self.bin_data(i))
    return tot / self.bin_area(i)
  def area_normalized_summed_data(self):
    """
    Returns the sum of the data of each bin divided by the area of the bin
    as a :class:`numpy.ndarray`.
    """
    a = [self.area_normalized_summed_bin_data(i) for i in self.range_used()]
    return numpy.array(a)
  def summed_data(self):
    """
    Returns the sum of the data for each bin as a :class:`numpy.ndarray`.
    """
    return numpy.array([flex.sum(d) for d in self.data()])
  def bin_center(self, i):
    """
    Return the center of bin `i` in :math:`\rho`.
    """
    return self.centers[i]
  def bin_d_center(self, i):
    """
    Return the center of bin `i` in d-spacing.
    """
    return self.centers_res[i]
  def bin_2theta_center(self, i):
    """
    Return the center of bin `i` in 2*theta.
    """
    return self.centers_2theta[i]


def usage():
  print "usage: powderx config.yml"
  sys.exit()
  
def plot_powder(sf, config, pltcfg, experiment, master=None,
                                                plot=None):
  """
  Returns an instance of :class:`pygmyplot.MyXYPlot` with intensity per
  detector area plotted over ``config['d_max']`` - ``config['d_min']``.

  Args:
    `sf`:
       structure factors as a :class:`cctbx.miller.array`
    `pltcfg`:
       mapping object with keys of
         - ``plot_points``: number of bins for the plot
         - ``n_ticks``: number of ticks for the x-axis
    `config`:
       mapping object with keys of
         - ``d_max``: maximum d-spacing in Angstroms
         - ``d_min``: minimum d-spacing in Angstroms
    `experiment`:
       mapping object with keys of
         - ``'WAVELENGTH'``: x-ray wavelength Angstroms
         - ``'DISTANCE'``: detector distance in mm
    `master`:
       Tkinter widget
    `plot`:
       a :class:`pygmyplot.MyPlot` to which the plot is added
  """
  if (master is not None) and (plot is not None):
    msg = "Either or both of master and plot should be None."
    raise GUIError(msg)

  max_min = sf.d_max_min()

  d_max = config['d_max']
  d_min = config['d_min']
  plot_bins = pltcfg['plot_points']
  n_ticks = pltcfg['x_ticks']

  I = sf.intensities()
  borders = _radialx.make_borders_rho(plot_bins, d_max, d_min, experiment)
  binner = Binner(I, borders, experiment)

  values = getattr(binner, config['values'])()
  x_labels = [("%4.1f" % c) for c in binner.centers_res]

  tick_step = int(math.floor(plot_bins / n_ticks))

  if plot is None:
    plot = pygmyplot.xy_plot(binner.centers, values, master=master)
  else:
    plot.plot(binner.centers, values)

  plot.axes.set_xticks(binner.centers[::tick_step])
  plot.axes.set_xticklabels(x_labels[::tick_step])
  plot.axes.set_xlabel('Resolution ($\\AA$)')
  plot.axes.set_ylabel('intensity')


  adj = {"left": pltcfg['left'],
         "right": pltcfg['right'],
         "top": pltcfg['top'],
         "bottom": pltcfg['bottom']}

  _radialx.plot_adjust(plot, adj)

  plot.canvas.draw()

  return plot
 
def powder_main():
  if len(sys.argv) != 2:
    usage()
  else:
    y = yaml.load(open(sys.argv[1]).read())

  uicfg = y['general']
  config = y['simulation']
  experiment = y['experiment']
  pltcfg = y['plot']

  powderx_version = __version__.rsplit(".", 1)[0]
  if uicfg['powderx_version'] != str(powderx_version):
    tplt = "Config file should be version '%s'."
    raise VersionError(tplt % powderx_version)

  log_level = getattr(logging, uicfg['verbosity'])
  logger = phyles.basic_logger(__name__, level=log_level)

  tk = TK.Tk()

  # logger.info("Reading mtz: '%s'", config['mtz_name'])
  # mtz_file = mtz.object(config['mtz_name'])

  # logger.info("Constructing Miller dict from mtz.")
  # mtz_miller_dict = mtz_file.as_miller_arrays_dict()
  # mtz_key = tuple(config['mtz_column'])
  # sf_mtz = mtz_miller_dict[mtz_key]

  logger.info("Reading pdb: '%s'", config['pdb_name'])
  pdb_inp = pdb.input(file_name=config['pdb_name'])

  logger.info("Creating structure factors from pdb.")
  structure = pdb_inp.xray_structure_simple()
  sf_pdb = structure.structure_factors(d_min=config['d_min']).f_calc()

  ec_x = config.get("extinction_correction_x", None)

  if ec_x is not None:
    # sf_mtz = sf_mtz.apply_shelxl_extinction_correction(ec_x,
    #                                       experiment['WAVELENGTH'])
    sf_pdb = sf_pdb.apply_shelxl_extinction_correction(ec_x,
                                          experiment['WAVELENGTH'])

  # logger.info("Plotting powder from mtz: '%s'", config['mtz_name'])
  # p = plot_powder(sf_mtz, config, pltcfg, experiment, master=tk)

  # logger.info("Plotting powder from pdb: '%s'", config['pdb_name'])
  # p = plot_powder(sf_pdb, config, pltcfg, experiment, master=tk)

  logger.info("Creating integrated intensity pattern from pdb.")
  pattern = integrated_intensity(sf_pdb,
                       experiment['WAVELENGTH'],
                       config['d_max'],
                       config['d_min'],
                       v=config['v'],
                       w=config['w'],
                       B=config['B'],
                       apply_Lp = config['apply_Lp'],
                       pattern_bins=config['pattern_shells'],
                       peakwidths=config['peak_widths'],
                       bin_reflections=config['combine_reflections'])

  p = pygmyplot.xy_plot(numpy.degrees(pattern[0]),
                        pattern[1], master=tk)
  p.axes.set_xlabel("$2\\theta$ (degrees)")
  p.axes.set_ylabel("Integrated Intensity")

  adj = {"left": pltcfg['left'],
         "right": pltcfg['right'],
         "top": pltcfg['top'],
         "bottom": pltcfg['bottom'],
         "wspace": None,
         "hspace": None}

  _radialx.plot_adjust(p, adj)

  tk.title(pltcfg['window_name'])

  p.refresh()

  _radialx.write_spectrum(config['pattern_name'],
                          config['pdb_name'],
                          pattern[0], pattern[1], False)

  TK.Button(tk, text="Quit", command=tk.destroy).pack()
  tk.mainloop()


# Compuers Educ. Vol. 13, No. 2. pp. 101-108, 1989
# special k --> 2 ((ln 2)/pi)^(1/2)
# SPECIAL_K --> $K_{s} = \dfrac{2 \sqrt{\ln 2}}{\sqrt{\pi}}$
SPECIAL_K = 2 * math.sqrt(math.log(2) / math.pi)
# k squared --> 4 ln 2
# K_SQ --> $K_{s}^{2} = 4 \ln 2$
K_SQ = 4 * math.log(2)
# half sigma k --> 1 / 4 sqrt(2 ln(2))
# HALF_SIGMA_K --> $K_{\sigma/2} = \dfrac{1}{4 \sqrt{2 \ln 2}}$
# HALF_SIGMA_K = 1 / (4 * math.sqrt(2 * math.log(2)))

def integrated_intensity(miller, wavelength, d_max, d_min,
                                 v=0.08, w=0.02, B=0, pattern_bins=250,
                                 peakwidths=3.0, bin_reflections=False,
                                 apply_Lp=False):
  """
  Returns a 2xN array, where the first row is 2-thetas and the second row
  is intensities.

  See ``Diffraction Basics, Part 2'' by James R. Connolly, 2012.

  - `wavelength`: wavelength in Angstroms
  - `d_max`: maximum d-spacing in Angstroms
  - `d_min`: minimum d-spacing in Angstroms
  - `v`: :math:`\\text{FWHM} = v + w\\tan\\theta` in degrees
  - `w`: :math:`\\text{FWHM} = v + w\\tan\\theta` in degrees
  - `N`: number of points in simulated spectrum
  - `peakwidths`: number of peak width-at-half-height over which to
    integrate a reflection
  """
  logger = logging.getLogger(__name__)
  miller = miller.resolution_filter(d_max=d_max, d_min=d_min)
  v = math.radians(v)
  w = math.radians(w)

  vol = miller.unit_cell().volume()

  logger.info('Getting multiplicities.')
  multis = miller.multiplicities()
  # $I = F \times F^{*}$
  logger.info('Getting intensities.')
  intensities = miller.intensities()
  # M corresponds to a Miller index (h,k,l)
  logger.info('Getting two_theta.')
  twothetas_M = miller.two_theta(wavelength)

  m_times_i = multis.data().as_double() * intensities.data()

  if bin_reflections:
    logger.info('Extracting arrays.')
    m_times_i = cctbx.miller.array(miller.set(), data=m_times_i)
    mbinner = m_times_i.setup_binner(n_bins=pattern_bins)
    logger.info('Creating selections.')
    msels = [mbinner.selection(i) for i in mbinner.range_used()]
    logger.info('Selecting M x F^2.')
    m_times_i = [m_times_i.select(i) for i in msels]
    logger.info('Summing M x F^2.')
    m_times_i = [m.sum() for m in m_times_i]
    m_times_i = numpy.array(m_times_i)
    logger.info('Selecting two-thetas.')
    twothetas_M = [twothetas_M.select(sel) for sel in msels]
    logger.info('Finding means of 2-thetas.')
    twothetas_M = [m.mean() for m in twothetas_M]
    twothetas_M = numpy.array(twothetas_M)
    size = m_times_i.size
    if B == 0:
      exp_neg_2B_stol_sq = None
    else:
      # B-factor: not necessary, can be applied during scaling
      # $\dfrac{\sin^{2}\theta}{\lambda^{2}}$
      logger.info('Selecting sin thetas over lambdas squared')
      stol_sq = miller.sin_theta_over_lambda_sq()
      stol_sq = [stol_sq.select(sel) for sel in msels]
      logger.info('Finding means of sin thetas over lambdas squared.')
      stol_sq = [m.mean() for m in stol_sq]
      stol_sq = numpy.array(stol_sq)
      print stol_sq.shape
      # $\exp\left(\dfrac{-2B\sin^{2}\theta}{\lambda^{2}}\right)$
      exp_neg_2B_stol_sq = numpy.exp(-2 * B * stol_sq)
  else:
    logger.info('Extracting arrays.')
    m_times_i = m_times_i.as_numpy_array()
    twothetas_M = twothetas_M.data().as_numpy_array()
    size = miller.size()
    if B == 0:
      exp_neg_2B_stol_sq = None
    else:
      # $\dfrac{\sin^{2}\theta}{\lambda^{2}}$
      stol_sq = miller.sin_theta_over_lambda_sq().data().as_numpy_array()
      # $\exp\left(\dfrac{-2B\sin^{2}\theta}{\lambda^{2}}\right)$
      exp_neg_2B_stol_sq = numpy.exp(-2 * B * stol_sq)

  thetas_M = twothetas_M / 2.0

  if apply_Lp:
    # Lorentz & polarization correction
    # See p. 192 Equation 2.71 of Pecharsky, VK
    # Fundamentals of powder diffraction  ISBN 0-387-24147-7
    # $\dfrac{1 + \cos^{2} 2\theta}{\sin^{2}\theta \cos\theta}$
    # note that $\cos^{2} \left ( 2\theta_{M} \right )$ is a constant
    LPs = ((1 + numpy.cos(twothetas_M)**2) / 
           ((numpy.sin(thetas_M)**2) * numpy.cos(thetas_M)))

    # as K in Connolly Diffraction Basics, Part 2
    """
    K_{hkl} = \dfrac{M_{hkl}}{V^{2}}
            \left | F_{hkl} \right |^{2}
            \left ( \dfrac{1 + \cos^{2}(2\theta)}
                          {\sin^{2}\theta cos\theta} \right )
    """
    Is = (LPs * m_times_i) / (vol**2)
  else:
    Is = m_times_i / (vol**2)

  print size

  if B != 0:
    # apply a B-factor
    Is = Is * exp_neg_2B_stol_sq

  # linear in $2\theta$
  twotheta_max = twothetas_M.max()
  twotheta_min = twothetas_M.min()
  # number of radians across entire region to be simulated
  twotheta_interval = twotheta_max - twotheta_min
  twothetas = numpy.linspace(twotheta_min,
                             twotheta_max,
                             pattern_bins)
  thetas = twothetas / 2.0

  # $w + (v \tan \theta)$
  HBs_sq = w + (v * numpy.tan(thetas_M))

  # full width of the peak at half-max
  HBs = numpy.sqrt(HBs_sq)
  # pygmyplot.scatter(twothetas_M, HBs)

  n_per_radian = pattern_bins / twotheta_interval

  integrated = numpy.zeros(pattern_bins, dtype=float)

  centers = (pattern_bins * 
             (twothetas_M - twotheta_min) / twotheta_interval)
  centers = numpy.round(centers).astype(int)

  delta_twothetas = numpy.empty_like(integrated)
  delta_twothetas_sq = numpy.empty_like(integrated)
  profile_f = numpy.empty_like(integrated)
  profile = numpy.empty_like(integrated)

  # sum the Gaussian profile functions
   #  Compuers Educ. Vol. 13, No. 2. pp. 101-108, 1989
  """
  \dfrac{2 \sqrt{\ln 2}}{H_{\mathrm{B}}\sqrt{\pi}}
  \exp\left[\left(\dfrac{-4\ln 2}{H_{\mathrm{B}}^2}\right)
          \left(2\theta - 2\theta_{\mathrm{B}}\right)^{2}\right]
  """
  for mdx in xrange(size):

    if mdx % 10000 == 1:
      logger.info('Completed %s of %s reflections.', mdx, size)

    # $K_{s}/H_{B} = \dfrac{2 \sqrt{\ln 2}}{H_{B} \sqrt{\pi}}$
    special_k_over_hb = SPECIAL_K / HBs[mdx]
    # $\dfrac{4 \ln 2}{H_{B}^{2}}$
    neg_k_sq_over_hb_sq = -K_SQ / HBs_sq[mdx]

    # half_sigma = HALF_SIGMA_K * HBs[mdx]
    # misnomer!
    # half_HB: half width of the peak at half-max in radians
    # $H_{B} / 2$
    half_HB = HBs[mdx] / 2
    # number of points in the half_peak
    half_peak = peakwidths * half_HB * n_per_radian
    center = centers[mdx]

    idx = max((0, center - int(math.floor(half_peak))))
    jdx = min((pattern_bins, center + int(math.ceil(half_peak))))

    # idx = max((0, center - 2))
    # jdx = min((N, center + 2))

    assert (jdx - idx) > 0

    # Lorenzian (Sato & Machii Computers & Chemistry Vol. 6 No 1. pp. 33-37, 1982.)
    """
    A_{J}^{L} = \dfrac{2}{\pi H_{B}}
    \left [1 + \dfrac{4}{H_{B}^{2}}
               \left ( 2\theta_{i} - 2\theta_{hk\ell} \right )^2
    \right ]^{-1}
    """
    # $\left(2\theta - 2\theta_{B}\right)$
    delta_twothetas[idx:jdx] = twothetas[idx:jdx] - twothetas_M[mdx]
    # $\left(2\theta - 2\theta_{B}\right)^{2}$
    delta_twothetas_sq[idx:jdx] = (delta_twothetas[idx:jdx] *
                                             delta_twothetas[idx:jdx])

    # Lorentzian $\left(2\theta - 2\theta_{B}\right)^{2}$
    profile_f[idx:jdx] = delta_twothetas_sq[idx:jdx]
    profile_f[idx:jdx] = profile_f[idx:jdx] * (4 / HBs_sq[mdx])
    profile_f[idx:jdx] = profile_f[idx:jdx] + 1
    # to -1 power
    """
    \left [1 + \dfrac{4}{H_{B}^{2}}
               \left ( 2\theta_{i} - 2\theta_{hk\ell} \right )^2
    \right ]^{-1}
    """
    profile_f[idx:jdx] = 1 / profile_f[idx:jdx]
    # $\dfrac{2}{\pi H_{B}}$
    profile_f[idx:jdx] = profile_f[idx:jdx] / (half_HB * math.pi)

    # profile_f = (1 / math.pi) * (half_HB /
    #                              ((delta_twothetas_sq) +
    #                               (HBs_sq[mdx] / 4)))

    # profile_f_exponents = neg_k_sq_over_hb_sq * delta_twothetas_sq
    # profile_f_gaussians = numpy.exp(profile_f_exponents)
    # profile_f = special_k_over_hb * profile_f_gaussians

    profile[idx:jdx] = profile_f[idx:jdx] * Is[mdx]
    integrated[idx:jdx] += profile[idx:jdx]

  return numpy.array([twothetas, integrated])
