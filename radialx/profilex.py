#! /usr/bin/env python

import os
import sys
import Tkinter as TK
import logging

from optparse import OptionParser
from tkFileDialog import askopenfilename
from tkMessageBox import askyesno, askyesnocancel

import numpy
import pyfscache

from PyDbLite import SQLite
from phyles import usage, last_made
from pygmyplot import xy_heat
from pygmyplot.pygmyplotlib import MyXYPlot

import _radialx
from _version import __version__

__module_name__ = "profilex"

TEXT = 'TEXT'
INTEGER = 'INTEGER'
REAL = 'REAL'
BLOB = 'BLOB'

class PB(object):
    # Create Progress Bar
    def __init__(self, parent, svar=None, height=5):
        self.parent = parent
        self._svar = svar
        self._g = TK.Frame(parent)
        self._g.pack()
        (w, h, x, y) = self.get_dims()
        self._canvas = TK.Canvas(self._g, width=w,
                                          height=height,
                                          bd=0,
                                          highlightthickness=0,
                                          relief=TK.SUNKEN)
        self._canvas.pack(side=TK.LEFT, expand=TK.YES, fill=TK.BOTH)
        self._tl = self._canvas.winfo_toplevel()

    def get_dims(self):
      (wh, x, y) = self._g.winfo_geometry().split('+')
      (w, h) = wh.split('x')
      return tuple(int(v) for v in  (w, h, x, y))

    def pack(self, *args, **kwargs):
      self._g.pack(*args, **kwargs)

    # Open Progress Bar
    def activate(self, msg='', ratio=0.0):
        self.update(msg, ratio)

    # Close Progress Bar
    def deactivate(self):
        self._svar.set('')
        self._canvas.delete(TK.ALL)

    # Update Progress Bar
    def update(self, msg, ratio):
        (w, h, x, y) = self.get_dims()
        self._canvas.config(width=w, height=h)
        self._canvas.delete(TK.ALL)
        self._canvas.create_rectangle(0, 0, w,
                                            h, fill='white')
        if ratio != 0.0:
          self._canvas.create_rectangle(0, 0, w * ratio,
                                              h, fill='blue')
        if self._svar is not None:
          self._svar.set(msg)
        self._tl.update()

class StdOutlet(object):
  def __init__(self, parent):
    self.parent = parent
  def __call__(self, msg):
    sys.stdout.write(msg + "\n")

def doopts():
  usage = 'usage: profilex configfile'
  parser = OptionParser(usage)
  parser.add_option("-m", "--model", dest="model",
                    help="unique pdb model",
                    metavar="MODEL")
  # parser.add_option("-i", "--inputs", dest="inputs",
  #          default='Inputs',
  #          help="directory where inputs are",
  #          metavar="INPUTS")
  return parser

def header(width=70):
  hline = "=" * width
  print hline
  print (" %s v.%s " % (__module_name__, __version__)).center(width)
  print hline
  print


def profile_main():
  DEFAULT_CACHE = ".radialx/cache"
  header()
  parser = doopts()
  (options, args) = parser.parse_args()

  try:
    config_file = args[0]
  except IndexError:
    usage(parser, 'No config file specified.')

  tk = TK.Tk()
  tk.title('profilex')
  tk.protocol("WM_DELETE_WINDOW", sys.exit)
  plot_f = TK.Frame(tk, relief=TK.RIDGE, border=2)
  plot_f.pack(side=TK.TOP, fill=TK.BOTH, expand=TK.YES)
  qb = TK.Button(tk, fg="#aa0000",
                     text="Quit", cursor="hand2",
                     command=sys.exit)
  qb.pack()
  task_pane = TK.Frame(tk)
  task_pane.pack(expand=TK.NO, fill=TK.X)
  statusvar = TK.StringVar()
  status = TK.Label(task_pane, textvariable=statusvar,
                               justify=TK.LEFT)
  status.pack(expand=TK.NO, fill=TK.BOTH, side=TK.LEFT)
  pb_f = TK.Frame(tk, relief=TK.RIDGE, border=2)
  pb_f.pack(side=TK.TOP, fill=TK.BOTH, expand=TK.NO)
  task_svar = TK.StringVar()
  TK.Label(tk, width=70, textvariable=task_svar).pack(side=TK.TOP)
  task_pb = PB(tk, svar=task_svar, height=8)
  task_pb.pack(side=TK.TOP, expand=TK.NO, fill=TK.BOTH)
  job_svar = TK.StringVar()
  TK.Label(tk, width=70, textvariable=job_svar).pack(side=TK.TOP)
  job_pb = PB(tk, job_svar, height=8)
  job_pb.pack(side=TK.TOP, expand=TK.NO, fill=TK.BOTH)
  # TK.Button(tk, text='Quit', command=tk.destroy).pack(side=TK.TOP)
  tk.update_idletasks()

  cfg = _radialx.load_config(config_file)
  general = cfg['general']
  spectradb = _radialx.load_config(general['spec_db'])

  # interpolate fields for spectra (e.g. %latest%)
  spec_dir = general['spec_dir']
  suffix = ("yml", "yaml", "spec", "json")
  field = "%latest%"
  latest = None
  for key, spec in spectradb.items():
    if spec['filename'] == field:
      if latest is None:
        latest = last_made(dirpath=spec_dir, suffix=suffix)
      spec['filename'] = latest

  imagesdb = _radialx.load_config(general['img_db'])

  if general['cache'] is None:
    cache = DEFAULT_CACHE
  else:
    cache = general['cache']
  cache = pyfscache.FSCache(cache)

  if general['mode'] == "averaging":
    config = cfg['averaging']
    groups_img = _radialx.load_config(general['img_groups'])
    groups_img = groups_img['images']
    groups_spec = _radialx.load_config(general['spec_groups'])
    groups_spec = groups_spec['spectra']
    cfg['groups'] = {'images' : groups_img, 'spectra' : groups_spec}
    spectra = _radialx.get_spectra(cfg, spectradb, "averaging")
    images = _radialx.get_images(cfg, imagesdb, "averaging")
    pltcfg = _radialx.load_config(config['plot_config'])
    plt = MyXYPlot(master=plot_f,
                   figsize=pltcfg['plot_size'],
                   dpi=pltcfg['dpi'])
    sp = _radialx.load_spectra(config, spectra)
    sp_values = sp.values()
    ms = _radialx.averages(config, images,
                                  pltcfg=pltcfg,
                                  plt=plt,
                                  job=job_pb.update,
                                  task=task_pb.update,
                                  cache=cache)
    ms_values = ms.values()
    TCs = sp_values + ms_values
    _radialx.sharpnesses(config, TCs)
    # if ((config['scaled_to'] is not None) and
    #     (len(images) + len(spectra) == 2)):
    if config['scaled_to'] is not None:
      if len(images) + len(spectra) == 2:
        db = SQLite.Database(general['score_db'])
        title = "Model Name"
        table = db.create("scores",
                          # pdb model that produced spectrum
                          ("model", TEXT),
                          # key from image database
                          ("image_key", TEXT),
                          # Ro score for the scaling
                          ("score", REAL),
                          # R**2 statistic
                          ("Rsq", REAL),
                          # fitted alpha scaling parameter
                          ("alpha", REAL),
                          # fitted m scaling parameter
                          ("m", REAL),
                          # fitted b scaling parameter
                          ("b", REAL),
                          # fitted B scaling parameter
                          ("Bfac", REAL),
                          # json of the image entry from the images db
                          ("image", TEXT),
                          # base64 of bz2 of model pdb file
                          ("pdb", TEXT),
                          # base64 of numpy serialized spectrum array
                          # in [ 2-theta, intensity ] pairs
                          ("spectrum", TEXT),
                          # json of the profilex averaging config section
                          ("config", TEXT),
                          # json of the profilex general config section
                          ("general", TEXT),
                          mode="open")

        if options.model is not None:
          if os.path.exists(options.model):
            options.model = None
          elif table(model=options.model):
            options.model = None
        title = "Unique Model Name"
        latest_pdb = last_made(dirpath=general['model_dir'],
                               suffix=".pdb")
        while options.model is None:
          options.model = askopenfilename(
                                    title=title,
                                    parent=tk,
                                    filetypes=[("PDB", "*.pdb")],
                                    initialdir=general['model_dir'],
                                    initialfile=latest_pdb)
          if not options.model:
            msg = "Must proceed with a\nunique model name.\n\nContinue?"
            if askyesno("Continue", msg, parent=tk):
              options.model = None
            else:
              sys.exit()
          else:
            rows = table(model=options.model)
            if rows:
              bn = os.path.basename(options.model)
              tplt = ("Model name '%s'\nis not unique.\n\n" +
                      "Overwrite score?")
              msg = tplt % bn
              overwrite = askyesnocancel("Overwrite Score", msg)
              if overwrite is None:
                options.model = None
              elif overwrite:
                if len(rows) > 1:
                  msg = ("%s Records with model\nnamed '%s'.\n\n" +
                         "Erase all?") % (len(rows), bn)
                  if askyesno("Erase Records", msg, parent=tk):
                    for row in rows:
                      del table[row['__id__']]
                  else:
                    config['save_score'] = False
              else:
                config['save_score'] = False
      else:
        table = None
        options.model = None
      config['pdb_model'] = options.model
      T = None
      Cs = []
      for TC in TCs:
        if TC['image_key'] == config['scaled_to']:
          T = TC
        else:
          Cs.append(TC)
      std_outlet = StdOutlet(tk)
      if T is None:
        tplt = "Unrecognized value for scaled_to: %s"
        msg = tplt % config['scaled_to']
        raise _radialx.ConfigError(msg)
      _radialx.scale_several(config, Cs, T, general, std_outlet, table)
    msc = ms.copy()
    msc.update(sp)
    pmsc = msc.reversed()
    nimgs = _radialx.plot_images(plt, pmsc, config, pltcfg, 0)
    statusvar.set("%s Images Processed and Plotted" % nimgs)
    task_pb.deactivate()
    job_pb.deactivate()
  elif general['mode'] == "difference":
    config = cfg['difference']
    config['images'] = [config['positive'], config['negative']]
    # adapting difference config to averaging config
    # maybe just change the settings file format?
    config['scaled_to'] = config['positive']
    config['roi'] = config['limits']
    config['roi_bins'] = config['bins']
    config['img_bins'] = config['bins']
    config['norm_type'] = None
    config['stretch'] = False
    images = _radialx.get_images(cfg, imagesdb, "difference")
    keys = [k for (k, d) in images]
    pltcfg = _radialx.load_config(config['plot_config'])
    plt = MyXYPlot(master=plot_f,
                   figsize=pltcfg['plot_size'],
                   dpi=pltcfg['dpi'])
    # _radialx.plot_spectra(config, spectra, pltcfg, plt, cache=cache)
    ms = _radialx.averages(config, images,
                                   pltcfg=pltcfg,
                                   plt=plt,
                                   job=job_pb.update,
                                   task=task_pb.update,
                                   cache=cache)
    TCs = ms.values()
    if TCs[0]['image_key'] == config['scaled_to']:
      T = TCs[0]
      C = TCs[1]
    else:
      T = TCs[1]
      C = TCs[0]
    Cs = [C]
    std_outlet = StdOutlet(tk)
    table = None
    _radialx.scale_several(config, Cs, T, general, std_outlet, table)
    msc = ms.copy()
    nimgs = _radialx.plot_images(plt, msc, config, pltcfg, 0)
    statusvar.set("%s Images Processed and Plotted" % nimgs)
    twoths = T['bin middles two-theta']
    avis = T['scaled'] - C['scaled']
    if avis.min() < 0:
      avis = avis - avis.min()
    spectrum_file = config['output'] + ".yml"
    _radialx.write_spectrum(spectrum_file,
                            config['output'], twoths, avis, False)
    task_pb.deactivate()
    job_pb.deactivate()
  elif general['mode'] == "recentering":
    config = cfg['recentering']
    image = imagesdb[config['recenter_image']]
    plt = MyXYPlot(master=plot_f, figsize=config['plot_size'])

    # result = {"coords" : coords,
    #           "vals" : vals,
    #           "exes" : exes,
    #           "whys" : whys,
    #           "max_xy" : max_xy}

    result = _radialx.recenter(general, config, image,
                                                plt=plt,
                                                job=job_pb.update,
                                                task=task_pb.update,
                                                cache=cache)
    # flip x and y for image space
    # title = "%s\n%s to %s %s (%s px grid)" % params
    title = "%s : %s" % (image['filename'], result['max_xy'])
    factor = (config['bin_factor'], config['bin_factor'])
    # flip exes and whys for image space
    xy_heat(result['coords'], result['vals'],
                              heat_up=1.0/3.0,
                              xlabs=result['exes'],
                              ylabs=result['whys'],
                              title=title)
  else:
    job_pb.update("The mode for the run is not recognized.", 1)

  tk.mainloop()

if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  profile_main()
