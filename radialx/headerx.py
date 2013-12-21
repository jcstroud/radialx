#! /usr/bin/env python

import sys
import os
import math

import _radialx

def headerx_usage():
  print "usage: headerx adxvfile"
  sys.exit(0)

def sanity_test(header, res):
  size1 = header['SIZE1']
  size2 = header['SIZE2']
  if 'BEAM_CENTER_X' not in header:
    header['BEAM_CENTER_X'] = header['PIXEL_SIZE'] * (size1 / 2.0)
  if 'BEAM_CENTER_Y' not in header:
    header['BEAM_CENTER_Y'] = header['PIXEL_SIZE'] * (size2 / 2.0)

  offset_x = header['BEAM_CENTER_X'] / header['PIXEL_SIZE']
  offset_y = header['BEAM_CENTER_Y'] / header['PIXEL_SIZE']
  x_center = offset_x
  y_center = size2 - offset_y
  x_px = _radialx.res2radpx(res, header) + x_center
  y_px = _radialx.res2radpx(res, header) + y_center
  x_px = int(math.floor(x_px))
  y_px = int(math.floor(y_px))
  x_cntr = int(math.floor(x_center))
  y_cntr = int(math.floor(y_center))
  print "%30s  ===============" % "==============="
  print "%30s " % "Sanity Test"
  print "%25s Angs: %s,%s px" % (res, x_px, y_cntr)
  print "%25s Angs: %s,%s px" % (res, x_cntr, y_px)
  print "%30s  ===============" % "==============="


def header_main():
  if len(sys.argv) not in (2, 4):
    headerx_usage()
  else:
    adxv_filename = sys.argv[1]
  try:
    bc_x, bc_y = sys.argv[2:4]
    bc_x = float(bc_x)
    bc_y = float(bc_y)
  except (IndexError, ValueError):
    bc_x, bc_y = None, None

  try:
    adxvfile = open(adxv_filename, 'rb')
  except IOError:
    print "Could not open file '%s'." % adxv_filename
    headerx_usage()
  
  header = _radialx.read_header(adxvfile)

  print
  print "%30s  ===============" % "==============="
  for itm in header.items():
    print "%30s: %s" % itm
  if bc_x is not None:
    header['BEAM_CENTER_X'] = bc_x
    header['BEAM_CENTER_Y'] = bc_y
  sanity_test(header, 4.7)
  print
