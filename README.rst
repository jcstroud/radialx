=========
 RadialX 
=========

Introduction
------------

RadialX is a python package for working with x-ray
powder diffraction data and for simulating
x-ray powder diffraction patterns from models.

The most complete documentation is available at
http://pythonhosted.org/radialx/.

Documented Functionalities
~~~~~~~~~~~~~~~~~~~~~~~~~~

At present, the only documented functionalities of RadialX are

1. The simulation of powder diffraction patterns
   from PDB files using the utility called **powderx**.

   If you wish to create simulated powder diffraction
   patterns, please see the documentation at
   http://pythonhosted.org/radialx/.

2. Displaying of diffraction image header information
   using the utility called **headerx**.

Undocumented Functionalities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Other, undocumented, functionalities correspond to "modes" of the
utility called **profilex**:

- *centering*: Finding the centers of powder diffraction images
  in adxv_ binary format.
- *averaging*: Radial integration (not averaging!) of one or
  more experimental powder diffraction patterns in adxv_ binary format.
- *averaging*: Scaling of several powder experimental or simulated
  diffraction patterns to a single experimental or simulated pattern.
- *difference*: Calculating the difference of two scaled
  radially integrated experimental powder diffraction patterns.

Although **profilex** is fully functional and heavily tested, its
user interface (i.e. config file format, etc.) is very likely to
change. For example, the modes will probably be split into different
different utilities.

It is hoped that the entire RadialX package
will be fully documented soon.

.. _adxv: http://www.scripps.edu/~arvai/adxv.html


Home Page & Repository
----------------------

Home Page: https://pypi.python.org/pypi/radialx/

Documentation: http://pythonhosted.org/radialx/

Repository: https://github.com/jcstroud/radialx
