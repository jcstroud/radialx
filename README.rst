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

Fully Documented Functionalities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At present, the only 100% documented functionalities of RadialX are

1. The simulation of powder diffraction patterns
   from PDB files using the utility called **powderx**.

2. Displaying of diffraction image header information
   using the utility called **headerx**.

Detailed usage for **powderx** and **headerx** are below.

Less Documentd Functionalities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Other, less documented, functionalities correspond to "modes" of the
utility called **profilex**:

- *centering* mode: Finding the centers of powder diffraction images
  in adxv_ binary format.
- *averaging* mode: Radial integration of one or
  more experimental powder diffraction patterns in adxv_ binary format.
- *averaging* mode: Scaling of several powder experimental or simulated
  diffraction patterns to a single experimental or simulated pattern.
- *difference* mode: Calculating the difference of two scaled
  radially integrated experimental powder diffraction patterns.

Although **profilex** is fully functional and heavily tested, its
user interface (i.e. config file format, etc.) is very likely to
change. For example, the modes will probably be split into different
different utilities and several config file parameters will
probably be renamed. The **profilex** utility is fairly well documented by
comments in the config files that populate the ``test/test-profilex``
directory of the source distribution, as explained below.

It is hoped that the entire RadialX package will be fully
and heavily documented soon.

.. _adxv: http://www.scripps.edu/~arvai/adxv.html


Home Page & Repository
----------------------

Home Page: https://pypi.python.org/pypi/radialx/

Documentation: http://pythonhosted.org/radialx/

Repository: https://github.com/jcstroud/radialx
