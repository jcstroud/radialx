=========
 RadialX 
=========

Introduction
------------

RadialX is a python package for working with x-ray
powder diffraction data and for simulating
x-ray powder diffraction patterns from models.

At present, the only documented functionality is

1. The simulation of powder diffraction patterns
   from PDB files using the utility called ``powderx``.

   If you wish to create simulated powder diffraction
   patterns, please see the documentation at
   http://pythonhosted.org/radialx/.

2. Displaying of diffraction image header information
   using the utility called ``headerx``.

Other, undocumented, functionality is

1. Finding the centers of powder diffraction images
2. Integration of powder diffraction data
3. Scaling of powder diffraction data with other data or
   simulated patterns
4. Finding the difference of two scaled powder diffraction
   patterns

These functions are found in the utility called ``profilex``.

It is hoped that the entire RadialX package
will be fully documented one day.


Home Page & Repository
----------------------

Home Page: https://pypi.python.org/pypi/radialx/

Documentation: http://pythonhosted.org/radialx/

Repository: https://github.com/jcstroud/radialx


Installation
------------
Dependencies
~~~~~~~~~~~~

The installation of RadialX and many other python packages will
be made easier by `pip`_. So, before going any further towards
installation, it is advisable to visit
follow the `pip installation instructions`_, including the
installation of Setuptools, which is essential.


The `CCTBX package`_ is needed to simulate powder diffraction patterns.
Thus, it will be necessary to have the full CCTBX package installed
and the ``cctbx.python`` executable in your path.

Other python dependencies are (in alphabetical order):

  - configobj
  - numpy
  - phyles
  - pil
  - PyDBLite
  - pyfscache
  - pygmyplot
  - pyyaml
  - scipy

The availability of each of these packages will be checked
during the make from the ``setup.py`` script.
Most can be installed by the python package
manager called `pip`_, if not already present on your system.


.. _`pip installation instructions`: http://www.pip-installer.org/en/latest/installing.html
.. _`CCTBX package`: http://cctbx.sourceforge.net/
.. _`pip`: https://pypi.python.org/pypi/pip

Download
~~~~~~~~

The best way to obtain RadialX is to download the source
code from the GitHub repository:

   % git clone https://github.com/jcstroud/radialx.git

This command automatically downloads and unpacks the complete
RadialX repository, which includes all of the code and some test data.


Build
~~~~~

It is advisable to look at the ``Makefile.inc`` file inside
the ``radialx`` directory to ensure that the settings reflect
your build environment. Most notably, ensure that the
``PYTHON`` setting points to the desired python version and
that the ``bin`` directory under the directory specified
by the ``PREFIX`` setting is in your path.

For example, if ``PREFIX`` is set to ``/usr/local``, then
ensure that ``/usr/local/bin`` is in your path.

Build and installation is easy::

   % cd radialx
   % make
   % sudo make install

The ``make`` command will automatically call ``setup.py`` and install
the utilities (``powderx``, ``headerx``, and ``profilex``) into
the appropriate location, specified by the ``PREFIX`` setting.


Example
-------

Example of how to use it and what it can do.
