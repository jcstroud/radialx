.. Created by phyles-quickstart.
   Add some items to the toctree.

radialx Documentation
=====================

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

Detailed usage for powderx_ and headerx_ are below.

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


Installation
------------

Dependencies
~~~~~~~~~~~~

pip & setuptools
++++++++++++++++

The installation of RadialX and many other python packages will
be made easier by `pip`_. So, before going any further towards
installation, it is advisable to follow the
`pip installation instructions`_, including the
installation of setuptools described therein.
Setuptools is essential to pip.

CCTBX
+++++

At this point the `CCTBX package`_ is *only* needed to
simulate powder diffraction patterns with the **powderx** utility.

For **powderx**, it will be necessary to have the full CCTBX package installed
and the ``cctbx.python`` executable in your path. `Downloads are available`_
for numerous operating systems, including Mac OS X, Windows 7 & XP, and
several flavors of Linux. Additionally, it is possible to build
CCTBX from a source bundle or, for the more ambitious,
the SVN repository at sourceforge.

Because of the unusual python interpreter behavior forced by
the cctbx.python executable,
it is necessary to have all dependencies (except CCTBX itself)
installed both to the ``cctbx.python`` interpreter and to a system python
interpreter (e.g. at ``/usr/local/bin/python``).

The difficulty here might be in using pip with CCTBX if
you are using one of the pre-built CCTBX distributions
called "cctbx+Python" or "cctbx plus",
wherein the CCTBX distribution python
will be different from your system python.

One way to overcome this difficulty is simply to install all of the
packages twice, once to the CCTBX python
and once to the system python.

Installing to the system python interpreter is easy with pip.
For example, to install the pyfscache_ package::

  % sudo pip install pyfscache

Note that "``%``" is the prompt and is not actually typed.

For the CCTBX python, things are slightly more complicated. First, when
following the `pip installation instructions`_, use ``cctbx.python``
with ``ez_setup.py`` and ``get-pip.py``. For example::

  % sudo cctbx.python ez_setup.py
  % sudo cctbx.python get-pip.py

Once this latter command completes, you'll see among the final lines of output
something similar to::

  Installing pip script to /opt/cctbx/Python.framework/Versions/2.7/bin

The directory path in this output points to the location of CCTBX's pip,
which can be used directly. Using pyfscache_ as an example::

  % sudo /opt/cctbx/Python.framework/Versions/2.7/bin/pip install pyfscache

.. _`pyfscache`: https://pypi.python.org/pypi/pyfscache/


Other Dependencies
++++++++++++++++++

Other python dependencies are (in alphabetical order):

  - configobj_
  - numpy_
  - phyles_
  - PIL_
  - PyDBLite_
  - pyfscache_
  - pygmyplot_
  - pyyaml_
  - scipy_

If not already present on your system, most (if not all) of
these dependencies can be installed by the python package
manager called `pip`_.
The availability of each of these packages will be checked
during the `build`_ of RadialX by the ``setup.py`` script.

.. _`pip installation instructions`: http://www.pip-installer.org/en/latest/installing.html
.. _`CCTBX package`: http://cctbx.sourceforge.net/
.. _`Downloads are available`: http://cci.lbl.gov/cctbx_build/
.. _`pip`: https://pypi.python.org/pypi/pip
.. _`configobj`: https://pypi.python.org/pypi/configobj/
.. _`numpy`: http://www.numpy.org/
.. _`phyles`: https://pypi.python.org/pypi/phyles/
.. _`PIL`: http://www.pythonware.com/products/pil/
.. _`PyDBLite`: http://www.pydblite.net/
.. _`pygmyplot`: https://pypi.python.org/pypi/pygmyplot
.. _`pyyaml`: http://pyyaml.org/
.. _`scipy`: http://scipy.org/

Download
~~~~~~~~

Because of the CCTBX dependency, it is not yet recommended
to install RadialX by using `pip`_.

For now, the best way to obtain RadialX is to download the source
code from the GitHub repository::

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

These settings only affect how RadialX is built and where it is installed,
not how it will execute once installed.

Once downloaded, build and installation are easy::

   % cd radialx
   % make
   % sudo make install

The ``make`` command will automatically call ``setup.py`` and install
the utilities (``powderx``, ``headerx``, and ``profilex``) into
the appropriate location, specified by the ``PREFIX`` setting.


Usage
-----

Complete examples of how to use all of the RadialX utilites are
currently in the ``test`` directory of the `source distribution`_.
These examples are documented by comments in the config files called
``powder.yml`` and ``profile.cfg``, the latter serving presently
as the only source of documentation for the **profilex** utility.

Detailed instructions for the headerx_ and powderx_ utilities follow.

headerx
~~~~~~~

The **headerx** utility is the most straightforward to use. First,
convert an image file from the synchrotron or a home-source detector
to an adxv_ binary file. This function is found under the
``File`` → ``Save..`` menu of adxv_. Ensure that the checkboxes
for "Image" and "Binary" are checked in the *Adxv Save*
window. I prefer to name these adxv binary files with the "``.bin``"
extension.

Using a filed called ``stsaa_119a_0_003.bin`` as an example::

  % headerx stsaa_119a_0_003.bin

This file is in the ``test/testdata`` directory and yields the following
output::

               ===============  ===============
                  HEADER_BYTES: 1024
                           DIM: 2
                    BYTE_ORDER: little_endian
                          TYPE: unsigned_short
                         SIZE1: 3072
                         SIZE2: 3072
                    PIXEL_SIZE: 0.10259
                           BIN: 2x2
                           ADC: slow
                   DETECTOR_SN: 911
                      BEAMLINE: 24_ID_C
                          DATE: Mon Jun  8 02:28:42 2009
                          TIME: 10.0
                      DISTANCE: 400.0
                     OSC_RANGE: 1.0
                           PHI: 47.0
                     OSC_START: 47.0
                      TWOTHETA: 0.0
                          AXIS: phi
                    WAVELENGTH: 0.9793
                 BEAM_CENTER_X: 157.11
                 BEAM_CENTER_Y: 156.05
                  TRANSMISSION: 10.0871
                          PUCK: C
                        SAMPLE: 2
                      RING_CUR: 102.2
                     RING_MODE: 0+24x1, ~1.3% Coupling
                  MD2_APERTURE: 30
                      UNIF_PED: 1500
          CCD_IMAGE_SATURATION: 65535
               ===============  ===============
                   Sanity Test 
                      4.7 Angs: 2357,1550 px
                      4.7 Angs: 1531,2376 px
               ===============  ===============

The "Sanity Test" is based on the header beam center. Hovering
the mouse over the given pixels in adxv should produce approximately
the given resolutions (depending on what adxv thinks is the beam center).

powderx
~~~~~~~

The **powderx** utility simulates powder diffraction patterns
from PDB files. These patterns are presented graphically and
also written to a file name designated by the user, as described
below.

The powderx Config File
+++++++++++++++++++++++

A yaml_ formatted config file controls the behavior of **powderx**.
This config file is specified as an argument on the command line::

  powderx powder.yml

An example config file named ``powder.yml`` is in the ``test/test-powder``
directory of the source distribution. The powder config file will be
referred to as "``powder.yml``" herein. The provided example file
has comments that briefly describe each parameter. It is suggested just
to copy and modify the example ``powder.yml`` file from the
``test/test-powder`` directory of the source distribution because
its format may change slightly between versions of RadialX.

An introduction to the `yaml config format`_ is given below and provides
everything users need to know about yaml to write a config file
for **powderx**. For the curious, the full yaml specification (version 1.2) can
be found at http://www.yaml.org/spec/1.2/spec.html.

The ``powder.yml`` file has three sections:

- ``general``: parameters that effect the user experience
- ``simulation``: parameters for the powder diffraction simulation
- ``plot``: parameters that modify the appearance of the plot
- ``experiment``: parameters of the simulated diffraction experiment

A detailed discussion of each section follows.

general
#######

Parameters in the ``general`` section effect the user experience to
a limited extent.

- ``powderx_version``: version number of the **powderx** program;
  it is critical for the config file version to match the version
  of the **powderx** program
- ``verbosity``: controls how verbose the output is;
  values may be ``DEBUG`` (most verbose), ``INFO``, ``WARNING``,
  ``ERROR``, or ``CRITICAL`` (least verbose)

simulation
##########

Of the three sections in ``powder.yml``, the ``simulation`` section
has the most parameters. Most of these parameters are self-explanatory.

- ``pdb_name``: pdb file from which to make a simulated pattern
- ``pattern_name``: the simulated pattern is written to a file
  of this name; the `simulated pattern format`_ is described below
- ``d_max``: maximum d-spacing (lowest resolution) for the
  simulation, given in Ångstroms.
- ``d_min``: minimum d-spacing (highest resolution) for the
  simulation, given in Ångstroms.
- ``extinction_correction_x``: an optional parameter refined during
  extinction correction, which is applied to the simulated pattern;
  this correction is discussed in the SHELXL 97 manual on page 7-7
  (http://shelx.uni-ac.gwdg.de/SHELX/shelx97.pdf); use ``null``
  or ``0`` if extinction correction is not desired
- ``v`` & ``w``: for the summation of reflections, the square
  of the full-width at half-max (FWHM) of a Lorentzian diffraction
  peak is equal to to v + w tan(θ)
- ``B``: isotropic temperature factor; if the simulated pattern
   is to be scaled with experimental data, then ``B`` should be
   set to 0 because it will be refined during scaling
- ``apply_Lp``: apply Lorentz polarization correction
  (``True`` or ``False``)
- ``pattern_shells``: number of points in the simulated pattern;
  each point represents the integrated intensity of the shell
- ``peak_widths``: the intensity of a reflection is taken to be
  0 beyond this number of FWHM from the center of the Lorentzian
  reflection peak
- ``combine _reflections``: reflections may be combined by resolution
  such that all the reflections within a shell
  (specified by ``pattern_shells``) are taken to have the
  same center and peak shape (i.e. the same FWHM),
  making the calculations significantly faster at
  the expense of a small decrease in accuracy;
  values for ``combine_reflections`` may be ``True`` or ``False``

plot
####

The ``plot`` section controls the appearance of the plot.

- ``window_name``: name of the plot window
- ``left``, ``right``, ``top``, ``bottom``:
  margins between plot and page border; note that axes labels
  are in the margins
- ``plot_points``: the data is rebinned simply for the purposes
  of the plot; the plot will have ``plot_points`` points
- ``x_ticks``: number of ticks on the x-axis; labeled
  by 2θ.


experiment
##########

A simulated diffraction pattern is the result of a simulated
experiment. The parameters of the ``experiment`` section specify
simulated experimental details.

- ``WAVELENGTH``: radiation wavelength
- ``DISTANCE``: the distance from the sample to the detector

Please note that yaml is case-sensitive, so these latter
two parameter names must be in all caps.


Summation
+++++++++

The simulated powder diffraction pattern is the spherical
summation of diffraction intensity over each resolution shell,
where the number of resolution shells is specified by the ``patern_shells``
setting of the **powderx** config file.
Each shell is summed independently. All shells have approximately the same
width in :math:`\varrho`, or :math:`\sin(\theta)/\lambda`, where
:math:`\theta` is the Bragg angle and :math:`\lambda` is
the wavelength.

Given that the structure factor for a Miller index, :math:`hk\ell`,
is :math:`F_{hk\ell}`, the corresponding intensity
:math:`I_{hk\ell}` is

.. math::
    :label: intensity

    I_{hk\ell} =
      \dfrac{M_{hk\ell}}{V^{2}} \cdot
      L_{p} \cdot
      F_{hk\ell} \cdot F_{hk\ell}^{*} \cdot
      \exp \left (
              \dfrac{-2B\sin^{2}\theta}{\lambda^{2}}
           \right )

Where :math:`V` is the unit cell volume, :math:`M_{hk\ell}` is
the multiplicity of the reflection, :math:`B` the isotropic
temperature factor, and :math:`F_{hk\ell}^{*}` is the
complex conjugate of :math:`F_{hk\ell}`.
The term :math:`L_{p}` is the Lorentz polarization correction:

.. math::
    :label: Lp

    L_{p} =
      \left [
        \dfrac{1 + \cos^{2}(2\theta)}
               {\sin^{2}\theta \cos\theta}
      \right ]

The Lorentz polarization correction may be aplied
to the simulated pattern using the ``apply_Lp`` setting
within the ``simulation`` section of the **powderx** config file.

If the simulated pattern is to be scaled to
experimental data using the **profilex** utility,
:math:`B` should be set to 0 because :math:`B` is
refined during scaling. Note that when :math:`B`
is set to 0, the expression for diffraction intensity
simplifies to

.. math::
    :label: zeroB

    I_{hk\ell} =
      \dfrac{M_{hk\ell}}{V^{2}} \cdot
      L_{p} \cdot
      F_{hk\ell} \cdot F_{hk\ell}^{*}

Diffraction intensities are distributed throughout a profile,
approximated by **powderx** as a `Lorentzian distribution`_ (also
called a "Cauchy distribution"). This distribution takes the form

.. math::
    :label: lorentzian

    A_{J}^{L} = \dfrac{2}{\pi H_{B}}
       \left [1 + \dfrac{4}{H_{B}^{2}}
                  \left ( 2\theta_{i} - 2\theta_{hk\ell} \right )^2
       \right ]^{-1}

Where :math:`A_{J}^{L}` is the intensity contribution
at angle :math:`\theta_{i}`, :math:`H_{B}` is the FWHM
of the peak, and :math:`\theta_{hk\ell}` is the angle of
diffraction for the Miller index :math:`hk\ell`.  
For the purposes of the simulation, values of :math:`\theta_{i}` are taken
from the middles (in :math:`\varrho`) of the shells.

The FWHM, or :math:`H_{B}`, can be expressed as a function of the Bragg
angle of the reflection:

.. math::
    :label: FWHM

    H_{B}^{2} = v \tan \theta_{hk\ell} + w

The two parameters of this expression, :math:`v` and :math:`w`,
may be adjusted using the **powderx** config file
settings ``v`` and ``w``, respectively.

Because it is computationally expensive to calculate the contribution
of every reflection to every shell, **powderx** allows the user
to limit the calculations to the shells in the neighborhood
of :math:`\theta_{i}` using the ``peak_widths`` setting. This setting
specifies the width of the neighborhood in multiples of the FWHM.

Combining Reflections
#####################

With a small sacrifice in accuracy, a significant
improvement in the speed of the summation
can be realized by combining reflections by bin before
applying Equation :eq:`lorentzian`. For each bin, intensities
are summed over each Miller index of the bin,
:math:`hk\ell(\text{bin})`:

.. math::
    :label: sum_bin

    I_{\text{bin}} = \sum I_{hk\ell(\text{bin})}

Each bin intensity :math:`I_{\text{bin}}` has a peak center,
:math:`2\theta_{\text{bin}}`, at the
middle of the bin as determined
by the taking the mean of the :math:`2\theta` angles
of the reflections therein. If :math:`N_{\text{bin}}` is
the number of reflections in the bin, the middle of the
bin is given by Equation :eq:`bin_middle`:

.. math::
    :label: bin_middle

    2\theta_{\text{bin}} =
       \dfrac
          {\sum 2\theta_{hk\ell(\text{bin})}}
          {N_{\text{bin}}}

When reflections are combined in this manner, Equation :eq:`lorentzian`
becomes

.. math::
    :label: combined_lorentzian

       A_{J}^{L} = \dfrac{2}{\pi H_{B}}
       \left [1 + \dfrac{4}{H_{B}^{2}}
                  \left ( 2\theta_{i} - 2\theta_{\text{bin}} \right )^2
       \right ]^{-1}

This approximation can be applied with the ``combine_reflections`` setting
of the **powderx** config file.

.. _`Lorentzian distribution`: http://en.wikipedia.org/wiki/Cauchy_distribution

File Formats
------------

Simulated Pattern Format
~~~~~~~~~~~~~~~~~~~~~~~~

The simulated integrated diffraction pattern is written to a yaml
formatted file specified by the ``pattern_name`` setting
in the ``simulation`` section of ``powder.yml``. In yaml terms, the
pattern is stored as a list of [2θ, intensity] pairs keyed by
the word "``pattern``".
A python program can make a 2-D numpy_ array from the pattern easily
if numpy_ (http://numpy.scipy.org/) and pyYAML_ (http://pyyaml.org/)
are installed.  For example, if the pattern is stored in the
file "``pattern.yml``"::

      import numpy
      import yaml
      ary = numpy.array(yaml.load(open('pattern.yml')))['pattern']

The array called "``ary``" is a Nx2 array, with each of the N rows being a
[2θ, intensity] pair.

More generally, the pattern starts on the fourth line of the yaml
file and each data line conforms to the following FORTRAN
formatted read::

      REAL X, Y
      READ '(5X, F10.0, 1X, F10.0)', X, Y

The following are the first six lines of a yaml simulated powder diffraction
pattern file::

      model : "../testdata/stg06-phi06.4-wc-03.8-rc1.0-m4-12.pdb"
      pattern :
        # [   2-theta, intensity ]
        - [  5.205029, 0.5671240 ]
        - [  5.285076, 0.5882654 ]
        - [  5.365124, 0.6002413 ]

.. _pyYAML: http://pyyaml.org/


YAML Config Format
~~~~~~~~~~~~~~~~~~

The ``powder.yml`` file has a simple structure, which can be understood
from the following listing::

   %YAML 1.2
   ---
   section_1 :
      parameter_a : value_a
      parameter_b : value_b
   section_2 :
      parameter_c : value_c

Here, the first line is optional and indicates to a yaml parser that
the file conforms to the yaml specification version 1.2. The second
line of three dashes indicates the beginning of a yaml document.
Each section name is on a line by itself and followed by a colon.
Each parameter key-value pair is *indented* relative to 
the section names. All parameter key-value pairs are indented the
same number of spaces.
A colon separates the parameter key from its
associated value.

Note that yaml is case sensitive.


Scaling Patterns
----------------

Scaling is achieved using the *averaging* mode of the **profilex**
utility, which is still not officially documented. However,
this functionality is available, so a brief discussion of the
method follows.

Method of Scaling
~~~~~~~~~~~~~~~~~

When scaling, the integrated pattern intensity in the resolution
range of interest (specfied by the ``roi`` setting) is partitioned
into the number of bins specified by the setting ``roi_bins``.
Each bin has a center at :math:`\varrho`, or
:math:`\sin(\theta)/\lambda`, where :math:`\theta` is the
Bragg angle. The scaled intensity :math:`I_{\varrho}^{\circ}`
for a bin with center :math:`\varrho`
is related to the expermental intensity :math:`I_{\varrho}`
for that bin by the equation

.. math::
    :label: scaled_intensity

    I_{\varrho}^{\circ} =
    \left \{ I_{\varrho} - \left (m\varrho + b \right ) \right \}
    \alpha \exp \left \{ -2B\varrho^{2} \right \}

Where :math:`\alpha`, :math:`m`, :math:`b`, and :math:`B` are fit
during scaling to the pattern specified by the ``scaled_to`` setting.
The parameter :math:`B` is the isotropic temperature factor and
:math:`\alpha` is the overall scale.

Background Correction
~~~~~~~~~~~~~~~~~~~~~

The parameters :math:`m` and :math:`b` in
Equation :eq:`scaled_intensity` estimate the contribution
of background scatter, such that the following equation describes
the background correction of experimental intensities
:math:`I_{\varrho}`:

.. math::
    :label: background_corrected

    I_{\varrho}^{\circ}
    \alpha^{-1} \exp \left \{ 2B\varrho^{2} \right \} =
    I_{\varrho} - \left (m\varrho + b \right ) 

Note that the expression :math:`m\varrho + b` in
Equation :eq:`background_corrected` defines a straight
line, meaning that the background scatter is coarsely estimated
to require a simple linear correction. This coarseness reduces
the risk of overfitting the background.

Background correction is applied with the ``background_correction``
setting using the the reference pattern specified by ``scaled_to``.
In other words, the background correction is estimated directly from
scaling.

Background correction will fail if an attempt is made to
scale more than one pattern to the reference pattern. The reason
can be seen on the left hand side of Equation
:eq:`background_corrected`, which implies that the reference
intensities (anlagous to the scaled intensities :math:`I_{\varrho}^{\circ}`)
must be scaled with :math:`\alpha` and corrected for
the isotropic temperature factor :math:`B`.
Since scaling produces unique values of these fitting parameters
for each pair of patterns, **profilex** will terminate with
an error if more than one pair is specified for scaling while
background correction is turned on. Scaling more than
two patterns simultaneously is not yet supported.

Experimental Considerations
+++++++++++++++++++++++++++

It should be emphasized that the proper experimental way
to correct for background is to take a background exposure
(*i.e.* an exposure with no sample but with a holder, etc.)
and subtract the blank from the otherwise equivalent exposure
of a sample. This subtraction can be achieved with the *difference*
mode of the **profilex** utility.


Goodness of Fit
~~~~~~~~~~~~~~~

The goodness of fit between patterns is measured if
a reference pattern is specified by the ``scaled_to`` setting.
The metric for goodness of fit is called :math:`R_{o}^{2}`,
which is calculated using the following equation:

.. math::
     :label: Ro_squared

     R_{o}^{2} =
       \dfrac{\sum_{\varrho}
                (I_{\varrho}^{\circ} - I_{\varrho}')^{2}}
             {\sum_{\varrho}
                (\left < I' \right > - I_{\varrho}')^{2}}

Where :math:`I_{\varrho}'` is the intensity of the bin
with a middle at :math:`\varrho` for
the reference pattern. The term :math:`\left < I' \right >`
is the average bin intensity for the reference pattern.

The :math:`R_{o}^{2}` metric has been described in
Proc Natl Acad Sci U S A. 2012 May 15;109(20):7717-22.


.. _`source distribution`: Download_
.. _`yaml`: http://www.yaml.org/


.. toctree::
   :maxdepth: 2
   :numbered:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
