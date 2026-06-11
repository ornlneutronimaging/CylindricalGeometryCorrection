Tutorials
=========

The following tutorials demonstrate how to use the neutron-geomcorr package for cylindrical geometry correction.
The tutorial notebooks are committed with executed outputs, so they can be browsed on GitHub without running
anything (development notebooks under ``notebooks/dev_*`` are kept output-free by convention).

Step-by-step series (FITS)
--------------------------

A four-part walkthrough splitting loading and correction into separate notebooks:

* `tutorial00 — loading a homogeneous dataset <https://github.com/ornlneutronimaging/CylindricalGeometryCorrection/blob/next/notebooks/tutorial00_homogeneous_loading.ipynb>`_
* `tutorial01 — correcting a homogeneous cylinder <https://github.com/ornlneutronimaging/CylindricalGeometryCorrection/blob/next/notebooks/tutorial01_homogeneous_apply.ipynb>`_
* `tutorial02 — loading an inhomogeneous (hollow) dataset <https://github.com/ornlneutronimaging/CylindricalGeometryCorrection/blob/next/notebooks/tutorial02_inhomogeneous_loading.ipynb>`_
* `tutorial03 — correcting an inhomogeneous cylinder <https://github.com/ornlneutronimaging/CylindricalGeometryCorrection/blob/next/notebooks/tutorial03_inhomogeneous_apply.ipynb>`_

Homogeneous Cylinder Tutorial
------------------------------

This tutorial demonstrates the correction process for homogeneous cylindrical samples.

`View Tutorial Notebook (Homogeneous) <https://github.com/ornlneutronimaging/CylindricalGeometryCorrection/blob/next/notebooks/tutorial_homogeneous_tiff.ipynb>`_

The tutorial covers:

* Loading TIFF images of homogeneous cylindrical samples
* Setting up geometry parameters (center and radius)
* Applying the homogeneous correction algorithm
* Visualizing the corrected results

Inhomogeneous Cylinder Tutorial
--------------------------------

This tutorial demonstrates the correction process for inhomogeneous cylindrical samples with inner and outer radii.

`View Tutorial Notebook (Inhomogeneous) <https://github.com/ornlneutronimaging/CylindricalGeometryCorrection/blob/next/notebooks/tutorial_inhomogeneous_tiff.ipynb>`_

The tutorial covers:

* Loading TIFF images of inhomogeneous cylindrical samples
* Setting up geometry parameters (center, inner radius, outer radius)
* Applying the inhomogeneous correction algorithm
* Visualizing the corrected results

Running the Tutorials
---------------------

To run these tutorials locally:

1. Clone the repository::

    git clone https://github.com/ornlneutronimaging/CylindricalGeometryCorrection.git
    cd CylindricalGeometryCorrection

2. Install the package with pixi::

    pixi install
    pixi shell

3. Launch Jupyter and open the tutorial notebooks::

    jupyter lab notebooks/
