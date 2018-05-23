[![Build Status](https://travis-ci.org/ornlneutronimaging/CylindricalGeometryCorrection.svg?branch=master)](https://travis-ci.org/ornlneutronimaging/CylindricalGeometryCorrection)

# CylindricalGeometryCorrection
This library produces a FITS image with the cylindrical geometry corrected (hollow or not).

# Principle
Any homegenous cylindrical sample placed in a beam (neutron for example) will show a much higher transmission signal
near the edge facing the beam, compare to the center. This is simply due to the fact that the beam has to go through more
material at the center compare to the side.

![image](documentation/source/_static/homogeneous_cylinder_2d_view.png)



# Tests
to run the tests
> pytest -v --cov=tests/

# Documentation


