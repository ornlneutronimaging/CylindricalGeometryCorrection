************
Introduction
************

This library has been created to remove the non constant thickness effect of a cylinder sample placed in our neutron
(works also for X-ray) beam. For such sample, the transmission signal will be much higher on the edge compare to the
center of the sample. Post analysis of such sample is then very challenging. This library creates a new image
corresponding to a sample of constant thickness. Thickness being the diameter of the original sample.

Principle
=========

Any homegenous cylindrical sample placed in a beam (neutron for example) will show a much higher transmission signal
near the edge seen by the beam, compare to the center. This is simply due to the fact that the beam has to go through
more material at the center compare to the side.

**ATTENTION** This library only works with cylinder placed in the vertical position!

The following figure illustrate the experimental set up and the signal measure for an homogeneous and inhomogeneous
sample

In order to correctly analyze data for those samples, one must cancel this cylindrical effect by "making" the sample
flat related to the direction of the beam.

The user needs to specify the position of the center as well as the radius of the cylinder. The program will then produce
an image corresponding to the same sample as if it was rectangular.

Such samples are called homogeneous because they are made of only one uniform and homogeneous material.

.. image:: _static/homogeneous_cylinder_2d_view.png

But program also work with inhomogenous sample for which the cylinder is hollow such as shown here.

.. image:: _static/inhomogeneous_cylinder_2d_view.png

Program works the same way, user needs to specify center and inner and outer radius of material sample.

In order to run, the program only requires the user to define

 * the center of the cylinder
 * the radius of the cylinder
 * (option) the radius of the inner cylinder