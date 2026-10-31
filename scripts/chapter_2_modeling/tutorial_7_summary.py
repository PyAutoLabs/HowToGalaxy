"""
Tutorial 7: Summary
===================

__Summary__

You have completed chapter 2. You can now define galaxy models and priors, connect them to data through Analysis
objects, and use searches to obtain and inspect fits.

The central challenge is balancing realism, reliable exploration and runtime. Visual inspection helps motivate
priors and reveal inadequate models. Linear profiles and MGEs can describe flexible light distributions with
fewer non-linear parameters. Data preparation determines which measurements constrain the model. Search chaining
lets different components be fitted in stages, while requiring care about the uncertainty of fixed components.

Try adapting the modelling examples in `autogalaxy_workspace/scripts/imaging` to your own data, or use the
simulators to make a controlled experiment. Once your fits are reliable, the workspace's results guides explain
how to load and analyse them in depth. Multiple-dataset modelling is another advanced topic for after this course.

Chapter 3 introduces pixelizations for reconstructing more irregular light, including features that a common-centre
MGE cannot describe. Chapter 4 develops modelling of scenes containing multiple galaxies. The optional searches
and speed tutorials are available when you need them; historical prior passing is not required reading.
"""
