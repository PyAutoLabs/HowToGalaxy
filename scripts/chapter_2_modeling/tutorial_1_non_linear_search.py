"""
Tutorial 1: Non-linear Search
=============================

In chapter 1, you adjusted a galaxy model by hand and inspected how well its image matched the data. Now we will
ask a search to do this for us. Our aim is still to learn about the galaxy: its size, shape and concentration.

We will fit one Sersic profile using Nautilus, a nested sampler. You do not need to learn several kinds of search
to start modelling galaxies. Alternative searches are covered in the optional searches tutorial.

__Contents__

- **Parameter Space:** The galaxy models we can explore.
- **Dataset and Mask:** The observation we will fit.
- **Model and Priors:** The components and possible parameter values.
- **Analysis:** The interface between a model and the data.
- **Nested Sampling:** Use Nautilus to explore possible models.
- **Result:** Inspect the fit and the uncertainty in its parameters.

__Parameter Space__

Imagine changing only the galaxy's size and concentration. Each pair of values describes a different galaxy.
All those possible pairs form a two-dimensional parameter space. Adding its centre and shape adds more dimensions.
A point in this space is one particular model, with a number assigned to every free parameter.

The likelihood function gives each point a score by comparing the model prediction with the data and its noise.
Plotting that score over parameter space would give a likelihood surface. High regions describe better fits;
the parameter space is the set of possible inputs, while the likelihood surface describes their scores.

__Non-linear Search__

A search tries many parameter combinations, evaluates their likelihoods and uses what it learns to explore
promising regions. This automates the repeated guesses you made in chapter 1. Relationships between parameters
make the task difficult: a larger, dimmer galaxy may resemble a smaller, brighter one. A search must explore
these possibilities, rather than adjust each parameter independently.
"""

# from autogalaxy import setup_notebook; setup_notebook()

from pathlib import Path
import numpy as np
import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset and Mask__

Load the imaging data and retain the region inside the circular mask. The simulator creates the data if needed.
The image, noise map and PSF together describe the observation; none specifies which galaxy model we must fit.
"""

dataset_name = "simple__sersic"
dataset_path = Path("dataset") / "imaging" / dataset_name
if ag.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run([sys.executable, "scripts/simulators/sersic.py"], check=True)

dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)
dataset = dataset.apply_mask(mask=mask)

"""
__Over Sampling__

The bright centre changes rapidly across a pixel. Adaptive over sampling evaluates it on a finer grid so that
numerical inaccuracies do not masquerade as poor models. The outskirts need fewer evaluations.
"""

over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[8, 4, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)
dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Model__

PyAutoFit supplies the model and search tools; PyAutoGalaxy supplies the galaxy physics and data analysis.
Passing a profile class to `af.Model` makes its parameters quantities to infer. This differs from the fixed
profile instances we created in chapter 1.

We use a linear Sersic profile. Its intensity is still fitted, but is calculated directly for each trial shape
instead of being another parameter for Nautilus to search. Tutorial 4 explains this in detail.
"""

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=ag.lp_linear.Sersic)
model = af.Collection(galaxies=af.Collection(galaxy=galaxy))
print(model.info)

"""
__Priors__

The model summary lists a prior for every free parameter. A prior describes which values we allow and how much
weight we give them before the fit. A uniform prior assigns equal weight within its limits. A Gaussian prior
favours values near its mean, with its width set by `sigma`.

The defaults come from `config/priors`. They cover a broad range of galaxies. Our cutout places the bright centre
near (0.0, 0.0), so a prior concentrated around that position is sensible. We can customise it as follows:
"""

model.galaxies.galaxy.bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
model.galaxies.galaxy.bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
print(model.info)

"""
A narrow prior can help a search, but can also exclude a good explanation of the data. Priors should reflect
reasonable knowledge and assumptions, not force the answer we hope to find. Tutorial 3 explores this balance.

__Analysis__

The data and model have separate jobs. The data describe what we measured and its uncertainty; the model describes
possible galaxies. An `Analysis` object connects them. It takes a trial model instance, predicts its image,
convolves it with the PSF and compares it with the measured image using the noise map. Its
`log_likelihood_function` returns the score that guides the search.

This is the same comparison you performed using `FitImaging` in chapter 1, now repeated automatically.
`use_jax=True` enables the numerical engine that accelerates these repeated calculations on a CPU or a compatible
GPU. You do not need to know how JAX works to use it.
"""

analysis = ag.AnalysisImaging(dataset=dataset, use_jax=True)

"""
__Nested Sampling__

Nautilus starts by drawing trial models from the priors. These are called live points. As it explores, it focuses
on regions containing higher-likelihood models. The samples describe both good-fitting parameter values and the
range of values consistent with the data under our model assumptions.

The `n_live` setting controls how thoroughly the search explores. We use 100 for this introductory example;
it is not a guarantee of a correct fit. Tutorial 2 explains practical search settings, including `n_batch`.
Nautilus has stopping criteria, so we do not have to prescribe a fixed number of trial models.
"""

search = af.Nautilus(n_live=100, n_batch=50)
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The result summary reports inferred parameter values and uncertainties. For example, a range of plausible
half-light radii tells us how precisely the data constrain the galaxy's size, given this model and its priors.
A best-fitting value alone does not tell us this.
"""

print(result.info)
aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
Look at the data, model image and residuals together. Does the model reproduce the central brightness and faint
outskirts? Residuals consistent with noise are encouraging; coherent bright or dark structures need investigation.
A search finishing is not evidence by itself that it found an adequate model or explored all good solutions.

The samples are available through `result.samples`. For now, inspecting the fit and summary is enough. Tutorial 2
introduces saved output and points to the workspace for detailed results analysis.

__Wrap Up__

We supplied data, a model and an Analysis object, then asked Nautilus to explore the model's parameter space.
The next tutorial uses these same ingredients for a more realistic galaxy and explains the practicalities of
running and inspecting a fit.
"""
