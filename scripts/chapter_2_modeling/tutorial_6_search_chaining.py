"""
Tutorial 6: Search Chaining
===========================

A neighbouring galaxy can complicate the fit to our target. We need to describe its light, but repeatedly
searching all its parameters while refining the target can waste effort and introduce difficult degeneracies.
Search chaining lets us fit different parts of the scene in stages.

First we fit a simple target profile and an MGE for the neighbour using a fast Prodigy optimisation. After checking
that fit, we fix the neighbour's fitted geometry and use Nautilus to fit a more flexible MGE for the target.
The second search can concentrate on the target without exploring the neighbour's geometry again.

This is the main benefit of chaining here: separating well-constrained components can make a realistic model
more robust and manageable. It does not depend on automatically narrowing the priors of the second search.

__Contents__

- **Dataset:** Keep both galaxies in the fit.
- **Search 1:** Obtain a useful description of the neighbour.
- **Inspect Before Chaining:** Check that the first result is suitable to fix.
- **Search 2:** Fix the neighbour's geometry and refine the target.
- **Limits and Pipelines:** Understand the assumptions and extend the workflow.
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

dataset_name = "extra_galaxy"
dataset_path = Path("dataset") / "imaging" / dataset_name
if ag.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run([sys.executable, "scripts/simulators/extra_galaxy.py"], check=True)

dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=6.0
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
__Keep Both Galaxies__

Unlike tutorial 5's noise-scaling example, this fit uses the original data and includes both galaxies in the model.
The target is near (0.0, 0.0), and the visible neighbour near (1.0, 3.5) arcseconds. On your own data, identify
these approximate centres from the image or with the workspace's data-preparation tools.

Include fine over sampling around both bright centres, since both contribute to the predicted image.
"""

over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[8, 4, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0), (1.0, 3.5)],
)
dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
__Search 1 Model__

The target starts as a linear Sersic. The neighbour uses 20 Gaussians with fixed widths and shared centre and
shape, following tutorial 4. The centre priors cover its visually identified location. All Gaussian intensities
are solved together with the target intensity each time the trial geometry is fitted.
"""

neighbor_gaussians = af.Collection(af.Model(ag.lp_linear.Gaussian) for _ in range(20))
sigmas = np.logspace(-2.0, np.log10(2.5), 20)
neighbor_centre_0 = af.UniformPrior(lower_limit=0.8, upper_limit=1.2)
neighbor_centre_1 = af.UniformPrior(lower_limit=3.3, upper_limit=3.7)
for i, gaussian in enumerate(neighbor_gaussians):
    gaussian.centre.centre_0 = neighbor_centre_0
    gaussian.centre.centre_1 = neighbor_centre_1
    gaussian.ell_comps = neighbor_gaussians[0].ell_comps
    gaussian.sigma = sigmas[i]
neighbor = af.Model(
    ag.Galaxy,
    redshift=0.5,
    light=af.Model(ag.lp_basis.Basis, profile_list=neighbor_gaussians),
)
target = af.Model(ag.Galaxy, redshift=0.5, bulge=ag.lp_linear.Sersic)
model_1 = af.Collection(galaxies=af.Collection(target=target, neighbor=neighbor))
print(model_1.info)

"""
__Fast First Search__

Prodigy improves trial models by following directions that increase their fit score, which includes the prior
weight as well as the likelihood. `MultiStartProdigy` tries several starting points so it is less dependent on
one initial guess. It uses JAX, the accelerated numerical engine already enabled in our Analysis objects.

Here we want a useful fitted neighbour, rather than a detailed range of uncertainty estimates. We use 24 starts
and a ceiling of 300 steps per start as a first pass. These settings are not a guarantee of a good result.
`batch_size=4` limits how many starts are evaluated together to keep memory use manageable.
"""

analysis_1 = ag.AnalysisImaging(dataset=dataset, use_jax=True)
search_1 = af.MultiStartProdigy(
    path_prefix=Path("howtogalaxy", "chapter_2"),
    name="tutorial_6_search_1",
    unique_tag=dataset_name,
    n_starts=24,
    batch_size=4,
    n_steps=300,
    iterations_per_quick_update=50,
)
result_1 = search_1.fit(model=model_1, analysis=analysis_1)
print(result_1.info)
aplt.subplot_fit_imaging(fit=result_1.max_log_likelihood_fit)

"""
__Inspect Before Chaining__

Look especially at the neighbour's centre and outskirts, and at the region where its light overlaps the target.
If systematic residuals remain, improve the first fit before fixing anything. More starts, a better neighbour
model or a more thorough search may be necessary. Chaining a poor result carries its mistakes into the next fit.

__Fix the Neighbour__

`result_1.instance` provides concrete fitted parameter values. Passing its neighbour into the next model fixes
that component's centre and shape, removing its four geometric parameters from the non-linear search.

This instance still contains linear Gaussian profiles: their intensities are solved again alongside the target
at each likelihood evaluation. We fix the neighbour's geometry, not a completely frozen image. Using converted
ordinary profiles would instead fix the brightnesses too, which is a different modelling choice.
"""

neighbor_fixed = result_1.instance.galaxies.neighbor

"""
__Search 2 Model__

Replace the target Sersic with a group of 30 Gaussians. This flexible radial light model has four non-linear
parameters: two centre coordinates and two elliptical components. A simultaneous fit with the neighbour's MGE
geometry free would have eight. Fixing the already-fitted neighbour therefore halves this geometric search space.

This example target is smooth. For a galaxy whose central and outer shapes differ, use the two-group construction
from tutorial 4; the same separation of the neighbour still helps. No result-based prior narrowing is needed.
"""

target_gaussians = af.Collection(af.Model(ag.lp_linear.Gaussian) for _ in range(30))
target_sigmas = np.logspace(-2.0, np.log10(6.0), 30)
target_centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
target_centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
for i, gaussian in enumerate(target_gaussians):
    gaussian.centre.centre_0 = target_centre_0
    gaussian.centre.centre_1 = target_centre_1
    gaussian.ell_comps = target_gaussians[0].ell_comps
    gaussian.sigma = target_sigmas[i]
target = af.Model(
    ag.Galaxy,
    redshift=0.5,
    light=af.Model(ag.lp_basis.Basis, profile_list=target_gaussians),
)
model_2 = af.Collection(galaxies=af.Collection(target=target, neighbor=neighbor_fixed))
print(model_2.info)

"""
__More Thorough Second Search__

Use Nautilus to explore the target's geometric parameters and estimate their uncertainties. The dataset and
likelihood definition are unchanged; the model and search have different roles in this second stage.
"""

analysis_2 = ag.AnalysisImaging(dataset=dataset, use_jax=True)
search_2 = af.Nautilus(
    path_prefix=Path("howtogalaxy", "chapter_2"),
    name="tutorial_6_search_2",
    unique_tag=dataset_name,
    n_live=200,
    n_batch=50,
)
result_2 = search_2.fit(model=model_2, analysis=analysis_2)
print(result_2.info)
aplt.subplot_fit_imaging(fit=result_2.max_log_likelihood_fit)

"""
Inspect both galaxies again. A refined target model can expose weaknesses in the neighbour that were less obvious
before. The first result is useful information, not a guarantee that this separation is valid.

__Benefits and Limits__

When a component can be fitted robustly, fixing its geometry removes unnecessary search directions and lets later
fits concentrate effort on the components of scientific interest. This can bring large improvements in reliability
and runtime, particularly in scenes with several galaxies. A fast first fit and a more thorough second fit also
avoid spending equal effort on every stage.

The second result is conditional on the fixed neighbour geometry. Its uncertainty does not automatically include
uncertainty in those fixed values. If blending makes the components strongly dependent, repeat the second fit with
plausible neighbour geometries or perform a final joint fit with the relevant parameters free. Inspecting each
stage is essential; more stages do not automatically make a fit more trustworthy.

__From Searches to Pipelines__

A pipeline packages repeated preparation, fits and checks into a reusable sequence. Chapter 4 applies these ideas
to systems with more galaxies. The workspace's `scripts/guides/modeling/chaining.py` provides further API examples.

The SLaM pipelines in PyAutoLens apply related component-by-component modelling to lensing. They are a useful
example of how far chaining can be developed, but are not required for learning galaxy morphology here.

__Wrap Up__

The chapter's tools work together: inspect the data, choose an adequate model, define its comparison through an
Analysis object, and organise searches so the difficult parts are explored reliably. Search chaining helps strike
the balance between flexibility and tractability without requiring every component to be fitted afresh at every stage.
"""
