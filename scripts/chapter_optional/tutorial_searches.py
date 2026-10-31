"""
Optional Tutorial: Alternative Searches
=======================================

You can complete HowToGalaxy using Nautilus without studying search algorithms in detail. Return here if you want
to understand the fast Prodigy stage of search chaining, or compare alternative approaches for your own work.

__Contents__

- **Optimisation:** Improve trial models towards a best fit with Prodigy.
- **Nested Sampling:** Explore parameter ranges with Nautilus.
- **MCMC:** Another way to estimate parameter distributions.

__Optimisation__

An optimiser repeatedly proposes changes to a trial galaxy, checks whether its score improves, and moves towards
better solutions. Imagine adjusting the size and shape so that the predicted light increasingly resembles the
image. Repeating this process aims to find the best-fitting model, rather than trying to describe every plausible
model in detail. Individual steps need not always improve the likelihood.

Prodigy uses gradients to choose useful directions. `MultiStartProdigy` repeats this from several starting models,
reducing the risk that one poor start determines the answer. Its score includes prior weight as well as likelihood
(a maximum-a-posteriori optimisation). It uses JAX, the numerical engine that accelerates these calculations and
provides the gradients; you do not need to study JAX to run it.

The result can give a useful galaxy size, but does not by itself tell us how precisely that size is measured.
It also cannot guarantee the best solution was found. Inspect residuals and test different starts or settings.
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

model = af.Collection(
    galaxies=af.Collection(
        galaxy=af.Model(ag.Galaxy, redshift=0.5, bulge=ag.lp_linear.Sersic)
    )
)
analysis = ag.AnalysisImaging(dataset=dataset, use_jax=True)
search = af.MultiStartProdigy(
    path_prefix=Path("howtogalaxy", "chapter_optional"),
    name="tutorial_searches_prodigy",
    unique_tag=dataset_name,
    n_starts=24,
    batch_size=4,
    n_steps=300,
)
result_prodigy = search.fit(model=model, analysis=analysis)
print(result_prodigy.info)
aplt.subplot_fit_imaging(fit=result_prodigy.max_log_likelihood_fit)

"""
__Nested Sampling__

Nautilus spends effort exploring the range of plausible parameters, supporting uncertainty estimates as well as
a best fit. This extra information is useful when making scientific measurements. It also estimates Bayesian
evidence, used for model comparison, and has stopping criteria rather than a prescribed number of steps.

Increase `n_live` when exploration is inadequate, but judge convergence from repeated fits and diagnostics as well.
No fixed live-point count guarantees reliable uncertainties for every model. The following runs the same galaxy
model with Nautilus so that you can inspect its fit and parameter ranges alongside the optimiser's result.
"""

search = af.Nautilus(
    path_prefix=Path("howtogalaxy", "chapter_optional"),
    name="tutorial_searches_nautilus",
    unique_tag=dataset_name,
    n_live=200,
    n_batch=50,
)
result_nautilus = search.fit(model=model, analysis=analysis)
print(result_nautilus.info)
aplt.subplot_fit_imaging(fit=result_nautilus.max_log_likelihood_fit)

"""
__MCMC__

Markov chain Monte Carlo methods explore distributions of parameters by proposing successive moves and accepting
or rejecting them according to their probabilities. Emcee uses an ensemble of walkers. Walkers can move to lower
likelihoods as well as higher ones; their aim is to sample the distribution, not only climb towards its peak.

A sensible starting region can help. Here the bounds describe an initial spread around values suggested by the
image; they do not replace the model priors. This demonstration uses a finite number of steps, which is not a
claim of convergence. Before interpreting uncertainty estimates, examine mixing, autocorrelation and whether
longer runs agree. Multiple walkers do not guarantee escape from separated regions of good fits.
"""

initializer = af.InitializerParamBounds(
    {
        model.galaxies.galaxy.bulge.centre.centre_0: (-0.05, 0.05),
        model.galaxies.galaxy.bulge.centre.centre_1: (-0.05, 0.05),
        model.galaxies.galaxy.bulge.effective_radius: (0.5, 1.5),
        model.galaxies.galaxy.bulge.sersic_index: (2.0, 5.0),
    }
)
search = af.Emcee(
    path_prefix=Path("howtogalaxy", "chapter_optional"),
    name="tutorial_searches_emcee",
    unique_tag=dataset_name,
    nwalkers=50,
    nsteps=1000,
    initializer=initializer,
)
result_emcee = search.fit(model=model, analysis=analysis)
print(result_emcee.info)
aplt.subplot_fit_imaging(fit=result_emcee.max_log_likelihood_fit)

"""
__Wrap Up__

A quick optimisation is useful when you need a fitted component or want to check a model setup. A sampler is useful
when you need ranges of plausible parameters and uncertainty estimates. Both require checking the result.
The workspace guide `autogalaxy_workspace/scripts/guides/modeling/searches.py` covers additional searches and settings.
For the core lessons, keep your attention on the galaxy model and the quality of its fit.
"""
