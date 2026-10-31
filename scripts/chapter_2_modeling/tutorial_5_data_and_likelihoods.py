"""
Tutorial 5: Data and Likelihoods
===============================

So far we have changed the galaxy model and the search. Now consider the other side of the fit: the measurements
we include and how they are compared with a prediction. The data, model and Analysis object have distinct jobs.
This separation lets the same modelling tools work with different observations and likelihood functions.

__Contents__

- **Masking:** Retain useful constraints without fitting unrelated objects.
- **Extra Galaxy Noise Scaling:** Reduce the weight of contaminated pixels.
- **Analysis:** Connect the prepared data to the model.
- **Interferometer Data:** Apply the same modelling pattern to visibilities.
- **Multiple Datasets:** A pointer to advanced joint modelling.
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
__Masking__

Our first fit used a 3.0-arcsecond radius. Here we use 6.0 arcseconds to include more of the faint outskirts.
Look beyond the bright centre: individually faint pixels can collectively constrain how quickly the light falls
away. A mask that is too tight may discard those constraints.

There is another reason to include the outskirts. A model may predict light outside the fitted region that the
real image does not contain. Masking that region hides the disagreement from the likelihood. A visually good fit
inside a small mask can therefore conceal a poor description of the galaxy as a whole.

A larger mask is not automatically better. It includes more pixels to calculate and may admit neighbouring galaxies,
background structure or defects our model does not describe. Choose the fitted region from the data and the
scientific question, then check that reasonable changes to it do not undermine the result.
"""

model = af.Collection(
    galaxies=af.Collection(
        galaxy=af.Model(ag.Galaxy, redshift=0.5, bulge=ag.lp_linear.Sersic)
    )
)
analysis = ag.AnalysisImaging(dataset=dataset, use_jax=True)

search = af.Nautilus(
    path_prefix=Path("howtogalaxy", "chapter_2"),
    name="tutorial_5_larger_mask",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=50,
)
result_mask = search.fit(model=model, analysis=analysis)
print(result_mask.info)
aplt.subplot_fit_imaging(fit=result_mask.max_log_likelihood_fit)

"""
__Contaminating Light__

A nearby galaxy is real signal, but it is not light from the target. If we fit it with the target's profile, the
search may distort the target's size or shape to account for it. We can exclude contaminated regions, reduce their
weight, or explicitly model the neighbour. The right choice depends on overlap and the information we need.

Load a second image containing a visible neighbour. The simulator also supplies a region identifying its light.
For real data, inspect and define this region yourself, for example with the workspace's data-preparation GUI.
Check the mask against the image rather than assuming a circular region removes every trace of contamination.
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
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Extra Galaxy Noise Scaling__

Noise scaling keeps the contaminated pixels in the grid but greatly reduces their influence on the fit. The
helper below sets their data to zero and their noise to very large values. This is a deliberate modification of
which measurements constrain the model, not a physical subtraction of the neighbouring galaxy.

The supplied file follows the workspace's mask-storage convention; `invert=True` makes `False` select the pixels
to scale. Apply this to the freshly loaded image before applying the main mask, so the grids have matching shapes.
"""

mask_extra_galaxies = ag.Mask2D.from_fits(
    file_path=dataset_path / "mask_extra_galaxies.fits",
    pixel_scales=0.1,
    invert=True,
)
dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=6.0,
)
dataset = dataset.apply_mask(mask=mask)
over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[8, 4, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)
dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)
aplt.subplot_imaging_dataset(dataset=dataset)

"""
Inspect the data, noise and signal-to-noise panels together. The neighbour's selected pixels should have almost
no weight, while the target remains constrained. This also retains a continuous grid, useful for some later
reconstruction methods. Light outside the selected region, including PSF wings, can still contaminate the fit.
Where the galaxies overlap substantially, fitting both is often preferable. Tutorial 6 illustrates that approach.

__Analysis as the Interface__

Our galaxy model does not change merely because we scale a region's noise. The new Analysis object receives the
prepared dataset and uses its noise map when calculating the likelihood. This is where our data choices enter
the model comparison. A search sees the returned score, not the reasons we chose a mask or excluded a neighbour.
"""

analysis = ag.AnalysisImaging(dataset=dataset, use_jax=True)

search = af.Nautilus(
    path_prefix=Path("howtogalaxy", "chapter_2"),
    name="tutorial_5_noise_scaling",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=50,
)
result_scaled = search.fit(model=model, analysis=analysis)
print(result_scaled.info)
aplt.subplot_fit_imaging(fit=result_scaled.max_log_likelihood_fit)

"""
Do not compare the absolute likelihood of this fit with the earlier dataset or a differently scaled noise map
as if only the galaxy model had changed. The measurements and their weighting also define the likelihood.

__Interferometer Data__

Chapter 1 introduced interferometer visibilities. The same galaxy model can predict those measurements, but the
comparison changes: `AnalysisInterferometer` Fourier transforms the model light and compares predicted visibilities
with the measured complex values and their uncertainties. Its likelihood is evaluated in visibility space.
The real-space mask defines the image grid used for predictions; it does not select image pixels as observations.

The pattern remains data + model + Analysis + search. See `autogalaxy_workspace/scripts/interferometer/modeling.py`
for a complete example after this course. The shared interface does not mean that different datasets have the
same noise properties or likelihood calculation.

__Multiple Datasets__

We can also combine Analysis objects. For independent observations, their likelihoods multiply, or equivalently
their log likelihoods add. A joint model can share parameters such as galaxy geometry while allowing others, such
as brightness in different wavebands, to differ between datasets.

The `autogalaxy_workspace/scripts/multi_dataset` examples develop this approach. It is advanced modelling:
finish HowToGalaxy first, then explore how shared parameters, calibration and any correlations between datasets
affect a joint analysis. No extra Python is needed here to understand the principle.

__Wrap Up__

The data describe measurements and uncertainty; the model describes possible galaxies; Analysis defines how to
compare them. The search explores that model using the resulting likelihood. Keeping these roles clear makes it
easier to decide whether a poor fit needs better data preparation, a different model or more thorough exploration.
"""
