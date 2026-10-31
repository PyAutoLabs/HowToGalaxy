"""
Tutorial 3: Why Modeling Is Hard
================================

The bulge and disk in tutorial 2 made our model more realistic, but also harder to fit. Their centres, shapes,
brightnesses and sizes can trade off. A search can find a plausible galaxy without finding the best explanation
available within the model.

Here we deliberately make exploration inadequate, then try three ways to improve it: tune priors, simplify the
model and search more thoroughly. Keep the images in view throughout. They tell us what the search is getting
right, what it is missing and which assumptions might help.

__Contents__

- **A Reference Fit:** Repeat the practicalities model for comparison.
- **An Inadequate Search:** Explore the same model with only 14 live points.
- **Local Maxima:** Understand how an apparently plausible fit can be wrong.
- **Prior Tuning:** Use visual impressions to guide exploration.
- **Reducing Complexity:** Test shared bulge and disk geometry.
- **Look Harder:** Give the full model more thorough exploration.
- **Benefits and Weaknesses:** Balance realism, reliability and runtime.
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

dataset_name = "simple"
dataset_path = Path("dataset") / "imaging" / dataset_name
if ag.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run([sys.executable, "scripts/simulators/simple.py"], check=True)

dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=2.5
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
__A Reference Fit__

The bright centre and extended disk are both visible. We repeat tutorial 2's model and search settings, so an
existing completed fit can be loaded. Treat this as a reference to compare against, not a proven global optimum.
On real data we would not know the correct parameter values in advance.
"""

model = af.Collection(
    galaxies=af.Collection(
        galaxy=af.Model(
            ag.Galaxy, redshift=0.5, bulge=ag.lp.Sersic, disk=ag.lp.Exponential
        )
    )
)
analysis = ag.AnalysisImaging(dataset=dataset, use_jax=True)
search = af.Nautilus(
    path_prefix=Path("howtogalaxy", "chapter_2"),
    name="tutorial_2_practicalities",
    unique_tag=dataset_name,
    n_live=200,
    n_batch=50,
    iterations_per_quick_update=2500,
)
result_reference = search.fit(model=model, analysis=analysis)
aplt.subplot_fit_imaging(fit=result_reference.max_log_likelihood_fit)

"""
__An Inadequate Search__

Now reduce the search to `n_live=14`, leaving the data and model unchanged. Fourteen is the minimum allowed
for this 13-parameter model: ten live points would be rejected before sampling. This deliberately poor setting makes
missing good solutions much more likely. Random searches differ between runs, so it cannot guarantee failure.
If it happens to find a good fit, that does not make 14 live points a reliable choice for this model.
"""

search = af.Nautilus(
    path_prefix=Path("howtogalaxy", "chapter_2"),
    name="tutorial_3_inadequate",
    unique_tag=dataset_name,
    n_live=14,
    n_batch=50,
)
result_inadequate = search.fit(model=model, analysis=analysis)
print(result_inadequate.info)
aplt.subplot_fit_imaging(fit=result_inadequate.max_log_likelihood_fit)

"""
__Compare the Fits__

Look for a centre that is too bright or too faint, or residual light following the disk. A reasonable-looking
parameter summary can conceal these failures. Compare the log likelihoods as well as the plots; here the data,
mask and likelihood definition are the same, so higher values mean a better fit.
"""

print("Reference:", result_reference.max_log_likelihood_fit.log_likelihood)
print("14 live points:", result_inadequate.max_log_likelihood_fit.log_likelihood)

"""
__Global and Local Maxima__

A local maximum is a solution better than nearby alternatives, with a better solution elsewhere in parameter
space. The global maximum is the highest likelihood available in that model. A search can concentrate around a
local peak and miss another: improving the fit nearby does not prove there is nowhere better to look.

If the reference has a higher likelihood, it directly demonstrates that the inadequate search missed better
models. It does not prove that the reference is the global maximum. If the two agree, inspect repeated runs and
more thorough searches before concluding the exploration is reliable.

Adding physical components increases the combinations we must explore. This is the central challenge of modelling:
we want enough flexibility to describe a real galaxy, while still exploring the model reliably.

__Approach 1: Prior Tuning__

Return to the image before changing settings. We can see roughly where the galaxy sits, its orientation and how
far its light extends. The default priors must cover many different galaxies; our visual impressions can guide a
search of this particular image. Below we create the bulge and disk separately to make those choices easy to read.
"""
bulge = af.Model(ag.lp.Sersic)
disk = af.Model(ag.lp.Exponential)


"""
The light peaks at the centre of the cutout. This suggests that both component centres lie close to (0.0, 0.0).
We use a narrow range around that position, leaving room for a small offset. This judgement comes from the image,
not from looking up the simulator's answer. On real data, first check the coordinate origin and any bright contaminants.
"""

bulge.centre_0 = af.UniformPrior(lower_limit=-0.05, upper_limit=0.05)
bulge.centre_1 = af.UniformPrior(lower_limit=-0.05, upper_limit=0.05)
disk.centre_0 = af.UniformPrior(lower_limit=-0.05, upper_limit=0.05)
disk.centre_1 = af.UniformPrior(lower_limit=-0.05, upper_limit=0.05)


"""
The central light is fairly round, while the outer disk is more elongated. The central contours suggest an angle
around 45 degrees; the outer contours lie closer to 30 degrees. The bulge's angle is less obvious because it is
rounder. These are rough impressions, not precise measurements, so our priors allow variation around them.

We convert approximate axis ratios of 0.9 and 0.7 and those angles into elliptical components. Allowing each
component its own shape matters: forcing them to agree would erase the visual distinction we are trying to model.
"""

bulge_ell_comps = ag.convert.ell_comps_from(axis_ratio=0.9, angle=45.0)
disk_ell_comps = ag.convert.ell_comps_from(axis_ratio=0.7, angle=30.0)

bulge.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
    mean=bulge_ell_comps[0], sigma=0.1, lower_limit=-1.0, upper_limit=1.0
)
bulge.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
    mean=bulge_ell_comps[1], sigma=0.1, lower_limit=-1.0, upper_limit=1.0
)

disk.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
    mean=disk_ell_comps[0], sigma=0.1, lower_limit=-1.0, upper_limit=1.0
)
disk.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
    mean=disk_ell_comps[1], sigma=0.1, lower_limit=-1.0, upper_limit=1.0
)


"""
The visible galaxy occupies only a few arcseconds. A half-light radius of tens of arcseconds would be a poor
starting guess for this cutout. We favour a compact bulge and a more extended disk, while retaining broad tails.

The half-light radius encloses half the profile's total light. It is not the radius where the image disappears
into the noise, so visual extent gives guidance rather than an exact radius. Positive limits exclude unphysical
sizes; the width of each prior leaves room for the data to refine our impression.
"""

bulge.effective_radius = af.TruncatedGaussianPrior(
    mean=0.8, sigma=0.5, lower_limit=0.0, upper_limit=np.inf
)
disk.effective_radius = af.TruncatedGaussianPrior(
    mean=1.5, sigma=0.8, lower_limit=0.0, upper_limit=np.inf
)


"""
The bright, concentrated centre motivates a bulge with a Sersic index around 3, whereas the disk's Exponential
profile has index 1. Neither a bright centre nor a fitted index uniquely identifies a galaxy's physical history.
This is a modelling assumption to test against the residuals, not a morphological classification imposed by eye.
"""

bulge.sersic_index = af.TruncatedGaussianPrior(
    mean=3.0, sigma=1.0, lower_limit=0.0, upper_limit=np.inf
)


"""
Now compose the model from these customised components. The names and physical profiles are unchanged;
we have changed the ranges and relative weights assigned to possible parameter values.
"""

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))


print(model.info)

"""
Run with 75 live points. This is fewer than the reference's 200, but more than the deliberately inadequate 14.
We are testing a practical remedy, changing both the priors and search effort; it is not an isolated experiment
measuring the effect of priors alone. Check whether the central and outer residuals improve.
"""

search = af.Nautilus(
    path_prefix=Path("howtogalaxy", "chapter_2"),
    name="tutorial_3_tuned_priors",
    unique_tag=dataset_name,
    n_live=75,
    n_batch=50,
)
result_tuned = search.fit(model=model, analysis=analysis)
print(result_tuned.info)
aplt.subplot_fit_imaging(fit=result_tuned.max_log_likelihood_fit)

"""
__Approach 2: Reducing Complexity__

The bulge and disk overlap and have similar centres, so sharing a centre is plausible. We go further and also
force them to have identical elliptical components. This removes four free parameters, leaving nine.

The image already hints at the risk: the outer disk looks flatter than the central bulge. An easier model to
search can still be a worse description of the galaxy. We deliberately test this assumption and inspect its cost.
Use fresh profiles so the tuned priors above do not carry into this separate experiment.
"""

bulge = af.Model(ag.lp.Sersic)
disk = af.Model(ag.lp.Exponential)
bulge.centre = disk.centre
bulge.ell_comps = disk.ell_comps
model = af.Collection(
    galaxies=af.Collection(
        galaxy=af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)
    )
)
print(model.info)

"""
Use 75 live points again. Inspect the residuals around the centre and along the disk. This simulation has
components with different shapes, so the shared-geometry model cannot reproduce both exactly, even with a
perfect search. Persistent structure can signal an inadequate model rather than inadequate exploration.
"""

search = af.Nautilus(
    path_prefix=Path("howtogalaxy", "chapter_2"),
    name="tutorial_3_reduced_complexity",
    unique_tag=dataset_name,
    n_live=75,
    n_batch=50,
)
result_reduced = search.fit(model=model, analysis=analysis)
print(result_reduced.info)
aplt.subplot_fit_imaging(fit=result_reduced.max_log_likelihood_fit)

"""
__Approach 3: Look Harder__

We can retain the original model and broad priors, and give Nautilus more live points. This is straightforward
and does not require deciding by eye how each galaxy should look. It costs more computation, and still cannot
guarantee that every good region will be found.

The optional fit below uses 300 live points. Uncomment it to run the more thorough comparison. It is left out of
the default execution because the earlier experiments already involve several fits.
"""

model = af.Collection(
    galaxies=af.Collection(
        galaxy=af.Model(
            ag.Galaxy, redshift=0.5, bulge=ag.lp.Sersic, disk=ag.lp.Exponential
        )
    )
)
search = af.Nautilus(
    path_prefix=Path("howtogalaxy", "chapter_2"),
    name="tutorial_3_look_harder",
    unique_tag=dataset_name,
    n_live=300,
    n_batch=50,
)
# result_thorough = search.fit(model=model, analysis=analysis)
# aplt.subplot_fit_imaging(fit=result_thorough.max_log_likelihood_fit)

"""
__Benefits and Weaknesses__

| Approach | Benefit | Weakness |
| --- | --- | --- |
| Tune priors (75 live points) | Use visible structure to avoid implausible regions. | Incorrect or overly narrow choices can bias the fit; individual tuning takes time for large samples. |
| Reduce complexity (75 live points) | Fewer dimensions can make exploration easier and faster. | Shared geometry can prevent the model from describing real differences between components. |
| Look harder (300 live points) | Keep the flexible model and apply the same approach to many galaxies. | More computation can become expensive and does not repair an unsuitable model. |

The 14-live-point run is deliberately inadequate; the 200-live-point practicalities run is our reference.
Compare actual residuals and likelihoods rather than expecting these settings to produce identical outcomes on
every run. These experiments illustrate choices, not a controlled ranking of runtime.

__Model Limitations__

Even a well-explored Sersic-plus-Exponential model cannot reproduce every real galaxy. A bar, spiral structure,
dust or bright star-forming knots can leave residuals no adjustment of these two smooth profiles will remove.
Look at the shape and location of those residuals before deciding that the search simply needs longer.

__Wrap Up__

Search chaining is especially useful because it helps strike a balance between realistic models, reliable
exploration and manageable runtimes. It is a central tool for moving beyond isolated, hand-tuned fits. Tutorial 6
shows it in practice; first, linear profiles give us another way to simplify exploration while retaining flexibility.
"""
