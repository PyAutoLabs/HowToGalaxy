"""
Tutorial 4: Linear Profiles
===========================

In tutorial 3, simplifying the bulge and disk geometry made the search easier but restricted the galaxies it could
represent. Linear light profiles offer a different opportunity: keep the same profile shapes and solve their
intensities directly, removing those dimensions from the non-linear search.

We will then combine many linear Gaussians into a multi Gaussian expansion (MGE). This gives us more freedom to
represent galaxy structure without asking Nautilus to search every Gaussian's brightness and size.

__Contents__

- **Linear Light Profiles:** Fit shapes while solving intensities directly.
- **Results and Intensities:** Retrieve and plot the solved profiles.
- **Galaxy Structure:** Motivate a more flexible description of the light.
- **Basis:** Build and inspect a group of Gaussians.
- **Two Gaussian Groups:** Represent different central and outer shapes.
- **Benefits and Limitations:** Understand what an MGE can and cannot describe.
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
__Linear Light Profiles__

For a chosen centre, shape, size and Sersic index, changing intensity simply scales a profile's image. Linear
algebra can therefore calculate the intensities that best fit the data for that trial geometry. We call this
calculation an inversion. The `ag.lp_linear` profiles use it automatically within the likelihood evaluation.

With an ordinary bulge and disk there are two intensity dimensions to search. Linear profiles remove both,
without removing either physical component. The intensities are still fitted; they are not fixed or ignored.
We also share the centres here, while retaining separate elliptical components because the image motivates them.
"""

bulge = af.Model(ag.lp_linear.Sersic)
disk = af.Model(ag.lp_linear.Exponential)
bulge.centre = disk.centre
model = af.Collection(
    galaxies=af.Collection(
        galaxy=af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)
    )
)
print(model.info)

"""
There is no intensity prior in this summary. At each trial geometry the intensities are solved conditionally;
this is not the same as sampling their full uncertainty distribution. Other parameter degeneracies remain, so
linear profiles make the search simpler without making it immune to failure.

__Search and Analysis__

The public modelling interface is unchanged. The Analysis object now includes the linear solve in its calculation
of the likelihood. This adds work per evaluation but can reduce the number of evaluations needed to explore the
model. Overall runtime depends on the data, hardware and search; there is no universal speed-up factor.
"""

analysis = ag.AnalysisImaging(dataset=dataset, use_jax=True)

search = af.Nautilus(
    path_prefix=Path("howtogalaxy", "chapter_2"),
    name="tutorial_4_linear_light_profiles",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=50,
)
result_linear = search.fit(model=model, analysis=analysis)
print(result_linear.info)
aplt.subplot_fit_imaging(fit=result_linear.max_log_likelihood_fit)

"""
__Intensities__

The result summary describes the searched parameters. To inspect the solved intensities, use the galaxies in
`max_log_likelihood_galaxies`. Their linear profiles have been converted to ordinary profiles with the solved
intensities. The fit also provides an explicit conversion property for this purpose.
"""

galaxies = result_linear.max_log_likelihood_galaxies
print(galaxies[0].bulge.intensity)
fit = result_linear.max_log_likelihood_fit
solved_galaxies = fit.galaxies_linear_light_profiles_to_light_profiles
print(solved_galaxies[0].disk.intensity)

"""
__Visualisation__

An unsolved linear profile has no brightness with which to draw its image. Plot the converted galaxies, or plot
the fit itself, which already contains the calculated model image. Compare the bulge and disk light with the data.
"""

aplt.plot_array(
    array=galaxies.image_2d_from(grid=dataset.grid), title="Solved galaxy image"
)
aplt.subplot_fit_imaging(fit=fit)

"""
__Galaxy Structure__

A single Sersic profile describes a smooth radial decline with one axis ratio and orientation. Real galaxies can
have a round central component and a flatter outer disk; their apparent flattening or orientation can change with
radius. A fixed Sersic shape cannot freely follow these changes, even if its central brightness is a good match.

An MGE adds Gaussians with different widths. Narrow Gaussians describe the central light and broader Gaussians
describe the outskirts. Their fitted intensities let the radial brightness change more freely than a single
Sersic law. Groups with different elliptical shapes can represent changes in flattening and isophotal twists,
where the orientation of equal-brightness contours varies across the galaxy.

This flexibility depends on the geometry we allow. Gaussians sharing a centre remain centrally symmetric even
when their angles differ. Asymmetric light requires offset centres or other freedom. An MGE does not automatically
capture every irregular feature simply because it contains many profiles.

__Basis__

A `Basis` groups linear profiles into one component. First build 30 Gaussians with fixed widths spanning 0.01
arcseconds (one tenth of the pixel scale) to 2.5 arcseconds. They share two centre coordinates and two elliptical
components, so only four parameters describe their non-linear geometry.

The width range is a modelling choice covering the scales of this example, not a statement that the profiles
end at the mask. Every Gaussian extends beyond its width and the mask boundary.
"""

total_gaussians = 30
sigma_list = np.logspace(
    np.log10(dataset.pixel_scales[0] / 10.0), np.log10(2.5), total_gaussians
)
centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
bulge_gaussians = af.Collection(
    af.Model(ag.lp_linear.Gaussian) for _ in range(total_gaussians)
)
for i, gaussian in enumerate(bulge_gaussians):
    gaussian.centre.centre_0 = centre_0
    gaussian.centre.centre_1 = centre_1
    gaussian.ell_comps = bulge_gaussians[0].ell_comps
    gaussian.sigma = sigma_list[i]
bulge = af.Model(ag.lp_basis.Basis, profile_list=bulge_gaussians)

"""
Before running a search, turn this model into a concrete trial instance using the median of each prior.
These are example parameter values, not fitted values. The linear solve will find intensities for this geometry,
but cannot change the geometry itself. This distinction is the reason we still need a non-linear search.
"""

trial_bulge = bulge.instance_from_prior_medians()
trial_galaxies = ag.Galaxies(galaxies=[ag.Galaxy(redshift=0.5, bulge=trial_bulge)])
trial_fit = ag.FitImaging(dataset=dataset, galaxies=trial_galaxies)
aplt.subplot_fit_imaging(fit=trial_fit)

"""
The default trial shape is round. Inspect where it leaves light along the elongated disk. More Gaussian widths
can improve a radial profile, but cannot turn a shared round shape into an ellipse or give the bulge and disk
different shapes. Those freedoms must be present in our model.

__Two Gaussian Groups__

Add ten Gaussians for a second group. They share the first group's centre but have their own elliptical components.
The two groups can now represent the rounder centre and flatter outer light with different weights at different
radii. Calling them `bulge` and `disk` is convenient, but does not prove a unique physical decomposition.
"""

disk_gaussians = af.Collection(af.Model(ag.lp_linear.Gaussian) for _ in range(10))
disk_sigmas = np.logspace(np.log10(dataset.pixel_scales[0] / 10.0), np.log10(2.5), 10)
for i, gaussian in enumerate(disk_gaussians):
    gaussian.centre.centre_0 = centre_0
    gaussian.centre.centre_1 = centre_1
    gaussian.ell_comps = disk_gaussians[0].ell_comps
    gaussian.sigma = disk_sigmas[i]
disk = af.Model(ag.lp_basis.Basis, profile_list=disk_gaussians)
model = af.Collection(
    galaxies=af.Collection(
        galaxy=af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)
    )
)
print(model.info)

"""
The 40 intensities are solved linearly and the widths are fixed. Two shared centre coordinates plus two shape
parameters for each group give six non-linear parameters. The longer model summary does not mean a larger search
space than the earlier Sersic-plus-Exponential model.

Fit this model with Nautilus and inspect the new result. The live-point setting is illustrative; assess its
reliability from the fit and repeat searches when needed.
"""

search = af.Nautilus(
    path_prefix=Path("howtogalaxy", "chapter_2"),
    name="tutorial_4_mge",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=50,
)
result_mge = search.fit(model=model, analysis=analysis)
print(result_mge.info)
aplt.subplot_fit_imaging(fit=result_mge.max_log_likelihood_fit)

"""
__MGE Result__

Use the new result to plot the fitted light and residuals. Its converted galaxies contain ordinary Gaussians with
all 40 solved intensities, so they can also be used for subsequent calculations and visualisation.
"""

galaxies_mge = result_mge.max_log_likelihood_galaxies
aplt.subplot_galaxies(galaxies=galaxies_mge, grid=dataset.grid)

"""
__Benefits and Limitations__

An MGE can follow a more flexible radial light distribution and, with suitable groups, changes in apparent shape.
It does this with a compact non-linear parameter space. Fixed widths and solved intensities remove difficult
search directions, although overlapping Gaussians and geometric parameters can still be degenerate.

The extra profiles and linear solve cost computation. An MGE is also less directly interpretable than a Sersic
index and effective radius: an individual Gaussian is a mathematical building block, not a separate stellar
population. Choose the representation for the scientific question, not only the appearance of its residuals.

The workspace example `autogalaxy_workspace/scripts/imaging/features/multi_gaussian_expansion/modeling.py`
explores more flexible MGE models. Our common-centre example cannot represent off-centre star-forming clumps or
arbitrary asymmetric structure. Additional centres add freedom, but choosing a centre for every knot is cumbersome.
Chapter 3 introduces pixelizations to reconstruct more irregular light without prescribing such components.

__Non-negative Intensities__

The linear solve constrains these light-profile intensities to be non-negative, including zero. This avoids
representing positive light through large positive and negative Gaussians that cancel one another. A zero
intensity simply means that a Gaussian is not needed for that trial geometry. This constraint does not guarantee
that the geometry or scientific interpretation is correct.

__Other Basis Functions__

Shapelets provide another way to expand a galaxy's light. Their different functions can describe other structure,
but their flexibility and computational cost must be assessed for the particular dataset. A worked example is in
`autogalaxy_workspace/scripts/imaging/features/shapelets/modeling.py`; it is optional background after this course.

__Wrap Up__

We can simplify the search while retaining, or increasing, flexibility in the light model. The next tutorial
turns back to the data: which measurements should enter the fit, how should contaminants be handled, and how does
the Analysis object turn those choices into a likelihood?
"""
