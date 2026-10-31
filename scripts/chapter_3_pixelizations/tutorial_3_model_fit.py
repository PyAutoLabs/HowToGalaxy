"""
Tutorial 3: Model Fit
=====================

A pixelization can describe a galaxy without assigning an analytic centre and shape to every visible feature.
However, analytic profiles remain useful when we want to describe its smooth components. In this tutorial we combine
an MGE with a pixelization and use search chaining to put the smooth light model in place first.

This is a continuation of Chapter 2's linear profiles. Gaussian widths are fixed, Gaussian intensities are solved
linearly, and the search explores their shared geometry. The second fit adds mesh brightnesses and searches for the
regularization coefficient. We actually perform both fits, so the second model inherits a fitted MGE.

Our smooth bulge-and-disc simulation is a controlled starting point. A pixelization need not reveal new clumps in
such data: inspect whether it captures genuine residual structure or merely compensates for modelling choices.
For irregular observed galaxies, the same procedure offers extra spatial freedom without choosing a separate
analytic profile for every knot.

__Contents__

- **Dataset:** Load, mask and oversample the image.
- **Search 1:** Fit the smooth galaxy light with an MGE.
- **Search 2:** Add a regularized pixelization using the fitted MGE geometry.
- **Search 3:** Refine the MGE geometry together with the pixelization.
- **Interpretation:** Inspect both the residuals and the component decomposition.
"""

from pathlib import Path
import subprocess
import sys
import numpy as np
import autogalaxy as ag
import autogalaxy.plot as aplt
import autoarray.plot as aaplt

dataset_path = Path("dataset") / "imaging" / "simple"
if ag.util.dataset.should_simulate(str(dataset_path)):
    subprocess.run([sys.executable, "scripts/simulators/simple.py"], check=True)
dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=2.0
)
dataset = dataset.apply_mask(mask=mask)
dataset = dataset.apply_over_sampling(over_sample_size_pixelization=1)
# Keep the smoke-test mesh smaller than its deliberately capped image.
mesh_pixels = min(25, min(dataset.shape_native) // 2)
mesh_shape = (mesh_pixels, mesh_pixels)
aplt.subplot_imaging_dataset(dataset=dataset)

import autofit as af

over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[8, 4, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)
dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
__Search 1: Fit the MGE__

An MGE combines Gaussians with fixed widths and linearly solved intensities. We use two Gaussian groups so that the
smooth model can describe light with different flattenings and orientations. Each group's Gaussians share their
elliptical shape; their centre is fixed at the known simulation centre in this example. For real data, the centre
should be inferred or justified independently.

The search therefore adjusts the shapes of these groups rather than a separate intensity and width for every
Gaussian. The helper below sets up those shared parameters. Look at `model_1.info` to inspect the resulting model.
This first fit supplies a useful starting description, not a guarantee that every smooth or compact feature has
been assigned to the correct physical component.
"""

mge = ag.model_util.mge_model_from(
    mask_radius=2.0,
    total_gaussians=20,
    gaussian_per_basis=2,
    centre_fixed=(0.0, 0.0),
    sigma_min=0.01,
)
model_1 = af.Collection(
    galaxies=af.Collection(galaxy=af.Model(ag.Galaxy, redshift=0.5, bulge=mge))
)
print(model_1.info)
search_1 = af.Nautilus(
    path_prefix=Path("howtogalaxy") / "chapter_3",
    name="mge_search_1",
    unique_tag="simple",
    n_live=100,
    n_batch=50,
)
result_1 = search_1.fit(
    model=model_1, analysis=ag.AnalysisImaging(dataset=dataset, use_jax=True)
)
aplt.subplot_fit_imaging(fit=result_1.max_log_likelihood_fit)

"""
__Search 2: Add the pixelization__

Inspect the MGE residuals before adding freedom. Compact coherent residuals might need a more flexible brightness
description, but PSF or background errors can also leave structure. A pixelization can absorb those errors too;
a better-looking fit does not identify their cause.

We now hold the MGE geometry at `result_1.instance` and add a rectangular brightness mesh. The mesh shape is fixed
before the search; JAX needs this array shape to remain static. The regularization coefficient is inferred, and the
mesh brightnesses are solved linearly for every proposal.

Using `.instance` makes this fit simpler because it does not need to rediscover the smooth galaxy geometry. The
MGE's linear Gaussian intensities are still solved together with the pixelized brightnesses. We are fitting one
combined image model, not subtracting a noisy observed image and pretending that subtraction is exact.
"""

pixelization = af.Model(
    ag.Pixelization,
    mesh=ag.mesh.RectangularUniform(shape=mesh_shape),
    regularization=af.Model(ag.reg.Constant),
)
pixelization.regularization.coefficient = af.LogUniformPrior(
    lower_limit=0.01, upper_limit=100.0
)
model_2 = af.Collection(
    galaxies=af.Collection(
        galaxy=af.Model(
            ag.Galaxy,
            redshift=0.5,
            bulge=result_1.instance.galaxies.galaxy.bulge,
            pixelization=pixelization,
        )
    )
)
search_2 = af.Nautilus(
    path_prefix=Path("howtogalaxy") / "chapter_3",
    name="mge_pixelization_search_2",
    unique_tag="simple",
    n_live=50,
    n_batch=50,
)
result_2 = search_2.fit(
    model=model_2, analysis=ag.AnalysisImaging(dataset=dataset, use_jax=True)
)
fit_2 = result_2.max_log_likelihood_fit
aplt.subplot_fit_imaging(fit=fit_2)
mapper_2 = fit_2.inversion.cls_list_from(ag.Mapper)[0]
aaplt.plot_mapper(
    mapper=mapper_2,
    solution_vector=fit_2.inversion.reconstruction_dict[mapper_2],
    title="Pixelized Component Alongside the MGE",
)

"""
__Search 3: Refine both components__

Fixing geometry helped us get started, but its uncertainty should not disappear from a final scientific analysis.
Using `result_1.model` returns the MGE geometry to the search with priors informed by the first fit. We also pass the
regularization model from the second fit. Their linear intensities continue to be solved for each proposal.

Search chaining is a practical way to initialize this combined model. Priors passed between searches should be
checked: if an initial analytic fit was inadequate, it can provide an overconfident starting point. In that case,
widen the relevant priors and investigate the competing solutions.
"""

model_3 = af.Collection(
    galaxies=af.Collection(
        galaxy=af.Model(
            ag.Galaxy,
            redshift=0.5,
            bulge=result_1.model.galaxies.galaxy.bulge,
            pixelization=result_2.model.galaxies.galaxy.pixelization,
        )
    )
)
search_3 = af.Nautilus(
    path_prefix=Path("howtogalaxy") / "chapter_3",
    name="mge_pixelization_search_3",
    unique_tag="simple",
    n_live=100,
    n_batch=50,
)
result_3 = search_3.fit(
    model=model_3, analysis=ag.AnalysisImaging(dataset=dataset, use_jax=True)
)
fit_3 = result_3.max_log_likelihood_fit
aplt.subplot_fit_imaging(fit=fit_3)
mapper_3 = fit_3.inversion.cls_list_from(ag.Mapper)[0]
aaplt.plot_mapper(
    mapper=mapper_3,
    solution_vector=fit_3.inversion.reconstruction_dict[mapper_3],
    title="Pixelized Component After Joint Geometry Refinement",
)

"""
__Wrap Up__

The MGE describes smooth light while the mesh gives the combined model more spatial freedom. Their decomposition
is not automatically unique: a pixelization can represent some light that an MGE could also explain. Regularization,
mesh resolution and analytic-model choices affect which component receives it.

Inspect the residuals, the pixelized component and the inferred MGE geometry together. A small residual map does not
guarantee unbiased component measurements. Check that scientific conclusions remain stable under reasonable mesh,
regularization and prior changes, and use the posterior samples when reporting uncertainty.

**🤖 Ask AI:** Why can the total galaxy image be well constrained while the division of its light between an MGE and
a pixelization remains uncertain?

This chapter has connected pixelized brightnesses to the linear intensities from Chapter 2, followed their mappings
into the image, explored regularization, and performed a chained model fit. MGE remains a useful modelling tool;
pixelization adds another way to represent galaxy structure when analytic components alone become cumbersome.
"""
