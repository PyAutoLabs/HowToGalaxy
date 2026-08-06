"""
Tutorial 5: Bayesian Formalism
==============================

In tutorials 1 to 4, we built an intuition for how pixelized galaxy reconstruction works: pixelizations place a
pixel-grid over the galaxy's image, mappers pair pixelization pixels with image-pixels, inversions solve for the
pixel fluxes that best fit the data, and regularization smooths the solution within a Bayesian framework.

This tutorial collects the linear algebra behind all of that in one place. The hands-on tutorials built intuition,
and now we write down the equations. We will construct, step-by-step and in code, every matrix and vector the
inversion uses, solve for the galaxy reconstruction ourselves and compute the Bayesian evidence by hand, comparing
our answer at the end to the `FitImaging` object which performs this calculation internally.

None of this is required to *use* **PyAutoGalaxy** for galaxy modeling. However, if you publish results which use a
pixelization, this is the calculation your paper's likelihood function section will describe, and understanding it
removes any sense that the galaxy reconstruction is a "black box".

The formalism follows Warren & Dye 2003 (https://arxiv.org/abs/astro-ph/0302587), hereafter WD03, with the data
vector notation of Nightingale & Dye 2015 (https://arxiv.org/abs/1412.7436), hereafter N15. These papers derived
the method for reconstructing gravitationally lensed sources, but the linear algebra is identical for a galaxy's
own light -- the only lensing-specific step (ray tracing image-pixel coordinates to a source-plane) is simply
absent, with the mesh laid directly over the galaxy's image. The Bayesian evidence was derived by Suyu et al. 2006
(https://arxiv.org/abs/astro-ph/0601493) and translated to **PyAutoGalaxy** notation in Dye et al. 2008
(https://arxiv.org/abs/0804.4002).

__Contents__

- **Initial Setup:** Load the dataset the previous tutorials fitted.
- **Dataset Auto-Simulation:** Automatically simulate the dataset if it does not already exist.
- **Mask:** Mask the data so the likelihood is evaluated only where the galaxy's light is.
- **Over Sampling:** Disable over sampling so the algebra stays simple.
- **Mesh Shape:** Fix the rectangular mesh's shape and note why edge pixels are zeroed.
- **Galaxy:** Create the galaxy whose pixelization pairs the mesh with `Constant` regularization.
- **Image Grid:** The masked image-pixel coordinates the mesh is laid over -- no ray tracing occurs.
- **Mesh Pixel Centres:** Overlay the rectangular mesh over the masked image grid.
- **Interpolation:** Pair every image-pixel with mesh pixels via bilinear interpolation.
- **Mapper:** Package the interpolation into a `Mapper` describing all image-to-mesh mappings.
- **Mapping Matrix:** Express the mappings as the 2D matrix $f$.
- **Blurred Mapping Matrix:** Convolve every column of $f$ with the imaging PSF.
- **Data Vector (D):** Compute the data vector $D$ from the blurred mapping matrix, data and noise-map.
- **Curvature Matrix (F):** Compute the curvature matrix $F$.
- **Unregularized Solve:** Solve $s = F^{-1} D$ and see the over-fitted mess this produces.
- **Regularization Matrix (H):** Compute the regularization matrix $H$ encoding the smoothness prior.
- **Galaxy Reconstruction (s):** Solve the regularized system $s = [F + H]^{-1} D$.
- **Image Reconstruction:** Map the reconstruction back to image resolution via the blurred mapping matrix.
- **Likelihood Function:** The five terms which combine into the log evidence.
- **Chi Squared:** The goodness-of-fit of the reconstructed image to the data.
- **Regularization Term:** The penalty $s^{T} H s$ applied by the smoothness prior.
- **Complexity Terms:** The log determinant terms which penalize complex galaxy reconstructions.
- **Noise Normalization Term:** The Gaussian noise normalization.
- **Log Evidence:** Combine all five terms into the log evidence.
- **Fit:** Compare our by-hand log evidence to the `FitImaging` object's internal calculation.
- **Wrap Up:** Summary and next steps.
"""

# from autogalaxy import setup_notebook; setup_notebook()

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import autogalaxy as ag
import autogalaxy.plot as aplt
from autoarray.inversion.plot.mapper_plots import plot_mapper

"""
__Initial Setup__

we'll use the same galaxy data as the previous tutorials, where:

 - The galaxy's bulge is an `Sersic`.
 - The galaxy's disk is an `Exponential`.

For simplicity, the model in this tutorial is a pixelization only, with no light profiles: the mesh absorbs the
bulge and disk together. In WD03's notation this means the light profile model image $b_{j}$ is zero everywhere. If
light profiles are included in the galaxy model, their PSF-convolved image is computed first and subtracted from
the data before the steps below -- nothing else about the formalism changes.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "imaging" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if ag.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/simulators/simple.py"],
        check=True,
    )

dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Mask__

The likelihood is only evaluated within a mask, which we define as the same 2.0" circle used in the previous
tutorials, containing all of the galaxy's light.
"""
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=2.0,
)

masked_dataset = dataset.apply_mask(mask=mask)

aplt.subplot_imaging_dataset(dataset=masked_dataset)

"""
__Over Sampling__

Over sampling splits every image pixel into a sub-grid of sub-pixels, each of which is paired with mesh pixels
individually. It improves accuracy, but it also multiplies the number of rows in every matrix below by the number
of sub-pixels.

To keep the algebra as easy to follow as possible, we disable over sampling by setting both sub-grid sizes to 1, so
each image pixel is represented by the single coordinate at its centre.
"""
masked_dataset = masked_dataset.apply_over_sampling(
    over_sample_size_lp=1,
    over_sample_size_pixelization=1,
)

"""
__Mesh Shape__

The `mesh_shape` defines the number of pixels in the rectangular mesh used to reconstruct the galaxy, set below
to 20 x 20 = 400 mesh pixels.

We use the `RectangularUniform` mesh, where all rectangular mesh pixels have the same size, rather than the
`RectangularAdaptDensity` mesh used in the previous tutorials. The uniform mesh keeps the geometry simple, and every
equation below applies unchanged to the adaptive meshes -- only the mesh pixel centres move.

By default, mesh pixels at the edge of the mesh are forced to solutions of zero flux by the linear algebra solver.
This prevents unphysical solutions where the mesh edge lights up to fit residuals, and does not change any of the
formalism below.
"""
mesh_pixels_yx = 20
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

"""
__Galaxy__

We create the galaxy whose `Pixelization` pairs the `RectangularUniform` mesh with `Constant` regularization (whose
role appears later, when we reach the matrix $H$).

The galaxy has no light profiles -- the mesh alone will reconstruct its bulge and disk.
"""
pixelization = ag.Pixelization(
    mesh=ag.mesh.RectangularUniform(shape=mesh_shape),
    regularization=ag.reg.Constant(coefficient=1.0),
)

galaxy = ag.Galaxy(redshift=0.5, pixelization=pixelization)

"""
__Image Grid__

In the lensing version of this formalism (see **PyAutoLens**), every image-pixel coordinate is first ray-traced to
a source-plane, and the mesh is laid over the traced coordinates. For a galaxy there is no lensing, so this step
simply does not exist: the mesh is laid directly over the galaxy's own image, and the grid the inversion uses is
the grid of masked image-pixel centres itself.

This grid is `masked_dataset.grids.pixelization`, with one coordinate at the centre of every masked image pixel
(because we disabled over sampling above).

(The `Mapper` attributes below carry `source_plane_` prefixes because **PyAutoGalaxy** and **PyAutoLens** share one
inversion implementation. For a galaxy, the "source plane" is just the image plane the galaxy lives in.)
"""
aplt.plot_grid(grid=masked_dataset.grids.pixelization, title="Masked Image Grid")

"""
__Mesh Pixel Centres__

To reconstruct the galaxy on a `RectangularUniform` mesh, we need the centres of its rectangular mesh pixels.

We compute these by overlaying a uniform rectangular grid over the masked image grid, sized so the mesh fully
contains the masked region without wasted edge pixels beyond it.
"""
from autoarray.inversion.mesh.mesh.rectangular_adapt_density import overlay_grid_from

mesh_grid = overlay_grid_from(
    shape_native=mesh_shape, grid=masked_dataset.grids.pixelization
)

"""
__Interpolation__

We now combine the two grids to create an `Interpolator`, which describes how every image-pixel coordinate maps to
the rectangular mesh pixels.

For a rectangular mesh the scheme is bilinear interpolation: every image pixel maps to the rectangular pixel it
lands in *and* its three nearest neighbours, with weights set by how close the coordinate is to each pixel centre.
Interpolation is what lets the mesh reconstruct smooth galaxy morphologies. We can print the mappings and weights
of the first image pixel to confirm it pairs with four mesh pixels.
"""
interpolator = pixelization.mesh.interpolator_from(
    source_plane_data_grid=masked_dataset.grids.pixelization,
    source_plane_mesh_grid=mesh_grid,
)

print(interpolator.mappings[0])
print(interpolator.weights[0])

"""
__Mapper__

The interpolator creates a `Mapper`, the object tutorials 1 and 2 introduced. It packages up the mapping between
every image pixel and every rectangular mesh pixel.

The key attribute is `pix_indexes_for_sub_slim_index`, mapping every image-pixel index (`sub_slim_index`) to the
mesh-pixel indexes (`pix_indexes`) it interpolates onto, alongside the number of mappings per image pixel and
their interpolation weights.
"""
mapper = ag.Mapper(interpolator=interpolator)

pix_indexes_for_sub_slim_index = mapper.pix_indexes_for_sub_slim_index

print(pix_indexes_for_sub_slim_index[0:9])
print(mapper.pix_sizes_for_sub_slim_index[0:9])
print(mapper.pix_weights_for_sub_slim_index[0:9])

"""
__Mapping Matrix__

The `mapping_matrix` expresses these image-pixel to mesh-pixel mappings as a single 2D matrix, with
dimensions `(total_image_pixels, total_mesh_pixels)`.

Each column is the "image" of one mesh pixel: entries are the interpolation weights for image pixels which map to
that mesh pixel and zero everywhere else.
"""
mapping_matrix = ag.util.mapper.mapping_matrix_from(
    pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
    pix_size_for_sub_slim_index=mapper.pix_sizes_for_sub_slim_index,
    pix_weights_for_sub_slim_index=mapper.pix_weights_for_sub_slim_index,
    pixels=mapper.pixels,
    total_mask_pixels=mapper.source_plane_data_grid.mask.pixels_in_mask,
    slim_index_for_sub_slim_index=mapper.slim_index_for_sub_slim_index,
    sub_fraction=mapper.over_sampler.sub_fraction,
)

plt.imshow(mapping_matrix, aspect=(mapping_matrix.shape[1] / mapping_matrix.shape[0]))
plt.show()
plt.close()

"""
Because each column is an image of zeros and interpolation weights, we can plot any column as a 2D image showing
all image pixels its mesh pixel maps to. For a mesh pixel near the mesh centre, these image pixels sit at the
centre of the galaxy, where its bulge is brightest.
"""
central_mesh_pixel = (mesh_shape[0] // 2) * mesh_shape[1] + mesh_shape[1] // 2

array_2d = ag.Array2D(
    values=mapping_matrix[:, central_mesh_pixel], mask=masked_dataset.mask
)

aplt.plot_array(array=array_2d, title="Image of Central Mesh Pixel")

"""
__Blurred Mapping Matrix__

The imaging data is blurred by the telescope's PSF, so the model must be too. Since each column of the mapping
matrix is an image, we simply convolve each column with the PSF via 2D convolution.

This produces the `blurred_mapping_matrix`, with the same dimensions `(total_image_pixels, total_mesh_pixels)`.

In WD03 this matrix is denoted $f_{ij}$, where $i$ runs over all $I$ mesh pixels and $j$ over all $J$ image
pixels. For example:

 - $f_{0, 2} = 0.3$ indicates that image-pixel $2$ maps to mesh-pixel $0$ with a weight of $0.3$ after PSF
   convolution.
 - $f_{4, 8} = 0$ indicates that image-pixel $8$ does not map to mesh-pixel $4$, even after PSF convolution.

(The indexing of the code's `mapping_matrix` is transposed relative to WD03's $f$: image pixels are the first index
in the code, but the second index in the equations.)

PSF blurring has an important consequence: it correlates neighbouring columns, so the images of nearby mesh pixels
now overlap far more than the interpolation alone produced.
"""
blurred_mapping_matrix = masked_dataset.psf.convolved_mapping_matrix_from(
    mapping_matrix=mapping_matrix, mask=masked_dataset.mask
)

plt.imshow(
    blurred_mapping_matrix,
    aspect=(blurred_mapping_matrix.shape[1] / blurred_mapping_matrix.shape[0]),
)
plt.colorbar()
plt.show()
plt.close()

"""
__Data Vector (D)__

We now pose the reconstruction as a linear inversion, converting the blurred mapping matrix, data and noise-map
into two objects: the data vector $D$ and the curvature matrix $F$.

The data vector has dimensions `(total_mesh_pixels,)` and is given by (WD03 / N15):

 $\vec{D}_{i} = \sum_{j=1}^{J} f_{ij} (d_{j} - b_{j}) / \sigma_{j}^2 \, \, .$

Where:

 - $d_{j}$ are the image-pixel data values.
 - $b_{j}$ are the model images of any light profiles in the galaxy model (zero here, because our model is a
   pixelization only).
 - $\sigma_{j}^2$ are the statistical uncertainties of each image pixel.

Each entry of $D$ is therefore the noise-weighted overlap between one mesh pixel's blurred image and the data:
it measures how much evidence the data provides for flux in that mesh pixel, with the PSF fully accounted for.
"""
data_vector = ag.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
    blurred_mapping_matrix=blurred_mapping_matrix,
    image=np.array(masked_dataset.data),
    noise_map=np.array(masked_dataset.noise_map),
)

plt.imshow(
    data_vector.reshape(data_vector.shape[0], 1), aspect=10.0 / data_vector.shape[0]
)
plt.colorbar()
plt.show()
plt.close()

"""
__Curvature Matrix (F)__

The curvature matrix has dimensions `(total_mesh_pixels, total_mesh_pixels)` and is given by (WD03):

 ${F}_{ik} = \sum_{j=1}^{J} f_{ij} f_{kj} / \sigma_{j}^2 \, \, .$

Every entry of $F$ is the noise-weighted overlap between the blurred images of two mesh pixels: $F_{ik}$ sums the
product of columns $i$ and $k$ of $f$ over all image pixels. For $F_{ik}$ to be non-zero, mesh pixels $i$ and $k$
must share at least one image pixel, which happens for neighbouring pixels via interpolation and for pixels
further apart via PSF blurring.

$F$ describes how degenerate pairs of mesh pixels are with one another: two mesh pixels whose blurred images
overlap heavily can trade flux between themselves whilst fitting the data almost equally well.
"""
curvature_matrix = ag.util.inversion.curvature_matrix_via_mapping_matrix_from(
    mapping_matrix=blurred_mapping_matrix, noise_map=masked_dataset.noise_map
)

plt.imshow(curvature_matrix)
plt.colorbar()
plt.show()
plt.close()

"""
__Unregularized Solve__

The inversion seeks the mesh-pixel fluxes $s$ (a vector with one entry per mesh pixel) that minimize the
chi-squared:

 $\chi^2 = \sum_{j=1}^{J} \bigg[ \frac{(\sum_{i=1}^{I} s_{i} f_{ij}) + b_{j} - d_{j}}{\sigma_{j}} \bigg]^2$

Setting the derivative of $\chi^2$ with respect to each $s_{i}$ to zero gives the linear system whose solution is
(equation 5 of WD03):

 $s = F^{-1} D$

We can solve this directly with NumPy. (Without regularization the curvature matrix is often singular, so the loop
below adds a tiny value to its diagonal to avoid a `LinAlgError` -- it is a numerical crutch, not part of the
formalism.)
"""
for i in range(curvature_matrix.shape[0]):
    curvature_matrix[i, i] += 1e-8

reconstruction = np.linalg.solve(curvature_matrix, data_vector)

plot_mapper(mapper=mapper, solution_vector=reconstruction)

"""
The reconstructed mesh-pixel fluxes are a noisy, unsmooth mess -- exactly the over-fitting we saw in tutorial 4
when we lowered the regularization coefficient towards zero. The linear inversion is fitting the noise in the
data, because this system of equations is ill-posed: we need a smoothness prior.

__Regularization Matrix (H)__

Regularization adds a linear regularization term $G_{L}$ to the merit function we minimize (equation 11 of WD03):

 $G = \chi^2 + \lambda \, G_{L}$

where $\lambda$ is the `regularization_coefficient` controlling the degree of smoothing. The `Constant` scheme uses
gradient regularization (equation 14 of WD03):

 $G_{L} = \sum_{i}^{I} \sum_{n=1}^{N} [s_{i} - s_{i, n}]^2$

In words: for every mesh pixel, compare its flux with each of its $N$ neighbours $n$, and penalize solutions where
the differences are large. This is precisely the "smoothness prior" of tutorial 4, now written as an equation.

To fold this into the linear algebra we define the regularization matrix $H$, with
dimensions `(total_mesh_pixels, total_mesh_pixels)` (equation 13 of WD03):

 $H_{ik} = \frac{1}{2} \frac{\partial^{2} G_{L}}{\partial s_{i} \partial s_{k}}$

$H$ has the coefficient $\lambda$ folded into it. Its non-zero off-diagonal entries mark pairs of mesh pixels
which are neighbours and therefore regularized with one another; most entries are zero because most mesh pixels
are not neighbours.
"""
regularization_matrix = ag.util.regularization.constant_regularization_matrix_from(
    coefficient=galaxy.pixelization.regularization.coefficient,
    neighbors=mapper.neighbors,
    neighbors_sizes=mapper.neighbors.sizes,
)

plt.imshow(regularization_matrix)
plt.colorbar()
plt.show()
plt.close()

"""
__Galaxy Reconstruction (s)__

$H$ enters the linear system as follows (equation 12 of WD03):

 $s = [F + H]^{-1} D$

We add the two matrices and solve again. The diagonal jitter used above is no longer needed, because $H$ makes the
system well-posed.
"""
curvature_reg_matrix = np.add(curvature_matrix, regularization_matrix)

reconstruction = np.linalg.solve(curvature_reg_matrix, data_vector)

plot_mapper(mapper=mapper, solution_vector=reconstruction)

"""
The reconstructed fluxes are now smooth and physical: regularization has suppressed the noisy solution and the
reconstruction actually looks like the galaxy's bulge and disk, without over-fitting the noise.

__Image Reconstruction__

Using the reconstructed mesh-pixel fluxes, we map the reconstruction back to image resolution via the blurred
mapping matrix (so the reconstructed image includes PSF blurring) to produce the model image of the galaxy.
"""
mapped_reconstructed_data = (
    ag.util.inversion.mapped_reconstructed_data_via_mapping_matrix_from(
        mapping_matrix=blurred_mapping_matrix, reconstruction=reconstruction
    )
)

mapped_reconstructed_data = ag.Array2D(
    values=mapped_reconstructed_data, mask=masked_dataset.mask
)

aplt.plot_array(array=mapped_reconstructed_data, title="Reconstructed Image")

"""
__Likelihood Function__

We now quantify the goodness-of-fit of the galaxy reconstruction, computing the quantity tutorial 4 called the
Bayesian evidence. The log evidence consists of five terms:

 $-2 \, \mathrm{ln} \, \epsilon = \chi^2 + s^{T} H s + \mathrm{ln} \, [ \mathrm{det} (F + H) ] - \mathrm{ln} \, [ \mathrm{det} (H) ] + \sum_{j=1}^{J} \mathrm{ln} \, [2 \pi (\sigma_{j})^2 ] \, .$

This expression was first derived by Suyu et al. 2006 (https://arxiv.org/abs/astro-ph/0601493), equation (19), and
is given in **PyAutoGalaxy** notation by Dye et al. 2008 (https://arxiv.org/abs/0804.4002), equation (5).

We now compute each term in turn.

__Chi Squared__

The first term is the $\chi^2$ statistic from the merit function above, computed as:

 - `model_data` = the reconstructed image of the galaxy (plus any light profile model images, zero here).
 - `residual_map` = (`data` - `model_data`)
 - `normalized_residual_map` = (`data` - `model_data`) / `noise_map`
 - `chi_squared_map` = (`normalized_residual_map`) ** 2.0
 - `chi_squared` = sum(`chi_squared_map`)

High chi-squared values indicate image pixels the reconstruction fits poorly, lowering the likelihood.
"""
model_image = mapped_reconstructed_data

residual_map = masked_dataset.data - model_image
normalized_residual_map = residual_map / masked_dataset.noise_map
chi_squared_map = normalized_residual_map**2.0

chi_squared = np.sum(chi_squared_map)

print(chi_squared)

"""
__Regularization Term__

The second term, $s^{T} H s$, is the $\lambda \, G_{L}$ regularization penalty evaluated at the solution: the
summed difference in flux between all neighbouring mesh pixels, weighted by the regularization coefficient (which
is already folded into $H$).

Less smooth solutions have larger values of this term and therefore lower likelihoods.
"""
regularization_term = np.matmul(
    reconstruction.T, np.matmul(regularization_matrix, reconstruction)
)

print(regularization_term)

"""
__Complexity Terms__

Up to this point, nothing has justified our choice of `regularization_coefficient=1.0`. We cannot choose it using
the two terms above, because increasing the coefficient smooths the solution more, which *both* worsens the
chi-squared *and* (for a fixed solution) raises the regularization penalty. Optimizing those two terms alone would
drive the coefficient to zero and put us right back at the over-fitted mess.

The two log determinant terms, $\mathrm{ln} \, [ \mathrm{det} (F + H) ]$ and $- \mathrm{ln} \, [ \mathrm{det} (H) ]$,
fix this. Together they measure how *complex* the galaxy reconstruction is -- roughly, how many effective degrees
of freedom the mesh uses after regularization correlates its pixels -- and penalize more complex solutions.
Lowering the regularization coefficient frees the mesh to use more of its flexibility, increasing this complexity
penalty.

These terms therefore counteract the chi-squared and regularization terms, so the highest evidence goes to
solutions which fit the data well with the *simplest* galaxy reconstruction. This is the Occam's razor behaviour
that tutorial 4 demonstrated empirically.
"""
log_curvature_reg_matrix_term = np.linalg.slogdet(curvature_reg_matrix)[1]
log_regularization_matrix_term = np.linalg.slogdet(regularization_matrix)[1]

print(log_curvature_reg_matrix_term)
print(log_regularization_matrix_term)

"""
__Noise Normalization Term__

The likelihood function assumes the imaging data consists of independent Gaussian noise in every image pixel, and
the final term is the normalization of those Gaussians: the sum of the log of every noise-map value squared.

Because the noise-map is fixed, this term is constant throughout galaxy modeling and has no impact on the model
we infer -- it simply normalizes the likelihood.
"""
noise_normalization = float(np.sum(np.log(2 * np.pi * masked_dataset.noise_map**2.0)))

print(noise_normalization)

"""
__Log Evidence__

We can now combine the five terms into the log evidence of the galaxy reconstruction.
"""
log_evidence = float(
    -0.5
    * (
        chi_squared
        + regularization_term
        + log_curvature_reg_matrix_term
        - log_regularization_matrix_term
        + noise_normalization
    )
)

print(log_evidence)

"""
__Fit__

Everything above is what the `FitImaging` object does internally when it fits a galaxy with a pixelization. We can
see this by performing the fit and comparing its `log_evidence` to ours.

The two values do not agree exactly, because the real fit improves on our simplified solve in two ways mentioned
along the way: it uses the positive-only solver (tutorial 3), which forbids the negative mesh-pixel fluxes our
unconstrained `np.linalg.solve` permits, and it zeroes the pixels at the edge of the mesh. Our unconstrained solve
exploits that extra (unphysical) freedom to push its chi-squared lower than the real solver allows, which is why
our by-hand log evidence comes out somewhat higher. Neither constraint changes the formalism -- the same $f$, $D$,
$F$ and $H$ feed a solver with extra conditions on $s$.
"""
galaxies = ag.Galaxies(galaxies=[galaxy])

fit = ag.FitImaging(dataset=masked_dataset, galaxies=galaxies)

print(fit.log_evidence)

aplt.subplot_fit_imaging(fit=fit)

"""
__Wrap Up__

We have walked through the complete linear algebra of a pixelized galaxy reconstruction:

 - The `mapping_matrix` and PSF-blurred mapping matrix $f$, whose columns are the blurred images of each
   mesh pixel.

 - The data vector $D$ and curvature matrix $F$, the noise-weighted overlaps of those images with the data and with
   each other.

 - The regularization matrix $H$, which encodes the smoothness prior, and the linear solve $s = [F + H]^{-1} D$ for
   the galaxy reconstruction.

 - The five terms of the Suyu et al. 2006 log evidence -- chi-squared, the regularization penalty, the two log
   determinant complexity terms and the noise normalization -- and their Bayesian interpretation as an Occam's
   razor which favours the simplest galaxy reconstruction the data allows.

During galaxy modeling, this whole calculation is one likelihood evaluation: the non-linear search varies the
model's parameters (for example the mesh resolution, the regularization coefficient, or the light profiles fitted
alongside the mesh), and each sample triggers the full solve and evidence computation above.

Two simplifications are worth remembering: real fits use over sampling (each image pixel contributes several
sub-pixel rows to $f$) and **PyAutoGalaxy** uses a positive-only solver for $s$ rather than the unconstrained
`np.linalg.solve` used here (see tutorial 3). The workspace
guide `autogalaxy_workspace/*/imaging/features/pixelization/likelihood_function.ipynb` repeats this walk-through
with additional visualization of every step, including how light profiles combine with the mesh.

In the next tutorial, we return to hands-on territory and use pixelizations in an actual model-fit, combining light
profiles and an inversion via search chaining.
"""
