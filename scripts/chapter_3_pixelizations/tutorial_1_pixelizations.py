"""
Tutorial 1: Pixelizations, Mappers and Inversions
=================================================

Look at a resolved galaxy image. Its broad light distribution may follow a smooth bulge and disc, while smaller regions contain bright star-forming knots, off-centre clumps or irregular extensions. A model can capture the overall galaxy yet leave these features clearly visible in the residual image.

In Chapter 2, we introduced linear light profiles and multi-Gaussian expansions (MGEs). Combining Gaussians of different widths gives us a more flexible radial brightness distribution than a single Sérsic profile. Gaussian groups with different ellipticities and orientations can also describe changes in flattening and isophotal twists. Fixed widths and linearly solved intensities provide this flexibility without introducing a large number of non-linear parameters.

Our worked example nevertheless imposed a particular geometry: the Gaussians within each galaxy’s group shared a common centre and elliptical shape. Adding more Gaussians improves our control over the brightness distribution around that centre, but does not create independently positioned knots or several components with distinct centres.

We could introduce offset Gaussian groups. The limitation is not that MGEs cannot represent multiple centres, but that increasingly complicated galaxies require us to choose and fit an increasingly elaborate arrangement of components.

Suppose the MGE reproduces the smooth galaxy light but leaves compact, structured residuals. These may indicate that the model needs more freedom to describe local features. They are not automatically evidence for additional galaxy components: an inaccurate PSF, background estimate, contaminating object or other data-treatment error can also leave structured residuals.

In this chapter, we introduce pixelized reconstruction as another way to describe the galaxy’s light. We infer brightness values across a mesh without assigning a separate analytic centre and shape to every feature. The predicted image includes telescope blurring, so the reconstruction is an inferred description of the galaxy rather than simply a copy of the observed pixels.

This builds on the linear profiles of Chapter 2. Instead of solving for Gaussian intensities, we solve for the mesh brightness values. Their extra freedom can describe irregular structure, but can also fit noise. Regularization supplies a prior that controls variations between neighbouring brightness values.

The reconstruction still depends on modelling choices, including the mesh, its resolution, regularization and the treatment of the data. We will learn to inspect how those choices affect both the reconstructed galaxy and the image residuals.

MGE and other analytic profiles remain useful for describing and measuring smooth galaxy components. Pixelizations complement them when more complicated structure needs to be represented, including in models that combine analytic components with a pixelized reconstruction.

We will build the connection between image coordinates, mesh indexes, reconstructed brightnesses and the model
image. Our worked dataset is deliberately a simple smooth bulge and disc: it makes these ingredients easy to see
before applying them to more complicated galaxy structure.

__Contents__

- **The image and its coordinates:** Load and mask the data.
- **A brightness mesh:** Distinguish image indexes from mesh indexes.
- **Following the mappings:** Highlight which image pixels receive each mesh contribution.
- **An inversion:** Infer brightnesses, accounting for noise and telescope blurring.
- **A model image:** Inspect the reconstruction and residuals separately.
- **Using FitImaging:** Put these ingredients into a galaxy model.
"""

"""
__The image and its coordinates__

The simulator supplies a Sérsic bulge and exponential disc. The circular mask selects the image pixels used in the
fit. We use one sample per image pixel for the indexed illustrations; accurate scientific modelling should also
check image integration. Each unmasked pixel has a position and a flattened index. These indexes describe the data,
not the separate mesh we are about to construct.
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

"""
__A brightness mesh__

Our rectangular mesh covers the observed galaxy. There is no ray tracing: image coordinates and reconstructed
coordinates refer to the same sky. The shared mapper API nevertheless calls its reconstruction coordinates
`source_plane_data_grid` and `source_plane_mesh_grid`.

`RectangularUniform` places brightness values at regularly spaced points. An image coordinate usually falls between
four mesh points, so bilinear interpolation combines their values with position-dependent weights. An image pixel
falling inside a rectangular cell does not simply take all its brightness from one mesh point.

Let's select small groups of image pixels near two different positions. Coordinate-based selection keeps this
illustration meaningful even when a quick smoke run uses a smaller dataset.
"""

image_grid = dataset.grids.pixelization
image_coordinates = np.asarray(image_grid)
indexes = [
    np.argsort(np.sum((image_coordinates - position) ** 2, axis=1))[:5].tolist()
    for position in [(0.0, 0.5), (0.0, -0.5)]
]
aaplt.plot_grid(
    grid=image_grid, indexes=indexes, title="Selected Image-Pixel Coordinates"
)
mesh = ag.mesh.RectangularUniform(shape=mesh_shape)
interpolator = mesh.interpolator_from(
    source_plane_data_grid=image_grid, source_plane_mesh_grid=None
)
mapper = ag.Mapper(
    interpolator=interpolator, regularization=ag.reg.Constant(coefficient=1.0)
)
image_index = indexes[0][0]
weights = np.asarray(mapper.mapping_matrix[image_index])
mesh_indexes = np.flatnonzero(np.abs(weights) > 1.0e-12)
aaplt.plot_grid(
    grid=mapper.source_plane_mesh_grid,
    indexes=[mesh_indexes.tolist()],
    title=f"Mesh Points Contributing to Image Pixel {image_index}",
)
print("Image position:", image_coordinates[image_index])
for mesh_index in mesh_indexes:
    print(
        f"Mesh index {mesh_index}: position={np.asarray(mapper.source_plane_mesh_grid[mesh_index])}, weight={weights[mesh_index]:.4f}"
    )

"""
__Following the mappings__

An image-pixel index and mesh index refer to different things. The mapper stores the relationship between them.
We can follow it in the opposite direction: select mesh points and find the image pixels receiving a contribution.

Matching colours identify the selected mesh groups and their associated image regions. These are interpolation
contributions, rather than a strict claim that every image coordinate lies inside a coloured cell. Nearby mesh
values can contribute to the same image pixel; that overlap is part of interpolation.

First inspect one mesh point near the centre, then compare a pair of nearby points with an offset point. Try moving
the offset position and watch which image pixels change colour.
"""

mesh_coordinates = np.asarray(mapper.source_plane_mesh_grid)
central_index = int(np.argmin(np.sum(mesh_coordinates**2, axis=1)))
pix_indexes = [[central_index]]
mappings = mapper.mappings_from(pix_indexes=pix_indexes)
aaplt.subplot_image_and_mapper(
    mapper=mapper,
    image=dataset.data,
    regions=mappings,
    region_labels=[str(central_index)],
    mesh_grid=mapper.source_plane_mesh_grid,
)
nearby_indexes = np.argsort(
    np.sum((mesh_coordinates - mesh_coordinates[central_index]) ** 2, axis=1)
)[:2].tolist()
offset_index = int(
    np.argmin(np.sum((mesh_coordinates - np.array([0.3, 0.3])) ** 2, axis=1))
)
pix_indexes = [nearby_indexes, [offset_index]]
mappings = mapper.mappings_from(pix_indexes=pix_indexes)
aaplt.subplot_image_and_mapper(
    mapper=mapper,
    image=dataset.data,
    regions=mappings,
    region_labels=[str(group) for group in pix_indexes],
    mesh_grid=mapper.source_plane_mesh_grid,
)

"""
__An inversion__

The mapper tells us how brightness would contribute to the image, but has not yet determined those brightnesses.
An inversion combines the mappings with the observed data, noise and PSF to infer the mesh values together.
Like the Gaussian intensities in Chapter 2, these values enter the predicted image linearly. Linear algebra handles
this large set of brightnesses without giving a non-linear search a free parameter for every mesh point.

We also supplied `Constant` regularization with coefficient `1.0`. This smoothness prior discourages unsupported
variations between neighbours. The next tutorial explores why it is needed and how to choose its strength.

Here there is only one mapper, so entry `j` in the reconstruction belongs to mesh index `j`. These values are jointly
inferred: interpolation and telescope blurring couple their effects on the image, while regularization couples
neighbouring brightnesses. They are not independent measurements of the selected coloured regions.
"""

inversion = ag.Inversion(dataset=dataset, linear_obj_list=[mapper])
reconstruction = np.asarray(inversion.reconstruction)
for group in pix_indexes:
    for mesh_index in group:
        print(
            f"Mesh index {mesh_index}: reconstructed brightness={reconstruction[mesh_index]:.6g}"
        )
aaplt.plot_mapper(
    mapper=mapper,
    solution_vector=reconstruction,
    mesh_grid=mapper.source_plane_mesh_grid,
    regions=[mapping.source_contours for mapping in mappings],
    region_labels=[str(group) for group in pix_indexes],
    title="Reconstructed Galaxy and Selected Mesh Indexes",
)

"""
__A model image__

The brightness reconstruction lives on the mesh. Mapping it onto the image grid and convolving with the PSF produces
the model image that we compare to the observations. Telescope blurring is already included in the inversion's solve;
it is not an extra correction applied after fitting an unblurred model to blurred data.

The next figures distinguish the unblurred prediction, blurred model and residuals. A compact reconstructed feature
can be less distinct in the observed image because the PSF spreads its light. Inspect coherent residuals as well as
the reconstruction: a plausible-looking galaxy does not establish that the model explains the data.
"""

image_before_blurring = inversion.mapped_reconstructed_data
model_image = inversion.mapped_reconstructed_operated_data
residual_map = dataset.data - model_image
aplt.plot_array(
    array=image_before_blurring, title="Reconstructed Image Before Telescope Blurring"
)
aplt.plot_array(array=model_image, title="Model Image Including Telescope Blurring")
aplt.plot_array(array=residual_map, title="Residuals: Data Minus Model")
aplt.plot_array(
    array=residual_map / dataset.noise_map, title="Residuals Divided by the Noise"
)

"""
__Using FitImaging__

We exposed the mapper and inversion to understand what they do. In everyday modelling, put a `Pixelization` in a
`Galaxy` and let `FitImaging` construct these ingredients. This evaluates a specified model; it is not a non-linear
search. The reconstruction still depends on the mesh, resolution, regularization and data treatment.

**🤖 Ask AI:** Why is a reconstructed brightness on the mesh different from a measured image-pixel value, even though
we are modelling a directly observed galaxy without lensing?
"""

pixelization = ag.Pixelization(
    mesh=ag.mesh.RectangularUniform(shape=mesh_shape),
    regularization=ag.reg.Constant(coefficient=1.0),
)
galaxy = ag.Galaxy(redshift=0.5, pixelization=pixelization)
fit = ag.FitImaging(dataset=dataset, galaxies=ag.Galaxies(galaxies=[galaxy]))
aplt.subplot_fit_imaging(fit=fit)

"""
__Wrap Up__

A mesh supplies brightness values, a mapper connects them to image coordinates, and an inversion infers them while
accounting for noise, telescope blurring and regularization. Next we investigate that regularization: a close image
fit alone does not tell us whether the inferred structure is trustworthy.
"""
