"""
Tutorial 2: Mappers
===================

In the previous tutorial, we used a pixelization to create a `Mapper`. However, it was not clear what a `Mapper`
does, why it was called a mapper and whether it was mapping anything at all!

Therefore, in this tutorial, we'll cover mappers in more detail.

__Contents__

- **Initial Setup:** Load the dataset for illustration.
- **Dataset Auto-Simulation:** Automatically simulate the dataset if it does not already exist.
- **Mappers:** Understand how mappers map image-plane pixels to pixelization pixels.
- **Mask:** Apply a mask and see how it affects the mapper.
- **Wrap Up:** Summary of mapper concepts.
"""

# from autogalaxy import setup_notebook; setup_notebook()

from pathlib import Path
import numpy as np
import autogalaxy as ag
import autogalaxy.plot as aplt
import autoarray.plot as aaplt

"""
__Initial Setup__

we'll use galaxy data, where:

 - The galaxy's bulge is an `Sersic`.
 - The galaxy's disk is an `Exponential`.
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

"""
Now, lets set up our `Grid2D` (using the image above).
"""
grid = ag.Grid2D.uniform(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales
)

"""
__Mappers__

We now setup a `Pixelization` and use it to create a `Mapper` via the image's grid, just like we did in
the previous tutorial.

We will make its pixelization resolution half that of the grid above.
"""
mesh = ag.mesh.RectangularBilinearAdaptDensity(
    shape=(dataset.shape_native[0] / 2, dataset.shape_native[1] / 2)
)

pixelization = ag.Pixelization(mesh=mesh)

interpolator = mesh.interpolator_from(
    source_plane_data_grid=grid, source_plane_mesh_grid=None
)

mapper = ag.Mapper(interpolator=interpolator)

"""
We now plot the `Mapper` alongside the image we used to generate the grid: the image on the left, and on the right the
rectangular mesh the image-pixel coordinates land on.
"""
aaplt.subplot_image_and_mapper(mapper=mapper, image=dataset.data)


"""
That shows us the two grids, but not the thing which links them. A mapper's actual job is to record, for every
pixelization pixel, which image-pixels fall inside it -- and we can ask it for exactly that.

`mapper.mappings_from` takes a list of pixelization-pixel index *groups* and returns one `Mapping` per group. Each
`Mapping` carries `source_contours` (the outline of the pixelization cell(s) in the group) and `image_contours` (the
outlines of the connected regions of image-pixels which map into it). Both are polygons in arc-seconds, so
`subplot_image_and_mapper` can draw them in matched colours: the cell on the right, the image-pixels it owns on the
left.

Lets map the pixelization pixels at the centre of the mesh to the image. The mesh above adapts its resolution to the
galaxy's light, so its central cells are far smaller than its outer ones -- a single central cell would be a speck on
the figure. We therefore take the 25 cells closest to the mesh centre as one group, which draws as one visible patch.
"""
mesh_grid = np.asarray(mapper.source_plane_mesh_grid)

distances = np.hypot(mesh_grid[:, 0], mesh_grid[:, 1])

pix_indexes = [np.argsort(distances)[:25]]

mappings = mapper.mappings_from(pix_indexes=pix_indexes)

aaplt.subplot_image_and_mapper(mapper=mapper, image=dataset.data, regions=mappings)

"""
On the right, a red patch of the pixelization. On the left, the same red marks the image-pixels which map into it,
sitting exactly on the galaxy's bright centre. The colour is the statement: these image-pixels and those cells are the
same thing, seen from the two sides of the mapper.

Notice that the image-plane patch is a little wider than the cells strictly contain. This is because the pairing is
not one-to-one: a bilinear interpolation scheme is used, so an image-pixel which lands just outside a cell is still
paired with it, with a weight.

Okay, so I think we can agree, mapper's map things! More specifically, they map pixelization pixels to multiple pixels 
in the observed image of a galaxy.

__Mask__

Finally, lets repeat the steps that we performed above, but now using a masked image. By applying a `Mask2D`, the 
mapper only maps image-pixels that are not removed by the mask. This removes the (many) image pixels at the edge of the 
image, where the galaxy is not present.

Lets just have a quick look at these edges pixels:

Lets use an circular `Mask2D`, which will capture the central galaxy light and clumps.
"""
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=2.0
)

dataset = dataset.apply_mask(mask=mask)
aplt.plot_array(array=dataset.data, title="Data")

"""
We can now use the masked grid to create a new `Mapper` (using the same rectangular pixelization as before).
"""
interpolator = mesh.interpolator_from(
    source_plane_data_grid=dataset.grids.pixelization, source_plane_mesh_grid=None
)
mapper = ag.Mapper(interpolator=interpolator)

"""
Lets plot it.
"""
aaplt.subplot_image_and_mapper(mapper=mapper, image=dataset.data)

"""
First, we can see a red circle of dots in both the image and pixelization, showing where the edge of the mask
maps to in the pixelization.

Now lets show that when we draw pixelization pixels, they still appear in the same place in the image. We take the 100
cells closest to the mesh centre and split them into four groups of 25, working outwards, so each group is drawn in
its own colour.
"""
mesh_grid = np.asarray(mapper.source_plane_mesh_grid)

distances = np.hypot(mesh_grid[:, 0], mesh_grid[:, 1])

order = np.argsort(distances)

pix_indexes = [order[index * 25 : (index + 1) * 25] for index in range(4)]

mappings = mapper.mappings_from(pix_indexes=pix_indexes)

aaplt.subplot_image_and_mapper(mapper=mapper, image=dataset.data, regions=mappings)

"""
Four groups, four colours, and each colour appears in exactly one place in the image -- unlike a strong lens, where
the same source region appears in several images, an unlensed galaxy maps one-to-one. Groups which are neighbours in
the pixelization own image-pixels which are neighbours in the image, so the colours nest outwards from the galaxy's
centre in both panels.

The mask has not moved anything: it has only removed the image-pixels outside it, so the cells beyond the mask's edge
now own nothing at all.

__Wrap Up__

In this tutorial, we learnt about mappers, and we used them to understand how the image and pixelization map to one 
another. Your exercises are:
        
 1) Think about how this could help us actually model galaxies. We have said we're going to reconstruct our galaxies
 on the pixel-grid. So, how does knowing how each pixel maps to the image actually help us? If you`ve not got
 any bright ideas, then worry not, that's exactly what we're going to cover in the next tutorial.
"""
