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
mesh = ag.mesh.RectangularAdaptDensity(
    shape=(dataset.shape_native[0] / 2, dataset.shape_native[1] / 2)
)

pixelization = ag.Pixelization(mesh=mesh)

interpolator = mesh.interpolator_from(
    source_plane_data_grid=grid, source_plane_mesh_grid=None
)

mapper = ag.Mapper(interpolator=interpolator)

"""
We now plot the `Mapper` alongside the image we used to generate the grid.

Using the `Visuals2D` object we are also going to highlight specific grid coordinates certain colors, such that we
can see how they map from the image grid to the pixelization grid. 
"""
indexes = [range(250), [150, 250, 350, 450, 550, 650, 750, 850, 950, 1050]]

aaplt.subplot_image_and_mapper(mapper=mapper, image=dataset.data)


"""
Using a mapper, we can now make these mappings appear the other way round. That is, we can input a pixelization pixel
index (of our rectangular grid) and highlight how all of the image-pixels that it contains map to the image-plane. 

Lets map a central source pixel to the image. We observe that for a given rectangular pixelization
pixel, there are four image pixels.
"""
pix_indexes = [[max(0, mapper.pixels // 2 - 1)]]

indexes = mapper.slim_indexes_for_pix_indexes(pix_indexes=pix_indexes)

aaplt.subplot_image_and_mapper(mapper=mapper, image=dataset.data)

"""
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

Now lets show that when we plot pixelization pixel indexes, they still appear in the same place in the image.
"""
pix_indexes = []

for pix_index in range(mapper.pixels):
    if mapper.slim_indexes_for_pix_indexes(pix_indexes=[[pix_index]])[0]:
        pix_indexes.append([pix_index])

    if len(pix_indexes) == 4:
        break

indexes = mapper.slim_indexes_for_pix_indexes(pix_indexes=pix_indexes)

aaplt.subplot_image_and_mapper(mapper=mapper, image=dataset.data)

"""
__Wrap Up__

In this tutorial, we learnt about mappers, and we used them to understand how the image and pixelization map to one 
another. Your exercises are:
        
 1) Think about how this could help us actually model galaxies. We have said we're going to reconstruct our galaxies
 on the pixel-grid. So, how does knowing how each pixel maps to the image actually help us? If you`ve not got
 any bright ideas, then worry not, that's exactly what we're going to cover in the next tutorial.
"""
