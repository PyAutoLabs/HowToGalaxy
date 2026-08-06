"""
Tutorial 3: Inversions
======================

In the previous two tutorials, we introduced:

 - `Pixelization`'s: which place a pixel-grid over the image data.
 - `Mappers`'s: which describe how each pixelization pixel maps to one or more image pixels.

However, none of this has actually helped us fit galaxy data or reconstruct the galaxy. This is the subject
of this tutorial, where the process of reconstructing the galaxy's light on the pixelization is called an `Inversion`.

__Contents__

- **Initial Setup:** Load the dataset for illustration.
- **Dataset Auto-Simulation:** Automatically simulate the dataset if it does not already exist.
- **Pixelization:** Create a pixelization and perform an inversion to reconstruct the galaxy.
- **Positive Only Solver:** Ensure the reconstruction has only positive intensity values.
- **Wrap Up:** Summary of inversion concepts.
- **Detailed Explanation:** In-depth explanation of the linear algebra behind inversions.
"""

# from autogalaxy import setup_notebook; setup_notebook()

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt
import autoarray.plot as aaplt

"""
__Initial Setup__

we'll use the same galaxy data as the previous tutorial, where:

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
Lets create a circular mask which contains the galaxy's emission:
"""
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=2.0
)

"""
We now create the masked imaging, as we did in the previous tutorial.
"""
dataset = dataset.apply_mask(mask=mask)

aplt.plot_array(array=dataset.data, title="Data")

"""
we again use the rectangular pixelization to create the mapper.

(Ignore the regularization input below for now, we will cover this in the next tutorial).
"""
mesh = ag.mesh.RectangularAdaptDensity(shape=(25, 25))

pixelization = ag.Pixelization(mesh=mesh)

interpolator = mesh.interpolator_from(
    source_plane_data_grid=dataset.grids.pixelization,
    source_plane_mesh_grid=None,
)

mapper = ag.Mapper(
    interpolator=interpolator,
    regularization=ag.reg.Constant(coefficient=1.0),
)

aaplt.subplot_image_and_mapper(mapper=mapper, image=dataset.data)

"""
__Pixelization__

Finally, we can now use the `Mapper` to reconstruct the galaxy via an `Inversion`. I'll explain how this works in a 
second, but lets just go ahead and create the inversion first. 
"""
inversion = ag.Inversion(dataset=dataset, linear_obj_list=[mapper])

"""
The inversion has reconstructed the galaxy's light on the rectangular pixel grid, which is called the 
`reconstruction`. 

This reconstruction can be mapped back to the same resolution as the image to produce the `mapped_reconstructed_operated_data`.
"""
print(inversion.reconstruction)
print(inversion.mapped_reconstructed_operated_data)

"""
Both of these can be plotted using `subplot_of_mapper`.

It is possible for an inversion to have multiple `Mapper`'s, therefore for certain figures we specify the index 
of the mapper we wish to plot. In this case, because we only have one mapper we specify the index 0.
"""
aplt.plot_array(
    array=inversion.mapped_reconstructed_operated_data, title="Reconstruction"
)
aaplt.subplot_of_mapper(inversion=inversion, mapper_index=0)

"""
There we have it, we have successfully reconstructed the galaxy using a rectangular pixel-grid, capturing the
structure of its light without assuming an analytic form for it.

Pretty great, huh? If you fitted this galaxy via search chaining at the end of chapter 2 (tutorials 9-10), you'll
remember how much effort a good model image took. With an inversion, we can do this with ease and without having to
perform model-fitting with 20+ parameters for the galaxy's light!

We will now briefly discuss how an inversion actually works, however the explanation given in this tutorial will be
overly-simplified. (The full linear algebra -- the mapping matrices, the linear solve and the equations behind
them -- is derived step-by-step in tutorial 5.) To be good at modeling you do not need to understand the details of
how an inversion works, you simply need to be able to use an inversion to model a galaxy.

To begin, lets consider some random mappings between our mapper`s pixelization pixels and the image.
"""
pix_indexes = [[445], [285], [313], [132], [11]]

indexes = mapper.slim_indexes_for_pix_indexes(pix_indexes=pix_indexes)

aaplt.subplot_image_and_mapper(mapper=mapper, image=dataset.data)

"""
These mappings are known before the inversion reconstructs the galaxy, which means before this inversion is performed 
we know two key pieces of information:

 1) The mappings between every pixelization pixel and a set of image-pixels.
 2) The flux values in every observed image-pixel, which are the values we want to fit successfully.

It turns out that with these two pieces of information we can linearly solve for the set of pixelization pixel fluxes 
that best-fit (e.g. maximize the log likelihood) our observed image. Essentially, we set up the mappings between
pixelization and image pixels as a large matrix and solve for the pixelization pixel fluxes in an analogous fashion to 
how you would solve a set of simultaneous linear equations. This process is called a `linear inversion`.

There are three more things about a linear inversion that are worth knowing:

 1) When performing fits using light profiles, we discussed how a `model_image` was generated by convolving the light
 profile images with the data's PSF. A similar blurring operation is incorporated into the inversion, such that it 
 reconstructs a galaxy (and therefore image) which fully accounts for the telescope optics and effect of the PSF.

 2) You may be familiar with image sub-gridding, which splits each image-pixel into a sub-pixel (if you are not 
 familiar then feel free to checkout the optional **HowToGalaxy** tutorial on sub-gridding. If a sub-grid is used, it is 
 the mapping between every sub-pixel -pixel that is computed and used to perform the inversion. This prevents 
 aliasing effects degrading the image reconstruction. By default **PyAutoGalaxy** uses sub-gridding of degree 4x4.

 3) The inversion`s solution is regularized. But wait, that`s what we'll cover in the next tutorial!

Finally, let me show you how easy it is to fit an image with an `Inversion` using a `FitImaging` object. Instead of 
giving the galaxy a light profile, we simply pass it a `Pixelization` and regularization, and pass it to a 
galaxies.
"""
pixelization = ag.Pixelization(
    mesh=ag.mesh.RectangularAdaptDensity(shape=(25, 25)),
    regularization=ag.reg.Constant(coefficient=1.0),
)

galaxy = ag.Galaxy(redshift=1.0, pixelization=pixelization)

galaxies = ag.Galaxies(galaxies=[galaxy])

"""
Then, like before, we pass the imaging and galaxies `FitImaging` object. 

We see some pretty good looking residuals, albeit there is faint flux leftover. We will consider how we can address 
this in the next tutorial. 

We can use the `subplot_of_galaxies` method to specifically visualize the inversion and plot the reconstruction.
"""
fit = ag.FitImaging(dataset=dataset, galaxies=galaxies)

aplt.subplot_fit_imaging(fit=fit)

"""
__Positive Only Solver__

All pixelized galaxy reconstructions use a positive-only solver, meaning that every pixelization pixel is only
allowed to reconstruct positive flux values. This ensures that the reconstruction is physical and that we don't
reconstruct negative flux values that don't exist in the real galaxy (a common unphysical systematic in methods
without this constraint).

It may be surprising to hear that this is a feature worth pointing out, but it turns out setting up the linear algebra
to enforce positive reconstructions is difficult to make efficient. A lot of development time went into making this
possible, where a bespoke fast non-negative linear solver was developed to achieve this.

Other methods in the literature often do not use a positive only solver, and therefore suffer from these
unphysical solutions, which can degrade the results of galaxy models in general.

__Wrap Up__

And, we're done, here are a few questions to get you thinking about inversions:

 1) The inversion provides the maximum log likelihood solution to the observed image. Is there a problem with seeking 
 the highest likelihood solution? Is there a risk that we're going to fit other things in the image than just the 
 galaxy? What happens if you reduce the `coefficient` of the regularization object above to zero?

 2) The exterior pixels in the rectangular pixel-grid have no image-pixels in them. However, they are still given a 
 reconstructed flux. Given these pixels do not map to the data, where is this value coming from?
 
__Detailed Explanation__

The linear algebra sketched above -- setting up the mappings as a matrix and solving for the pixelization pixel
fluxes -- is derived in full in tutorial 5 of this chapter, where we build every matrix by hand and perform the
solve ourselves. The file `autogalaxy_workspace/*/imaging/features/pixelization/likelihood_function.ipynb` gives a
further visual step-by-step guide of the process alongside equations and references to literature on the subject.
"""
