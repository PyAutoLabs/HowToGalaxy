"""
Tutorial 6: Masking
===================

We have learnt everything we need to know about non-linear searches to model a galaxy and infer a good
model solution. Now, lets consider masking in more detail, something we have not given much consideration previously.
We'll also learn a neat trick to improve the speed and accuracy of a non-linear search.

__Contents__

- **Initial Setup:** Load the dataset and apply a mask.
- **Mask:** Apply a custom mask to focus the fit on specific regions.
- **Model + Search + Analysis:** Fit the model using the custom mask.
- **Discussion:** How the mask affects the fit quality and run time.
- **Wrap Up:** Summary of masking strategies for galaxy modeling.
"""

# from autogalaxy import setup_notebook; setup_notebook()

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt
import autofit as af

"""
__Initial Setup__

we'll use the same galaxy data as tutorials 1 & 2, where:

 - The galaxy's `LightProfile` is a `Sersic`.
"""
dataset_name = "simple__sersic"
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
        [sys.executable, "scripts/simulators/sersic.py"],
        check=True,
    )


dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

"""
__Mask__

In tutorials 1 and 2 we used a 3.0" circular mask. 

However, there is very faint flux emitted at the outskirts of the galaxy, which the model will benefit from fitting
by using a larger mask.
"""
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=6.0
)

dataset = dataset.apply_mask(mask=mask)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Over Sampling__

Apply adaptive over sampling to ensure the calculation is accurate, you can read up on over-sampling in more detail via 
the `autogalaxy_workspace/*/guides/advanced/over_sampling.ipynb` notebook.
"""
over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[8, 4, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
__Model + Search + Analysis__

Lets fit the data using this mask, by creating the search as per usual. Note that the `imaging` data with this mask
applied is passed into the `AnalysisImaging` object, ensuring that this is the mask the model-fit uses. 
"""
galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=ag.lp.Sersic)
model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

search = af.Nautilus(
    path_prefix=Path("howtogalaxy", "chapter_2"),
    name="tutorial_6_with_custom_mask",
    unique_tag=dataset_name,
    n_live=80,
    n_batch=50,  # GPU batching and VRAM use explained in chapter 2 tutorial 2.
)

analysis = ag.AnalysisImaging(dataset=dataset, use_jax=True)

search.fit(model=model, analysis=analysis)

"""
__Discussion__

So, we can choose the mask we use in a model-fit. We know that we want the mask to remove as little of the galaxy's light,
but is this the 'right' mask? What is the 'right' mask? Maybe we want a bigger mask? a smaller mask?

When it comes to choosing a mask, we are essentially balancing two things: computational run-time and accuracy. When we
use a bigger mask the model-fit will take longer to perform. Why? Because a bigger mask includes more image-pixels 
in the analysis, and for every additional image-pixel we have to compute its light, blur it with the PSF, compare
it to the data, etc.
 
If run-time was not a consideration we would always choose a bigger mask, for two reasons:

 1) The galaxy may have very faint emission that when you choose the mask you simply do not notice. Overly aggressive 
 masking runs the risk of us inadvertently masking out some of the galaxy's light, which would otherwise better
 constrain the model!
    
 2) When the data is fitted with a model image, the fit is performed only within the masked region. For certain galaxies
 it is possible that it may produce extraneous emission outside of the masked region that is not actually observed in 
 the data itself. If this region had not been masked-out, the model would create residuals in these locations and 
 reduce the value of likelihood appropriately, whereas if it is masked out this reduction in likelihood is 
 not fed through to the analysis. 

As you use **PyAutoGalaxy** more you will get a feel for how fast a model-fit will run given the quality of data,
model complexity, non-linear search settings, etc. As you develop this intuition, I recommend that you always aim to 
use as large of a mask as possible (whilst still achieving reasonable run-times). Aggressive masking will make 
**PyAutoGalaxy** run very fast, but could lead you to infer an incorrect model! 

At the end of this chapter, where we introduce 'non-linear search chaining', we will see how chaining makes it
possible to use tighter masks in earlier searches to achieve faster run times.

If your data includes the light of additional galaxies nearby you may have no choice but to use a smaller
circular mask, because it is important these objects do not interfere with the fit.

In fact, you can draw custom masks that remove their light entirely. You may now wish to checkout
the `autogalaxy_workspace/*/imaging/data_preparation` package. This includes tools for creating custom masks and
marking the positions on a galaxy (via a GUI) so you can use them in a model-fit.

__Wrap Up__

There is one thing you should bear in mind in terms of masking:

 1) Customizing the mask for the analysis of one galaxy gets the analysis running fast and can provide accurate 
 non-linear sampling. However, for a large sample of galaxies, this high level of customization may take a lot of time. 
"""
