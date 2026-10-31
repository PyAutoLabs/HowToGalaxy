"""
Tutorial 0: Visualization
========================

Plot a simulated image with **PyAutoGalaxy**, compare views of the dataset, and adjust the display so you
can inspect its structure. This tutorial introduces the plotting tools used throughout the course.

__Contents__

- **Directories:** Check where the tutorial looks for files.
- **Dataset:** Identify the simulated image we will use.
- **Dataset Auto-Simulation:** Generate the example if needed, load it and plot the image.
- **Subplots:** Compare several views of the dataset.
- **Plot Customization:** Adjust colour scaling and figure size.
- **Visuals:** Select a region of the image with a mask.
- **Wrap Up:** Review the plotting tools and continue to Tutorial 1.

"""

# from autogalaxy import setup_notebook; setup_notebook()

"""
__Directories__

Run this tutorial from the `HowToGalaxy` repository folder, which contains `config/`, `dataset/` and `scripts/`.
Relative paths in the tutorial are resolved from your current working directory.

The next cell prints that directory. Check that it points to your `HowToGalaxy` folder before continuing.

If you have not downloaded the tutorials, get them from [the HowToGalaxy repository](https://github.com/PyAutoLabs/HowToGalaxy).
"""
from pathlib import Path

print(Path.cwd())

"""
If the directory is incorrect, replace the example path below with the location of your repository and uncomment
the two lines. Then rerun the cell above to check the directory.
"""
# import os
# os.chdir("/path/to/HowToGalaxy")

"""
__Dataset__

We will use a simulated image of a galaxy described by a Sérsic light profile. The next tutorial explains how
this profile represents the distribution of a galaxy’s light.

The following path identifies the dataset folder within `HowToGalaxy`. If the data are missing, the next section
generates them.
"""
import autogalaxy as ag
import autogalaxy.plot as aplt

dataset_path = Path("dataset", "imaging", "simple__sersic")

"""
__Dataset Auto-Simulation__

The next cell runs the simulator if the example dataset needs to be generated. You do not need to run the
simulator separately.
"""
if ag.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    # Use the same Python interpreter as this tutorial.
    subprocess.run(
        [sys.executable, "scripts/simulators/sersic.py"],
        check=True,
    )


"""
We now load three FITS files into an `Imaging` object: the image, its noise map and the point-spread function
(PSF). The noise map describes the uncertainty in each pixel; the PSF describes how the image of a point source
is blurred.

Each pixel spans 0.1 arcseconds on each side, matching the simulator settings. Loading these files does not
fit a model to the image.
"""
dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

"""
Plot the simulated image using `aplt.plot_array()`. Look for the galaxy’s bright centre and its light fading
towards the outskirts.

The simulation includes blurring by the PSF and noise, so the image includes both instrumental blurring and random fluctuations.
"""
aplt.plot_array(array=dataset.data, title="Dataset Image")

"""
__Subplots__

A subplot brings several views of the dataset into one figure. The next cell displays the image, noise map,
PSF and signal-to-noise map, alongside additional diagnostic panels.

Compare the image with its logarithmic view, labelled “Data (log10)”. The different colour scaling helps
reveal faint structure alongside bright features; it does not change the underlying data.

For now, focus on these image panels and the noise map. We will introduce the numerical settings shown in
the other panels as they become relevant.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Plot Customization__

Change a figure's appearance by passing options to `aplt.plot_array()`. Here we use a logarithmic colour
scale to make faint structure easier to see. We create a 7-by-7-inch figure with Matplotlib and pass its axes
to the plotting function.

Compare this plot with the first image. Logarithmic scaling changes how positive pixel values map to colours;
it does not change the image or make faint features more statistically significant.
"""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 7))
aplt.plot_array(
    array=dataset.data,
    title="Dataset Image (Log10)",
    use_log10=True,
    ax=ax,
)
plt.show()
plt.close(fig)

"""
Try changing `use_log10=True` to `False` and rerun the cell to return to a linear colour scale. To enlarge
the figure, change `figsize=(7, 7)` to `(9, 9)`. Restore the starting values to reproduce the plot above.

You can also set `colormap="gray"` to display the image in greyscale. For persistent defaults, explore
`config/visualize/general.yaml`; its `colormap` entry sets the default colour map and
`subplot_shape_to_figsize_factor` controls subplot sizes. The per-call options above are enough for this lesson.

**🤖 Ask AI:** Why can logarithmic colour scaling make faint structure easier to see without increasing its
signal-to-noise ratio?
"""

"""
__Visuals__

A mask selects which pixels to use. Here an annular mask keeps a ring between 0.3 and 3.0 arcseconds from
the image centre, excluding both the bright centre and the outer image.

The next cell plots this selected region. With the default plotting settings, the display zooms to the
unmasked region. Masking selects pixels; it does not subtract the galaxy's light.
"""
mask = ag.Mask2D.circular_annular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    inner_radius=0.3,
    outer_radius=3.0,
)

masked_dataset = dataset.apply_mask(mask=mask)

aplt.plot_array(array=masked_dataset.data, title="Data With Annular Mask")

"""
Compare the masked plot with the original image: the central light is excluded from this view, not removed
from the original dataset. We keep the masked version in `masked_dataset` so `dataset` remains available.
"""

"""
__Wrap Up__

You have plotted an image, compared dataset panels and changed the display without changing the pixel values.
Use `aplt.plot_array()` for an individual image and `aplt.subplot_imaging_dataset()` for a dataset overview.

Next, open Tutorial 1: Grids and Galaxies to learn how coordinates and light profiles produce a galaxy image.
"""
