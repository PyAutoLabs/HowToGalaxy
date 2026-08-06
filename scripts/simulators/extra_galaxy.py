"""
Simulator: Extra Galaxy
=======================

This script simulates `Imaging` of a galaxy using light profiles where:

 - The main galaxy's bulge is an `Sersic`.
 - There is one extra galaxy nearby, whose light is an `ExponentialSph`, located a few arc-seconds
   from the main galaxy.

This dataset is used in chapter 4 tutorial 1, which shows the two approaches to dealing with an
extra galaxy whose light contaminates the fit: noise scaling its emission out of the data, or
including it in the model via the extra galaxies API.

__Contents__

- **Dataset Paths:** Set the output path for the simulated dataset.
- **Grid:** Create a 2D grid with adaptive over-sampling for simulation.
- **Galaxies:** Define the main galaxy and the nearby extra galaxy used for simulation.
- **Output:** Save the simulated dataset to FITS files.
- **Visualize:** Output subplot and image PNGs of the simulated dataset.
- **Mask Extra Galaxies:** Build and save `mask_extra_galaxies.fits` covering the extra galaxy's light.
- **Plane Output:** Save the Galaxies object as a JSON file.
- **Extra Galaxies Centres:** Save the extra galaxy's (y,x) centre as a JSON file.

__Start Here Notebook__

If any code in this script is unclear, refer to the `simulators/simple.ipynb` notebook.
"""

# from autogalaxy import setup_notebook; setup_notebook()

from pathlib import Path

import numpy as np

import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a descriptive name.
"""
dataset_type = "imaging"
dataset_name = "extra_galaxy"

dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Grid__

Simulate the image using a (y,x) grid with the adaptive over sampling scheme.

The grid is larger than the 100 x 100 pixels used by other simulators in this folder, so that the extra galaxy,
which is offset from the main galaxy centre of (0.0", 0.0"), is comfortably contained in the image.

The adaptive over sampling scheme is centred on both the main galaxy and the extra galaxy, ensuring the light of
both is over sampled accurately.
"""
grid = ag.Grid2D.uniform(
    shape_native=(150, 150),
    pixel_scales=0.1,
)

extra_galaxy_centre = (1.0, 3.5)

over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0), extra_galaxy_centre],
)

grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

"""
Simulate a simple Gaussian PSF for the image.
"""
psf = ag.Convolver.from_gaussian(
    shape_native=(11, 11), sigma=0.1, pixel_scales=grid.pixel_scales
)

"""
Create the simulator for the imaging data, which defines the exposure time, background sky, noise levels and psf.
"""
simulator = ag.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
)

"""
__Galaxies__

Setup the main galaxy with a bulge (elliptical Sersic) for this simulation.
"""
galaxy = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=1.0,
        effective_radius=0.8,
        sersic_index=4.0,
    ),
)

"""
Setup the extra galaxy, a smaller galaxy whose light is a spherical Exponential profile, offset a few arc-seconds
from the main galaxy.

Note that its redshift is the same as the main galaxy, which is not necessarily the case in real observations.
If it is at a different redshift, the tools for masking or modeling extra galaxies are equipped to handle this.
"""
extra_galaxy = ag.Galaxy(
    redshift=0.5,
    light=ag.lp.ExponentialSph(
        centre=extra_galaxy_centre, intensity=2.0, effective_radius=0.5
    ),
)

"""
Use these galaxies to generate the image for the simulated `Imaging` dataset.
"""
galaxies = ag.Galaxies(galaxies=[galaxy, extra_galaxy])
aplt.plot_array(array=galaxies.image_2d_from(grid=grid), title="Image")

"""
Pass the simulator galaxies, which creates the image which is simulated as an imaging dataset.
"""
dataset = simulator.via_galaxies_from(galaxies=galaxies, grid=grid)

"""
Plot the simulated `Imaging` dataset before outputting it to fits.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Output__

Output the simulated dataset to the dataset path as .fits files.
"""
aplt.fits_imaging(
    dataset=dataset,
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    overwrite=True,
)

"""
__Visualize__

Output a subplot of the simulated dataset, the image and the galaxies quantities to the dataset path as .png files.
"""
aplt.subplot_imaging_dataset(
    dataset=dataset, output_path=dataset_path, output_format="png"
)
aplt.plot_array(
    array=dataset.data, title="Data", output_path=dataset_path, output_format="png"
)

aplt.subplot_galaxies(
    galaxies=galaxies, grid=grid, output_path=dataset_path, output_format="png"
)

"""
__Mask Extra Galaxies__

Build and output a `mask_extra_galaxies.fits` covering the extra galaxy's emission, so that the chapter 4
tutorial which uses this dataset can load the mask directly without a separate data-preparation step. For real
data, a user would create this mask themselves by inspecting the image (the `autogalaxy_workspace`'s
`data_preparation` package includes a GUI for drawing it).

The circle is sized to ~3x the extra galaxy's `effective_radius`, which comfortably covers the light extent of
the `ExponentialSph` profile used above. The geometry is derived from the same centre + radius defined for the
extra galaxy in this script, so it stays in sync with any future tweak to those values.

`Mask2D.circular` honours the `PYAUTO_SMALL_DATASETS=1` env var, so the mask automatically shrinks alongside the
small-dataset image and never raises an out-of-bounds error.
"""
extra_galaxies_mask = np.zeros(dataset.shape_native, dtype=bool)

for centre, radius in [
    (extra_galaxy_centre, 3.0 * 0.5),
]:
    circle = ag.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        centre=centre,
        radius=radius,
        invert=True,  # True inside the circle (i.e. masked region)
    )
    extra_galaxies_mask = np.logical_or(extra_galaxies_mask, circle.native)

mask_extra_galaxies = ag.Mask2D(
    mask=extra_galaxies_mask,
    pixel_scales=dataset.pixel_scales,
)

aplt.fits_array(
    array=mask_extra_galaxies,
    file_path=dataset_path / "mask_extra_galaxies.fits",
    overwrite=True,
)

"""
__Plane Output__

Save the `Galaxies` in the dataset folder as a .json file, ensuring the true light profiles and galaxies
are safely stored and available to check how the dataset was simulated in the future.

This can be loaded via the method `galaxies = ag.from_json()`.
"""
ag.output_to_json(
    obj=galaxies,
    file_path=Path(dataset_path, "galaxies.json"),
)

"""
__Extra Galaxies Centres__

Save the (y,x) centre of the extra galaxy as a `Grid2DIrregular` JSON file. The chapter 4 tutorial loads this
file to fix the extra galaxy's light profile centre when composing the model. For real data, a user would mark
these centres themselves on the image (the `autogalaxy_workspace`'s `data_preparation` package shows how).
"""
extra_galaxies_centres = ag.Grid2DIrregular(values=[extra_galaxy_centre])

ag.output_to_json(
    obj=extra_galaxies_centres,
    file_path=Path(dataset_path, "extra_galaxies_centres.json"),
)

"""
The dataset can be viewed in the folder `dataset/imaging/extra_galaxy`.
"""
