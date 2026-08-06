"""
Simulator: Interferometer
=========================

This script simulates `Interferometer` data of a galaxy, as would be observed by a radio or sub-mm
interferometer like ALMA or the JVLA, where:

 - The galaxy's bulge is an `Sersic`.

Unlike CCD imaging, an interferometer does not observe an image of the galaxy. It measures "visibilities", which
are the Fourier transform of the sky brightness sampled at a set of points in the "uv-plane", where each point
corresponds to the separation of a pair of antennas in the array.

It is used to illustrate interferometer data in the HowToGalaxy lecture series. HowToGalaxy does not cover
interferometry beyond this glimpse; the `autogalaxy_workspace/scripts/interferometer` package is the dedicated
resource for uv-plane galaxy modeling.

__Contents__

- **Dataset Paths:** Set the output path for the simulated dataset.
- **Simulate:** Simulate the image using a (y,x) real-space grid and a synthetic set of uv-plane baselines.
- **Galaxies:** Define the galaxy Sersic light profile used for simulation.
- **Output:** Save the simulated dataset to FITS files.
- **Visualize:** Output subplot and dirty-image PNGs of the simulated dataset.
- **Plane Output:** Save the Galaxies object as a JSON file.

__Start Here Notebook__

If any code in this script is unclear, refer to the `autogalaxy_workspace/*/interferometer/simulator.ipynb`
notebook.
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
dataset_type = "interferometer"
dataset_name = "simple"

dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Simulate__

For interferometer data, the galaxy's image is evaluated in real space on a (y,x) grid and then Fourier
transformed to the uv-plane, where it is compared to the observed visibilities.

Interferometers do not observe galaxies in a way where over sampling is necessary, therefore the grid uses no
over sampling.
"""
grid = ag.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.1,
)

"""
To perform the Fourier transform we need the wavelengths of the baselines, which describe where in the uv-plane
each visibility samples the sky's Fourier transform.

For real data these are determined by the layout of the interferometer's antennas. The `autogalaxy_workspace`
bundles baselines of real instruments (e.g. the Square Mile Array (SMA) and ALMA). For this tutorial dataset we
instead draw a small synthetic set of baselines from a Gaussian distribution in the uv-plane, with a scale
comparable to the SMA's (a few hundred visibilities), keeping the simulation fast and self-contained.
"""
rng = np.random.default_rng(1)

total_visibilities = 200

uv_wavelengths = rng.normal(loc=0.0, scale=1.0e5, size=(total_visibilities, 2))

"""
To simulate the interferometer dataset we first create a simulator, which defines the exposure time, noise levels
and Fourier transform method used in the simulation.

We use the `TransformerDFT`, an exact Discrete Fourier Transform which is fast for datasets with a low number of
visibilities like this one.
"""
simulator = ag.SimulatorInterferometer(
    uv_wavelengths=uv_wavelengths,
    exposure_time=300.0,
    noise_sigma=1000.0,
    transformer_class=ag.TransformerDFT,
)

"""
__Galaxies__

Setup the galaxy with a bulge (elliptical Sersic) for this simulation.
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
Use these galaxies to generate the image for the simulated `Interferometer` dataset.
"""
galaxies = ag.Galaxies(galaxies=[galaxy])
aplt.plot_array(array=galaxies.image_2d_from(grid=grid), title="Image")

"""
Pass the simulator galaxies, which creates the image plotted above and simulates it as an interferometer dataset.
"""
dataset = simulator.via_galaxies_from(galaxies=galaxies, grid=grid)

"""
Plot the simulated interferometer dataset's dirty images before outputting it to fits.
"""
aplt.subplot_interferometer_dirty_images(dataset=dataset)

"""
__Output__

Output the simulated dataset to the dataset path as .fits files.
"""
aplt.fits_interferometer(
    dataset=dataset,
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    overwrite=True,
)

"""
__Visualize__

Output a subplot of the simulated dataset's dirty images and the galaxies quantities to the dataset path as
.png files.
"""
aplt.subplot_interferometer_dirty_images(
    dataset=dataset, output_path=dataset_path, output_format="png"
)

aplt.subplot_galaxies(
    galaxies=galaxies, grid=grid, output_path=dataset_path, output_format="png"
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
The dataset can be viewed in the folder `dataset/interferometer/simple`.
"""
