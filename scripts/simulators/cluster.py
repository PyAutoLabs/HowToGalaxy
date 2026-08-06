"""
Simulator: Cluster
==================

This script simulates `Imaging` of a cluster field: a brightest cluster galaxy (BCG) surrounded by ten
lower-luminosity member galaxies. It is used in chapter 4 of the **HowToGalaxy** lectures, where the member
population is modeled via a **catalogue**: a CSV of member centres and luminosities whose photometry pins
the faint galaxies while only a shared normalization stays free.

This script simulates `Imaging` of a cluster field where:

 - The BCG's bulge is an elliptical `Sersic` (de Vaucouleurs-like).
 - Ten member galaxies have `SersicSph` light profiles whose intensities follow their catalogue
   luminosities.

__Contents__

- **Dataset Paths:** Set the output path for the simulated dataset.
- **Grid:** Create a 2D grid with adaptive over-sampling for simulation.
- **Galaxies:** Define the BCG and member galaxy light profiles used for simulation.
- **Output:** Save the simulated dataset to FITS files.
- **Member Catalogue CSV:** Write `scaling_galaxies.csv` (y, x, luminosity) — the tutorial's input.
- **Visualize:** Output subplot and image PNGs of the simulated dataset.
- **Plane Output:** Save the Galaxies object and BCG centre as JSON files.

__Start Here Notebook__

If any code in this script is unclear, refer to the `simulators/simple.ipynb` notebook.
"""

# from autogalaxy import setup_notebook; setup_notebook()

import csv
from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a descriptive name.
"""
dataset_type = "cluster"
dataset_name = "simple"

dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Grid__

The cluster field spans a much wider area than the single-galaxy datasets of earlier chapters, so the grid
is 250 x 250 pixels (25" x 25" at 0.1" per pixel).

The BCG sits at the centre of the field and the ten member galaxies are scattered across it. Their centres
and luminosities are defined here because both the light profiles and the member catalogue CSV are built
from them below.
"""
grid = ag.Grid2D.uniform(
    shape_native=(250, 250),
    pixel_scales=0.1,
)

bcg_centre = (0.0, 0.0)

member_centres = [
    (5.5, -6.5),
    (-7.5, 3.0),
    (3.0, 8.0),
    (8.0, 5.0),
    (-6.5, -8.0),
    (-2.5, 6.5),
    (7.0, -2.0),
    (-8.0, 8.5),
    (2.0, -8.5),
    (-4.0, -3.5),
]

member_luminosities = [0.40, 0.32, 0.25, 0.20, 0.16, 0.13, 0.10, 0.08, 0.06, 0.05]

"""
Simulate the image using a (y,x) grid with the adaptive over sampling scheme, centred on every galaxy in
the field (the BCG and all ten members).
"""
over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=[bcg_centre] + member_centres,
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

Setup the BCG: a bright, extended de Vaucouleurs-like Sersic at the cluster centre. In the chapter 4
cluster tutorial it is the one galaxy modeled individually, with a free MGE.
"""
bcg = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=bcg_centre,
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        intensity=1.5,
        effective_radius=2.5,
        sersic_index=4.0,
    ),
)

"""
Setup the ten member galaxies, whose central intensities equal their catalogue luminosities — so the
rendered image visibly traces the catalogue, and the tutorial's shared-normalization tier
(intensity = scale * luminosity) can recover the truth with `scale = 1`.
"""
members = []
for centre, luminosity in zip(member_centres, member_luminosities):
    members.append(
        ag.Galaxy(
            redshift=0.5,
            bulge=ag.lp.SersicSph(
                centre=centre,
                intensity=luminosity,
                effective_radius=0.6,
                sersic_index=3.0,
            ),
        )
    )

"""
Use these galaxies to generate the image for the simulated `Imaging` dataset.
"""
galaxies = ag.Galaxies(galaxies=[bcg] + members)
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
__Member Catalogue CSV__

Write the member catalogue to `scaling_galaxies.csv` with the three-column `y, x, luminosity` schema used
by the `autogalaxy_workspace` cluster package (and shared with the lensing workspace's cluster package,
where the same catalogue drives member MASSES via a scaling relation; here it drives member LIGHT). The
tutorial loads it with `ag.galaxy_table_from_csv` — the catalogue-loading API that makes the member
population a row-append away from scaling up.
"""
with open(dataset_path / "scaling_galaxies.csv", "w", newline="") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(["y", "x", "luminosity"])
    for centre, luminosity in zip(member_centres, member_luminosities):
        writer.writerow([centre[0], centre[1], luminosity])

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
__Plane Output__

Save the `Galaxies` in the dataset folder as a .json file, ensuring the true light profiles and galaxies
are safely stored and available to check how the dataset was simulated in the future.

This can be loaded via the method `galaxies = ag.from_json()`.

The BCG centre is also saved as a .json file, mirroring the real-cluster workflow where the BCG centre(s)
are recorded separately from the member catalogue.
"""
ag.output_to_json(
    obj=galaxies,
    file_path=Path(dataset_path, "galaxies.json"),
)

ag.output_to_json(
    obj=ag.Grid2DIrregular([bcg_centre]),
    file_path=Path(dataset_path, "bcg_centres.json"),
)

"""
The dataset can be viewed in the folder `dataset/cluster/simple`.
"""
