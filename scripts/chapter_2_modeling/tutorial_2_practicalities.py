"""
Tutorial 2: Practicalities
==========================

A galaxy can contain a compact bulge and an extended disk. We now fit both, while learning how to run a search,
manage its memory use and inspect its output. These practical skills let us assess a model as the fit progresses.

__Contents__

- **Dataset and Mask:** Inspect a galaxy with a bulge and disk.
- **Model:** Make the model more realistic.
- **Search Settings:** Balance exploration, batching and output updates.
- **Analysis and Run Times:** Understand the cost of a fit.
- **VRAM Use:** Check memory before fitting.
- **Output Folder:** Find summaries, images and saved results.
- **Unique Identifier:** Understand separate fits and resuming.
- **Results and Configuration:** Inspect the fit and find further tools.
"""

# from autogalaxy import setup_notebook; setup_notebook()

from pathlib import Path
import numpy as np
import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset and Mask__

Load the imaging data and retain the region inside the circular mask. The simulator creates the data if needed.
The image, noise map and PSF together describe the observation; none specifies which galaxy model we must fit.
"""

dataset_name = "simple"
dataset_path = Path("dataset") / "imaging" / dataset_name
if ag.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run([sys.executable, "scripts/simulators/simple.py"], check=True)

dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=2.5
)
dataset = dataset.apply_mask(mask=mask)

"""
__Over Sampling__

The bright centre changes rapidly across a pixel. Adaptive over sampling evaluates it on a finer grid so that
numerical inaccuracies do not masquerade as poor models. The outskirts need fewer evaluations.
"""

over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[8, 4, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)
dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Model__

Look at the bright central light and the more extended emission around it. A single Sersic profile need not
capture both structures. We represent the bulge with a Sersic and the disk with an Exponential profile, as in the
simulator. Real galaxies need not separate this neatly, but this controlled example lets us practise modelling.

Here we use ordinary light profiles, so both intensities are searched along with the shapes. There are 13 free
parameters: seven for the bulge and six for the disk. Greater realism has also given the search more to explore.
Tutorial 4 returns to linear profiles and shows how to reduce this burden.
"""

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=ag.lp.Sersic, disk=ag.lp.Exponential)
model = af.Collection(galaxies=af.Collection(galaxy=galaxy))
print(model.info)

"""
__Search Settings__

We use 200 live points for this more complex model. More live points generally improve exploration at the cost
of more work; fewer may be faster but can miss good solutions. The appropriate value depends on the model and
data, and must be judged from the results rather than a universal rule.

`n_batch=50` lets the search evaluate batches of trial models using the accelerated likelihood. This can make good
use of a GPU, but larger batches need more memory. Batching does not replace adequate exploration with live points.

`path_prefix` groups related fits, `name` identifies this tutorial and `unique_tag` identifies the dataset.
Together they organise the output folders described below.

__Iterations Per Update__

`iterations_per_quick_update=2500` controls quick output updates after roughly this many likelihood evaluations.
Full updates have their own setting, `iterations_per_full_update`. The two update types do different amounts of
work, so this does not mean that every saved product is regenerated every 2500 evaluations.

Frequent plotting and disk writes can dominate a fast fit. Increase the update interval if output takes too much
of the runtime; decrease it if you need more frequent feedback while diagnosing a fit.
"""

search = af.Nautilus(
    path_prefix=Path("howtogalaxy", "chapter_2"),
    name="tutorial_2_practicalities",
    unique_tag=dataset_name,
    n_live=200,
    n_batch=50,
    iterations_per_quick_update=2500,
)

"""
__Analysis__

The model defines possible bulges and disks; `AnalysisImaging` compares their predicted light with this masked
dataset. The search itself does not need to know how a galaxy image is calculated.
"""

analysis = ag.AnalysisImaging(dataset=dataset, use_jax=True)

"""
__Run Times__

Two important costs are how expensive the likelihood is to evaluate and how many evaluations the search needs.
More pixels, finer over sampling and more elaborate models can increase the first; difficult parameter spaces
and more thorough search settings can increase the second.

With JAX, compilation adds an initial cost, after which compiled calculations are reused. Batches can evaluate
several models together, so timing one ordinary `FitImaging` call is not a reliable prediction of search runtime.
Use a representative run on your own machine and inspect `search.summary`, allowing for compilation, search work
and output overhead. There is no fixed runtime or evaluations-per-parameter rule that applies to every fit.

__VRAM Use__

A GPU has a finite amount of memory, often called VRAM. Larger datasets, models and batches need more of it.
The following estimates memory for the likelihood calculation at the chosen batch size:
"""

analysis.print_vram_use(model=model, batch_size=search.batch_size)

"""
This is a likelihood estimate, not a promise about the total memory required by the process. Compilation,
other arrays and other programs also need space. On a CPU, available system memory is the relevant limit.
If memory is tight, try reducing `n_batch` from 50 to 10 before removing useful data or simplifying the science.

__Model Fit__

Run the fit below. A notebook cell waits until it finishes, but output appears on disk while it runs.
You can instead run `python scripts/chapter_2_modeling/tutorial_2_practicalities.py` from the repository root.
Scripts and notebooks save to the same output location when run with the same settings and working directory.
A script does not inherently make the inference faster; it is often more convenient for long unattended fits.
"""

result = search.fit(model=model, analysis=analysis)
print(result.info)

"""
__Output Folder__

Open `output` in the repository root. The layout below shows the main products of a completed fit; the exact
images and optional files depend on configuration and which updates have run.

```text
output/
+-- howtogalaxy/
    +-- chapter_2/
        +-- simple/                         # unique_tag
            +-- tutorial_2_practicalities/  # name
                +-- <unique_identifier>/
                    +-- model.info
                    +-- model.results
                    +-- search.summary
                    +-- search.log
                    +-- image/
                    |   +-- dataset.png
                    |   +-- dataset.fits
                    |   +-- fit.png
                    |   +-- fit.fits
                    +-- files/
                        +-- model.json
                        +-- galaxies.json
                        +-- search.json
                        +-- samples.csv
                        +-- samples_info.json
                        +-- samples_summary.json
                        +-- covariance.csv
                        +-- dataset_identity.json
                        +-- search_internal/
```

`model.info` records the model and priors. `model.results` summarises inferred parameters and uncertainties.
`image/fit.png` lets you inspect the current best fit without writing Python. `search.summary` helps assess search
progress and runtime, while `search.log` records messages. The `files` directory stores machine-readable results;
search-specific checkpoint files support resuming an interrupted fit.

__Unique Identifier__

The identifier is derived from identifying model and search settings, including the tag. It is not a random name.
Repeating the same setup finds the same output, allowing an interrupted search to resume when its checkpoint is
intact, or a completed result to load rather than run again. Changing identifying settings creates another folder.

`unique_tag=dataset_name` keeps datasets separate. The identifier does not automatically hash every data pixel,
mask choice or Analysis setting. If you change these while keeping the same model and search, use a new name or
tag so that you cannot accidentally reuse an old result. Keep enough information to reproduce each analysis.

__Results__

The result gives us the maximum-likelihood fit and galaxies, plus samples from which parameter estimates and
uncertainties are calculated. For this course, our priority is learning to obtain reliable fits.
"""

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
Inspect the central bulge, disk and residuals. A visually good fit is encouraging, but another search may find a
better solution. The next tutorial explores that failure mode.

Once you can fit real data reliably, use `autogalaxy_workspace/scripts/guides/results/start_here.py` to learn
how to load, manipulate and combine results, including database tools for larger samples.

__Parameter Plots__

The output also includes corner plots showing ranges of plausible parameters and relationships between them.
For example, the bulge's intensity and effective radius can trade off while producing similar images.
The `corner_anesthetic` plotting function can show these distributions. With `latent=True`, it displays derived
quantities such as axis ratio and position angle instead of the sampled elliptical components.

Labels are controlled by `config/notation.yaml`. Component names become superscripts, distinguishing the bulge
from the disk. Search plot switches, including the optional latent corner plot, are in
`config/visualize/plots_search.yaml`. These plots become useful once you have checked the fit itself.

__Configuration and Storage__

The `config` directory controls defaults for priors, search settings, plotting and output. The
`info_whitespace_length` setting in `config/general.yaml` adjusts the spacing of model and result summaries.
Restart a notebook kernel when changing configuration that is loaded at startup.

`config/output.yaml` controls which products are retained. In particular, `search_internal: false` removes
search internals after completion to save space; keep them if you need the search-specific state later.
Do not remove the checkpoints of an unfinished fit you intend to resume.

__Wrap Up__

We have fitted a more realistic model and learnt where to inspect its progress, results and resource use.
Now we will deliberately give the search too little exploration and learn how to diagnose and improve the fit.
"""
