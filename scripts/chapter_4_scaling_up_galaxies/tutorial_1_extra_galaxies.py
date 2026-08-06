"""
Tutorial 1: Extra Galaxies
==========================

Welcome to chapter 4 of **HowToGalaxy**, where we scale up galaxy modeling beyond a single galaxy.

In chapters 1 to 3, every dataset we studied had the same simple anatomy: one galaxy, alone at the centre of the
image, whose light we modeled with parametric profiles, basis functions or pixelizations. This is the cleanest
possible configuration, and it was the right place to learn the fundamentals of light profiles, non-linear searches
and Bayesian inference.

Real galaxies are rarely this tidy. Galaxies live in crowded fields: a galaxy may have a faint companion a few
arc-seconds away, be blended with a close neighbour of comparable brightness, or sit in a cluster containing
hundreds of members. The science of galaxy modeling scales up through this hierarchy, and so must our modeling:

- **Extra galaxies (this tutorial)**: a single galaxy of interest, with one or more nearby interloper galaxies
  whose light may contaminate the analysis.

- **Multi-galaxy blends (next tutorial)**: systems where two or more galaxies are blended together and all are
  subjects of the analysis, so no single galaxy can be called "the" galaxy.

- **Cluster fields (final tutorial)**: images containing many galaxies, for example a brightest cluster galaxy
  surrounded by a population of cluster members, which must be modeled together.

The same ladder exists in strong gravitational lensing, where interlopers, multi-galaxy deflectors and cluster-scale
lenses complicate the analysis in analogous ways. The **HowToLens** lectures scale up lens modeling through an
equivalent chapter 4, using the same **PyAutoFit** API you will learn here.

In this first tutorial, we take the first step up in scale: a galaxy with one extra galaxy nearby. We will learn
how to decide whether the extra galaxy matters, and the two approaches to dealing with it when it does: removing
its light from the data, or including it in the model.

__Contents__

- **Initial Setup:** Load the imaging dataset of a galaxy with an extra galaxy and inspect the interloper.
- **Dataset Auto-Simulation:** Automatically simulate the dataset if it does not already exist.
- **The Decision:** The core question: does the extra galaxy's light overlap the region of the image we fit?
- **Mask:** Define a circular mask large enough to include the extra galaxy's emission.
- **Approach 1 Noise Scaling:** Remove the extra galaxy's light from the fit by scaling its data and noise values.
- **Noise Scaling Fit:** Fit a model to the noise-scaled data, without the extra galaxy in the model.
- **Approach 2 Extra Galaxies Model:** Include the extra galaxy's light in the model explicitly.
- **Extra Galaxy Centres:** Why the extra galaxy's centre is fixed to its observed light centre.
- **Extra Galaxies Model Composition:** Compose the model including the extra galaxy via the extra galaxies API.
- **Extra Galaxies Fit:** Fit the model which includes the extra galaxy.
- **Which Approach When:** Guidance on choosing between noise scaling and explicit modeling.
- **Wrap Up:** Summary of the script and next steps.
"""

from autogalaxy import jax_wrapper  # Sets JAX environment before other imports

# from autogalaxy import setup_notebook; setup_notebook()

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt
import autofit as af

"""
__Initial Setup__

Lets load the `Imaging` dataset we'll fit in this tutorial. It is similar to the `simple__sersic` dataset used
throughout chapter 2, where:

 - The galaxy's bulge is an `Sersic`.

However, there is one addition: an extra galaxy, with its own light (an `ExponentialSph`), located a few
arc-seconds from the main galaxy.
"""
dataset_name = "extra_galaxy"
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
        [sys.executable, "scripts/simulators/extra_galaxy.py"],
        check=True,
    )

dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

"""
When we plot the dataset, the familiar sight of a single galaxy at the centre of the image is joined by a blob of
light in the upper-right of the image.

This is the extra galaxy. It is not the galaxy we are studying: it is an interloper, a galaxy that happens to lie
close to ours on the sky. Interlopers like this are extremely common in real imaging of galaxies, and everyone who
models galaxies has to decide what to do about them.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__The Decision__

The decision of what to do about an extra galaxy hinges on one question:

**Does its light overlap the region of the image we fit?**

The extra galaxy's emission may extend into the region of the image we fit. If it does, and we fit a model
containing only the main galaxy, the model has no component that can produce this emission. The fit will respond by
distorting the main galaxy's light profiles to soak it up — for example inflating the `effective_radius` or skewing
the `ell_comps` towards the interloper — biasing every parameter we infer about the galaxy's structure.

If the extra galaxy's light does not reach the fitted region (it is faint, or far away, or both), we can simply
ignore it. When it does, we have two approaches:

- **Approach 1 (remove its light)**: We can remove the extra galaxy's emission from the fit entirely, without
  adding anything to the model.

- **Approach 2 (model it explicitly)**: If its light blends too closely with the main galaxy's emission to cleanly
  remove, we include the extra galaxy in the model, with its own light profile.

We will now perform both, and at the end of the tutorial discuss when each is appropriate.

__Mask__

We first define the circular mask used to fit the data. In chapter 2 we typically used a 2.5" - 3.0" mask, which
tightly contained the galaxy.

Here, we use a larger 6.0" mask, so that the region containing the extra galaxy is included in the fit. If we
simply shrank the mask to exclude the extra galaxy, we would also throw away pixels containing the main galaxy's
outer emission, and the mask's hard edge could still cut through the extra galaxy's light.
"""
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=6.0,
)

dataset = dataset.apply_mask(mask=mask)

"""
Plotting the masked dataset confirms the extra galaxy's emission is inside the mask, and will therefore impact the
model-fit unless we do something about it.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Approach 1 Noise Scaling__

Our first approach removes the extra galaxy's light from the fit.

The most obvious way to do this would be to mask the extra galaxy's pixels, removing them from the fit entirely.
However, removing pixels changes the fit in subtle ways: their coordinates are no longer used when evaluating the
galaxy's light, and for certain models (e.g. the pixelized reconstructions of chapter 3) removing interior pixels
creates discontinuities in the pixelization that introduce unexpected systematics.

Instead, we use **noise scaling**: the pixels stay in the fit, but their data values are set to zero and their
noise-map values are increased to very large values. A pixel with enormous noise contributes negligibly to the
likelihood, so the extra galaxy's light cannot influence the model, while the pixels themselves remain part of the
fit's geometry.

To do this we need a mask of the extra galaxy's region. For real data, you would create this yourself by
inspecting the image (the `autogalaxy_workspace`'s `data_preparation` package includes a GUI for drawing it); for
this simulated dataset the simulator script has already output a `mask_extra_galaxies.fits` circle covering the
extra galaxy.

We reload the dataset first, because noise scaling must be applied before the circular mask.
"""
dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

mask_extra_galaxies = ag.Mask2D.from_fits(
    file_path=dataset_path / "mask_extra_galaxies.fits",
    pixel_scales=0.1,
    invert=True,  # Note that we invert the mask here as `True` means a pixel is scaled.
)

dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=6.0,
)

dataset = dataset.apply_mask(mask=mask)

"""
Plotting the dataset shows the extra galaxy's emission has vanished: its data values are zero and the
signal-to-noise of its pixels is effectively zero, so the fit will simply ignore that region of the image.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Noise Scaling Fit__

We now fit this noise-scaled dataset with a model that does **not** include the extra galaxy. The model is a
single linear `Sersic` bulge, using the linear light profiles introduced in chapter 2's linear profiles
tutorial.
"""
bulge = af.Model(ag.lp_linear.Sersic)

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

search = af.Nautilus(
    path_prefix=Path("howtogalaxy", "chapter_4"),
    name="tutorial_1_extra_galaxies_noise_scaling",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=50,  # GPU batching and VRAM use explained in chapter 2 tutorial 2.
)

analysis = ag.AnalysisImaging(dataset=dataset, use_jax=True)

print(
    "The non-linear search has begun running - checkout the workspace/output/howtogalaxy/chapter_4/tutorial_1_extra_galaxies_noise_scaling"
    " folder for live output of the results, images and model."
    " This Jupyter notebook cell with progress once search has completed - this could take some time!"
)

result_noise_scaling = search.fit(model=model, analysis=analysis)

print("The search has finished run - you may now continue the notebook.")

"""
Plotting the maximum log likelihood fit shows the galaxy is fitted well, with the noise-scaled region contributing
nothing to the residuals.

The extra galaxy's light has been dealt with, and no complexity was added to the model: the fit had exactly the
same free parameters as a fit to an isolated galaxy. This is the great appeal of noise scaling.
"""
aplt.subplot_fit_imaging(fit=result_noise_scaling.max_log_likelihood_fit)

"""
__Approach 2 Extra Galaxies Model__

Our second approach includes the extra galaxy in the model, fitting its light explicitly so we no longer need to
remove it from the data. Its emission is subtracted by the model itself, including any faint light that spills
towards the main galaxy which a noise-scaling mask cannot cleanly separate.

We reload the dataset and apply the 6.0" circular mask, but this time we do **not** apply noise scaling, because
the extra galaxy's emission is now something the model itself will fit.
"""
dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=6.0,
)

dataset = dataset.apply_mask(mask=mask)

"""
__Extra Galaxy Centres__

To include the extra galaxy in the model, we input the centre of its light, as observed in the image.

In principle, we could add the extra galaxy to the model with a completely free centre, and let the non-linear
search figure out where it is. In practice this is a bad idea: the extra parameters make parameter space more
complex, and fits commonly go wrong in characteristic ways. For example, the extra galaxy's light profile may
wander off and try to fit part of the main galaxy's asymmetric emission instead of the interloper, leaving the
interloper unfitted and the main galaxy's light model biased.

Fixing each extra galaxy's light centre to its observed light centre removes these failure modes and keeps the
model as simple as possible. The observed centre is an excellent estimate of the true centre, because a galaxy's
brightest pixel closely traces the peak of its light distribution.

For real data you would measure these centres from the image (the `autogalaxy_workspace`'s `data_preparation`
package shows how, including a GUI for marking them); for this simulated dataset the simulator has output them to
a .json file, which we load below.
"""
extra_galaxies_centres = ag.Grid2DIrregular(
    ag.from_json(file_path=dataset_path / "extra_galaxies_centres.json")
)

print(extra_galaxies_centres)

"""
__Extra Galaxies Model Composition__

We compose the main galaxy model exactly as before.
"""
bulge = af.Model(ag.lp_linear.Sersic)

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

"""
We now compose the model of the extra galaxy, using the dedicated `extra_galaxies` modeling API.

For each extra galaxy centre (there is only one here, but the loop below scales to any number) we create a `Galaxy`
model with:

 - A linear `SersicSph` light profile, with its `centre` fixed to the observed centre [2 free
   parameters: `effective_radius` and `sersic_index`, as the `intensity` is solved for by the linear inversion].

The extra galaxies are grouped into their own `af.Collection`, which is passed to the overall model via its
`extra_galaxies` input, alongside the `galaxies` collection containing the main galaxy. This is the same API used
throughout the `autogalaxy_workspace` for extra galaxies, and it is how **PyAutoGalaxy** knows these galaxies are
nuisance components of the fit rather than the subject of the analysis.
"""
# Extra Galaxies:

extra_galaxies_list = []

for extra_galaxy_centre in extra_galaxies_centres:

    extra_galaxy = af.Model(
        ag.Galaxy,
        redshift=0.5,
        bulge=ag.lp_linear.SersicSph,
    )

    extra_galaxy.bulge.centre = extra_galaxy_centre

    extra_galaxies_list.append(extra_galaxy)

extra_galaxies = af.Collection(extra_galaxies_list)

# Overall Model:

model = af.Collection(
    galaxies=af.Collection(galaxy=galaxy), extra_galaxies=extra_galaxies
)

"""
The `info` attribute confirms the model includes the extra galaxy, with its fixed centre and its
free `effective_radius` and `sersic_index` parameters.
"""
print(model.info)

"""
__Extra Galaxies Fit__

We fit this model with the same search set up as before. The model has only two more free parameters than the
noise-scaling fit, thanks to the fixed centre and linear light profile, so the fit remains fast.
"""
search = af.Nautilus(
    path_prefix=Path("howtogalaxy", "chapter_4"),
    name="tutorial_1_extra_galaxies_model",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=50,  # GPU batching and VRAM use explained in chapter 2 tutorial 2.
)

analysis = ag.AnalysisImaging(dataset=dataset, use_jax=True)

print(
    "The non-linear search has begun running - checkout the workspace/output/howtogalaxy/chapter_4/tutorial_1_extra_galaxies_model"
    " folder for live output of the results, images and model."
    " This Jupyter notebook cell with progress once search has completed - this could take some time!"
)

result_extra_galaxies = search.fit(model=model, analysis=analysis)

print("The search has finished run - you may now continue the notebook.")

"""
Plotting the maximum log likelihood fit shows the extra galaxy's emission is now fitted and subtracted by its own
light profile, leaving the main galaxy's light cleanly fitted by the `Sersic` bulge.
"""
aplt.subplot_fit_imaging(fit=result_extra_galaxies.max_log_likelihood_fit)

"""
The result's `info` shows the inferred `effective_radius` and `sersic_index` of the extra galaxy, alongside the
main galaxy's parameters, which are no longer at risk of being biased by the interloper's light.
"""
print(result_extra_galaxies.info)

"""
__Which Approach When__

We have seen the two extremes of dealing with an extra galaxy. Choosing between them comes back to the question at
the start of this tutorial — does its light overlap the region we fit? — which in practice is settled by two
properties of the interloper:

- **Distance from the main galaxy's emission**: An extra galaxy whose light is well separated from the main galaxy
  (as in this tutorial) can be cleanly noise-scaled away. If its light blends into the main galaxy's emission,
  noise scaling would also delete main-galaxy emission we need to fit — biasing exactly the outer isophotes that
  constrain the `effective_radius` and `sersic_index` — and the light must be modeled instead.

- **Brightness**: A faint interloper whose emission barely rises above the noise can often be ignored entirely, or
  noise-scaled with no consequence. A bright one must be removed or modeled, or it will bias the fit. The brighter
  the interloper, the further its light extends above the noise, and the more likely it blends with the main
  galaxy.

When in doubt, fit both approaches and compare the inferred models of the main galaxy: if its parameters shift
appreciably, the extra galaxy's light is leaking into the fit and explicit modeling is the safer choice.

A middle ground also exists and is fully supported by the API: noise-scale the interloper's bright central regions
but also include it in the model, so its faint outer light under the main galaxy is still subtracted. The
`extra_galaxies` collection simply contains whatever galaxies you give it.

Finally, the `SersicSph` profile we used for the extra galaxy can be swapped for any light model from the earlier
chapters. In particular, a Multi Gaussian Expansion basis (chapter 2) captures irregular interloper morphologies
with no extra non-linear parameters, and is the recommended choice once the number of extra galaxies grows beyond
a handful — see `autogalaxy_workspace/*/imaging/features/extra_galaxies/modeling.py` for this extension.

__Wrap Up__

In this tutorial, we took the first step up in scale from the single galaxy of chapters 1 to 3, and learnt:

1. Real galaxies live in crowded fields, and interloping extra galaxies are the first complication real data
   throws at us.

2. Whether an extra galaxy matters hinges on one question: does its **light** overlap the region of the image we
   fit? If it does, an unmodeled interloper biases the main galaxy's inferred structure.

3. Noise scaling removes an extra galaxy's light from the fit without adding model complexity, by zeroing its data
   and inflating its noise, while keeping the pixels in the fit's geometry.

4. The `extra_galaxies` modeling API includes extra galaxies in the model with their own light profiles, with
   their centres fixed to the observed light centres to keep parameter space simple and well behaved.

5. Which approach is appropriate depends on the interloper's distance from the main galaxy's emission and its
   brightness, and the two approaches can be combined.

Throughout, the extra galaxy was a nuisance: something to remove or account for, so that our analysis of the main
galaxy remained accurate. In the next tutorial we meet systems where that framing breaks down entirely, because a
second galaxy is not a nuisance but a co-equal subject of the analysis, blended with the first and of comparable
brightness. There, no single galaxy is "the" galaxy, and the model must treat them all on an equal footing.
"""
