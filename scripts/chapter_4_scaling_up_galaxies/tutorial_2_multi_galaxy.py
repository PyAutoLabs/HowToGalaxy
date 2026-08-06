"""
Tutorial 2: Multi-Galaxy Blends
===============================

In the previous tutorial, we learned how to deal with extra galaxies near the galaxy we care about — nuisance
objects whose light contaminates the data but which are not themselves the subject of our study. We removed their
emission from the analysis, or gave them a heavily restricted model, and the single galaxy we were studying
remained the star of the show.

In this tutorial, we meet systems where that picture breaks down entirely: **multi-galaxy blends**, where two
(or more) galaxies of comparable brightness overlap on the sky and *every one of them* is a science subject.
Neither galaxy is a minor contaminant we can mask away or simplify — they are **co-dominant**, and each needs
its own free light model, fitted simultaneously in a single analysis.

How do such systems arise physically? There are two main channels:

- **Interacting pairs and mergers**: two galaxies at the same redshift caught orbiting one another or in the
  act of merging. Their projected separation is small compared to the extent of their light, so their surface
  brightness distributions genuinely overlap — and the interaction itself (tidal features, triggered star
  formation, morphological disturbance) is often exactly the science we want to measure, which is why both
  galaxies are subjects of the fit.

- **Chance projections**: two physically unrelated galaxies at different redshifts that happen to lie along
  nearly the same line of sight. There is no physical interaction, but on the image their light still blends,
  and measuring either galaxy's morphology or photometry requires decomposing the blend.

In both cases the modeling challenge is the same: the flux in each pixel is the sum of both galaxies'
emission (blurred by the PSF), so the only principled way to measure either galaxy is to fit a model in which
both galaxies are present at once. This tutorial shows how to compose and fit that model, how to stop its
parameter space growing out of control, and what the data can — and cannot — tell us about a blend.

__Contents__

- **Initial Setup:** Load imaging of two blended galaxies.
- **Dataset Auto-Simulation:** Automatically simulate the dataset if it does not already exist.
- **Mask:** Define a mask which encloses the light of both galaxies.
- **Over Sampling:** Centre the adaptive over sampling grid on every galaxy, not just one.
- **Why Not Fit Them Separately?:** Why masking out one galaxy and fitting the other biases both measurements.
- **Model:** Compose a model with one free light model per galaxy, and count its parameters.
- **Fixing the Centres:** Fix each galaxy's centre to its observed light centre, and why this is standard.
- **Model Fit:** Fit the two-galaxy model to the data with a non-linear search.
- **Result:** Decompose the blend into each galaxy's light and measure per-galaxy photometry.
- **Light Decomposition Degeneracy:** The total flux is well constrained, but its split between the galaxies is not.
- **Wrap Up:** Summary and the road to cluster fields.
"""

from autogalaxy import jax_wrapper  # Sets JAX environment before other imports

# from autogalaxy import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Initial Setup__

We begin with `Imaging` of two galaxies, separated by 2.0", whose light blends together:

 - The first galaxy's bulge is a `Sersic` centred at (0.0", -1.0"), with effective radius 0.8" and Sersic
   index 2.5.
 - The second galaxy's bulge is a `Sersic` centred at (0.0", 1.0"), with effective radius 0.6" and Sersic
   index 3.0.

Because both galaxies have Sersic indices well above 1, their light falls off slowly with radius — each
galaxy's outer envelope extends far past the midpoint between them, so a significant fraction of the flux in
every central pixel comes from *both* galaxies at once. This is the blend we must decompose.

__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
dataset_name = "sersic_x2"
dataset_path = Path("dataset") / "imaging" / dataset_name

if ag.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/simulators/sersic_x2.py"],
        check=True,
    )

dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

"""
When we plot the data, the difference from every dataset we have modeled so far is obvious: there are two
bright galaxies, and between them their light merges into a continuous bridge of emission. There is no radius
at which we could draw a boundary and say "this flux belongs to galaxy 0, that flux to galaxy 1".
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Mask__

We define a 3.0" circular mask centred between the two galaxies. For a blended system, the mask must enclose
the light of **both** galaxies — the fit is going to decompose the blend, so it needs to see the whole blend.
A mask sized by eye around either galaxy individually would cut through the other galaxy's light, guaranteeing
a biased fit.
"""
mask_radius = 3.0

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

"""
__Over Sampling__

We use the adaptive over sampling scheme introduced in earlier chapters, which evaluates the steep central
regions of a galaxy's light at high resolution. The one multi-galaxy specific point is that the adaptive grid
is centred on **every** galaxy, not just one — each galaxy has its own steep central light profile needing
accurate evaluation, and `centre_list` takes as many centres as we give it.

The two centres below are the observed centres of the two galaxies, which for this simulated dataset we know
exactly. For real data you would measure them from the image itself — the `autogalaxy_workspace`'s
`multi_galaxy` package loads them from a `galaxy_centres.json` file, which a GUI in the workspace writes from
mouse clicks on the image.
"""
galaxy_centres = [(0.0, -1.0), (0.0, 1.0)]

over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 2],
    radial_list=[0.3, 0.6],
    centre_list=galaxy_centres,
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Why Not Fit Them Separately?__

Before composing the joint model, it is worth asking the obvious question: why not just fit each galaxy on its
own? We could mask out galaxy 1, fit galaxy 0, then swap and repeat — turning one hard problem into two easy
ones, each identical to the single-galaxy fits of chapter 2.

The answer is that this biases **both** measurements, for two compounding reasons:

- **Contaminated flux**: whatever mask we draw, the pixels we keep still contain the other galaxy's light —
  the blend has no clean boundary. The fit attributes that extra flux to the galaxy being modeled, inflating
  its measured brightness and dragging its effective radius and Sersic index towards the contaminating
  neighbour. The neighbour's light is brightest exactly on the side facing it, so the fitted ellipticity and
  centre are skewed in that direction too.

- **Amputated flux**: the mask that removes the neighbour also removes part of the modeled galaxy's own outer
  envelope, precisely where the two overlap. The fit never sees that light, so it underestimates how far the
  galaxy extends — biasing the effective radius and Sersic index in the *opposite* direction to the first
  effect. The two biases do not cancel; they combine into measurements that are simply wrong, in ways that
  depend on the mask choice.

The joint fit has neither problem. Every pixel in the mask is modeled as the sum of both galaxies' light, so
no flux is wrongly attributed and none is thrown away. Decomposing the blend *is* the fit — this is the whole
point of the multi-galaxy regime, and it is why both galaxies must be in the model even if only one of them is
the galaxy we ultimately care about scientifically.
"""

"""
__Model__

We now compose the model, and here the multi-galaxy regime shows its teeth: every galaxy needs its own free
light model, so the model has one `Galaxy` entry per galaxy.

We build each galaxy in a loop over the observed centres and store them in a dictionary as `galaxy_0`,
`galaxy_1`, etc. This list-based composition scales to any number of blended galaxies, and it is the same API
the `autogalaxy_workspace`'s `multi_galaxy` package uses — so nothing needs re-learning later.

For each galaxy's light we use a Multi Gaussian Expansion (MGE) made of linear light profiles, built by the
utility function `ag.model_util.mge_model_from` (this hides the longer `Basis` composition API we stepped
through in the linear profiles tutorial of chapter 2). Each galaxy's 20 Gaussians add just **4 non-linear
parameters** — a shared centre and shared elliptical components, with every `sigma` fixed to log-spaced values
and every `intensity` solved for by the linear inversion.

The `centre_prior_is_uniform=True` input places a uniform prior of width 0.2" on each galaxy's centre,
centred on its observed light centre — so `galaxy_0`'s centre priors are centred on (0.0", -1.0") and
`galaxy_1`'s on (0.0", 1.0"). This is essential in a blend: if both galaxies had identical broad centre
priors, the model would not know which component is which, and the search would waste time exploring
solutions where the two galaxies have swapped places (or worse, piled on top of one another).

The MGE choice matters more here than anywhere we have used it before, because the multi-galaxy regime
multiplies whatever light model we choose by the number of galaxies:

 - Two full `Sersic` bulges (`ag.lp.Sersic`): **14** free parameters (7 each).
 - Two linear `Sersic` bulges (`ag.lp_linear.Sersic`, intensities solved for): **12** free parameters.
 - Two MGEs: **8** free parameters (4 each) — and each MGE is far more flexible than a single Sersic,
   capturing the asymmetries and radially-varying ellipticity that real (especially interacting!) galaxies
   show.

A light model that is both cheap and flexible is close to essential in this regime — this is exactly where
the MGE shines. On top of the parameter count, solving the intensities by linear algebra has a second, subtler
benefit for blends that we will return to at the end of this tutorial.
"""
galaxy_dict = {}

for i, centre in enumerate(galaxy_centres):

    bulge = ag.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        centre_prior_is_uniform=True,
        centre=(centre[0], centre[1]),
        sigma_min=dataset.pixel_scales[0] / 10.0,
    )

    galaxy_dict[f"galaxy_{i}"] = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

model = af.Collection(galaxies=af.Collection(**galaxy_dict))

"""
The model's `info` shows `galaxy_0` and `galaxy_1` each carrying their own free MGE — the signature of the
multi-galaxy regime — and its `prior_count` gives the total number of free parameters.
"""
print(model.info)

print(f"Free parameters (free centres): {model.prior_count}")

"""
The count is **8**: each galaxy contributes its MGE's 2 centre parameters and 2 elliptical component
parameters. Every further blended galaxy will add 4 more — the growth is linear, and with plain Sersic light
profiles instead of MGEs it would grow at 7 per galaxy instead.

__Fixing the Centres__

Before fitting, we make one change that is standard practice for blended systems: we **fix each galaxy's
centre to its observed light centre**, removing 2 free parameters per galaxy. The previous tutorial fixed the
interloper's centre for the same reason; in a blend the case is even stronger.

Why? With a single galaxy, the data constrains the centre extremely well: it is simply the peak of the light,
and there is only one plausible culprit for every photon. With blended light this breaks down. Each pixel's
flux is the sum of both galaxies' light, so a small shift of one galaxy's centre can be compensated by changes
to the other galaxy's model — its ellipticity, its Gaussians' relative intensities, or a shift of its own
centre. The free centres become degenerate with one another and with the rest of the model, and the search
wanders these degeneracies, converging slowly and sometimes settling in unphysical corners of parameter space
where one model galaxy has drifted between the two real ones.

Fixing each centre to the observed light peak resolves this at minimal cost. Unlike almost every other light
profile parameter, the centre is something we can measure directly and robustly from the image before any
fitting — a peak position is hard to fake, even in a blend. (Measuring a genuine offset — say, of a nucleus
displaced during an interaction — is real science, but it is a *follow-up* fit performed after a robust model
with fixed centres has been found.)

The `centre_fixed` input of `mge_model_from` fixes every Gaussian's centre to the input tuple, so the centre
is no longer a free parameter with a prior.
"""
galaxy_dict = {}

for i, centre in enumerate(galaxy_centres):

    bulge = ag.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        centre_fixed=(centre[0], centre[1]),
        sigma_min=dataset.pixel_scales[0] / 10.0,
    )

    galaxy_dict[f"galaxy_{i}"] = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

model = af.Collection(galaxies=af.Collection(**galaxy_dict))

"""
Printing the model's `info` again, each centre is now listed as a fixed value with no prior, and the free
parameter count has dropped from 8 to **4** — just each galaxy's two elliptical components. Pause on that
number: we are about to decompose a fully blended pair of galaxies, capturing both morphologies with 40
Gaussians, by searching a 4-dimensional parameter space. Compare it to the 14 dimensions two full Sersics
would have cost, and it is clear how the combination of linear profiles, the MGE and fixed centres is what
makes this regime tractable at all.
"""
print(model.info)

print(f"Free parameters (fixed centres): {model.prior_count}")

"""
__Model Fit__

We fit the model with the nested sampling algorithm `Nautilus`, as in previous chapters. Thanks to the MGE and
the fixed centres the parameter space is small and well behaved, so `n_live=100` (the chapter 2 default) is
ample.

Note that the `AnalysisImaging` object is completely unchanged from single-galaxy fitting — the multi-galaxy
regime changed the model composition, and nothing else.
"""
search = af.Nautilus(
    path_prefix=Path("howtogalaxy", "chapter_4"),
    name="tutorial_2_multi_galaxy",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=50,  # GPU batching and VRAM use explained in chapter 2 tutorial 2.
)

analysis = ag.AnalysisImaging(dataset=dataset, use_jax=True)

print(
    "The non-linear search has begun running - checkout the workspace/output/howtogalaxy/chapter_4/tutorial_2_multi_galaxy"
    " folder for live output of the results, images and model."
    " This Jupyter notebook cell with progress once search has completed - this could take some time!"
)

result = search.fit(model=model, analysis=analysis)

"""
__Result__

The result's `info` shows the inferred model, with each galaxy's parameters listed under its own `galaxy_0` /
`galaxy_1` entry.
"""
print(result.info)

"""
The fit subplot shows the joint model reproducing the full blend — both galaxies and the bridge of overlapping
light between them — with residuals at the noise level.
"""
aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
The core deliverable of a multi-galaxy fit is the **decomposition**: `subplot_fit_imaging_of_galaxy` shows,
for each galaxy in turn, its modeled light on its own and the data with the *other* galaxy's model subtracted.
This is the closest thing to an image of each galaxy as it would appear without its companion — something no
mask could ever have given us.
"""
for i in range(len(galaxy_centres)):
    aplt.subplot_fit_imaging_of_galaxy(fit=result.max_log_likelihood_fit, galaxy_index=i)

"""
With the blend decomposed, per-galaxy photometry is direct: each galaxy's model image contains only its own
light, so summing it gives that galaxy's flux uncontaminated by its neighbour.
"""
galaxies = result.max_log_likelihood_galaxies

for i, galaxy in enumerate(galaxies):
    image = galaxy.image_2d_from(grid=dataset.grids.lp)
    print(f"galaxy_{i}: total model flux = {float(image.array.sum()):.3f}")

"""
__Light Decomposition Degeneracy__

The photometry above comes with an important caveat, and it is the deepest lesson of the multi-galaxy regime.

In the pixels where the two galaxies overlap, the data constrains only the **sum** of their light. A model
that brightens galaxy 0's outer envelope and dims galaxy 1's by the same amount produces a nearly identical
image, and therefore a nearly identical likelihood. The *total* flux of the blend is pinned down superbly —
every photon is accounted for — but the *split* of that flux between the two galaxies is constrained only by
the shapes of the profiles, which is a much weaker lever. The closer the pair and the larger their overlap,
the weaker it gets.

You can see this degeneracy directly in the posterior. When you run this tutorial for real (rather than
skimming the generated output), open the search's corner plot — the `.png` in the output folder's `image`
directory, or plot it yourself with `aplt.corner_anesthetic(samples=result.samples)` — and focus on the panels
pairing `galaxy_0`'s parameters against `galaxy_1`'s. You will see tilted, anti-correlated contours: when one
galaxy's model claims more of the shared light, the other's gives it up. Each galaxy's flux, effective radius
and shape measurements inherit the full width of this degeneracy, so their error bars are larger — sometimes
much larger — than an equivalent isolated galaxy's would be. Honest science on blended systems propagates
these widened uncertainties rather than quoting the best-fit split as if it were certain, and this is
precisely why we fit blends with a sampler that maps the full posterior rather than an optimizer that returns
a single point: the best-fit sits somewhere on the degeneracy ridge and tells you nothing about the ridge's
length.

This is also the subtler benefit of linear light profiles promised earlier. The flux ratio between the two
galaxies — the quantity most degenerate in a blend — lives in the `intensity` values, and the inversion solves
those exactly at every likelihood evaluation. The non-linear search never has to explore the intensity
degeneracy stochastically; it is handled by linear algebra, leaving the sampler to map only the (much milder)
degeneracies between the galaxies' shapes. Fitting a blend with standard (non-linear) light profiles forfeits
this, and is markedly slower and less reliable as a result.

__Wrap Up__

In this tutorial, we modeled a system of two blended galaxies simultaneously. Let's summarise what we've
learnt:

- **Co-dominant galaxies**: When two or more galaxies of comparable brightness overlap on the sky, every one
  of them needs its own free light model — unlike the nuisance neighbours of the previous tutorial, none can
  be masked away or simplified. Such blends arise from interacting pairs and mergers at one redshift, or
  chance projections of unrelated galaxies along the line of sight.

- **Separate fits are biased**: Masking one galaxy out and fitting the other contaminates the fit with the
  neighbour's flux while amputating the target's own overlapping light — both measurements come out wrong.
  The joint fit, in which every pixel is modeled as the sum of both galaxies, is the only principled
  decomposition.

- **Parameter accounting**: Two full Sersics would cost 14 free parameters; linear profiles cut this to 12,
  the MGE to 8, and fixing each galaxy's centre to its observed light centre to just 4 — with the MGE's
  intensities solved by linear algebra rather than sampled. Cheap, flexible light models are what make the
  multi-galaxy regime tractable, because every additional galaxy multiplies the cost.

- **Fixed centres**: Blended light makes free centres degenerate — one galaxy's centre shift can be absorbed
  by changes to the other's model. Fixing each centre to the observed light peak, the one parameter we can
  measure robustly before fitting, is the standard trick for taming this.

- **Light decomposition degeneracy**: The data constrains the total flux of the blend far better than its
  split between the galaxies, producing anti-correlated posteriors between the two galaxies' parameters and
  inflating the uncertainties on all per-galaxy measurements. Map it with a sampler, and propagate it.

Everything here scaled comfortably to two galaxies, and the loop-based composition would carry us to three or
four. But the ladder keeps climbing: in the next tutorial we reach **cluster fields**, where a brightest
cluster galaxy sits among tens to hundreds of member galaxies. Giving every member its own free model — even
a 2-parameter one — cannot scale that far, and modeling them one blend at a time would reintroduce every bias
we just eliminated. The answer is catalogue-driven composition: the member population is built automatically
from a catalogue of measured positions and photometry, and modeled collectively. That is where we go next.
"""
