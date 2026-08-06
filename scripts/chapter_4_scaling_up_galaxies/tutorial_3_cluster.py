"""
Tutorial 3: Cluster
===================

Throughout this chapter we have been scaling up: from a single galaxy with extra galaxies nearby, to blended
multi-galaxy systems where every galaxy received its own free light model.

This tutorial climbs the final rung: **cluster fields**, the richest environments in which galaxies live. An
image of a galaxy cluster contains:

- **A brightest cluster galaxy (BCG)**: the giant elliptical that sits at the cluster's centre. BCGs are the
  most massive galaxies in the Universe, built up over billions of years by swallowing their neighbours, and
  their extended envelopes hold a record of that assembly history.

- **Tens to hundreds of member galaxies**: the cluster's population of (mostly elliptical) galaxies,
  scattered across a field many times wider than the single-galaxy images of earlier chapters.

- **Intra-cluster light (ICL)**: a faint, diffuse glow of stars stripped from member galaxies, filling the
  space between them. It blends smoothly into the BCG's outer envelope, which is why careful modeling of the
  BCG's light is the starting point of every ICL measurement.

Why model all of this light? Three science cases drive the cluster regime:

- **BCG growth**: the size and shape of a BCG's outer envelope constrains how much of its mass was accreted
  from merging satellites — but measuring that envelope requires subtracting every member galaxy whose light
  overlaps it.

- **Intra-cluster light**: the ICL is only measurable once the BCG and members are modeled and removed;
  what remains is the diffuse component.

- **Member photometry**: the luminosity of each member, uncontaminated by its neighbours and the BCG's
  envelope, is what luminosity functions and scaling relations are built from.

(A quick note for readers heading towards gravitational lensing: galaxy clusters are also the Universe's
most powerful strong lenses, and the lensing version of this scale — where a catalogue like the one below
drives member *masses* and the fit is to point-source image positions rather than galaxy light — lives in
chapter 4 of the **HowToLens** lectures.)

__The Scaling Problem__

The previous tutorials composed models the way we have all chapter: one `af.Model(ag.Galaxy)` per galaxy,
each with its own free light profile. A blend of two or three galaxies is fine. A cluster is not:

- **Parameters**: ten members with a free spherical Sersic each (5 parameters per galaxy) is 50 free
  parameters before we even touch the BCG; a hundred members is 500. No non-linear search can sample such a
  space reliably.

- **Information**: the faint members do not contain enough signal to constrain four free parameters each —
  most of those dimensions would be unconstrained noise.

- **Practicality**: hand-writing Python model code for hundreds of galaxies is not sensible, and no
  astronomer works that way. What an observer actually has is a **catalogue**: a table listing where each
  member is and how bright it is.

The cluster regime therefore changes how the model is composed. The BCG — the galaxy whose structure we
care most about — is modeled individually and richly, with a free MGE, exactly as a single galaxy would be.
The member population is driven by the catalogue: each member's centre is fixed to its catalogue position,
its shape is fixed to sensible values, and its intensity is TIED to its catalogue luminosity through a
single shared free normalization. Adding a member is a row append; the model's dimensionality does not grow
with the population.

__Contents__

- **The Scaling Problem:** Why one-free-model-per-galaxy cannot scale to a cluster's member population.
- **Dataset:** Load the simulated cluster field (1 BCG + 10 members).
- **Dataset Auto-Simulation:** Automatically simulate the dataset if it does not already exist.
- **Member Catalogue:** Load the member centres and luminosities from `scaling_galaxies.csv`.
- **Masking:** Mask the wide cluster field and over-sample every galaxy's centre.
- **Model:** Compose the two-tier cluster model — free BCG MGE + catalogue-driven member tier.
- **Search + Analysis:** Configure the Nautilus non-linear search and the analysis.
- **Model Fit:** Run the fit.
- **Result:** Inspect the fit, including the member-subtracted BCG decomposition.
- **Per-Member Results:** Access each member's fitted light — catalogue-scale photometry.
- **Refinements:** Where the composition goes next — freeing tier shapes, promoting bright members.
- **Wrap Up:** The chapter and the **HowToGalaxy** lectures conclude.
"""

from autogalaxy import jax_wrapper  # Sets JAX environment before other imports

# from autogalaxy import setup_notebook; setup_notebook()

from pathlib import Path

import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset__

We fit a simulated cluster field kept deliberately small so it runs quickly — 1 BCG and 10 member galaxies
in a 25" x 25" field — but containing every ingredient of the cluster regime. A real cluster simply has more
members, and we will see that scaling the model up to hundreds of them does not add a single free parameter.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "cluster" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if ag.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/simulators/cluster.py"],
        check=True,
    )


dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

"""
Plotting the dataset shows what a cluster field looks like in imaging: the bright, extended BCG dominating
the centre, with the fainter members scattered across the frame. In real data the diffuse intra-cluster
light would fill the space between them.

Note the field of view compared to earlier chapters — the members sit up to ~10" from the centre, so the
image is far larger than the ~6" cutouts we fitted before.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Member Catalogue__

The member population enters the model through a catalogue file, `scaling_galaxies.csv`, with one row per
member galaxy and three columns:

 - `y`, `x`: the member's centre in arc-seconds — where the galaxy is.
 - `luminosity`: the member's luminosity — how bright it is. Any consistent luminosity units work, because
   (as we will see below) only a shared normalization of the luminosities is fitted.

This is exactly what an observer measures: a photometry catalogue. In a real analysis this CSV comes
straight from source-extraction software or a survey database — it is spreadsheet-editable, and adding a
member to the model means appending a row, not writing Python.

We load it with `ag.galaxy_table_from_csv`, the catalogue-loading API.
"""
scaling_table = ag.galaxy_table_from_csv(
    file_path=dataset_path / "scaling_galaxies.csv"
)

member_centres = scaling_table.centres.in_list
member_luminosities = scaling_table.luminosities

print(f"Members in catalogue: {len(member_centres)}")
print(f"First member centre: {member_centres[0]}")
print(f"First member luminosity: {float(member_luminosities[0])}")

"""
The BCG's centre is recorded in its own file, `bcg_centres.json` — in a real cluster there may be more than
one dominant galaxy (e.g. a second bright elliptical) that deserves individual modeling, so their centres
are kept separate from the member catalogue.
"""
bcg_centres = ag.from_json(file_path=dataset_path / "bcg_centres.json")

print(f"BCG centre: {bcg_centres[0]}")

"""
__Masking__

We mask the field generously — the members span the frame, and the model must account for every galaxy
inside the mask (an unmodeled galaxy inside the mask would bias the fit, as the first tutorial of this
chapter showed).

The over-sampling scheme is centred on every galaxy in the field — the BCG and all ten members — so each
galaxy's steep central light gradient is evaluated accurately.
"""
mask_radius = 11.0

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 2],
    radial_list=[0.3, 0.6],
    centre_list=list(bcg_centres) + list(member_centres),
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Model__

The model has two tiers, and this two-tier composition is the cluster regime's signature:

**Tier 1 — the BCG, modeled richly.** The BCG is the science target, so it gets the most flexible light
model we have: a free Multi-Gaussian Expansion (the MGE introduced in chapter 2), composed via
`ag.model_util.mge_model_from`. Its 20 Gaussians can capture the BCG's extended envelope, radially-varying
ellipticity and isophotal twists — exactly the structures that BCG growth and intra-cluster light studies
measure. This costs the usual handful of non-linear parameters (centre + elliptical components; the
Gaussian intensities are solved linearly).

**Tier 2 — the members, modeled cheaply from the catalogue.** Each member gets a spherical Sersic
(`SersicSph`) generated in a loop over the catalogue rows:

 - its `centre` is FIXED to the catalogue (y, x) position;
 - its shape (`effective_radius`, `sersic_index`) is FIXED to values typical of cluster ellipticals;
 - its `intensity` is TIED to the catalogue luminosity through one shared free parameter:

       intensity_i = intensity_scale * luminosity_i

The single `intensity_scale` prior is defined once, outside the loop, and every member's intensity is an
arithmetic expression of it — an extension of the prior linking we used in earlier chapters (pairing two
parameters with `=`, e.g. `bulge.centre = disk.centre`), now doing population-scale work. The whole 10-member tier therefore contributes ONE free parameter to the non-linear
search, and would still contribute one with 200 members.

Physically, `intensity_scale` is the conversion between the catalogue's luminosity units and the image's
flux units: the catalogue fixes each member's brightness *relative* to the others (which photometry measures
well), and the fit solves for the one overall normalization.
"""
# BCG:

bulge = ag.model_util.mge_model_from(
    mask_radius=3.0,
    total_gaussians=20,
    centre_prior_is_uniform=True,
    centre=(bcg_centres[0][0], bcg_centres[0][1]),
    sigma_min=dataset.pixel_scales[0] / 10.0,
)

galaxy_dict = {"bcg": af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)}

# Members: one shared free normalization for the whole tier.

intensity_scale = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

for i, (centre, luminosity) in enumerate(zip(member_centres, member_luminosities)):

    bulge = af.Model(ag.lp.SersicSph)
    bulge.centre = tuple(centre)
    bulge.intensity = intensity_scale * float(luminosity)  # tied to the catalogue
    bulge.effective_radius = 0.6
    bulge.sersic_index = 3.0

    galaxy_dict[f"member_{i}"] = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

model = af.Collection(galaxies=af.Collection(**galaxy_dict))

"""
Printing `model.info` confirms the composition: the BCG's MGE parameters are free, every member's intensity
shows as tied to the single shared `intensity_scale`, and the total dimensionality stays small despite the
model containing 11 galaxies. This is the punchline of the cluster workflow — catalogues decouple the
model's physical richness from the dimensionality of the search.
"""
print(model.info)

"""
__Search + Analysis__

The analysis is the same `AnalysisImaging` used on every rung of the ladder — the cluster regime changed how
the model is *composed*, not how it is *fitted*. We fit with the Nautilus nested sampler, as throughout the
lectures, which returns the full posterior: the errors on the BCG's structural parameters and on
`intensity_scale` are what a cluster paper reports.
"""
search = af.Nautilus(
    path_prefix=Path("howtogalaxy", "chapter_4"),
    name="tutorial_3_cluster",
    unique_tag=dataset_name,
    n_live=150,
    n_batch=50,
)

analysis = ag.AnalysisImaging(dataset=dataset, use_jax=True)

"""
__Model Fit__

Run the fit. Despite the model containing 11 galaxies, the parameter space is barely larger than a single
galaxy's — this is the catalogue tier doing its job.
"""
print(
    "The non-linear search has begun running - checkout the workspace/output/howtogalaxy/chapter_4/tutorial_3_cluster"
    " folder for live output of the results, images and model."
    " This Jupyter notebook cell with progress once search has completed - this could take some time!"
)

result = search.fit(model=model, analysis=analysis)

print("The search has finished run - you may now continue the notebook.")

"""
__Result__

The `info` attribute shows the resulting model — the BCG's MGE parameters with their errors, and the single
inferred `intensity_scale` shared by the member tier.
"""
print(result.info)

"""
Plotting the maximum likelihood fit shows the full cluster model reproducing the field: BCG and all ten
members fitted simultaneously.
"""
aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
The plot below is the one that motivates cluster light modeling: the fit decomposed by galaxy, showing the
BCG's light (galaxy index 0) in isolation — that is, with every member galaxy's light accounted for by its
own model component rather than contaminating the BCG's.

This member-subtracted BCG is the measurement BCG-growth and intra-cluster light studies are built on: its
outer envelope can now be traced without the members' light biasing it, and whatever diffuse light the model
cannot attribute to any galaxy is the ICL candidate signal.
"""
aplt.subplot_fit_imaging_of_galaxy(fit=result.max_log_likelihood_fit, galaxy_index=0)

"""
__Per-Member Results__

The member tier's single free parameter was `intensity_scale`, but the maximum likelihood instance contains
the fully-realised light profile of every member — the tie to the catalogue has been applied, so each member
carries the intensity its luminosity implies. Looping over the members tabulates the fitted photometry of
the whole population, exactly as a member-photometry analysis of a real cluster would.
"""
instance = result.max_log_likelihood_instance

for i in range(len(member_centres)):
    member = getattr(instance.galaxies, f"member_{i}")
    print(
        f"Member {i}: centre={member.bulge.centre}, "
        f"intensity={float(member.bulge.intensity):.3f}"
    )

"""
The simulator built every member with `intensity = luminosity` — a true shared normalization of 1.0. We can
check the fit recovered it by dividing any member's fitted intensity by its catalogue luminosity.

(A real, full run recovers a value close to 1.0; if you are running this script in the fast test mode used
for automated checks, the search is bypassed and the printed value is a prior median, not meaningful.)
"""
recovered_scale = float(instance.galaxies.member_0.bulge.intensity) / float(
    member_luminosities[0]
)

print(f"Recovered member-tier intensity_scale = {recovered_scale:.3f} (truth: 1.0)")

"""
__Refinements__

This tutorial used the simplest catalogue tier — fixed shapes, one shared normalization — which is the right
starting point. Real cluster analyses then reach for two refinements, both of which preserve the
row-append scaling:

- **Free the tier's shape**: promote the shared `effective_radius` and `sersic_index` from fixed values to
  parameters shared by the whole tier. The tier then costs 3 free parameters instead of 1 — still
  independent of the member count.

- **Promote bright members**: give the brightest few members their own free light models alongside the BCG,
  since the data constrains them well. Promotion costs the full per-galaxy parameters, so promote sparingly
  — brightest first, and only while the data keeps constraining them.

Both are demonstrated in `autogalaxy_workspace/*/cluster/modeling.py`, which fits this same dataset.

__Wrap Up__

This tutorial completes the chapter's ladder of scale, and with it the **HowToGalaxy** lectures. Lets recap
the ladder one final time:

- **One galaxy** (chapters 1-3): light profiles, non-linear searches, linear profiles and the MGE, and
  pixelized reconstructions — the core toolkit, applied to a single galaxy at the centre of its image.

- **Extra galaxies**: nearby galaxies whose light contaminates the target's, noise-scaled out of the fit or
  included in the model so the target's photometry is unbiased.

- **Multi-galaxy blends**: systems of overlapping galaxies modeled simultaneously, each with its own free
  light model.

- **Cluster fields** (this tutorial): a BCG modeled richly plus a member population too numerous for free
  models, driven instead by a catalogue — fixed centres, luminosity-tied intensities, one shared free
  normalization. Model complexity decoupled from population size.

The theme of the chapter has been that scaling up is about *composition*, not new fitting machinery: the
same `af.Model` / `af.Collection` API, the same `AnalysisImaging`, the same searches — arranged so that the
information in the data decides which parameters are free.

And that theme closes the series: you now know how to simulate and fit galaxy imaging, compose models from
simple Sersics to MGEs and pixelizations, run and interpret non-linear searches, and scale all of it from
one galaxy to a cluster field. Where to go next:

- `autogalaxy_workspace`: the destination for real science — the `imaging`, `multi_galaxy` and `cluster`
  packages mirror this chapter's ladder with production-ready scripts (including interferometer data,
  multi-wavelength fitting and ellipse fitting, which the lectures did not cover), and its `guides` cover
  every API in depth.

- **HowToLens**: the companion lecture series for strong gravitational lensing, where galaxies like these
  become lenses — its chapter 4 climbs this same ladder on the mass side, with the cluster rung fitting
  point-source image positions driven by the very same catalogue machinery.

Congratulations on finishing the **HowToGalaxy** lectures — now go model some galaxies!
"""
