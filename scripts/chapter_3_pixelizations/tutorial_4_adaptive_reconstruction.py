r"""
Tutorial 4: Adaptive Reconstruction
===================================

Combine Delaunay meshes, Hilbert brightness adaptation and adaptive Split regularization for directly observed galaxies.

__Contents__

- Adaptive galaxy reconstruction
- Returning to the compact-galaxy problem
- From image coordinates to a reconstruction mesh
- Constructing the Delaunay galaxy mesh
- Interpolating within a triangle
- Regularization on an irregular mesh
- Building an adapt image
- Placing points with a Hilbert image mesh
- Controlling the brightness weighting
- Keeping resolution in the faint regions
- Reconstructing with the Hilbert mesh
- Comparing overlay and Hilbert reconstructions
- Smoothing on an irregular mesh
- Reconstructing with ConstantSplit
- Allowing different smoothing strengths
- Controlling the transition with `signal_scale`
- Exploring the adaptive coefficients
- Does adaptive regularization address the fit problem?
- Wrap Up

__Adaptive galaxy reconstruction__

In **Fit Problems**, we identified two limitations of our reconstruction: too few mesh points across compact galaxy structure, and a single regularization coefficient that must compromise between the bright centre and faint outskirts.

We will now introduce methods that address these problems.

Our first step is to replace the rectangular galaxy mesh with a **Delaunay mesh**. Its points need not lie in regular rows and columns, allowing us to vary their spacing across the galaxy.

We then need to decide where to place those points. We will select coordinates over the observed image, using an **image mesh**. Those same sky coordinates become the reconstruction vertices: there is no ray tracing.

This gives us two useful ways to allocate resolution:

- An evenly spaced overlay provides a simple starting mesh, independent of the galaxy brightness.
- A brightness-adaptive image mesh places more points in bright image regions, directing more resolution towards the galaxy emission.

Finally, we will introduce adaptive regularization, allowing the smoothing strength to vary across the reconstructed galaxy.

These choices work together. Placing more points across a bright component gives the reconstruction enough resolution to describe it; allowing appropriate brightness differences between those points lets it use that resolution.

We will proceed through:

- Constructing an image-plane overlay.
- Connecting its sky coordinates to build a Delaunay mesh.
- Understanding interpolation and regularization on that mesh.
- Using a fitted image to guide Hilbert brightness adaptation.
- Introducing Split and adaptive regularization.
- Comparing reconstructions on the same dataset.
- Preparing for the combined analytic and pixelized model in Tutorial 5.

---

__Returning to the compact-galaxy problem__

We return to the very compact galaxy from **Fit Problems**. Keeping the same example lets us examine whether the adaptive methods address the difficulties we have just investigated.

The setup below reproduces that dataset, including its PSF and Gaussian noise realization. This makes the tutorial runnable independently.
"""

import numpy as np
import matplotlib.pyplot as plt
import autogalaxy as ag
import autogalaxy.plot as aplt
import autoarray.plot as aaplt

simulation_grid = ag.Grid2D.uniform(
    shape_native=(150, 150),
    pixel_scales=0.05,
)
simulation_grid = simulation_grid.apply_over_sampling(
    over_sample_size=8,
)

psf = ag.Convolver.from_gaussian(
    shape_native=(11, 11),
    sigma=0.05,
    pixel_scales=0.05,
)

true_source = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.15),
        intensity=0.2,
        effective_radius=0.1,
        sersic_index=4.0,
    ),
)

true_galaxies = ag.Galaxies(galaxies=[true_source])

simulator = ag.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=False,
)

noiseless = simulator.via_galaxies_from(
    galaxies=true_galaxies,
    grid=simulation_grid,
)

noise_sigma = 0.05
noise_values = np.random.default_rng(1).normal(
    loc=0.0,
    scale=noise_sigma,
    size=simulation_grid.shape_native,
)

dataset = ag.Imaging(
    data=ag.Array2D.no_mask(
        values=np.asarray(noiseless.data.native) + noise_values,
        pixel_scales=0.05,
    ),
    noise_map=ag.Array2D.no_mask(
        values=np.full(simulation_grid.shape_native, noise_sigma),
        pixel_scales=0.05,
    ),
    psf=psf,
)

mask = ag.Mask2D.circular(
    shape_native=simulation_grid.shape_native,
    pixel_scales=0.05,
    radius=3.0,
)

dataset = dataset.apply_mask(mask=mask)
dataset = dataset.apply_over_sampling(
    over_sample_size_pixelization=2,
)

aplt.subplot_imaging_dataset(dataset=dataset)

r"""
We keep the image sampling and PSF fixed. This lets us inspect mesh placement and regularization separately from any choice of accompanying analytic profiles.

We know the simulated galaxy profile, but we will not use its brightness distribution to tell the adaptive mesh where to place its points. Brightness adaptation will use a fitted model image, as it would for observed data.

---

__From image coordinates to a reconstruction mesh__

Previously, we constructed a rectangular mesh over the galaxy image.

For our Delaunay reconstruction, we instead begin by selecting coordinates in the **image plane**. These coordinates form the image mesh. These are also the coordinates at which we infer the galaxy brightness. The shared mapper API calls them `source_plane_mesh_grid`, but here they refer to the same sky as the image.

The simplest example is an `Overlay`, which lays a coarse, evenly spaced grid over the image and retains the points inside the mask.
"""

image_mesh = ag.image_mesh.Overlay(
    shape=(25, 25),
)

image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=dataset.mask,
)

print(
    "Number of image-mesh points inside the mask:",
    image_plane_mesh_grid.shape[0],
)

aplt.plot_array(
    array=dataset.data,
    grid=image_plane_mesh_grid,
    title="Image-Plane Mesh: Uniform Overlay",
)

r"""
These are mesh coordinates, rather than a selection of the only image pixels we will fit. All unmasked data pixels still contribute to the reconstruction.

The overlay’s shape describes the grid before masking. Its retained number of points is therefore smaller than `25 × 25` for our circular mask.

We will connect these coordinates directly into a triangulation.

The overlay still does not inspect brightness. Changing an extended galaxy into a compact one leaves these coordinates unchanged for a fixed mask. Switching triangulation alone therefore does not allocate extra resolution to the bright centre. We will establish a baseline reconstruction before introducing brightness-based placement.

__Constructing the Delaunay galaxy mesh__

We now connect the selected mesh points into a **Delaunay triangulation**.

Unlike a rectangular mesh, the triangles can have different sizes and orientations. Their vertices follow the selected image-mesh coordinates, allowing finer sampling in some galaxy regions than others.

The galaxy brightness values are assigned to the **vertices**, rather than one constant brightness being assigned to each triangle. We will examine how brightness is calculated between the vertices in the next block.

The helper below constructs a Delaunay reconstruction using our image-plane overlay. It also lets us vary the regularization coefficient.
"""

dataset = dataset.apply_sparse_operator()


def fit_delaunay_overlay(coefficient):
    pixelization = ag.Pixelization(
        mesh=ag.mesh.Delaunay(
            pixels=image_plane_mesh_grid.shape[0],
        ),
        regularization=ag.reg.Constant(
            coefficient=coefficient,
        ),
    )

    source_galaxy = ag.Galaxy(
        redshift=0.5,
        pixelization=pixelization,
    )

    galaxies = ag.Galaxies(galaxies=[source_galaxy])

    adapt_images = ag.AdaptImages(
        galaxy_image_plane_mesh_grid_dict={
            source_galaxy: image_plane_mesh_grid,
        },
    )

    return ag.FitImaging(
        dataset=dataset,
        galaxies=galaxies,
        adapt_images=adapt_images,
    )


overlay_fit = fit_delaunay_overlay(coefficient=1.0)
overlay_mapper = overlay_fit.inversion.cls_list_from(ag.Mapper)[0]

aaplt.plot_mapper(
    mapper=overlay_mapper,
    solution_vector=overlay_fit.inversion.reconstruction,
    mesh_grid=overlay_mapper.source_plane_mesh_grid,
    zoom_to_brightest=False,
    title="Delaunay Reconstruction from an Image-Plane Overlay",
)

aplt.subplot_fit_imaging(fit=overlay_fit)

r"""
`AdaptImages` supplies the image-mesh coordinates associated with the pixelized galaxy. Here, we are supplying an overlay; we have not yet introduced brightness adaptation.

Inspect the mesh points around the reconstructed galaxy. For this overlay, their spacing is initially uniform; the Hilbert example below will change that distribution.

The coefficient of `1.0` is only a starting value. We should not judge whether this mesh improves the fit until we have explored its regularization.

---

__Interpolating within a triangle__

How does a triangular mesh predict the brightness at an image coordinate?

Recall our rectangular mesh: bilinear interpolation combined the brightness values at four surrounding mesh points.

For a Delaunay mesh, we locate the triangle containing the coordinate and interpolate between its **three vertices**. Each vertex contributes according to a weight determined by the coordinate’s position within the triangle.

At a vertex, that vertex supplies the brightness. At the midpoint of an edge, the two edge vertices contribute equally. At the triangle’s centroid, all three vertices contribute equally.

The brightness therefore varies continuously across each triangle. It is not a collection of uniformly bright triangular patches.

We can inspect the mapping weights for an image pixel, using the same approach as tutorial 1.
"""

image_index = int(np.argmax(np.asarray(dataset.data.slim)))

mapping_weights = np.asarray(overlay_mapper.mapping_matrix[image_index])

source_indexes = np.flatnonzero(np.abs(mapping_weights) > 1.0e-12)

reconstruction = np.asarray(overlay_fit.inversion.reconstruction)

for source_index in source_indexes:
    weight = mapping_weights[source_index]
    brightness = reconstruction[source_index]

    print(
        f"Galaxy vertex {source_index}: "
        f"weight={weight:.5g}, "
        f"brightness={brightness:.5g}, "
        f"contribution={weight * brightness:.5g}"
    )

print(
    "Predicted image-pixel brightness before PSF blurring:",
    float(np.dot(mapping_weights, reconstruction)),
)

r"""
We have enabled image-pixel oversampling. Each sub-pixel sample is interpolated within its triangle, and those contributions are combined for the image pixel. Consequently, the printed row can contain contributions from more than three galaxy vertices if its samples fall in different triangles.

The PSF then redistributes the predicted light between image pixels, as in our rectangular reconstruction.

Interpolation still does not determine the vertex brightnesses. Those are solved by the inversion, with regularization controlling how freely they can vary.

---

__Regularization on an irregular mesh__

We begin with the same `Constant` regularization scheme used earlier.

For the rectangular mesh, an interior point had four immediate neighbours. On a Delaunay mesh, neighbours are connected by triangle edges. The number of neighbours can vary from one vertex to another; belonging to a triangular mesh does not mean every vertex has only three neighbours.

Regularization penalizes brightness differences between connected vertices. One coefficient controls the strength of this smoothing across the mesh.

The neighbour connections and spacings differ from the rectangular case, so we should not assume that its preferred coefficient transfers to the Delaunay mesh.

Let’s evaluate a range of coefficients.
"""

overlay_coefficients = np.logspace(-3, 3, 13)
overlay_evidences = []
best_evidence = -np.inf

# Keep only the preferred full fit; other scan entries are scalar evidences.
for coefficient in overlay_coefficients:
    coefficient = float(coefficient)
    current_fit = fit_delaunay_overlay(coefficient=coefficient)
    evidence = float(current_fit.log_evidence)
    overlay_evidences.append(evidence)
    if evidence > best_evidence:
        best_evidence = evidence
        overlay_fit = current_fit

overlay_evidences = np.asarray(overlay_evidences)
if not np.all(np.isfinite(overlay_evidences)):
    raise ValueError("Inspect non-finite evidence values before continuing.")

best_index = int(np.argmax(overlay_evidences))
if best_index in (0, len(overlay_coefficients) - 1):
    raise ValueError("Extend the coefficient range to bracket the overlay peak.")

overlay_coefficient = float(overlay_coefficients[best_index])

fig, ax = plt.subplots()
ax.semilogx(
    overlay_coefficients,
    overlay_evidences - np.max(overlay_evidences),
    marker="o",
)
ax.set_xlabel("Regularization coefficient")
ax.set_ylabel("Log evidence relative to the highest sampled value")
ax.set_title("Delaunay Overlay: Constant Regularization")
plt.show()

print("Preferred sampled coefficient:", overlay_coefficient)

aplt.subplot_fit_imaging(fit=overlay_fit)


r"""
As before, inspect the peak and refine the coefficient sampling around it before making a close quantitative comparison.

Now examine the galaxy centre and bright image residuals. Has the overlay placed enough vertices across the compact component? Does the preferred smoothing reproduce its steep brightness gradient?

We have changed the mesh geometry, but two limitations remain. The overlay does not respond to the galaxy brightness, and one coefficient still controls regularization everywhere.

Next, we will use a fitted image to guide where the image-mesh points are placed, concentrating more sampling on the galaxy emission.

__Building an adapt image__

Our overlay samples the image mask uniformly. It does not distinguish bright galaxy emission from background noise.

To direct more mesh points towards the galaxy emission, we need an image describing where that emission is located. We call this the **adapt image**.

We use the model image from our initial reconstruction. Unlike the observed image, it does not contain the individual noise fluctuations that the model leaves unexplained. It provides a fitted estimate of where the galaxy light lies.
"""

adapt_image = overlay_fit.model_data

aplt.plot_array(
    array=overlay_fit.model_data,
    title="Adapt Image: Fitted Galaxy Light",
)

aplt.subplot_fit_imaging(fit=overlay_fit)

r"""
Compare the adapt image with the data and residuals. Does it locate the bright galaxy regions correctly, even if it does not yet reproduce their sharpest features?

The initial reconstruction need not be perfect to provide useful guidance. However, if it misses a galaxy component entirely, the adapt image may allocate too little resolution to that component. We should therefore inspect the starting fit before using it to guide the next mesh.

This example reconstructs one galaxy entirely on a mesh. In a combined MGE–pixelization model, decide which light the pixelization should describe before choosing its adapt image. The total model can be dominated by a smooth bulge; it need not be the best guide for irregular outer components. Inspect neighbours and residuals rather than treating a noisy residual map as known positive galaxy light.

We will keep this adapt image fixed while comparing mesh settings. It guides the placement of points; it does not fix the brightness values reconstructed on the new mesh.

---

__Placing points with a Hilbert image mesh__

We now replace the evenly spaced overlay with a **Hilbert image mesh**.

The Hilbert approach orders locations along a space-filling path through the image. Brightness-derived weights determine how points are distributed along that path, allocating more sampling to regions with larger weights.

The path itself is not the galaxy mesh. The resulting coordinates are connected directly into a Delaunay mesh, just as our overlay coordinates were.

Let’s construct a brightness-weighted image mesh using the same number of points as our retained overlay. This allows us to examine a change in their distribution without deliberately increasing their total number.
"""

hilbert_pixels = image_plane_mesh_grid.shape[0]

hilbert_image_mesh = ag.image_mesh.Hilbert(
    pixels=hilbert_pixels,
    weight_power=1.0,
    weight_floor=0.05,
)

hilbert_image_plane_mesh_grid = hilbert_image_mesh.image_plane_mesh_grid_from(
    mask=dataset.mask,
    adapt_data=adapt_image,
)

for label, mesh_grid in [
    ("Uniform Overlay", image_plane_mesh_grid),
    ("Brightness-Weighted Hilbert", hilbert_image_plane_mesh_grid),
]:
    print(f"{label}: {mesh_grid.shape[0]} image-mesh points")

    aplt.plot_array(
        array=dataset.data,
        grid=mesh_grid,
        title=label,
    )

r"""
Compare the spacing of the points across the bright galaxy regions and the surrounding background.

The overlay distributes its points evenly across the image mask. The Hilbert mesh uses the adapt image to concentrate more points on the emission, while the non-zero weight floor retains some sampling in faint regions.

These selected sky coordinates will be the reconstruction vertices.

The point distribution now responds to the fitted image brightness. As the galaxy becomes more compact, a brightness-weighted mesh can place more points across that compact emission. Its effective resolution remains limited by the image sampling, PSF and signal-to-noise; more vertices alone do not recover arbitrary small-scale structure.

Next, let’s examine how its weights are calculated.

---

__Controlling the brightness weighting__

Two parameters control how the adapt image becomes a weight map: `weight_power` and `weight_floor`.

The calculation first takes the absolute adapt-image values and divides them by their maximum. For our positive model image, this gives normalized brightnesses between zero and one.

It then raises those values to `weight_power` and replaces any weight below `weight_floor` with that minimum value.

Consider three powers:

- `0.0` gives equal weights, removing the brightness preference.
- `1.0` retains the normalized brightness weighting.
- A value greater than `1.0` increases the contrast between bright and faint regions.

For example, a normalized brightness of `0.5` becomes `0.03125` when raised to the fifth power, whereas the brightest value remains `1.0`. Larger powers therefore concentrate the sampling more strongly on the brightest emission.

Let’s inspect the weight maps and resulting point distributions together.
"""

for weight_power in [0.0, 1.0, 5.0]:
    trial_image_mesh = ag.image_mesh.Hilbert(
        pixels=hilbert_pixels,
        weight_power=weight_power,
        weight_floor=0.05,
    )

    weights = trial_image_mesh.weight_map_from(
        adapt_data=adapt_image,
    )

    weight_map = ag.Array2D(
        values=weights,
        mask=dataset.mask,
    )

    trial_mesh_grid = trial_image_mesh.image_plane_mesh_grid_from(
        mask=dataset.mask,
        adapt_data=adapt_image,
    )

    aplt.plot_array(
        array=weight_map,
        vmin=0.0,
        vmax=1.0,
        title=f"Weight Map: Power {weight_power:g}",
    )

    aplt.plot_array(
        array=dataset.data,
        grid=trial_mesh_grid,
        title=f"Image-Mesh Points: Power {weight_power:g}",
    )

r"""
Follow how increasing the power changes the allocation of points. Does the mesh resolve the brightest parts more densely? How much sampling remains across the fainter emission?

A higher power is not automatically better. Concentrating too many points on the brightest peaks can leave too few to describe the rest of the galaxy.

The weight floor helps moderate this effect. Here, every location has a weight of at least `0.05`. This is a minimum weight, not a statement that five per cent of the mesh points must lie in the background.

Next, we will vary that floor and reconstruct the galaxy with the Hilbert mesh, checking the residuals and evidence rather than judging the mesh by its appearance alone.

__Keeping resolution in the faint regions__

Increasing `weight_power` concentrates points on the brightest parts of the adapt image. However, the galaxy may also have faint extended emission that needs to be reconstructed.

The **weight floor** prevents those regions from receiving arbitrarily small weights. After applying the power, any weight below the floor is raised to that value.

For example, with a floor of `0.05`, weights of `0.001` and `0.02` both become `0.05`, while a weight of `0.8` remains unchanged.

Let’s compare three floors while keeping the power and number of mesh points fixed.
"""

for weight_floor in [0.0, 0.05, 0.5]:
    trial_image_mesh = ag.image_mesh.Hilbert(
        pixels=hilbert_pixels,
        weight_power=5.0,
        weight_floor=weight_floor,
    )

    weights = trial_image_mesh.weight_map_from(
        adapt_data=adapt_image,
    )

    weight_map = ag.Array2D(
        values=weights,
        mask=dataset.mask,
    )

    trial_mesh_grid = trial_image_mesh.image_plane_mesh_grid_from(
        mask=dataset.mask,
        adapt_data=adapt_image,
    )

    aplt.plot_array(
        array=weight_map,
        vmin=0.0,
        vmax=1.0,
        title=f"Weight Map: Floor {weight_floor:g}",
    )

    aplt.plot_array(
        array=dataset.data,
        grid=trial_mesh_grid,
        title=f"Image-Mesh Points: Floor {weight_floor:g}",
    )

r"""
Compare the points around the brightest peaks, along the fainter galaxy emission, and in the background.

Increasing the floor reduces the contrast between their weights. This retains more sampling away from the brightest emission, but leaves fewer points available to concentrate on its peaks.

Because the background can cover much more area than the galaxy light, even a modest floor can affect the allocation substantially. We should inspect the resulting mesh rather than interpret the floor as a fixed fraction of points assigned to faint regions.

For our first reconstruction, we return to the mesh already constructed with `weight_power=1.0` and `weight_floor=0.05`. These are starting choices to investigate, rather than settings we have established as optimal.

---

__Reconstructing with the Hilbert mesh__

We now use the Hilbert coordinates to build a Delaunay galaxy mesh.

The reconstruction proceeds as before: construct the triangulation at the selected sky coordinates, and solve for its vertex brightnesses. The difference is where we placed the image-mesh points.

We retain `Constant` regularization for now, so that we can investigate brightness-based mesh placement before introducing adaptive smoothing.
"""


def fit_delaunay_hilbert(coefficient):
    pixelization = ag.Pixelization(
        mesh=ag.mesh.Delaunay(
            pixels=hilbert_image_plane_mesh_grid.shape[0],
        ),
        regularization=ag.reg.Constant(
            coefficient=coefficient,
        ),
    )

    source_galaxy = ag.Galaxy(
        redshift=0.5,
        pixelization=pixelization,
    )

    galaxies = ag.Galaxies(galaxies=[source_galaxy])

    adapt_images = ag.AdaptImages(
        galaxy_image_dict={
            source_galaxy: adapt_image,
        },
        galaxy_image_plane_mesh_grid_dict={
            source_galaxy: hilbert_image_plane_mesh_grid,
        },
    )

    return ag.FitImaging(
        dataset=dataset,
        galaxies=galaxies,
        adapt_images=adapt_images,
    )


hilbert_fit = fit_delaunay_hilbert(
    coefficient=overlay_coefficient,
)

hilbert_mapper = hilbert_fit.inversion.cls_list_from(ag.Mapper)[0]

aaplt.plot_mapper(
    mapper=hilbert_mapper,
    solution_vector=hilbert_fit.inversion.reconstruction,
    mesh_grid=hilbert_mapper.source_plane_mesh_grid,
    zoom_to_brightest=False,
    title="Hilbert Mesh: Initial Galaxy Reconstruction",
)

aplt.subplot_fit_imaging(fit=hilbert_fit)

r"""
Inspect the mesh around the compact galaxy centre. Has the brightness weighting placed more vertices across the region where its brightness changes rapidly?

Then inspect the residuals around the bright galaxy regions. Additional resolution gives the reconstruction more freedom to represent their structure, but regularization still determines how freely neighbouring brightnesses can differ.

We have initially reused the overlay’s preferred coefficient. As tutorial 2 showed, changing the mesh can change the preferred smoothing strength. We therefore need to explore regularization again before comparing the two reconstructions.

---

__Comparing overlay and Hilbert reconstructions__

Let’s scan the regularization coefficient for the Hilbert mesh, using the same range as the overlay.

The adapt image and Hilbert mesh coordinates remain fixed during this scan. Each coefficient produces a new reconstruction on that mesh.
"""

hilbert_evidences = []
best_evidence = -np.inf

# Keep only the preferred full fit; other scan entries are scalar evidences.
for coefficient in overlay_coefficients:
    coefficient = float(coefficient)
    current_fit = fit_delaunay_hilbert(coefficient=coefficient)
    evidence = float(current_fit.log_evidence)
    hilbert_evidences.append(evidence)
    if evidence > best_evidence:
        best_evidence = evidence
        hilbert_fit = current_fit

hilbert_evidences = np.asarray(hilbert_evidences)
if not np.all(np.isfinite(hilbert_evidences)):
    raise ValueError("Inspect non-finite evidence values before continuing.")

best_index = int(np.argmax(hilbert_evidences))
if best_index in (0, len(overlay_coefficients) - 1):
    raise ValueError("Extend the coefficient range to bracket the hilbert peak.")

hilbert_coefficient = float(overlay_coefficients[best_index])

reference_evidence = max(
    float(np.max(overlay_evidences)),
    float(np.max(hilbert_evidences)),
)

fig, ax = plt.subplots()

for label, values in [
    ("Overlay", overlay_evidences),
    ("Hilbert", hilbert_evidences),
]:
    ax.semilogx(
        overlay_coefficients,
        values - reference_evidence,
        marker="o",
        label=label,
    )

ax.set_xlabel("Regularization coefficient")
ax.set_ylabel("Log evidence relative to the highest sampled value")
ax.legend()
plt.show()

r"""
Both curves use the same reference value, preserving the difference between their peaks. Compare their preferred coefficients as well as their peak heights.

We can now inspect each mesh at its own preferred sampled coefficient.
"""

for label, current_fit in [
    ("Overlay", overlay_fit),
    ("Hilbert", hilbert_fit),
]:
    print(f"{label}: log evidence={float(current_fit.log_evidence):.3f}")

    aplt.subplot_fit_imaging(fit=current_fit)


print(
    "Hilbert minus overlay log evidence:",
    float(hilbert_fit.log_evidence - overlay_fit.log_evidence),
)

r"""
A positive difference favours this Hilbert reconstruction over this overlay reconstruction under the comparison we have made. A negative difference favours the overlay. Refine the coefficient sampling around both peaks before interpreting a small difference.

Connect the evidence comparison to the residuals. Does the brightness-adaptive mesh better reproduce the compact emission? Are the fainter regions still adequately described?

This tests one Hilbert weighting choice, using a fixed adapt image. It does not establish that every Hilbert mesh outperforms every overlay.

We have now addressed where the galaxy points are placed. Next, we will address how to regularize their brightnesses on an irregular mesh, before allowing the smoothing strength to vary between bright and faint regions.

__Smoothing on an irregular mesh__

Our Delaunay mesh has vertices with different neighbour counts, edge lengths and directions.

With neighbour-based regularization, we penalize brightness differences along those edges. Consequently, the pattern of comparisons changes with the triangulation: one vertex might have several nearby neighbours, while another has fewer neighbours at larger distances.

**Split regularization** provides another way to measure local brightness variations.

Around each galaxy vertex, it defines four nearby sampling positions arranged in a cross. It interpolates the reconstructed brightness at those positions and compares them with the brightness at the central vertex. The size of the cross follows the local mesh scale.

This gives each vertex a common pattern of local comparisons, instead of relying only on its particular set of connected neighbours.

The additional sampling positions are not new galaxy brightness parameters. Their brightnesses are interpolated from the existing reconstruction. We therefore change how smoothing is applied without increasing the number of brightness values solved by the inversion.

It is also important to distinguish **Split** from **adaptive** regularization:

- `ConstantSplit` uses the cross-based comparisons with one coefficient across the mesh.
- Adaptive Split regularization uses those comparisons while allowing the smoothing strength to vary across the galaxy.

We will first examine `ConstantSplit`, then introduce the brightness-dependent coefficients.

---

__Reconstructing with ConstantSplit__

Let’s keep the Hilbert mesh and adapt image fixed and change the regularization scheme.

We generalize our fitting helper so that it accepts a regularization object. This will let us use the same reconstruction setup for both constant and adaptive Split regularization.
"""


def fit_hilbert_with_regularization(regularization):
    pixelization = ag.Pixelization(
        mesh=ag.mesh.Delaunay(
            pixels=hilbert_image_plane_mesh_grid.shape[0],
        ),
        regularization=regularization,
    )

    source_galaxy = ag.Galaxy(
        redshift=0.5,
        pixelization=pixelization,
    )

    galaxies = ag.Galaxies(galaxies=[source_galaxy])

    adapt_images = ag.AdaptImages(
        galaxy_image_dict={
            source_galaxy: adapt_image,
        },
        galaxy_image_plane_mesh_grid_dict={
            source_galaxy: hilbert_image_plane_mesh_grid,
        },
    )

    return ag.FitImaging(
        dataset=dataset,
        galaxies=galaxies,
        adapt_images=adapt_images,
    )


r"""
The coefficient preferred by neighbour-based regularization need not be preferred by Split regularization. We therefore repeat the coefficient scan.
"""

split_evidences = []
best_evidence = -np.inf

# Keep only the preferred full fit; other scan entries are scalar evidences.
for coefficient in overlay_coefficients:
    coefficient = float(coefficient)
    current_fit = fit_hilbert_with_regularization(
        regularization=ag.reg.ConstantSplit(coefficient=coefficient)
    )
    evidence = float(current_fit.log_evidence)
    split_evidences.append(evidence)
    if evidence > best_evidence:
        best_evidence = evidence
        split_fit = current_fit

split_evidences = np.asarray(split_evidences)
if not np.all(np.isfinite(split_evidences)):
    raise ValueError("Inspect non-finite evidence values before continuing.")

best_index = int(np.argmax(split_evidences))
if best_index in (0, len(overlay_coefficients) - 1):
    raise ValueError("Extend the coefficient range to bracket the split peak.")

split_coefficient = float(overlay_coefficients[best_index])

print("Preferred sampled Split coefficient:", split_coefficient)

aplt.subplot_fit_imaging(fit=split_fit)


print(
    "ConstantSplit minus Constant log evidence:",
    float(split_fit.log_evidence - hilbert_fit.log_evidence),
)

r"""
Compare the galaxy and residuals with the previous Hilbert reconstruction. We have kept the mesh fixed, so the changes arise from the regularization scheme and its selected coefficient.

As before, refine the coefficient sampling around the peaks before interpreting small evidence differences.

Split changes how local brightness variations are penalized, but `ConstantSplit` still applies one coefficient everywhere. It therefore retains the centre-versus-outskirts compromise identified in **Fit Problems**.

---

__Allowing different smoothing strengths__

We now let the regularization strength depend on the galaxy brightness.

The adapt image supplies a brightness-based signal estimate for each galaxy vertex through its mappings to the image. These signals are normalized so that the highest has a value of one.

We use that information to transition between two coefficients:

- `inner_coefficient` controls smoothing in high-signal regions.
- `outer_coefficient` controls smoothing in low-signal regions.

Despite their names, these coefficients are not assigned using distance from the galaxy centre. A bright off-centre component can receive the high-signal coefficient too.

For our compact galaxy, we want to explore weaker smoothing in the bright region and stronger smoothing in the faint outskirts. The `signal_scale` parameter controls how the normalized signal determines the transition between them.

We use `AdaptSplitPower` with `power=1.0`, which keeps the coefficient convention consistent with `ConstantSplit`. Setting its inner and outer coefficients equal then recovers the constant-strength Split scheme.

Let’s first try a deliberately contrasting pair of coefficients to make the spatial variation visible.
"""

adaptive_regularization = ag.reg.AdaptSplitPower(
    inner_coefficient=split_coefficient / 10.0,
    outer_coefficient=split_coefficient * 10.0,
    signal_scale=1.0,
    power=1.0,
)

adaptive_fit = fit_hilbert_with_regularization(
    regularization=adaptive_regularization,
)

adaptive_mapper = adaptive_fit.inversion.cls_list_from(ag.Mapper)[0]

adaptive_weights = adaptive_mapper.regularization.regularization_weights_from(
    linear_obj=adaptive_mapper,
)

aaplt.plot_inversion_reconstruction(
    mapper=adaptive_mapper,
    pixel_values=adaptive_weights,
    grid=adaptive_mapper.source_plane_mesh_grid,
    cb_unit="",
    zoom_to_brightest=False,
    title="Adaptive Split: Regularization Weights",
)

aaplt.plot_mapper(
    mapper=adaptive_mapper,
    solution_vector=adaptive_fit.inversion.reconstruction,
    mesh_grid=adaptive_mapper.source_plane_mesh_grid,
    zoom_to_brightest=False,
    title="Adaptive Split: Galaxy Reconstruction",
)

aplt.subplot_fit_imaging(fit=adaptive_fit)

r"""
Compare the regularization-weight map with the reconstructed galaxy. Smaller weights should occur where the adapt image indicates stronger galaxy emission, allowing greater local brightness variation. Larger weights encourage more smoothing in low-signal regions.

These weights control the regularization penalty; they are not uncertainties on the reconstructed brightnesses.

The trial coefficients were chosen to illustrate the mechanism. We have not yet established that they improve the fit or maximize the evidence. Next, we will examine the transition controlled by `signal_scale` and explore the adaptive coefficients systematically.

__Controlling the transition with `signal_scale`__

The inner and outer coefficients set the smoothing strengths at the high- and low-signal ends of the galaxy. How do we assign a strength to the regions between them?

We first raise each normalized galaxy signal to `signal_scale`. With `power=1.0`, the regularization weight is then:

\[
w = c_{\mathrm{inner}}\,s
  + c_{\mathrm{outer}}\,(1-s),
\]

where \(s\) is the normalized signal **after** applying `signal_scale`.

At \(s=1\), the weight equals the inner coefficient. At \(s=0\), it equals the outer coefficient. Intermediate values blend the two.

Let’s plot this relationship while keeping the coefficients fixed.
"""

normalized_signal = np.linspace(0.0, 1.0, 200)

inner_coefficient = split_coefficient / 10.0
outer_coefficient = split_coefficient * 10.0

fig, ax = plt.subplots()

for signal_scale in [0.5, 1.0, 3.0]:
    scaled_signal = normalized_signal**signal_scale

    weights = inner_coefficient * scaled_signal + outer_coefficient * (
        1.0 - scaled_signal
    )

    ax.plot(
        normalized_signal,
        weights,
        label=f"signal_scale={signal_scale:g}",
    )

ax.set_xlabel("Normalized galaxy signal before scaling")
ax.set_ylabel("Regularization weight")
ax.legend()
plt.show()

r"""
For a normalized signal between zero and one, increasing `signal_scale` reduces its scaled value. In this example, that moves its weight towards the stronger outer coefficient. Only the highest-signal regions retain weights close to the weaker inner coefficient.

Decreasing `signal_scale` has the opposite effect: more of the galaxy receives weights closer to the inner coefficient.

These weights are based on the fixed adapt image and its mappings. They are not recalculated from each newly reconstructed brightness distribution during this experiment.

Also distinguish `signal_scale` from the Hilbert mesh’s `weight_power`. Both transform brightness-derived information, but they control different things: `weight_power` changes **where mesh points are placed**, whereas `signal_scale` changes **how their brightnesses are regularized**.

---

__Exploring the adaptive coefficients__

We now have three regularization parameters: the inner coefficient, outer coefficient and signal scale.

Let’s explore a small grid of their values, keeping the PSF, Hilbert mesh and adapt image fixed. This makes the experiment more expensive than our one-coefficient scans, so we retain only the evidence values and the best fit encountered.

We centre the coefficient grid on the preferred `ConstantSplit` value. The grid includes equal inner and outer coefficients, allowing the adaptive scheme to recover constant-strength smoothing.

We also allow either coefficient to be larger. Although weaker smoothing in bright regions motivates the scheme, we should inspect what the evidence actually prefers.
"""

adaptive_coefficient_grid = split_coefficient * np.logspace(-1, 1, 5)
signal_scales = [0.5, 1.0, 3.0]

adaptive_trials = []
best_adaptive_fit = None
best_adaptive_parameters = None
best_adaptive_evidence = -np.inf

for inner_coefficient in adaptive_coefficient_grid:
    for outer_coefficient in adaptive_coefficient_grid:
        for signal_scale in signal_scales:
            regularization = ag.reg.AdaptSplitPower(
                inner_coefficient=float(inner_coefficient),
                outer_coefficient=float(outer_coefficient),
                signal_scale=signal_scale,
                power=1.0,
            )

            current_fit = fit_hilbert_with_regularization(
                regularization=regularization,
            )
            log_evidence = float(current_fit.log_evidence)

            if not np.isfinite(log_evidence):
                raise ValueError(
                    "Inspect the non-finite adaptive evidence before continuing."
                )

            parameters = (
                float(inner_coefficient),
                float(outer_coefficient),
                signal_scale,
            )

            adaptive_trials.append((*parameters, log_evidence))

            if log_evidence > best_adaptive_evidence:
                best_adaptive_evidence = log_evidence
                best_adaptive_parameters = parameters
                best_adaptive_fit = current_fit

best_inner, best_outer, best_signal_scale = best_adaptive_parameters

print("Preferred sampled inner coefficient:", best_inner)
print("Preferred sampled outer coefficient:", best_outer)
print("Preferred sampled signal scale:", best_signal_scale)
print("Highest sampled log evidence:", best_adaptive_evidence)

r"""
This performs 75 inversions. It is an introductory exploration, rather than a precise optimization of all three parameters.

Check whether a preferred coefficient lies at either end of its range. If so, extend that range before concluding that the optimum has been located. The three signal-scale values are also widely spaced and may require further exploration.

If the preferred inner and outer coefficients are equal, changing `signal_scale` has no effect: the weight is constant everywhere.

For a complete analysis, a non-linear search can explore these regularization parameters. Tutorial 5 demonstrates search chaining for a combined MGE–pixelization model, using constant regularization to isolate the component-decomposition question.

---

__Does adaptive regularization address the fit problem?__

Let’s inspect the highest-evidence adaptive reconstruction we have evaluated and compare it with `ConstantSplit`.

We keep the mesh fixed between these fits. This isolates the change from one smoothing strength to brightness-dependent smoothing.
"""

comparison_fits = [
    ("ConstantSplit", split_fit),
    ("Adaptive Split", best_adaptive_fit),
]

source_vmax = max(
    float(np.max(np.asarray(current_fit.inversion.reconstruction)))
    for _, current_fit in comparison_fits
)

for label, current_fit in comparison_fits:
    mapper = current_fit.inversion.cls_list_from(ag.Mapper)[0]

    aaplt.plot_mapper(
        mapper=mapper,
        solution_vector=current_fit.inversion.reconstruction,
        mesh_grid=mapper.source_plane_mesh_grid,
        vmin=0.0,
        vmax=source_vmax,
        zoom_to_brightest=False,
        title=f"{label}: Galaxy Reconstruction",
    )

    aplt.subplot_fit_imaging(fit=current_fit)

print(
    "Adaptive minus constant Split log evidence:",
    best_adaptive_evidence - float(split_fit.log_evidence),
)

r"""
Return to the two regions that motivated this experiment.

At the bright centre, does the adaptive reconstruction preserve a sharper brightness peak and reduce the corresponding image residuals? In the outskirts, does it suppress fluctuations without removing faint emission needed to explain the image?

Inspect the preferred regularization weights alongside those changes.
"""

best_adaptive_mapper = best_adaptive_fit.inversion.cls_list_from(ag.Mapper)[0]

best_adaptive_weights = best_adaptive_mapper.regularization.regularization_weights_from(
    linear_obj=best_adaptive_mapper,
)

aaplt.plot_inversion_reconstruction(
    mapper=best_adaptive_mapper,
    pixel_values=best_adaptive_weights,
    grid=best_adaptive_mapper.source_plane_mesh_grid,
    cb_unit="",
    zoom_to_brightest=False,
    title="Preferred Adaptive Regularization Weights",
)

r"""
A higher log evidence shows that this adaptive setting improves the reconstruction statistic for the fixed PSF and mesh. It is not, by itself, a model-comparison evidence marginalized over the additional regularization parameters.

If substantial residuals remain, inspect where they occur. Adaptive regularization still cannot supply resolution missing from the mesh, and both adaptive steps depend on the initial adapt image.

We have now explored the mesh placement and smoothing separately. Next, Tutorial 5 explores a complementary strategy: fitting smooth light with an MGE and additional light with a pixelization. This can reduce the burden on the mesh, but makes the interpretation of individual components depend on how their shared light is divided.

__Wrap Up__

We have separated two choices that a single regularization coefficient cannot make: where the mesh has resolution,
and how strongly its brightnesses are smoothed. The Hilbert mesh uses a fitted image to guide placement, while
adaptive Split regularization changes the local smoothing strengths on that mesh.

Judge both changes using the residuals and evidence on the same dataset. The adapt image, mesh size, PSF and noise
remain assumptions to inspect. A higher sampled log evidence does not identify the reliability of an individual knot or show
that the adaptive parameters have been fully explored.

Our Sérsic example is deliberately difficult for a coarse mesh, but an analytic profile can describe it directly.
The lesson is not to replace every analytic model with a pixelization. In the final tutorial we fit an MGE and a
pixelization together, examining both the quality of the total image fit and the ambiguity of the decomposition.

**🤖 Ask AI:** How do brightness-weighted mesh placement and adaptive regularization address different limitations,
and why can neither recover information erased by telescope blurring?

__Env__

Keep full-size datasets so the compact-galaxy experiment retains its resolved structure and evidence peaks.

ENV: full_datasets
"""
