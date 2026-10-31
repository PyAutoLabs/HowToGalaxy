Look at a resolved galaxy image. Its broad light distribution may follow a smooth bulge and disc, while smaller regions contain bright star-forming knots, off-centre clumps or irregular extensions. A model can capture the overall galaxy yet leave these features clearly visible in the residual image.

In Chapter 2, we introduced linear light profiles and multi-Gaussian expansions (MGEs). Combining Gaussians of different widths gives us a more flexible radial brightness distribution than a single Sérsic profile. Gaussian groups with different ellipticities and orientations can also describe changes in flattening and isophotal twists. Fixed widths and linearly solved intensities provide this flexibility without introducing a large number of non-linear parameters.

Our worked example nevertheless imposed a particular geometry: the Gaussians within each galaxy’s group shared a common centre and elliptical shape. Adding more Gaussians improves our control over the brightness distribution around that centre, but does not create independently positioned knots or several components with distinct centres.

We could introduce offset Gaussian groups. The limitation is not that MGEs cannot represent multiple centres, but that increasingly complicated galaxies require us to choose and fit an increasingly elaborate arrangement of components.

Suppose the MGE reproduces the smooth galaxy light but leaves compact, structured residuals. These may indicate that the model needs more freedom to describe local features. They are not automatically evidence for additional galaxy components: an inaccurate PSF, background estimate, contaminating object or other data-treatment error can also leave structured residuals.

In this chapter, we introduce pixelized reconstruction as another way to describe the galaxy’s light. We infer brightness values across a mesh without assigning a separate analytic centre and shape to every feature. The predicted image includes telescope blurring, so the reconstruction is an inferred description of the galaxy rather than simply a copy of the observed pixels.

This builds on the linear profiles of Chapter 2. Instead of solving for Gaussian intensities, we solve for the mesh brightness values. Their extra freedom can describe irregular structure, but can also fit noise. Regularization supplies a prior that controls variations between neighbouring brightness values.

The reconstruction still depends on modelling choices, including the mesh, its resolution, regularization and the treatment of the data. We will learn to inspect how those choices affect both the reconstructed galaxy and the image residuals.

MGE and other analytic profiles remain useful for describing and measuring smooth galaxy components. Pixelizations complement them when more complicated structure needs to be represented, including in models that combine analytic components with a pixelized reconstruction.

# Files

- [Tutorial 1: Pixelizations](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/main/notebooks/chapter_3_pixelizations/tutorial_1_pixelizations.ipynb) — Meshes, indexed mappings, inversions and telescope blurring.

- [Tutorial 2: Bayesian Regularization](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/main/notebooks/chapter_3_pixelizations/tutorial_2_bayesian_regularization.ipynb) — Interpolation, smoothness priors, evidence and resolution experiments.

- [Tutorial 3: Model Fit](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/main/notebooks/chapter_3_pixelizations/tutorial_3_model_fit.ipynb) — An MGE fit followed by a combined MGE and pixelized model.
