"""
Tutorial 8: Need For Speed
==========================

In this chapter, we have learnt how to model galaxies and how to balance complexity and realism to ensure that we
infer a good model.

For fitting more complex models, the final challenge that we face is keeping the run-time low. One can easily end
up in a situation where a model-fit takes days, or longer, to fit just one image. For fitting complex models and high
resolution datasets this is somewhat unavoidable. However, it is worth us discussing what drives the long run-times of
the modeling process and how we might speed it up.

We have kept an eye on the log likelihood evaluation times and therefore expected overall run-time of the fits
performed in this chapter. Lets quickly remind ourselves of the main factors that drive the run-time, and how we
might reduce it.

The overall run-time of a model-fit is, to a good approximation, a simple product of two numbers:

 - The number of log likelihood evaluations the non-linear search performs before it converges.

 - The time each individual log likelihood evaluation takes.

Everything we discuss below reduces one (or both) of these numbers.

__Contents__

- **Searching Non-linear Parameter Space:** What drives the number of likelihood evaluations a search performs.
- **Cost Per Evaluation:** What drives the time a single log likelihood evaluation takes.
- **JAX:** How **PyAutoGalaxy** compiles the likelihood function with JAX and runs it on CPUs and GPUs.
- **Run Time Estimation:** How to estimate the overall run-time of a fit before committing to it.
- **Data Quantity:** How the amount of data fitted, and the mask applied to it, drives run-times.
- **Wrap Up:** Summary of the script and next steps.

__Searching Non-linear Parameter Space__

The time it takes for the non-linear search to sample parameter space and find the high likelihood models is driven by:

 - Dimensionality: A more complex parameter space (e.g. more parameters) takes longer to search.
 - Priors: The broader the priors on each parameter the longer the search.
 - Settings: Non-linear search settings which sample parameter space more thoroughly (e.g. a higher `n_live`) lead to
   longer run-times.

When we use only one search to fit a model, we are somewhat restricted in how we can try to achieve faster run
times by changing these 3 aspects of the search. We have already seen one powerful exception: the linear light
profiles of tutorial 5 remove every `intensity` parameter from the non-linear parameter space, reducing its
dimensionality and removing some of its most difficult degeneracies at the same time.

The final two tutorials of this chapter (tutorials 9 and 10) introduce 'non-linear search chaining', which fits
a model using multiple searches that are performed back-to-back, passing the results of earlier searches to
later ones. A key motivation for this is that it gives us a lot more flexibility in juggling the dimensionality,
priors and settings so as to perform faster and more efficient modeling.

In the optional **HowToGalaxy** tutorial `chapter_optional/tutorial_searches.py` we discuss other non-linear searches
supported by **PyAutoGalaxy** which use a different approach to sample parameter space than `Nautilus`. For those
familiar with statistical inference, this includes maximum likelihood estimators and MCMC algorithms.

For galaxy modeling, there are maximum likelihood estimator methods (e.g. the Levenberg-Marquardt) that for simple
models (e.g. a single `Sersic`) can often reliably infer the maximum likelihood model, using ten times or fewer
likelihood evaluations than `Nautilus` and therefore running over ten times faster. However, these methods do
not infer reliable errors and are prone to becoming trapped in local maxima. Nevertheless, users modeling large
galaxy samples may wish to investigate these methods.

__Cost Per Evaluation__

The second number is the time a single log likelihood evaluation takes, which is set by every operation
**PyAutoGalaxy** performs to fit the galaxy data with a model, for example:

 - Computing the intensity values of every light profile on the image's grid of (y,x) coordinates.
 - Over-sampling: sub-dividing image pixels into finer sub-pixels where the light profile changes rapidly (e.g. the
   centre of the galaxy), which multiplies the number of intensity evaluations performed.
 - Convolving the model image of the galaxies with the telescope's PSF to compare it to the data. The larger the
   PSF, the more expensive this convolution becomes.
 - For linear light profiles, basis functions like the MGE of tutorial 5 and the pixelizations of chapter 3, solving
   the linear algebra of the inversion, which computes the `intensity` values that best fit the data.

More complex fitting techniques therefore cost more per evaluation: a couple of standard light profiles are cheap,
an MGE costs more (its inversion solves for many Gaussians), and pixelized reconstructions cost the most.
As we saw in tutorial 5, this trade often still favours the more advanced technique, because the simpler parameter
space it produces reduces the number of evaluations the search needs.

__JAX__

How are these operations made fast? **PyAutoGalaxy** uses JAX (https://github.com/jax-ml/jax), Google's numerical
computing library, which just-in-time (JIT) compiles the entire log likelihood function into optimized machine code.

JIT compilation means the first time the likelihood function is called, JAX traces through every operation it
performs (light profile evaluation, over-sampling, PSF convolution, the linear algebra of an inversion) and compiles
them into a single optimized program. This compilation is a one-off overhead, typically taking seconds to a couple
of minutes depending on model complexity. Every one of the tens of thousands of likelihood evaluations that follow
then reuses the compiled program and runs dramatically faster than ordinary Python. When a search begins you will
see a log message like `JAX jit compiling vectorized (vmap) likelihood function, could take seconds or minutes...`
-- that is the compilation happening.

JAX's second superpower is that the same compiled code runs on either a CPU or a GPU. On a GPU, likelihood
evaluations are not only faster individually but can be batched, with many evaluated simultaneously (the
`n_batch` search input we have set throughout this chapter controls the size of these batches). GPU speed-ups
over CPU of an order of magnitude or more are common, and they grow with the size of the dataset and model. If you
have access to a GPU (even a modest laptop GPU), it is the single biggest speed-up available to you.

You do not need to do anything to switch this on. JAX installs with `autogalaxy` by default, and every
`AnalysisImaging` object we have created in this chapter defaults to `use_jax=True` and the search compiles and
batches the likelihood function automatically. You can force the plain NumPy path with
`ag.AnalysisImaging(dataset=dataset, use_jax=False)` (or by setting the environment variable
`PYAUTO_DISABLE_JAX=1`), which is mainly useful for debugging, as NumPy errors and stack traces are much easier
to read than JAX ones.

The one trade-off to keep in mind is the compilation overhead itself. For a full model-fit performing tens of
thousands of likelihood evaluations it is negligible. For a quick one-off calculation (e.g. fitting galaxies to
data once via a `FitImaging`, as we did in chapter 1) compilation can take longer than the calculation it speeds
up, which is why short interactive computations sometimes feel slower than you might expect the first time they run.

Long-time users may remember that **PyAutoGalaxy** previously used `numba`, which recompiles individual Python
functions into fast machine code. JAX supersedes it: rather than accelerating functions one at a time, it compiles
and optimizes the likelihood function end-to-end, and adds GPU support and batching on top.

Therefore, **PyAutoGalaxy** is pretty well optimized and there are no 'low hanging fruit' speed ups available by
writing the code in a different language.

__Run Time Estimation__

Because run-time is the product of the two numbers above, we can estimate it before committing to a fit:

 - Time the log likelihood evaluation: perform a fit of the model to the data once (e.g. via a `FitImaging`) and
   time it, remembering to discard the first call, which includes JIT compilation.

 - Estimate the number of evaluations: for `Nautilus` a reasonable rule of thumb is of order 10000 evaluations per
   free parameter, with simpler parameter spaces (e.g. linear light profiles, chained searches with tight priors)
   converging in fewer.

Multiplying the two gives a ballpark overall run-time. In tutorial 2 we used exactly this arithmetic: a log
likelihood evaluation time of ~0.05 seconds and 50000-100000 evaluations gave expected run-times of roughly 30
minutes on CPU and 10 minutes on GPU. If your own estimate comes out at days or weeks, that is the moment to
simplify the model, tighten the search settings, reach for search chaining or find a GPU -- before starting the
fit, not after.

On a GPU, there is one other pre-flight check: `analysis.print_vram_use(model=model, batch_size=search.batch_size)`
estimates whether the fit will fit within the GPU's memory for your chosen batch size.

__Data Quantity__

The final factor driving run-speed is the quantity of data that is fitted. For every image-pixel that we fit,
we have to compute the light profile intensities and convolve them with the telescope's PSF. The larger that PSF is,
the more convolution operations we have to perform too.

In the previous exercises, we used images with a pixel scale of 0.1". This value is relatively low resolution: most
Hubble Space Telescope images have a pixel scale of 0.05", which is four times the number of pixels! Some telescopes
observe at scales of 0.03" or, dare I say it, 0.01". At these resolutions things can *really* slow down, if we
do not think carefully about run speed beforehand.

There are ways that we can reduce the number of image-pixels we fit, via masking. If we mask out more of the image,
we will fit fewer pixels and **PyAutoGalaxy** will run faster. If you want the best, most perfect model possible,
aggressive masking and cutting the data in this way is a bad idea, as discussed in tutorial 6.

__Wrap Up__

This tutorial simply wanted to get you thinking about *why* a model takes as long to fit as it does: the number
of likelihood evaluations the search performs, times the cost of each evaluation, with JAX's compiled likelihood
function (and a GPU, if you have one) driving the latter down as far as it will go.

The remaining tutorials of this chapter attack the other half of the product: search chaining (tutorial 9) and prior
passing (tutorial 10) reduce the number of evaluations needed to fit complex models, laying the groundwork for the
pixelized reconstructions of chapter 3.
"""

# from autogalaxy import setup_notebook; setup_notebook()
