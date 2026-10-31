"""
Optional Tutorial: Need for Speed
=================================

Return to this tutorial when a fit is too slow for your scientific needs. Faster likelihood calculations help,
but a well-organised model and search often matter just as much. Hardware cannot make an inappropriate model fit.

__Contents__

- **Search Effort:** Reduce unnecessary exploration without hiding uncertainty.
- **Likelihood Cost:** Understand the calculation for each trial model.
- **JAX and Batching:** Account for compilation, hardware and memory.
- **Measuring Runtime:** Use representative runs rather than fixed rules.
- **Data Quantity:** Retain the measurements needed for your science.

__Search Effort__

More free parameters, broad priors and difficult degeneracies can increase the work needed to explore a model.
More live points generally make exploration more thorough at additional cost. There is no reliable universal
rule such as a fixed number of likelihood evaluations per parameter.

Linear profiles remove intensities from the non-linear search. Search chaining can fix well-constrained component
geometry so subsequent searches concentrate on the remaining problem. Both can improve reliability as well as
runtime. Narrow priors or fewer live points may shorten a run while concealing good solutions: check the result.

Prodigy can be useful for quickly finding a fitted component, as in chapter 2 tutorial 6. It does not replace a
sampler when you need uncertainty estimates. The optional searches tutorial explains the distinction.

__Likelihood Cost__

Each evaluation predicts the galaxy light, accounts for the instrument and compares it with the data. Costs
include profile evaluation, over sampling, PSF convolution and, for linear profiles or pixelizations, a linear solve.
For interferometry, transforming predicted light into visibilities introduces different costs.

More profiles or reconstruction elements usually mean more work, but the overall fit can still be faster if the
resulting parameter space is easier to search. Measure the full workflow rather than judging only one operation.

__JAX and Batching__

JAX compiles supported likelihood calculations into a form that can be reused efficiently on CPUs or compatible
GPUs. The first compilation costs time. Changed shapes or model structures can require compilation again, so
this cost is not necessarily paid only once during an entire session.

Nautilus can evaluate batches of trial models; gradient optimisers can evaluate several starts together. This
parallel work may suit a GPU well, but a GPU is not automatically faster for every small dataset or model. Device
support, numerical precision, memory and compilation all matter. Installing the appropriate accelerator runtime
is also necessary; having a GPU in the machine alone does not mean JAX is using it.

For Nautilus, `n_batch` controls batching. For `MultiStartProdigy`, explicitly choose `batch_size`; setting it to
`None` processes all starts together and may require much more memory. The practicalities tutorial shows
`analysis.print_vram_use(model=model, batch_size=search.batch_size)`. Treat it as a likelihood memory estimate,
not the total process requirement or a guarantee that gradient calculations will fit in the same memory.

A NumPy path remains available via `use_jax=False` for supported analyses, useful for debugging or comparison.
Gradient searches require the accelerated differentiable path. Older advice based only on ordinary Python or
Numba likelihood timings does not describe the compiled, batched workflow.

__Measuring Runtime__

Run a representative model on the machine you intend to use. Separate initial compilation from subsequent
execution, and include search overhead and output updates when estimating the time to completion. If timing
individual JAX operations, wait for the device to finish: dispatch time alone is not execution time.

Timing a standalone `FitImaging` call is not a benchmark of the compiled batched search. A rough estimate based on
cost per evaluation times evaluation count is useful only when both numbers refer to the same execution path;
for batches, use batch throughput rather than multiplying by a single-call timing.

Inspect `search.summary` and logs, then compare representative runs with different batch sizes or model choices.
Do not compare a compilation-heavy first run against a cached or resumed run as though both did the same work.
Frequent visualisation and file writes can also dominate short fits; adjust update intervals when appropriate.

__Data Quantity__

Halving the pixel scale at fixed field of view gives four times as many pixels. Finer over sampling, a larger PSF
or a larger fitted region can add work too. However, retaining faint outskirts can be essential to constrain a
galaxy's size. Do not mask away informative residuals merely to make a fit faster.

__Wrap Up__

Use the data-and-likelihood lesson to decide which measurements belong in the fit, then assess numerical accuracy
and runtime together. Speed is useful when it lets you test more credible models and check them more carefully.
"""
