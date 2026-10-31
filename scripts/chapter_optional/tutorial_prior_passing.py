"""
Optional Historical Tutorial: Prior Passing
===========================================

This is historical background, not part of the recommended HowToGalaxy learning path. Older versions of the
PyAutoGalaxy / PyAutoLens workflow automatically narrowed priors between searches. That default behaviour has
been disabled. New readers are discouraged from investing time in the old prior-passing recipes: learn the
component-separation approach in chapter 2 tutorial 6 instead.

__Contents__

- **Historical Behaviour:** Why older tutorials describe Gaussian priors centred on results.
- **Current Behaviour:** Distinguish fixed instances, default priors and explicit centred models.
- **Why It Is Optional:** Understand the assumptions before reusing information from a fit.

__Historical Behaviour__

Older workflows used a first fit to create Gaussian priors for a later one. The Gaussian mean was based on the
first result, and its width used uncertainty and configurable width modifiers. An absolute modifier supplied a
width in parameter units; a relative modifier scaled with the inferred value. This was intended to guide later
searches towards plausible regions without manually copying every parameter.

This history explains older references to `result.model` as automatic prior passing. Those descriptions are not
instructions for the current API, and should not be used to interpret a modern chained fit.

__Current Behaviour__

`result.instance` supplies concrete fitted parameter values. Using a component from it fixes those values in the
next model. Linear-profile intensities remain solved inside the likelihood unless converted to ordinary profiles.
This is the behaviour used in the core chaining tutorial.

`result.model` instead supplies a model with default priors for free parameters; it does not automatically centre
those priors on the previous fit. Fixed parameters and model structure still matter, so inspect `model.info`.

Explicit centred-model tools such as `result.model_centred`, `model_centred_absolute` and `model_centred_relative`
remain available for advanced use. Automatic default narrowing has been disabled; the capability has not been
removed entirely. See `autogalaxy_workspace/scripts/guides/modeling/chaining.py` for the current API.

__Why It Is Optional__

Narrowing priors around an earlier solution can exclude better solutions or carry its mistakes forward. Reusing
the same data to construct a prior and then fitting those data again also needs careful statistical interpretation;
it is not automatically equivalent to a joint analysis with independent prior information.

__Wrap Up__

Search chaining remains valuable without this step. It lets well-constrained components be dealt with separately,
reduces the parameter space of later fits and allows different search effort at different stages. Those are the
ideas to retain for modelling real galaxies.
"""
