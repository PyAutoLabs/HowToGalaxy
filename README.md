# HowToGalaxy

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/2026.9.26.1/start_here.ipynb)

[Start Here on Colab](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/2026.9.26.1/start_here.ipynb) |
[Installation Guide](https://pyautogalaxy.readthedocs.io/en/latest/installation/overview.html) |
[PyAutoGalaxy readthedocs](https://pyautogalaxy.readthedocs.io/en/latest/index.html) |
[Browse Chapter 1 With Images](markdown/README.md) |
[autogalaxy_workspace](https://github.com/PyAutoLabs/autogalaxy_workspace)

<img src="https://github.com/Jammy2211/PyAutoLogo/blob/main/gifs/pyautogalaxy.gif?raw=true" width="900" />

Welcome to **HowToGalaxy** — the tutorial lecture series for [PyAutoGalaxy](https://github.com/PyAutoLabs/PyAutoGalaxy),
an open-source library for modeling the light of galaxies.

**HowToGalaxy** teaches new users how to model galaxy morphologies from scratch. It assumes minimal prior
knowledge of astronomy or statistics and takes you from first principles all the way to using
**PyAutoGalaxy** for professional scientific research.

For experienced scientists who already know the fundamentals of galaxy light profile fitting and Bayesian
modeling, the [autogalaxy_workspace](https://github.com/PyAutoLabs/autogalaxy_workspace) examples will be
more appropriate — they are concise and assume the concepts taught in **HowToGalaxy** as background.

## Chapters

- `chapter_1_introduction` — An introduction to galaxy morphology and **PyAutoGalaxy**: grids, light
  profiles, galaxies, simulated imaging data, and fitting.
- `chapter_2_modeling` — Bayesian inference, non-linear searches, and how to fit a galaxy model to CCD
  imaging data with **PyAutoGalaxy**, ending with search chaining and automated pipelines.
- `chapter_3_pixelizations` — Pixelized reconstructions (inversions) for galaxies with irregular
  morphologies, including the Bayesian formalism underpinning them.
- `chapter_4_scaling_up_galaxies` — Scaling galaxy modeling up beyond a single galaxy: extra galaxies,
  blended multi-galaxy systems, and cluster fields.
- `chapter_optional` — Optional tutorials on alternative non-linear searches and other advanced topics.

**HowToGalaxy** currently sits at four chapters. Each chapter will take around a day to work through.
We recommend completing chapters 1 and 2, then applying what you've learned to real galaxy modeling in the
`autogalaxy_workspace` before returning for the more advanced material in chapters 3 and 4.

## Getting Started

### Run in Google Colab (nothing to install)

Every tutorial opens in Google Colab in one click. There is nothing to install and no local Python
environment to set up — **PyAutoGalaxy** installs itself in the notebook's first cell. In Colab you *run*
the tutorial: edit the code, change the model, and see the output for yourself.

The `markdown` links are the same tutorial already executed and rendered on GitHub, with its real
output figures inline. Nothing runs and nothing installs — you just read it. They are good for
skimming a tutorial before running it, or for reading on a phone. Markdown pages currently exist only
for the chapter 1 tutorials listed with a `markdown` link below; every other tutorial is Colab-only.

**[Start Here](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/2026.9.26.1/start_here.ipynb)** — a one-page overview of the whole series.

- **[Chapter 1: Introduction](scripts/chapter_1_introduction/README.md)** — An introduction to galaxy morphology and **PyAutoGalaxy**: grids, light profiles, galaxies, simulated imaging data, and fitting.
  - Tutorial 0: Visualization — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/2026.9.26.1/notebooks/chapter_1_introduction/tutorial_0_visualization.ipynb) / [markdown](markdown/chapter_1_introduction/tutorial_0_visualization.md))
  - Tutorial 1: Grids And Galaxies — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/2026.9.26.1/notebooks/chapter_1_introduction/tutorial_1_grids_and_galaxies.ipynb) / [markdown](markdown/chapter_1_introduction/tutorial_1_grids_and_galaxies.md))
  - Tutorial 2: Data — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/2026.9.26.1/notebooks/chapter_1_introduction/tutorial_2_data.ipynb) / [markdown](markdown/chapter_1_introduction/tutorial_2_data.md))
  - Tutorial 3: Fitting — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/2026.9.26.1/notebooks/chapter_1_introduction/tutorial_3_fitting.ipynb) / [markdown](markdown/chapter_1_introduction/tutorial_3_fitting.md))
  - Tutorial 4: Methods — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/2026.9.26.1/notebooks/chapter_1_introduction/tutorial_4_methods.ipynb))
  - Tutorial 5: Summary — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/2026.9.26.1/notebooks/chapter_1_introduction/tutorial_5_summary.ipynb) / [markdown](markdown/chapter_1_introduction/tutorial_5_summary.md))
- **[Chapter 2: Modeling](scripts/chapter_2_modeling/README.md)** — Bayesian inference, non-linear searches, and how to fit a galaxy model to CCD imaging data with **PyAutoGalaxy**, ending with search chaining and automated pipelines.
  - Tutorial 1: Non-linear Search — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/2026.9.26.1/notebooks/chapter_2_modeling/tutorial_1_non_linear_search.ipynb))
  - Tutorial 2: Practicalities — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/2026.9.26.1/notebooks/chapter_2_modeling/tutorial_2_practicalities.ipynb))
  - Tutorial 3: Realism and Complexity — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/2026.9.26.1/notebooks/chapter_2_modeling/tutorial_3_realism_and_complexity.ipynb))
  - Tutorial 4: Dealing with Failure — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/2026.9.26.1/notebooks/chapter_2_modeling/tutorial_4_dealing_with_failure.ipynb))
  - Tutorial 5: Linear Profiles — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/2026.9.26.1/notebooks/chapter_2_modeling/tutorial_5_linear_profiles.ipynb))
  - Tutorial 6: Masking — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/2026.9.26.1/notebooks/chapter_2_modeling/tutorial_6_masking.ipynb))
  - Tutorial 7: Results — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/2026.9.26.1/notebooks/chapter_2_modeling/tutorial_7_results.ipynb))
  - Tutorial 8: Need for Speed — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/2026.9.26.1/notebooks/chapter_2_modeling/tutorial_8_need_for_speed.ipynb))
  - Tutorial 9: Search Chaining — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/2026.9.26.1/notebooks/chapter_2_modeling/tutorial_9_search_chaining.ipynb))
  - Tutorial 10: Prior Passing — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/2026.9.26.1/notebooks/chapter_2_modeling/tutorial_10_prior_passing.ipynb))
- **[Chapter 3: Pixelizations](scripts/chapter_3_pixelizations/README.md)** — Pixelized reconstructions (inversions) for galaxies with irregular morphologies, including the Bayesian formalism underpinning them.
  - Tutorial 1: Pixelizations — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/2026.9.26.1/notebooks/chapter_3_pixelizations/tutorial_1_pixelizations.ipynb))
  - Tutorial 2: Mappers — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/2026.9.26.1/notebooks/chapter_3_pixelizations/tutorial_2_mappers.ipynb))
  - Tutorial 3: Inversions — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/2026.9.26.1/notebooks/chapter_3_pixelizations/tutorial_3_inversions.ipynb))
  - Tutorial 4: Bayesian Regularization — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/2026.9.26.1/notebooks/chapter_3_pixelizations/tutorial_4_bayesian_regularization.ipynb))
  - Tutorial 5: Bayesian Formalism — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/2026.9.26.1/notebooks/chapter_3_pixelizations/tutorial_5_bayesian_formalism.ipynb))
  - Tutorial 6: Model Fit — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/2026.9.26.1/notebooks/chapter_3_pixelizations/tutorial_6_model_fit.ipynb))
- **[Chapter 4: Scaling Up Galaxies](scripts/chapter_4_scaling_up_galaxies/README.md)** — Scaling galaxy modeling up beyond a single galaxy: extra galaxies, blended multi-galaxy systems, and cluster fields.
  - Tutorial 1: Extra Galaxies — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/2026.9.26.1/notebooks/chapter_4_scaling_up_galaxies/tutorial_1_extra_galaxies.ipynb))
  - Tutorial 2: Multi Galaxy — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/2026.9.26.1/notebooks/chapter_4_scaling_up_galaxies/tutorial_2_multi_galaxy.ipynb))
  - Tutorial 3: Cluster — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/2026.9.26.1/notebooks/chapter_4_scaling_up_galaxies/tutorial_3_cluster.ipynb))
- **[Optional Chapter](scripts/chapter_optional/)** — Optional tutorials on alternative non-linear searches and other advanced topics.
  - Tutorial: Alternative Searches — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/2026.9.26.1/notebooks/chapter_optional/tutorial_searches.ipynb))

Model-fits run considerably faster on a GPU. In Colab, enable one via *Runtime* → *Change runtime type*
→ *Hardware accelerator* before running a notebook.

### Run on your own machine

Follow the
[PyAutoGalaxy installation guide](https://pyautogalaxy.readthedocs.io/en/latest/installation/overview.html),
then clone this repository:

```bash
git clone https://github.com/PyAutoLabs/HowToGalaxy.git
cd HowToGalaxy
```

The tutorials are distributed as both Jupyter notebooks (`notebooks/`) and Python scripts (`scripts/`).
We recommend the notebooks for reading — images and plots render inline, and you can step through small
code blocks interactively. Use the Python scripts for actual **PyAutoGalaxy** use, which is the workflow
chapter 3 onwards transitions you to.

## Before Chapter 1

Before starting chapter 1, complete `scripts/chapter_1_introduction/tutorial_0_visualization.py`
(or the equivalent notebook). This confirms your **PyAutoGalaxy** installation, walks you through how
images and figures display in Jupyter, and configures matplotlib for the rest of the tutorial series.

## Repository Structure

- `scripts/` — Runnable Python tutorial scripts, one subfolder per chapter.
- `notebooks/` — Jupyter notebook versions of the scripts (auto-generated; see below).
- `config/` — **PyAutoGalaxy** configuration YAML files used by the tutorials.
- `dataset/` — Tutorial datasets are generated at runtime by scripts in `scripts/simulators/` —
  no `.fits` files are committed.
- `output/` — Model-fit results (generated at runtime, not committed).

## Notebooks vs Scripts

Notebooks in `notebooks/` are generated from the Python files in `scripts/`. **Always edit the \`\`.py\`\`
scripts, never the notebooks directly.** The `# %%` markers in each script alternate between code and
markdown cells, which [PyAutoHands](https://github.com/PyAutoLabs/PyAutoHands) uses to produce the
`.ipynb` files.

## Relationship to autogalaxy_workspace

[autogalaxy_workspace](https://github.com/PyAutoLabs/autogalaxy_workspace) is the main user-facing
workspace for **PyAutoGalaxy** — concise examples, guides, and science templates aimed at users who have
a working understanding of galaxy morphology and light profile fitting. **HowToGalaxy** is the teaching
companion. Many tutorials in chapters 2–4 reference `autogalaxy_workspace` scripts as the next place to
go after the relevant concept has been introduced.

## Citations

If you use **HowToGalaxy** or **PyAutoGalaxy** in your research, please cite the references listed in
`CITATIONS.rst`.

## Community & Support

Support for **PyAutoGalaxy** is available via our Slack workspace. Slack is invitation-only; send an email
if you'd like an invite.

For installation issues, bug reports, or feature requests, raise an issue on the
[PyAutoGalaxy GitHub issues page](https://github.com/PyAutoLabs/PyAutoGalaxy/issues) (for library issues)
or the [HowToGalaxy GitHub issues page](https://github.com/PyAutoLabs/HowToGalaxy/issues) (for tutorial
content issues).
