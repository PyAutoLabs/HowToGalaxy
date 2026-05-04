# HowToGalaxy

[Installation Guide](https://pyautogalaxy.readthedocs.io/en/latest/installation/overview.html) |
[PyAutoGalaxy readthedocs](https://pyautogalaxy.readthedocs.io/en/latest/index.html) |
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
  imaging data with **PyAutoGalaxy**.
- `chapter_3_search_chaining` — Chaining multiple non-linear searches together to build automated galaxy
  modeling pipelines for complex systems.
- `chapter_4_pixelizations` — Pixelized source reconstructions (inversions) for galaxies with irregular
  morphologies.
- `chapter_optional` — Optional tutorials on alternative non-linear searches and other advanced topics.

**HowToGalaxy** currently sits at four chapters. Each chapter will take around a day to work through.
We recommend completing chapters 1 and 2, then applying what you've learned to real galaxy modeling in the
`autogalaxy_workspace` before returning for the more advanced material in chapters 3 and 4.

## Getting Started

You can run the tutorials on your own machine by following the
[PyAutoGalaxy installation guide](https://pyautogalaxy.readthedocs.io/en/latest/installation/overview.html),
then cloning this repository:

```bash
git clone https://github.com/PyAutoLabs/HowToGalaxy.git
cd HowToGalaxy
```

Alternatively, every tutorial can be opened directly in Google Colab via the links in each chapter's
`README.md`.

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
markdown cells, which [PyAutoBuild](https://github.com/PyAutoLabs/PyAutoBuild) uses to produce the
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
