"""
Start Here: HowToGalaxy
=======================

Welcome to **HowToGalaxy** — the `PyAutoGalaxy` tutorial lecture series on modeling galaxy morphology.

This script gives you a one-page overview of the series and points you to the first tutorial.

__HowToGalaxy and Natural Language__

**PyAutoGalaxy** can be used by describing an analysis in natural language to an AI coding agent, which writes and runs
the **PyAutoGalaxy** code for you. Natural language does not remove the need for expertise: to direct an analysis well,
and to recognise when a result is wrong, you still need to understand the science, the modeling assumptions and
how uncertainty is quantified.

That is what **HowToGalaxy** is for. It teaches galaxy structure and morphology from first principles, with real **PyAutoGalaxy** Python that you
run and explore yourself, so that you can then describe a galaxy model in natural language with expertise, or write it
in Python directly. Python is not going anywhere: every lecture is ordinary **PyAutoGalaxy** code.

You can also study with the [PyAutoGalaxy AI Assistant](https://github.com/PyAutoLabs/autogalaxy_assistant) in teacher
mode, which explains concepts, equations, code and results as you go. Open Claude Code, Codex or another AI coding
agent and paste:

> I want to work through the HowToGalaxy lectures (https://github.com/PyAutoLabs/HowToGalaxy) with the
> PyAutoGalaxy Assistant: https://github.com/PyAutoLabs/autogalaxy_assistant
> First clone that repository, cd into it and follow its AGENTS.md.
>
> Enter teacher mode. Show me where to find the lectures and how to use Jupyter Notebook or
> Markdown, then help me with questions as I go.

__HowToGalaxy__

**HowToGalaxy** is a four-chapter guide which takes you, step by step, from knowing nothing about galaxy
light profile fitting to being able to model real galaxy imaging data with **PyAutoGalaxy** for scientific
research.

- Chapter 1: Introduction to galaxy morphology — grids, light profiles, galaxies, simulated data, and how
  fitting works.
- Chapter 2: Practical galaxy modelling — data, models, likelihoods, linear profiles and search chaining.
- Chapter 3: Pixelized galaxy reconstruction — irregular morphology, inversions and regularization.
- Chapter 4: Scaling up galaxies — modelling multiple galaxies and cluster fields.

Each chapter is organised into numbered tutorial files: `chapter_<N>_<name>/tutorial_<M>_<topic>.py` (Python script)
or the matching `.ipynb` in `notebooks/`. Tutorials build on each other within a chapter and assume you have
completed the earlier ones.

__Recommended Path__

We recommend working through the tutorials in order:

1. First run `scripts/chapter_1_introduction/tutorial_0_visualization.py` — this checks your installation
   and sets up matplotlib for the rest of the series.
2. Work through `chapter_1_introduction` tutorial-by-tutorial.
3. Work through `chapter_2_modeling`.
4. At this point you can start modeling real galaxies using scripts in the separate
   `autogalaxy_workspace` repository — you know enough to be productive.
5. Come back for `chapter_3_pixelizations` and `chapter_4_scaling_up_galaxies` when you want to tackle more
   complex systems.

__Notebooks vs Scripts__

Every tutorial exists as both a Python script (under `scripts/`) and a Jupyter notebook (under `notebooks/`).
The notebooks are ideal for reading because plots render inline between small blocks of code. The Python
scripts are more convenient for actual **PyAutoGalaxy** use — which chapter 3 onwards starts to assume.

The notebooks are auto-generated from the Python scripts, so if you want to make changes, **edit the
Python scripts, not the notebooks.**

__Next Step__

Open `scripts/chapter_1_introduction/tutorial_0_visualization.py` (or the `.ipynb` equivalent) and start there.
"""

# from autogalaxy import setup_notebook; setup_notebook()

print(__doc__)
