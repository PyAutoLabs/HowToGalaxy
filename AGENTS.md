# HowToGalaxy — Agent Instructions

This is the **HowToGalaxy** tutorial lecture series for **PyAutoGalaxy**, a Python library for galaxy
morphology modeling. Tutorials teach new users how to model galaxy light from first principles. It is
the teaching companion to `../autogalaxy_workspace`. These are the canonical, agent-agnostic
instructions for this repo.

## Repository Structure

- `scripts/` — Runnable Python tutorial scripts:
  - `chapter_1_introduction/` — Grids, light profiles, galaxies, data, fitting
  - `chapter_2_modeling/` — Non-linear searches, Bayesian inference, galaxy modeling
  - `chapter_3_search_chaining/` — Search chaining, prior passing, automated pipelines
  - `chapter_4_pixelizations/` — Pixelized reconstruction, inversions, regularization
  - `chapter_optional/` — Alternative searches and advanced topics
  - `simulators/` — Simulator scripts that generate the tutorial datasets at runtime
- `notebooks/` — Jupyter versions, generated from `scripts/` (do not edit directly)
- `config/` — PyAutoGalaxy configuration YAML
- `dataset/` — Empty in the repo; written at runtime by the simulator scripts
- `output/` — Model-fit results (generated at runtime, not committed)

## Running Scripts

Scripts are run **from the repo root** so relative paths to `dataset/` and `output/` resolve. A
tutorial that needs a dataset invokes the matching `scripts/simulators/` script via `subprocess` if
the dataset folder is absent — no manual simulate-then-run step.

```bash
python scripts/chapter_1_introduction/tutorial_1_grids_and_galaxies.py
```

Fast mode for integration: `PYAUTO_TEST_MODE=1` skips sampling (`=2` also bypasses; combine with
`PYAUTO_SKIP_FIT_OUTPUT=1 PYAUTO_SKIP_VISUALIZATION=1 PYAUTO_SKIP_CHECKS=1 PYAUTO_FAST_PLOTS=1` for a
fast smoke run). **Dataset realism:** `PYAUTO_SMALL_DATASETS` is deliberately **not** used in
HowToGalaxy — tutorials assume the full-resolution simulated datasets (unlike the workspaces, which
cap grids/masks).

## Testing

On CI, every PR is gated on Python **3.12 and 3.13** by `smoke_tests.yml` (runs
`python .github/scripts/run_smoke.py`, driven by `smoke_tests.txt` + `config/build/env_vars.yaml` —
the definition of green), `navigator_check.yml` (PyAutoBuild's reusable navigator-catalogue check;
see *Notebooks vs Scripts*), and `url_check.yml` (link checking). The smoke and navigator jobs check
out **PyAutoBuild** as a sibling and run the PyAuto* libraries from the **same-named branch** of each
source repo, so a HowToGalaxy PR is validated against matching library branches.

## Sandboxed / restricted runs

If `numba` or `matplotlib` cannot write to the default cache locations, point them at writable dirs:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python scripts/chapter_1_introduction/tutorial_1_grids_and_galaxies.py
```

## Notebooks vs Scripts

Notebooks in `notebooks/` are **generated** from the `.py` scripts via PyAutoBuild. **Always edit the
`.py` scripts, never the `.ipynb` directly.** The `# %%` marker alternates code and markdown cells.
Regenerate from the repo root:

```bash
PYTHONPATH=../PyAutoBuild/autobuild python3 ../PyAutoBuild/autobuild/generate.py howtogalaxy
```

The `howtogalaxy` project target is registered in PyAutoBuild (`run_all.py`, `navigator.py`,
`config/`). The navigator catalogue — `llms-full.txt` + `workspace_index.json` — is what
`navigator_check.yml` gates; it is rebuilt by the same PyAutoBuild generate/merge flow that builds
the notebooks. Commit regenerated notebooks and catalogue alongside the script changes.

## Bulk-edit safety

When editing the same region across many scripts in one pass, only rewrite the targeted region.
**Never produce a whole-file write unless you have read the entire current file** — a whole-file
write from a header skim silently deletes every section below the header.

## Scientific Context

When a tutorial benefits from framing a galaxy concept against a real scientific application —
light profiles + MGE, regularisation, pixelisation, bulge/halo decomposition, kinematics — pull from
the `autolens_assistant` literature wiki at https://github.com/PyAutoLabs/autolens_assistant
(`wiki/literature/` — concepts, entities, sources). If cloned as a sibling, read it locally at
`../autolens_assistant/wiki/literature/`. It is currently lensing-focused, but several galaxy-modelling
concepts (source reconstruction, bulge/halo decomposition) are directly useful.

## Related Repos

- `../PyAutoGalaxy` — source library.
- `../autogalaxy_workspace` — the user-facing workspace (tutorials point here as the next destination).
- `../PyAutoBuild` — notebook generation + CI tooling.
- `../autolens_assistant` — science-assistant workspace (literature wiki; see *Scientific Context*).

## Task Workflows

**`[API Update]` issues:** find every renamed/moved/removed/changed public API, update each tutorial
script (preserving the teaching prose), run `python .github/scripts/run_smoke.py`, and fix `[FAIL]`
entries until the summary passes; regenerate notebooks + catalogue after. **General issues:** edit
only files in `scripts/` (never `notebooks/`), preserve docstrings and explanations, test, then
regenerate. Flag any change that affects `autogalaxy_workspace` or the source libraries in your PR.

## Clean state

Never rewrite history on a repo with a remote (no `git init` over a tracked tree, no force-push to
`main`, no rebasing pushed shared branches). To reset a dirty tree the only correct sequence is:

```bash
git fetch origin
git reset --hard origin/main
git clean -fd
```
