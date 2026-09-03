# HowToGalaxy — Agent Instructions

This is the **HowToGalaxy** tutorial lecture series for **PyAutoGalaxy**, a Python library for galaxy
morphology modeling. Tutorials teach new users how to model galaxy light from first principles. It is
the teaching companion to `../autogalaxy_workspace`. These are the canonical, agent-agnostic
instructions for this repo.

## Repository Structure

- `scripts/` — Runnable Python tutorial scripts:
  - `chapter_1_introduction/` — Grids, light profiles, galaxies, data, fitting
  - `chapter_2_modeling/` — Non-linear searches, Bayesian inference, galaxy modeling
  - `chapter_3_pixelizations/` — Pixelized reconstruction, inversions, regularization, the Bayesian
    formalism
  - `chapter_4_scaling_up_galaxies/` — Extra galaxies, blended multi-galaxy systems, cluster fields
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
fast smoke run). **Dataset realism:** automated runs **do** cap datasets —
`config/build/profile_smoke.yaml` sets `PYAUTO_SMALL_DATASETS: "1"` for every script, the same as the
workspaces. (This paragraph previously claimed the opposite; the claim was untrue and went unnoticed
because the chapters that break under the cap were never in the smoke list.) Tutorials must therefore
work at **both** resolutions: never hardcode an index or a shape derived from the full-resolution
dataset. `chapter_3_pixelizations/tutorial_3_inversions.py` is the cautionary case — it sized its mesh
from `dataset.shape_native`, giving 10000 mesh pixels at full resolution but 256 under the cap, so its
fixed `pix_indexes` ran off the end.

## Testing

On CI, every PR is gated on Python **3.12 and 3.13** by `smoke_tests.yml` (runs
`python .github/scripts/run_smoke.py`, which runs **every** script under `scripts/` except the
exclusions in `config/build/no_run.yaml`, with per-script env from `config/build/profile_smoke.yaml` —
the definition of green), `navigator_check.yml` (PyAutoHands's reusable navigator-catalogue check;
see *Notebooks vs Scripts*), and `url_check.yml` (link checking). The smoke and navigator jobs check
out **PyAutoHands** as a sibling and run the PyAuto* libraries from the **same-named branch** of each
source repo, so a HowToGalaxy PR is validated against matching library branches.

## Sandboxed / restricted runs

If `numba` or `matplotlib` cannot write to the default cache locations, point them at writable dirs:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python scripts/chapter_1_introduction/tutorial_1_grids_and_galaxies.py
```

## Notebooks vs Scripts

Notebooks in `notebooks/` are **generated** from the `.py` scripts via PyAutoHands. **Always edit the
`.py` scripts, never the `.ipynb` directly.** The `# %%` marker alternates code and markdown cells.
Regenerate from the repo root:

```bash
PYTHONPATH=../PyAutoHands/autohands python3 ../PyAutoHands/autohands/generate.py howtogalaxy
```

The `howtogalaxy` project target is registered in PyAutoHands (`run_all.py`, `navigator.py`,
`config/`). The navigator catalogue — `llms-full.txt` + `workspace_index.json` — is what
`navigator_check.yml` gates; it is rebuilt by the same PyAutoHands generate/merge flow that builds
the notebooks. Commit regenerated notebooks and catalogue alongside the script changes.

## Bulk-edit safety

When editing the same region across many scripts in one pass, only rewrite the targeted region.
**Never produce a whole-file write unless you have read the entire current file** — a whole-file
write from a header skim silently deletes every section below the header.

## Scientific Context

When a tutorial benefits from framing a galaxy concept against a real scientific application —
Sersic profiles, light profiles + MGE, pixelisation, bulge/disk decomposition, isophotes,
scaling relations, kinematics — pull from the `autogalaxy_assistant` literature wiki at
https://github.com/PyAutoLabs/autogalaxy_assistant (`wiki/literature/` — concept pages,
survey/instrument entities, per-topic annotated bibliographies, every citation verified). If
cloned as a sibling, read it locally at `../autogalaxy_assistant/wiki/literature/`.

## Related Repos

- `../PyAutoGalaxy` — source library.
- `../autogalaxy_workspace` — the user-facing workspace (tutorials point here as the next destination).
- `../PyAutoHands` — notebook generation + CI tooling.
- `../autogalaxy_assistant` — the PyAutoGalaxy science-assistant workspace (literature wiki; see *Scientific Context*).

## Task Workflows

**`[API Update]` issues:** find every renamed/moved/removed/changed public API, update each tutorial
script (preserving the teaching prose), run `python .github/scripts/run_smoke.py`, and fix `[FAIL]`
entries until the summary passes; regenerate notebooks + catalogue after. **General issues:** edit
only files in `scripts/` (never `notebooks/`), preserve docstrings and explanations, test, then
regenerate. Flag any change that affects `autogalaxy_workspace` or the source libraries in your PR.

<!-- repos_sync:history:begin -->
## Never rewrite history

Never rewrite pushed history on any repo with a remote — no `git init` over a
tracked repo, no force-push to `main`, no fresh-start "Initial commit", no
`filter-repo` / `filter-branch` / `rebase -i` on pushed branches. To get a
clean tree: `git fetch origin && git reset --hard origin/main && git clean -fd`.
<!-- repos_sync:history:end -->

<!-- repos_sync:deliverable:begin -->
## Sessions end at their deliverable

A session ends when it reports its deliverable — never arm anything that
outlives the turn to wait for CI, a review or a merge: no `send_later`, no
`subscribe_pr_activity`, no `CronCreate`, no `ScheduleWakeup`, no `/loop`, no
`RemoteTrigger` create/update/run. Judge once, report, stop; the human re-runs
`/prm` (or the batch review) when it is green. Measured: five batch members
armed hourly check-ins on 2026-08-31, and a mobile `/prm` re-armed a 60-minute
`send_later` hourly all night on 2026-09-03 with no task active, draining usage.
<!-- repos_sync:deliverable:end -->
