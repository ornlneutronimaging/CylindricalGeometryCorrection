# CylindricalGeometryCorrection

Corrects the non-uniform neutron path length through cylindrical samples in
transmission imaging (solid and hollow cylinders).

## Names

- PyPI/conda package: `neutron-geomcorr` (conda channel: `neutronimaging`)
- Python module: `neutron_geomcorr` (src layout under `src/`)
- GitHub repo: `CylindricalGeometryCorrection`

## Environment and commands

Pixi-managed — run everything through `pixi run`; never `pip install` into the
environment.

- `pixi run test` — pytest with coverage
- `pixi run build-docs` / `pixi run test-docs` — Sphinx html / doctest builds
  (sources in `documentation/source/`)
- `pixi run pre-commit run --files <files>` — lint (ruff, codespell, gitleaks,
  yamllint, taplo)
- Two environments: `default` and `jupyter` (notebook work) — see
  `[tool.pixi.environments]`

## Packaging

- Version comes from versioningit (git tags); `src/neutron_geomcorr/_version.py`
  is generated — never edit it.
- `pixi run conda-build` is a task chain: `backup-toml` → `sync-version`
  (writes the static version pixi build needs into pyproject.toml) →
  `conda-build-command` → `reset-toml` (restores pyproject.toml). If a build
  dies midway, check for a leftover `pyproject.toml.bak` / dirty
  pyproject.toml before doing anything else.
- `pixi run conda-publish` uploads to the `neutronimaging` anaconda.org
  channel via `pixi upload`.

## Branch flow

`next` (default, development) → `qa` → `main`. PRs target `next`.

## Conventions and caveats

- The corrected output of `GeometryCorrection` is `(height, 2*outer_radius - 1)`:
  the two edge columns (zero chord length) are trimmed.
- TIFF/FITS loading is direct tifffile/astropy (NeuNorm was removed in #58;
  this package does not need the scipp-based NeuNorm 2.0).
- `documentation/derivation.md` describes the *intended* Beer-Lambert
  correction; the implemented correction is linear chord-length division —
  reconciling them is tracked in issue #57. Do not present the docs and the
  code as equivalent.
- Notebooks: tutorial notebooks under `notebooks/` are committed WITH executed
  outputs so users browsing GitHub see expected results; dev notebooks should
  stay output-free and small (`check-added-large-files` is capped high for
  historical reasons — do not add more large notebooks).
- Tests must not depend on optional viz packages; image fixtures live in
  `tests/data/` (a real directory — it replaced an earlier symlink into
  `notebooks/data`, which broke Windows checkouts).
