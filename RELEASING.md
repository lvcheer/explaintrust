# Release checklist

No release is published automatically. A maintainer must complete and verify
each step.

1. Confirm the working tree is clean. Recheck that the intended project name is
   available on PyPI; a prior availability check does not reserve it.
2. Choose the release version and update it consistently in
   `pyproject.toml`, `explaintrust/__init__.py`, and `CITATION.cff`.
3. Add `date-released` to `CITATION.cff`. Move relevant entries from
   `CHANGELOG.md`'s Unreleased section into a dated
   release section and update its comparison links.
4. Run the quality checks:

   ```bash
   python -m pytest -q
   python examples/demo.py
   python experiments/calibrate_thresholds.py
   python experiments/benchmark_real_data.py
   python article/scripts/generate_figures.py
   python docs/build_technical_brief.py
   python -m experiments.build_all_results
   python -m experiments.build_all_results --check
   git diff --check
   quarto render article
   ```

   Review every changed canonical result and generated public artifact before
   continuing. Write mode updates generated targets only; if a checked manual
   claim is stale, revise that prose deliberately and rerun check mode.

5. Build and inspect the distribution from a clean tree:

   ```bash
   python -m pip install -e ".[docs,release]"
   python -m build
   python -m twine check dist/*
   ```

6. Install the wheel in a fresh environment and run an import smoke test.
7. Commit the release metadata, create an annotated `vX.Y.Z` tag, and push the
   commit and tag.
8. Create a GitHub release from the tag and attach the wheel and source archive.
9. Publish to TestPyPI first. Verify installation and the project page before
   publishing the same artifacts to PyPI.

Prefer PyPI trusted publishing rather than long-lived API tokens. Configure the
publisher in the PyPI project settings before adding an automated upload
workflow.
