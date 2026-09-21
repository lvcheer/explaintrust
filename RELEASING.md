# Release checklist

No release is published automatically. A maintainer must complete and verify
each step.

1. Confirm the working tree is clean. Recheck that the intended project name is
   available on PyPI; a prior availability check does not reserve it.
2. Choose the release version and update it consistently in
   `pyproject.toml`, `explaintrust/__init__.py`, and `CITATION.cff`.
3. Add `date-released` to `CITATION.cff`. Move relevant entries from
   `CHANGELOG.md`'s Unreleased section into a dated
   release section and update its comparison links. Update
   `RELEASE_NOTES.md` with user-visible changes, protocol changes,
   non-comparability notes, and known limitations.
4. Review `LICENSE` and `THIRD_PARTY_LICENSES.md` against the direct dependency
   metadata in the candidate environment. The repository must not vendor new
   third-party source, data, fonts, or media without recording its license.
5. Run the quality checks:

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

6. Build and inspect the distribution from a clean tree:

   ```bash
   python -m pip install -e ".[docs,release]"
   python -m build
   python -m twine check dist/*
   ```

7. Install the wheel and source distribution separately in fresh environments;
   run the import smoke test and `python -m pip check` in each.
8. Commit the release metadata, create an annotated `vX.Y.Z` tag, and push the
   commit and tag.
9. Create a GitHub release from the tag and attach the wheel and source archive.
10. Publish to TestPyPI first. Verify installation and the project page before
   publishing the same artifacts to PyPI.

For a release candidate, use a PEP 440 version such as `0.2.0rc1` consistently
and tag it `v0.2.0rc1`. Do not reuse candidate artifacts for the final release:
update the version, rebuild from the final tag, and repeat the clean-install
checks.

Prefer PyPI trusted publishing rather than long-lived API tokens. Configure the
publisher in the PyPI project settings before adding an automated upload
workflow.
