# Contributing

Thank you for helping improve explaintrust. Changes should preserve the
distinction between measuring model behaviour and making causal or normative
claims about an explanation.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[app,test]"
python -m pytest -q
```

Run the reference pipeline with:

```bash
python examples/demo.py
```

## Change expectations

- Add a regression or property test for every metric or explainer change.
- State the attribution type and output space expected by a metric.
- Label project-specific diagnostics and distinguish them from published metric
  definitions.
- Keep seeded operations reproducible and record sample counts and thresholds.
- Do not interpret SHAP or LIME output as causal evidence without a separate
  identification argument.
- Update generated experiment outputs and article numbers when their producing
  code changes.

Names exported from `explaintrust.__all__` are the supported public API.
Breaking changes require a major version bump; during `0.x`, incompatible API
changes require a minor version bump and a changelog entry.

Before opening a pull request, run the tests, the headless demo, and
`git diff --check`. For release-related changes, also follow `RELEASING.md`.
