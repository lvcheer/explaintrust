# Changelog

All notable changes are recorded here. This project follows
[Semantic Versioning](https://semver.org/) and the
[Keep a Changelog](https://keepachangelog.com/) structure.

## [Unreleased]

## [0.1.0] - 2026-09-04

### Added

- Initial explainers, diagnostic metrics, trust report, Streamlit demo,
  synthetic experiments, and real-data benchmark.
- JSON report export with method context and preprocessing audit metadata.
- Configurable, validated report thresholds.
- Held-out synthetic calibration and refreshed real-data benchmark outputs.
- GitHub Actions tests for Python 3.9 and 3.12.

### Changed

- SHAP, LIME, and perturbation metrics now use a consistent model-output space.
- LIME coefficients are returned in original feature units.
- Infidelity normalization and max-sensitivity now match their documented
  baselines and formulas.
- Rank-based comparisons use attribution magnitude; subgroup consistency is no
  longer described as direct distribution-shift validation.
- Positive overall verdicts were replaced with the more limited “no issues
  detected” wording; unavailable checks produce insufficient evidence.

### Security

- Removed unsafe pickle/joblib model upload from the Streamlit application.

[Unreleased]: https://github.com/lvcheer/explaintrust/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/lvcheer/explaintrust/releases/tag/v0.1.0
