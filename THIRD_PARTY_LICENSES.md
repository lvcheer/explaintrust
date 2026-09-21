# Third-party licenses

explaintrust is distributed under the MIT License in `LICENSE`. The repository
does not vendor third-party source code, fonts, or benchmark datasets. Python
dependencies are installed separately and remain governed by their own
licenses.

The direct dependency metadata was reviewed for the `0.2.0` release on
21 September 2026. This table records the project or package license reported
by the reviewed distributions or upstream package metadata; binary wheels may
contain additional notices, which remain authoritative in the installed
distribution. XGBoost was not installed in the core verification environment,
so its declared license must be rechecked when testing that optional extra.

| Dependency | Use | Reported license |
|---|---|---|
| NumPy | runtime | BSD-3-Clause, with bundled-component notices |
| pandas | runtime | BSD-3-Clause, with bundled-component notices |
| SciPy | runtime | BSD-3-Clause, with bundled-component notices |
| scikit-learn | runtime | BSD-3-Clause |
| SHAP | runtime | MIT |
| LIME | runtime | BSD |
| tqdm | runtime | MPL-2.0 and MIT |
| Streamlit | optional application | Apache-2.0 |
| Plotly | optional application | MIT |
| Matplotlib | optional application and figures | PSF-based Matplotlib license |
| XGBoost | optional model | Apache-2.0 |
| ReportLab | optional documentation build | BSD |

The Adult and Diabetes source datasets are downloaded only when their benchmark
loader is run; their source files are ignored by Git and are not redistributed
in the wheel or source archive. Experiment outputs derived from those datasets
remain subject to the source dataset terms described by their providers.

Before each release, inspect the exact installed distributions and preserve all
license files and notices included in their wheels. Development and release
tools such as pytest, build, and twine are not runtime dependencies and are not
redistributed with explaintrust.
