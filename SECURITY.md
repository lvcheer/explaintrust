# Security policy

## Reporting a vulnerability

Please report vulnerabilities through the repository's GitHub Security tab
using a private security advisory. Do not include exploit details or sensitive
datasets in a public issue.

## Data and model safety

- The Streamlit application accepts CSV data but does not accept serialized
  Python models. Pickle and joblib files can execute arbitrary code when loaded
  and are intentionally unsupported.
- A locally run app processes uploaded CSV content on the machine hosting the
  app. A publicly deployed instance would process data on that deployment's
  server; users should not upload confidential or regulated data unless the
  operator has documented appropriate controls.
- Exported reports may contain feature names, class labels, and preprocessing
  metadata. Review them before sharing.

The current supported line is the latest code on `main`. Security fixes will be
documented in `CHANGELOG.md`.
