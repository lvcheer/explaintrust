# Project page and interactive article

The Quarto site combines a concise explaintrust project page with the
interactive article "Why your SHAP plot might be lying to you". The library,
experiment outputs, and two-page technical brief are the reproducible technical
artifacts; none is peer reviewed yet.

**Numerical refresh completed (2026-09-21).** The canonical
`figures/article_results.json` bundle records the seeded configuration, package
versions, SHAP context, per-instance measurements, and article summaries. The
article claims, compatibility JSON, and figures are generated from that bundle.

## One-time setup

1. Install Quarto and the documentation dependency:

   ```bash
   python -m pip install -e ".[app,docs]"
   ```

   Quarto installation instructions: https://quarto.org/docs/get-started/

2. Regenerate the figures and data (from the repo root):

   ```bash
   python3 article/scripts/generate_figures.py
   ```

   This writes `article/figures/article_results.json`, then updates the article
   claims, `conversion.json` (the data behind the centerpiece interactive),
   `conversion_flip.png`, and `endpoints.png` from that one result bundle.

3. Rebuild the two-page technical brief:

   ```bash
   python docs/build_technical_brief.py
   ```

4. Verify that all tracked outputs match their canonical sources:

   ```bash
   python -m experiments.build_all_results --check
   ```

## Preview / render

From the `article/` directory:

```bash
quarto preview   # live preview with hot reload
quarto render    # build static site into _site/
```

The interactive cell (`{ojs}`) requires the figures to be present and is best
checked in `quarto preview`.

## Publishing

`_quarto.yml` is pre-configured for GitHub Pages (`output-dir: _site`). Update
the `repo-url` / links in `index.qmd` to your own repo, then publish the
`_site/` directory (e.g. via the `quarto publish gh-pages` command).

## Maintenance checklist

The article is a complete first draft. Before publishing an update:

1. Regenerate the result bundle, figures, and technical brief after any
   explainer, metric, or headline-evidence change.
2. Verify the OJS toggle renders in `quarto preview` (it loads
   `figures/conversion.json`).
3. Run `python -m experiments.build_all_results --check` to verify every
   generated numerical claim and artifact.
4. Render the site and check desktop and mobile layouts.
