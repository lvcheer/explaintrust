# Article: "Why your SHAP plot might be lying to you"

An explorable (interactive) article built with Quarto. It is the *public-facing*
half of the explaintrust project — the peer-reviewed / citable half is the
library itself.

## One-time setup

1. Install Quarto: https://quarto.org/docs/get-started/
2. Regenerate the figures and data (from the repo root):

   ```bash
   python3 article/scripts/generate_figures.py
   ```

   This writes `article/figures/conversion.json` (the data behind the
   centerpiece interactive), `conversion_flip.png`, and `endpoints.png`.

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

## What to write next

The `index.qmd` is a complete skeleton: the thesis, section structure, the two
figure slots, and one working interactive are in place. The `<!-- TODO -->`
markers show exactly where prose needs to be written. Suggested order:

1. Fill §"The picture that feels like understanding" and §"Four ways…" prose.
2. Verify the OJS toggle renders in `quarto preview` (it loads
   `figures/conversion.json`).
3. Write §"A worked example" around `endpoints.png`.
4. Add your own references to `references.bib`.
