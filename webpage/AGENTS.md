# webpage

## Purpose
- Static public site for sysone-bench, served at `https://sysone.sdad.pro/`.
- Presents published benchmark results to humans and to AI answer engines.

## Ownership
- Owns site structure, copy, visual design, generated site data, GEO files, and Vercel deploy config.
- Does not own result artifacts, ground truth, coverage rules, or model adapters. Those belong to
  `results/`, `datasets/`, `ops/`, and `runners/`.

## Local Contracts
- No framework and no build step. Files ship as-is; there is no bundler, transpiler, or package.json.
- `data/results.json` is generated. Never edit it by hand. Regenerate with `python webpage/build-data.py`.
- `build-data.py` must stay runnable from any working directory. It resolves `ops/panel_coverage.py`
  relative to its own file, not the caller's cwd.
- `build-data.py` reads only checksum-verified run directories. A failing checksum drops the
  measurement rather than publishing it.
- Counts come from `ops/panel_coverage.py` so the site cannot drift from the repository's coverage
  number. Never hand-type a count, score, or total into markup.
- Copy counts use `<span data-count="key">`. `app.js` fills them from `results.json` using the keys
  `measurements`, `dirs`, `decisions`, `suites`, `scope`, `unmeasured`. The inner text is only a
  no-JS fallback and is expected to drift; it is never the source of truth.
- Share metadata in `<head>` and anything inside JSON-LD cannot be data-driven in a buildless site.
  Those must carry no count or score that changes when a model is added. Keep them method-level.
- Site counts must stay internally consistent: 43 distinct measurements, 46 verified run directories,
  51-entry scope, so 8 entries are unmeasured and each needs a recorded reason in `app.js` `EXCLUDED`
  and in `llms.txt`.
- The site brands itself only as `sysone-bench`. Do not add external panel, leaderboard, or edition
  attribution to site copy, `llms.txt`, or the generated data keys.
- The closed-API reference is drawn as a labelled rule or row, never merged into the open-weights ranking.
- No composite or averaged cross-suite score. No radar chart. Suite spread is shown per suite instead.
- Truncated axes must state the truncation on the chart itself.
- Charts are hand-built inline SVG in `app.js`. Keep them accessible: `role`, `aria-label`, and a
  `<title>` per mark.
- Smooth scrolling uses Lenis plus GSAP ScrollTrigger. GSAP `ScrollSmoother` is a paid Club plugin and
  is not available; do not reintroduce it.
- `html` must keep `scroll-behavior: auto` and `overscroll-behavior-y: none`. Native smooth scrolling
  fights Lenis and produces rubber-band scrolling. Drive Lenis with `lerp`, not a long `duration`.
- Anything pinned with GSAP must not use `pinSpacing: false` against a margin-offset heading, which
  collapses the flow and overlaps the following paragraph.
- Wide fixed-pixel charts, such as the heatmap, live inside an `overflow-x: auto` wrapper so they stay
  scrollable on narrow viewports. Page-level horizontal overflow must stay false.
- Reveal animations must resolve to full opacity in both normal and `prefers-reduced-motion` modes.
  No content may be left stranded invisible.
- No secrets, tokens, or keys in this directory. `vercel.json` carries headers and caching only.
- `favicon.svg` is the icon source of truth. `icon-512.png` and `apple-touch-icon.png` are rasters of
  it, and the touch icon must stay full-bleed because iOS applies its own mask.
- `og.png` is a static PNG, so anything printed on it goes stale the moment another model lands.
  It must therefore claim **no volatile fact**: not the measurement count, not the best open score,
  not any ranking or top-N. It carries only the stable method facts: sealed case count, scored
  decisions per model, closed-API reference, seed. Do not add a count back without also solving
  per-deploy card regeneration.
- Never draw a magnitude the data does not support. Equal-length bars for the nine suites were
  removed for exactly this reason; the suites are listed as a ruled list instead.
- Never type a score into the card by hand. It is 1200x630 and nothing may overflow that frame or
  collide with the footer rule; assert both.
- `llms-full.txt` is generated from `data/results.json`. It is the single self-contained answer source
  for AI engines and must never be hand-edited or left to drift from the site. Regenerate it whenever
  the data changes, alongside `og.png`.
- Every path declared in a `<link rel="icon">`, `apple-touch-icon`, or `og:image` tag must resolve to a
  committed file. An advertised-but-missing `og.png` is a broken share card, so verify each returns
  HTTP 200 after adding it.

## Work Guidance
- Design direction is an editorial data essay: warm paper, near-black ink, hairline rules, no cards,
  no panels, no shadows, no grain. It should read like a printed research report.
- Type: Newsreader for display and body, IBM Plex Mono for every number, label, and axis. Numerals
  are the primary graphic, so set them large and let the prose stay in a narrow measure.
- One accent only: oxblood `--accent` for emphasis and open-weights figures. `--closed` slate is
  reserved for the closed-API reference and must never be reused for an open model. Reimplementations
  are `--caveat` grey.
- The ranked field list is `.models` / `.model`, not a card grid. Columns are rank, name, hairline
  bar, score, flags. The bar is a 3px rule that fills, never a slab.
- `app.js` owns data colour on chart marks. CSS owns only `fill-opacity` and hover. Setting `fill` in
  CSS on `.bar` flattens the compare-top-5 encoding to one grey, so do not reintroduce it.
- Wide fixed-pixel charts need their own scroll wrapper or they widen the whole page on mobile.
- Section headings use an oxblood rule above them; charts sit on a single hairline top border.
- Keep long model names clear of their bars. Label gutters must exceed the widest truncated name at
  the rendered font size, or names run underneath the marks.
- Run a local static server for preview. `file://` will not work because `app.js` fetches the data file.

## Verification
- `node --check webpage/app.js` for syntax.
- Preview on `http://127.0.0.1:8899`. Console must be clean.
- Check at 1440px and 390px: page-level horizontal overflow must be false at both, and
  `.heatmap-scroll` must still scroll.
- Walk the full page at both widths and confirm no `.reveal` is left below full opacity, in both
  normal and `prefers-reduced-motion` modes.
- Exercise both leaderboard sort directions, both suite compare modes, and the model selector. The
  compare-top-5 view is the only place five data colours appear, so it is the regression check for
  the bar-fill layering.
- Confirm `build-data.py` reproduces `data/results.json` byte-identically apart from intended key changes.
- Verify `favicon.svg`, `icon-512.png`, `apple-touch-icon.png`, `og.png` and `llms-full.txt` all
  return HTTP 200, and that the icon is legible at 16px.
- `llms.txt` must link `llms-full.txt`, and the counts quoted in both must match `results.json`.
- Grep the directory for removed branding before publishing.

## Child DOX Index
- No child docs. This directory is a leaf.
