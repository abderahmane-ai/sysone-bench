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
- Anything pinned with GSAP must not use `pinSpacing: false` against a margin-offset heading, which
  collapses the flow and overlaps the following paragraph.
- Wide fixed-pixel charts, such as the heatmap, live inside an `overflow-x: auto` wrapper so they stay
  scrollable on narrow viewports. Page-level horizontal overflow must stay false.
- Reveal animations must resolve to full opacity in both normal and `prefers-reduced-motion` modes.
  No content may be left stranded invisible.
- No secrets, tokens, or keys in this directory. `vercel.json` carries headers and caching only.

## Work Guidance
- Design direction is Instrument: dark neutral surfaces, Archivo with JetBrains Mono for numerals,
  fine grain, one accent hue re-hued per section band, a fixed measurement rail on desktop.
- The fixed rail is hidden below 900px. Verify mobile separately at 390px.
- Keep bars narrow and centred in their band. Cap chart width so a 1000-unit viewBox does not stretch
  into slabs across a wide card.
- Rotated axis labels need padding below the plot or they clip.
- Run a local static server for preview. `file://` will not work because `app.js` fetches the data file.

## Verification
- `node --check webpage/app.js` for syntax.
- Preview on `http://127.0.0.1:8899` and check the console is clean, page horizontal overflow is false,
  all 43 monogram marks render, all 9 suite multiples render, and the rail shows 7 ticks.
- Walk the full page at 1440px and 390px to confirm no reveal is stuck faded, then screenshot the
  suites, heatmap, precision, coverage, and footer bands.
- Confirm `build-data.py` reproduces `data/results.json` byte-identically apart from intended key changes.
- Grep the directory for removed branding before publishing.

## Child DOX Index
- No child docs. This directory is a leaf.
