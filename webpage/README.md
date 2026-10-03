# webpage

Static site for sysone-bench. No framework, no build step. Deploys to Vercel as-is.

    npx vercel --prod        # or connect the directory in the Vercel dashboard

Custom domain: `sysone.sdad.pro`.

## Regenerating the data

Every number on the site is rendered from `data/results.json`, and that file is generated from
checksum-verified run directories. Nothing is typed by hand.

    python webpage/build-data.py

It calls `ops/panel_coverage.py` for the authoritative counts, so the site's headline number and the
repository's coverage number cannot drift apart. If a run's checksum fails, the measurement is
dropped rather than shown.

## Files

| File | Purpose |
|---|---|
| `index.html` | Structure and JSON-LD (dataset, publisher, FAQ) |
| `style.css` | Editorial design system: paper, ink, hairlines. One oxblood accent |
| `app.js` | Hand-built SVG charts, ranked field list, filters, reveal-on-scroll |
| `data/results.json` | Generated. Do not edit by hand |
| `llms.txt` | Plain-language summary for AI answer engines |
| `robots.txt` | Explicitly permits GPTBot, ClaudeBot, PerplexityBot and others |
| `sitemap.xml` | Single-URL sitemap |
| `vercel.json` | Security headers, asset caching, CORS on the data file |

## Local preview

    cd webpage && python3 -m http.server 8899

Then open `http://127.0.0.1:8899`. A server is required, not `file://`, because `app.js` fetches
`data/results.json`.

## Design

An editorial data essay. Warm paper, near-black ink, hairline rules, no cards, no shadows, no grain.
Newsreader for prose, IBM PlexMono for every number. The numbers are the graphic.

## Charts

Five, all inline SVG built by `app.js`:

1. **Field** - all 43 measurements as a ranked list with hairline bars, plus a strip plot in the hero
2. **Per-suite** - grouped bars for one model, or the top five
3. **Heatmap** - 43 models against 9 suites, horizontally scrollable
4. **Precision** - tev1-08b on CPU fp32 against T4 fp16, on a truncated axis that says so
5. **Coverage** - verified directories against distinct measurements

Axes that do not start at zero state it on the chart.
