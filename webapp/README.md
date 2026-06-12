# FIREKit Hub — Web App

The FIREKit product hub: a Vercel-deployable Next.js site with interactive
introductions to all nine products, live demo results, the user manual
(English + 中文), the ecosystem overview, and the end-to-end pipeline
walkthrough. The original self-contained dashboard is served at
[`/dashboard/`](public/dashboard/).

## Pages

| Route | Content |
|---|---|
| `/` | Hero, live demo stats, interactive architecture diagram, searchable product grid |
| `/products` | Searchable, layer-filterable product explorer |
| `/products/[slug]` | Per-product page: overview + features, live demo charts/tables, manual section, full design doc |
| `/manual`, `/manual/zh` | Rendered user manual with table of contents (English / 中文) |
| `/ecosystem` | Clickable architecture + full ecosystem overview doc |
| `/pipeline` | End-to-end pipeline stages and latest run results |
| `/dashboard/` | The original vanilla-JS FIREKit Hub dashboard, unchanged |

## Content integration

The site renders content straight from the monorepo. `scripts/sync-content.mjs`
(run automatically by `npm run build` via `prebuild`) copies:

- `../docs/products/*.md` and the manuals → `content/`
- `../hub/data/*.json` (deterministic demo output) → `content/data/`
- `../hub/` (dashboard) → `public/dashboard/`

The synced snapshot is committed, so builds also work standalone. After
re-running `python3 run_all.py` in the repo root, run `npm run sync-content`
to refresh the live numbers.

## Develop

```bash
npm install
npm run dev      # http://localhost:3000
npm run build    # production build (syncs content first)
```

## Deploy to Vercel

1. Import the repository in Vercel.
2. Set **Root Directory** to `webapp` (framework auto-detects as Next.js).
3. Deploy — no environment variables required.
