#!/usr/bin/env node
/**
 * Sync content from the FIREKit monorepo into the webapp.
 *
 * Copies:
 *   ../docs/products/*.md      -> content/products/
 *   ../docs/MANUAL.md          -> content/MANUAL.md
 *   ../docs/MANUAL.zh-CN.md    -> content/MANUAL.zh-CN.md
 *   ../docs/ECOSYSTEM_OVERVIEW.md -> content/ECOSYSTEM_OVERVIEW.md
 *   ../hub/data/*.json         -> content/data/
 *   ../hub/{index.html,app.js,style.css,data} -> public/dashboard/
 *
 * The synced output is committed to git so Vercel builds work even when the
 * monorepo sources are not available (e.g. Root Directory builds without
 * "Include source files outside of the Root Directory"). When the sources
 * are present, running this script refreshes the snapshot.
 */
import { cpSync, existsSync, mkdirSync, readdirSync, copyFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const webapp = resolve(here, "..");
const repo = resolve(webapp, "..");

const docs = join(repo, "docs");
const hub = join(repo, "hub");

if (!existsSync(docs) || !existsSync(hub)) {
  console.log("[sync-content] monorepo sources not found, keeping committed snapshot.");
  process.exit(0);
}

const content = join(webapp, "content");
mkdirSync(join(content, "products"), { recursive: true });
mkdirSync(join(content, "data"), { recursive: true });

for (const f of readdirSync(join(docs, "products")).filter((f) => f.endsWith(".md"))) {
  copyFileSync(join(docs, "products", f), join(content, "products", f));
}
for (const f of ["MANUAL.md", "MANUAL.zh-CN.md", "ECOSYSTEM_OVERVIEW.md"]) {
  if (existsSync(join(docs, f))) copyFileSync(join(docs, f), join(content, f));
}
for (const f of readdirSync(join(hub, "data")).filter((f) => f.endsWith(".json"))) {
  copyFileSync(join(hub, "data", f), join(content, "data", f));
}

const dashboard = join(webapp, "public", "dashboard");
mkdirSync(dashboard, { recursive: true });
for (const f of ["index.html", "app.js", "style.css"]) {
  if (existsSync(join(hub, f))) copyFileSync(join(hub, f), join(dashboard, f));
}
cpSync(join(hub, "data"), join(dashboard, "data"), { recursive: true });

console.log("[sync-content] synced docs, hub data, and dashboard.");
