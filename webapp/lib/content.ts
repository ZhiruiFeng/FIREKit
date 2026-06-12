import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import type { HubData } from "./types";

const CONTENT = join(process.cwd(), "content");

export function readProductDoc(docFile: string): string {
  return readFileSync(join(CONTENT, "products", docFile), "utf-8");
}

export function readContentFile(name: string): string {
  return readFileSync(join(CONTENT, name), "utf-8");
}

export function readHubData(slug: string): HubData | null {
  const path = join(CONTENT, "data", `${slug}.json`);
  if (!existsSync(path)) return null;
  return JSON.parse(readFileSync(path, "utf-8")) as HubData;
}

/**
 * Extract a `### <section> <Name> — ...` subsection of the manual
 * (e.g. section "4.9") up to the next `###` or `##` heading.
 */
export function extractManualSection(manual: string, section: string): string | null {
  const lines = manual.split("\n");
  const start = lines.findIndex((l) => l.startsWith(`### ${section} `));
  if (start === -1) return null;
  let end = lines.length;
  for (let i = start + 1; i < lines.length; i++) {
    if (/^##+ /.test(lines[i]) && !lines[i].startsWith("####")) {
      // stop at next ## or ### heading, but allow nested #### and code fences
      if (insideCodeFence(lines, start + 1, i)) continue;
      end = i;
      break;
    }
  }
  return lines.slice(start, end).join("\n").trim();
}

function insideCodeFence(lines: string[], from: number, at: number): boolean {
  let fences = 0;
  for (let i = from; i < at; i++) {
    if (lines[i].startsWith("```")) fences++;
  }
  return fences % 2 === 1;
}

export interface TocEntry {
  depth: number;
  text: string;
  id: string;
}

export function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[`*_]/g, "")
    .replace(/[^\p{L}\p{N}\s-]/gu, "")
    .trim()
    .replace(/\s+/g, "-");
}

/** Build a table of contents from ##/### headings, skipping code fences. */
export function buildToc(markdown: string): TocEntry[] {
  const toc: TocEntry[] = [];
  let inFence = false;
  for (const line of markdown.split("\n")) {
    if (line.startsWith("```")) {
      inFence = !inFence;
      continue;
    }
    if (inFence) continue;
    const m = /^(##{1,2}) (.+)$/.exec(line);
    if (m) {
      const text = m[2].trim();
      toc.push({ depth: m[1].length, text, id: slugify(text) });
    }
  }
  return toc;
}
