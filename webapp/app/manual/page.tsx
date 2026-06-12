import type { Metadata } from "next";
import Link from "next/link";
import { readContentFile } from "@/lib/content";
import DocPage from "@/components/DocPage";

export const metadata: Metadata = {
  title: "User Manual",
  description:
    "The FIREKit user manual: setup, core workflow, per-product guides with working code, and the end-to-end pipeline walkthrough.",
};

export default function ManualPage() {
  const manual = readContentFile("MANUAL.md");
  return (
    <DocPage
      markdown={manual}
      aside={
        <div className="mb-6">
          <Link
            href="/manual/zh"
            className="inline-block rounded-lg border border-white/10 px-3 py-1.5 text-xs text-zinc-300 transition-colors hover:border-orange-400/40 hover:text-orange-200"
          >
            中文版 →
          </Link>
        </div>
      }
    />
  );
}
