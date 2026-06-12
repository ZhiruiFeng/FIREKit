import type { Metadata } from "next";
import ArchitectureDiagram from "@/components/ArchitectureDiagram";
import DocPage from "@/components/DocPage";
import { readContentFile } from "@/lib/content";

export const metadata: Metadata = {
  title: "Ecosystem Overview",
  description:
    "How the nine FIREKit products fit together: architecture, development phases, integration patterns, and success metrics.",
};

export default function EcosystemPage() {
  const overview = readContentFile("ECOSYSTEM_OVERVIEW.md");
  return (
    <div>
      <div className="mx-auto max-w-6xl px-4 pt-12 sm:px-6">
        <h1 className="text-3xl font-bold text-zinc-50">Ecosystem</h1>
        <p className="mt-3 max-w-2xl text-zinc-400">
          Click any product to explore it. Data flows bottom-up from the foundation
          layer to execution.
        </p>
        <div className="mt-8">
          <ArchitectureDiagram />
        </div>
      </div>
      <DocPage markdown={overview} />
    </div>
  );
}
