import Link from "next/link";
import { notFound } from "next/navigation";
import type { Metadata } from "next";
import { PRODUCTS, PRODUCT_BY_SLUG, LAYERS } from "@/lib/products";
import {
  readProductDoc,
  readContentFile,
  readHubData,
  extractManualSection,
} from "@/lib/content";
import Markdown from "@/components/Markdown";
import Tabs from "@/components/Tabs";
import MetricGrid from "@/components/MetricGrid";
import HubChart from "@/components/HubChart";
import DataTable from "@/components/DataTable";

export function generateStaticParams() {
  return PRODUCTS.map((p) => ({ slug: p.slug }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}): Promise<Metadata> {
  const { slug } = await params;
  const product = PRODUCT_BY_SLUG.get(slug);
  if (!product) return {};
  return { title: product.name, description: product.tagline };
}

export default async function ProductPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const product = PRODUCT_BY_SLUG.get(slug);
  if (!product) notFound();

  const data = readHubData(product.slug);
  const designDoc = readProductDoc(product.docFile);
  const manualSection = extractManualSection(
    readContentFile("MANUAL.md"),
    product.manualSection
  );

  const idx = PRODUCTS.findIndex((p) => p.slug === product.slug);
  const prev = PRODUCTS[(idx + PRODUCTS.length - 1) % PRODUCTS.length];
  const next = PRODUCTS[(idx + 1) % PRODUCTS.length];

  const overview = (
    <div className="space-y-10">
      <div>
        <h2 className="mb-4 text-lg font-semibold text-zinc-100">Key features</h2>
        <ul className="grid gap-3 sm:grid-cols-2">
          {product.features.map((f) => (
            <li
              key={f}
              className="flex gap-3 rounded-xl border border-white/10 bg-white/[0.03] p-4 text-sm leading-relaxed text-zinc-300"
            >
              <span className="text-orange-400" aria-hidden>
                ◆
              </span>
              {f}
            </li>
          ))}
        </ul>
      </div>
      {data && (
        <div>
          <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2">
            <h2 className="text-lg font-semibold text-zinc-100">
              Latest demo results
            </h2>
            <span className="text-xs text-zinc-500">
              generated {data.generated_at.slice(0, 10)} · status: {data.status}
            </span>
          </div>
          <MetricGrid metrics={data.summary_metrics} />
          {data.notes && (
            <p className="mt-4 rounded-xl border border-white/5 bg-white/[0.02] p-4 text-sm leading-relaxed text-zinc-400">
              {data.notes}
            </p>
          )}
        </div>
      )}
      <div className="rounded-xl border border-white/10 bg-white/[0.03] p-4 text-sm text-zinc-400">
        <span className="font-medium text-zinc-200">Run it yourself:</span>{" "}
        <code className="rounded bg-white/10 px-1.5 py-0.5 font-mono text-[13px] text-orange-200">
          cd {product.pkg} && python3 -m {product.pkg}.demo
        </code>{" "}
        — a deterministic, seeded, offline demo that finishes in under a minute and
        feeds the live dashboard.
      </div>
    </div>
  );

  const liveDemo = data ? (
    <div className="space-y-6">
      {data.charts.map((c) => (
        <HubChart key={c.id} chart={c} />
      ))}
      {data.tables.map((t) => (
        <DataTable key={t.id} table={t} />
      ))}
    </div>
  ) : (
    <p className="text-sm text-zinc-500">No demo data available.</p>
  );

  const tabs = [
    { label: "Overview", content: overview },
    { label: "Live Demo", content: liveDemo },
    ...(manualSection
      ? [{ label: "Manual", content: <Markdown>{manualSection}</Markdown> }]
      : []),
    { label: "Design Doc", content: <Markdown>{designDoc}</Markdown> },
  ];

  return (
    <div className="mx-auto max-w-5xl px-4 py-12 sm:px-6">
      <nav className="mb-8 text-sm text-zinc-500">
        <Link href="/products" className="hover:text-orange-300">
          Products
        </Link>{" "}
        / <span className="text-zinc-300">{product.name}</span>
      </nav>

      <header className="mb-10">
        <div className="flex flex-wrap items-center gap-4">
          <span
            className="flex h-14 w-14 items-center justify-center rounded-2xl text-3xl"
            style={{ background: `${product.accent}1f` }}
          >
            {product.icon}
          </span>
          <div>
            <h1 className="text-3xl font-bold text-zinc-50">{product.name}</h1>
            <div className="mt-1 flex flex-wrap items-center gap-2 text-xs">
              <span className="rounded-full bg-white/10 px-2.5 py-0.5 font-mono text-zinc-300">
                v{product.version}
              </span>
              <span
                className="rounded-full px-2.5 py-0.5 font-medium"
                style={{ background: `${product.accent}26`, color: product.accent }}
              >
                {LAYERS[product.layer].label} layer
              </span>
            </div>
          </div>
        </div>
        <p className="mt-5 max-w-3xl text-lg leading-relaxed text-zinc-400">
          {product.blurb}
        </p>
      </header>

      <Tabs tabs={tabs} />

      <nav className="mt-16 flex justify-between border-t border-white/10 pt-6 text-sm">
        <Link href={`/products/${prev.slug}`} className="text-zinc-400 hover:text-orange-300">
          ← {prev.name}
        </Link>
        <Link href={`/products/${next.slug}`} className="text-zinc-400 hover:text-orange-300">
          {next.name} →
        </Link>
      </nav>
    </div>
  );
}
