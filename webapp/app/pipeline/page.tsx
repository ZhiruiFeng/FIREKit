import type { Metadata } from "next";
import Link from "next/link";
import { readHubData } from "@/lib/content";
import { PRODUCTS } from "@/lib/products";
import MetricGrid from "@/components/MetricGrid";
import HubChart from "@/components/HubChart";
import DataTable from "@/components/DataTable";

export const metadata: Metadata = {
  title: "End-to-End Pipeline",
  description:
    "The FIREKit pipeline chains all nine products on one shared synthetic universe — from data generation to a validated portfolio backtest.",
};

const STAGES: { slug: string; role: string }[] = [
  { slug: "datastream", role: "generates and quality-checks the shared universe" },
  { slug: "alphalab", role: "mines and ranks alpha factors on the panel" },
  { slug: "sentimentpulse", role: "scores synthetic news into sentiment features" },
  { slug: "signalml", role: "trains walk-forward ML models on factors + sentiment" },
  { slug: "portfolioengine", role: "allocates signal-tilted portfolio weights" },
  { slug: "riskguard", role: "applies vol targeting and the drawdown circuit breaker" },
  { slug: "executioncore", role: "executes the rebalance with TWAP and cost models" },
  { slug: "deeptrader", role: "trains an RL agent on the strategy return stream" },
  { slug: "vectorforge", role: "validates everything with a portfolio backtest" },
];

export default function PipelinePage() {
  const data = readHubData("pipeline");

  return (
    <div className="mx-auto max-w-5xl px-4 py-12 sm:px-6">
      <h1 className="text-3xl font-bold text-zinc-50">The end-to-end pipeline</h1>
      <p className="mt-3 max-w-3xl leading-relaxed text-zinc-400">
        <code className="rounded bg-white/10 px-1.5 py-0.5 font-mono text-[13px] text-orange-200">
          python3 pipeline.py
        </code>{" "}
        chains all nine products on one shared, deterministic synthetic universe.
        Each stage consumes the previous stage&apos;s output — exactly how the
        products are meant to integrate in a real system.
      </p>

      <ol className="mt-10 space-y-0">
        {STAGES.map((stage, i) => {
          const p = PRODUCTS.find((x) => x.slug === stage.slug)!;
          return (
            <li key={stage.slug} className="flex gap-4">
              <div className="flex flex-col items-center">
                <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-orange-400/40 bg-orange-500/10 font-mono text-sm text-orange-300">
                  {i + 1}
                </span>
                {i < STAGES.length - 1 && <span className="w-px flex-1 bg-white/10" />}
              </div>
              <div className="pb-6">
                <Link
                  href={`/products/${p.slug}`}
                  className="font-semibold text-zinc-100 hover:text-orange-200"
                >
                  {p.icon} {p.name}
                </Link>
                <span className="text-sm text-zinc-400"> — {stage.role}</span>
              </div>
            </li>
          );
        })}
      </ol>

      {data && (
        <div className="mt-12 space-y-8">
          <div>
            <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2">
              <h2 className="text-xl font-semibold text-zinc-100">Latest run</h2>
              <span className="text-xs text-zinc-500">
                generated {data.generated_at.slice(0, 10)} · status: {data.status}
              </span>
            </div>
            <MetricGrid metrics={data.summary_metrics} />
          </div>
          {data.charts.map((c) => (
            <HubChart key={c.id} chart={c} />
          ))}
          {data.tables.map((t) => (
            <DataTable key={t.id} table={t} />
          ))}
          {data.notes && (
            <p className="rounded-xl border border-white/5 bg-white/[0.02] p-4 text-sm leading-relaxed text-zinc-400">
              {data.notes}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
