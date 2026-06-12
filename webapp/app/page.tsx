import Link from "next/link";
import ArchitectureDiagram from "@/components/ArchitectureDiagram";
import ProductExplorer from "@/components/ProductExplorer";
import { PRODUCTS } from "@/lib/products";
import { readHubData } from "@/lib/content";

const HIGHLIGHT_METRICS: [string, string][] = [
  // [product slug, metric label] — headline numbers pulled from live demo data
  ["vectorforge", "Sharpe ratio"],
  ["alphalab", "Best rank IC"],
  ["signalml", "Ensemble rank IC"],
  ["riskguard", "Max DD (protected)"],
];

export default function Home() {
  const liveStats = PRODUCTS.map((p) => readHubData(p.slug)).filter((d) => d !== null);
  const generatedAt = liveStats[0]?.generated_at?.slice(0, 10);

  return (
    <div className="mx-auto max-w-6xl px-4 sm:px-6">
      {/* Hero */}
      <section className="py-16 text-center sm:py-24">
        <p className="mb-4 inline-block rounded-full border border-orange-400/30 bg-orange-500/10 px-4 py-1 text-xs font-medium text-orange-300">
          🔥 Nine products · one ecosystem · ~1000 tests green
        </p>
        <h1 className="mx-auto max-w-3xl text-4xl font-bold tracking-tight text-zinc-50 sm:text-6xl">
          Build your path to{" "}
          <span className="bg-gradient-to-r from-orange-400 to-amber-300 bg-clip-text text-transparent">
            financial freedom
          </span>
        </h1>
        <p className="mx-auto mt-6 max-w-2xl text-lg leading-relaxed text-zinc-400">
          FIREKit is a comprehensive ecosystem of interconnected tools for building
          AI-powered quantitative trading systems — from clean point-in-time data and
          1000x-faster backtesting to ML signals, portfolio optimization, risk
          management, and execution.
        </p>
        <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
          <Link
            href="/products"
            className="rounded-xl bg-orange-500 px-6 py-3 text-sm font-semibold text-zinc-950 transition-colors hover:bg-orange-400"
          >
            Explore the products
          </Link>
          <Link
            href="/manual"
            className="rounded-xl border border-white/15 px-6 py-3 text-sm font-semibold text-zinc-200 transition-colors hover:border-orange-400/50 hover:text-orange-200"
          >
            Read the manual
          </Link>
          <a
            href="/dashboard"
            className="rounded-xl border border-white/15 px-6 py-3 text-sm font-semibold text-zinc-200 transition-colors hover:border-orange-400/50 hover:text-orange-200"
          >
            Live dashboard ↗
          </a>
        </div>
      </section>

      {/* Live demo stats strip */}
      <section className="rounded-2xl border border-white/10 bg-white/[0.03] p-6">
        <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-zinc-400">
            Live demo results
          </h2>
          {generatedAt && (
            <span className="text-xs text-zinc-500">
              generated {generatedAt} by the deterministic demo pipeline
            </span>
          )}
        </div>
        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          {HIGHLIGHT_METRICS.map(([slug, label]) => {
            const data = liveStats.find((d) => d.product === slug);
            const metric = data?.summary_metrics.find((m) => m.label === label)
              ?? data?.summary_metrics[0];
            if (!data || !metric) return null;
            return (
              <Link
                key={slug}
                href={`/products/${slug}`}
                className="group rounded-xl border border-white/5 p-4 transition-colors hover:border-orange-400/40"
              >
                <div className="font-mono text-2xl text-orange-300">
                  {metric.value}
                  {metric.unit ? <span className="text-sm text-zinc-500"> {metric.unit}</span> : null}
                </div>
                <div className="mt-1 text-xs text-zinc-400">{metric.label}</div>
                <div className="mt-2 text-xs font-medium text-zinc-500 group-hover:text-orange-300">
                  {data.title} →
                </div>
              </Link>
            );
          })}
        </div>
      </section>

      {/* Architecture */}
      <section className="py-16">
        <h2 className="text-2xl font-bold text-zinc-100">The ecosystem at a glance</h2>
        <p className="mt-2 max-w-2xl text-sm text-zinc-400">
          Data flows bottom-up: the foundation layer feeds the intelligence layer,
          whose signals become risk-managed portfolios that the execution layer turns
          into orders. Hover a layer to focus it; click any product to dive in.
        </p>
        <div className="mt-8">
          <ArchitectureDiagram />
        </div>
      </section>

      {/* Product explorer */}
      <section className="pb-8" id="products">
        <h2 className="text-2xl font-bold text-zinc-100">Explore the products</h2>
        <p className="mt-2 mb-8 max-w-2xl text-sm text-zinc-400">
          Every product is an independently installable Python package with its own
          test suite and a deterministic demo that feeds the live dashboard.
        </p>
        <ProductExplorer />
      </section>

      {/* Pipeline teaser */}
      <section className="mt-8 rounded-2xl border border-orange-400/20 bg-gradient-to-br from-orange-500/10 to-transparent p-8">
        <h2 className="text-xl font-bold text-zinc-100">One command, nine products</h2>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-zinc-400">
          The end-to-end pipeline chains every product on one shared universe:
          DataStream generates and cleans data, AlphaLab mines factors, SentimentPulse
          scores news, SignalML trains walk-forward models, PortfolioEngine allocates,
          RiskGuard protects, ExecutionCore executes, DeepTrader learns, and
          VectorForge validates it all with a portfolio backtest.
        </p>
        <Link
          href="/pipeline"
          className="mt-5 inline-block rounded-xl bg-orange-500 px-5 py-2.5 text-sm font-semibold text-zinc-950 transition-colors hover:bg-orange-400"
        >
          See the pipeline results →
        </Link>
      </section>
    </div>
  );
}
