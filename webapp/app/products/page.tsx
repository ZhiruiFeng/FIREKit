import type { Metadata } from "next";
import ProductExplorer from "@/components/ProductExplorer";

export const metadata: Metadata = {
  title: "Products",
  description:
    "All nine FIREKit products: VectorForge, DataStream, AlphaLab, SignalML, SentimentPulse, DeepTrader, PortfolioEngine, RiskGuard, and ExecutionCore.",
};

export default function ProductsPage() {
  return (
    <div className="mx-auto max-w-6xl px-4 py-12 sm:px-6">
      <h1 className="text-3xl font-bold text-zinc-50">Products</h1>
      <p className="mt-3 mb-10 max-w-2xl text-zinc-400">
        Nine interconnected products covering the full quant workflow: data,
        backtesting, factor research, ML signals, sentiment, RL agents, allocation,
        risk, and execution. Search or filter by layer.
      </p>
      <ProductExplorer />
    </div>
  );
}
