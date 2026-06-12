"use client";

import Link from "next/link";
import { useState } from "react";
import { PRODUCTS, LAYERS, LAYER_ORDER, type Layer } from "@/lib/products";

const FLOW: Record<Layer, string> = {
  execution: "orders & fills",
  "portfolio-risk": "sized, risk-checked weights",
  intelligence: "signals & forecasts",
  foundation: "clean point-in-time data & backtests",
};

/**
 * Clickable layered architecture of the ecosystem: data flows bottom-up from
 * Foundation to Execution. Hovering a layer highlights it; every product chip
 * links to its detail page.
 */
export default function ArchitectureDiagram() {
  const [active, setActive] = useState<Layer | null>(null);
  const layersTopDown = [...LAYER_ORDER].reverse();

  return (
    <div className="space-y-2">
      {layersTopDown.map((layer, i) => {
        const products = PRODUCTS.filter((p) => p.layer === layer);
        const dimmed = active !== null && active !== layer;
        return (
          <div key={layer}>
            <div
              onMouseEnter={() => setActive(layer)}
              onMouseLeave={() => setActive(null)}
              className={`rounded-2xl border p-4 transition-all duration-200 ${
                dimmed
                  ? "border-white/5 bg-white/[0.01] opacity-50"
                  : "border-white/10 bg-white/[0.03]"
              } ${active === layer ? "border-orange-400/40 bg-orange-500/[0.06]" : ""}`}
            >
              <div className="mb-3 flex flex-wrap items-baseline gap-x-3">
                <span className="text-sm font-semibold uppercase tracking-wider text-orange-300">
                  {LAYERS[layer].label}
                </span>
                <span className="text-xs text-zinc-400">{LAYERS[layer].description}</span>
              </div>
              <div className="flex flex-wrap gap-2">
                {products.map((p) => (
                  <Link
                    key={p.slug}
                    href={`/products/${p.slug}`}
                    className="group inline-flex items-center gap-2 rounded-xl border border-white/10 bg-zinc-900/80 px-3 py-2 text-sm transition-colors hover:border-orange-400/50 hover:bg-zinc-800"
                  >
                    <span>{p.icon}</span>
                    <span className="font-medium text-zinc-100 group-hover:text-orange-200">
                      {p.name}
                    </span>
                    <span className="hidden text-xs text-zinc-500 sm:inline">
                      v{p.version}
                    </span>
                  </Link>
                ))}
              </div>
            </div>
            {i < layersTopDown.length - 1 && (
              <div className="flex items-center justify-center gap-2 py-1 text-xs text-zinc-500">
                <span aria-hidden>▲</span>
                <span>{FLOW[layersTopDown[i + 1]]}</span>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
