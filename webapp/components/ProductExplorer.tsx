"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { PRODUCTS, LAYERS, LAYER_ORDER, type Layer } from "@/lib/products";

export default function ProductExplorer() {
  const [query, setQuery] = useState("");
  const [layer, setLayer] = useState<Layer | "all">("all");

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return PRODUCTS.filter((p) => {
      if (layer !== "all" && p.layer !== layer) return false;
      if (!q) return true;
      return [p.name, p.tagline, p.blurb, ...p.features]
        .join(" ")
        .toLowerCase()
        .includes(q);
    });
  }, [query, layer]);

  return (
    <div>
      <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-center">
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search products and features…"
          className="w-full rounded-xl border border-white/10 bg-white/[0.04] px-4 py-2.5 text-sm text-zinc-100 placeholder-zinc-500 outline-none transition-colors focus:border-orange-400/50 sm:max-w-sm"
        />
        <div className="flex flex-wrap gap-1.5">
          {(["all", ...LAYER_ORDER] as const).map((l) => (
            <button
              key={l}
              onClick={() => setLayer(l)}
              className={`rounded-full px-3 py-1.5 text-xs font-medium transition-colors ${
                layer === l
                  ? "bg-orange-500 text-zinc-950"
                  : "border border-white/10 text-zinc-300 hover:border-orange-400/40"
              }`}
            >
              {l === "all" ? "All" : LAYERS[l].label}
            </button>
          ))}
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {filtered.map((p) => (
          <Link
            key={p.slug}
            href={`/products/${p.slug}`}
            className="group flex flex-col rounded-2xl border border-white/10 bg-white/[0.03] p-5 transition-all hover:-translate-y-0.5 hover:border-orange-400/40 hover:bg-white/[0.05]"
          >
            <div className="flex items-center gap-3">
              <span
                className="flex h-10 w-10 items-center justify-center rounded-xl text-xl"
                style={{ background: `${p.accent}1f` }}
              >
                {p.icon}
              </span>
              <div>
                <div className="font-semibold text-zinc-100 group-hover:text-orange-200">
                  {p.name}
                </div>
                <div className="text-xs text-zinc-500">
                  v{p.version} · {LAYERS[p.layer].label}
                </div>
              </div>
            </div>
            <p className="mt-3 text-sm leading-relaxed text-zinc-400">{p.tagline}</p>
            <span className="mt-auto pt-4 text-xs font-medium text-orange-300/80 group-hover:text-orange-300">
              Explore →
            </span>
          </Link>
        ))}
        {filtered.length === 0 && (
          <p className="col-span-full py-8 text-center text-sm text-zinc-500">
            No products match “{query}”.
          </p>
        )}
      </div>
    </div>
  );
}
