import Link from "next/link";
import { PRODUCTS } from "@/lib/products";

export default function Footer() {
  return (
    <footer className="mt-20 border-t border-white/10">
      <div className="mx-auto grid max-w-6xl gap-8 px-4 py-10 sm:grid-cols-3 sm:px-6">
        <div>
          <div className="flex items-center gap-2 text-lg font-bold text-zinc-100">
            <span aria-hidden>🔥</span> FIREKit
          </div>
          <p className="mt-2 max-w-xs text-sm text-zinc-500">
            Build a set of toolkits for financial freedom. From backtest to production,
            build your path to financial independence.
          </p>
        </div>
        <div>
          <div className="mb-3 text-sm font-semibold text-zinc-200">Products</div>
          <ul className="grid grid-cols-2 gap-1.5 text-sm">
            {PRODUCTS.map((p) => (
              <li key={p.slug}>
                <Link href={`/products/${p.slug}`} className="text-zinc-400 hover:text-orange-300">
                  {p.name}
                </Link>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <div className="mb-3 text-sm font-semibold text-zinc-200">Resources</div>
          <ul className="space-y-1.5 text-sm">
            <li><Link href="/manual" className="text-zinc-400 hover:text-orange-300">User Manual</Link></li>
            <li><Link href="/manual/zh" className="text-zinc-400 hover:text-orange-300">用户手册 (中文)</Link></li>
            <li><Link href="/ecosystem" className="text-zinc-400 hover:text-orange-300">Ecosystem Overview</Link></li>
            <li><Link href="/pipeline" className="text-zinc-400 hover:text-orange-300">End-to-End Pipeline</Link></li>
            <li><a href="/dashboard" className="text-zinc-400 hover:text-orange-300">Live Dashboard ↗</a></li>
          </ul>
        </div>
      </div>
      <div className="border-t border-white/5 py-4 text-center text-xs text-zinc-600">
        FIREKit — nine products, one ecosystem. MIT licensed.
      </div>
    </footer>
  );
}
