"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

const LINKS = [
  { href: "/", label: "Home" },
  { href: "/products", label: "Products" },
  { href: "/ecosystem", label: "Ecosystem" },
  { href: "/pipeline", label: "Pipeline" },
  { href: "/manual", label: "Manual" },
  { href: "/dashboard", label: "Live Dashboard", external: true },
];

export default function Nav() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  const isActive = (href: string) =>
    href === "/" ? pathname === "/" : pathname.startsWith(href);

  return (
    <header className="sticky top-0 z-50 border-b border-white/10 bg-zinc-950/80 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center gap-4 px-4 py-3 sm:px-6">
        <Link href="/" className="flex items-center gap-2 text-lg font-bold text-zinc-100">
          <span aria-hidden>🔥</span> FIREKit
        </Link>
        <nav className="ml-auto hidden items-center gap-1 md:flex">
          {LINKS.map((l) =>
            l.external ? (
              <a
                key={l.href}
                href={l.href}
                className="rounded-lg px-3 py-1.5 text-sm text-zinc-300 transition-colors hover:bg-white/5 hover:text-orange-200"
              >
                {l.label} ↗
              </a>
            ) : (
              <Link
                key={l.href}
                href={l.href}
                className={`rounded-lg px-3 py-1.5 text-sm transition-colors ${
                  isActive(l.href)
                    ? "bg-orange-500/15 text-orange-300"
                    : "text-zinc-300 hover:bg-white/5 hover:text-orange-200"
                }`}
              >
                {l.label}
              </Link>
            )
          )}
        </nav>
        <button
          onClick={() => setOpen(!open)}
          aria-label="Toggle navigation"
          className="ml-auto rounded-lg border border-white/10 px-3 py-1.5 text-sm text-zinc-300 md:hidden"
        >
          ☰
        </button>
      </div>
      {open && (
        <nav className="flex flex-col gap-1 border-t border-white/10 px-4 py-3 md:hidden">
          {LINKS.map((l) =>
            l.external ? (
              <a key={l.href} href={l.href} className="rounded-lg px-3 py-2 text-sm text-zinc-300">
                {l.label} ↗
              </a>
            ) : (
              <Link
                key={l.href}
                href={l.href}
                onClick={() => setOpen(false)}
                className={`rounded-lg px-3 py-2 text-sm ${
                  isActive(l.href) ? "bg-orange-500/15 text-orange-300" : "text-zinc-300"
                }`}
              >
                {l.label}
              </Link>
            )
          )}
        </nav>
      )}
    </header>
  );
}
