"use client";

import { useState, type ReactNode } from "react";

export default function CodeBlock({
  language,
  code,
  children,
}: {
  language?: string;
  code: string;
  children: ReactNode;
}) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard unavailable (e.g. insecure context) — ignore
    }
  }

  return (
    <div className="group relative my-4 overflow-hidden rounded-xl border border-white/10 bg-zinc-950">
      <div className="flex items-center justify-between border-b border-white/5 px-4 py-1.5">
        <span className="text-xs text-zinc-500">{language || "code"}</span>
        <button
          onClick={copy}
          className="rounded-md px-2 py-0.5 text-xs text-zinc-400 transition-colors hover:bg-white/10 hover:text-zinc-100"
        >
          {copied ? "Copied ✓" : "Copy"}
        </button>
      </div>
      <pre className="overflow-x-auto p-4 text-[13px] leading-relaxed text-zinc-200">
        {children}
      </pre>
    </div>
  );
}
