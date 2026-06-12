"use client";

import { useState, type ReactNode } from "react";

export interface Tab {
  label: string;
  content: ReactNode;
}

export default function Tabs({ tabs }: { tabs: Tab[] }) {
  const [active, setActive] = useState(0);
  return (
    <div>
      <div className="mb-6 flex flex-wrap gap-1.5 border-b border-white/10 pb-px">
        {tabs.map((t, i) => (
          <button
            key={t.label}
            onClick={() => setActive(i)}
            className={`rounded-t-lg px-4 py-2 text-sm font-medium transition-colors ${
              active === i
                ? "border-b-2 border-orange-400 text-orange-300"
                : "text-zinc-400 hover:text-zinc-200"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>
      {tabs.map((t, i) => (
        <div key={t.label} hidden={active !== i}>
          {t.content}
        </div>
      ))}
    </div>
  );
}
