import type { HubMetric } from "@/lib/types";

export default function MetricGrid({ metrics }: { metrics: HubMetric[] }) {
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
      {metrics.map((m) => (
        <div
          key={m.label}
          className="rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3"
        >
          <div className="truncate font-mono text-lg text-orange-300" title={String(m.value)}>
            {m.value}
            {m.unit ? <span className="ml-1 text-xs text-zinc-400">{m.unit}</span> : null}
          </div>
          <div className="mt-0.5 text-xs text-zinc-400">{m.label}</div>
        </div>
      ))}
    </div>
  );
}
