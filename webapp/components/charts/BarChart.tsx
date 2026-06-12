"use client";

import { useState } from "react";
import type { HubChart } from "@/lib/types";

const W = 720;
const H = 300;
const PAD = { top: 16, right: 16, bottom: 64, left: 56 };

function fmt(v: number): string {
  if (Math.abs(v) >= 1e4) return `${(v / 1e3).toFixed(0)}k`;
  if (Math.abs(v) >= 1) return v.toFixed(2);
  return v.toPrecision(3);
}

export default function BarChart({ chart }: { chart: HubChart }) {
  const [hover, setHover] = useState<number | null>(null);
  const series = chart.series[0];
  if (!series || series.data.length === 0) return null;

  const vals = series.data;
  const lo = Math.min(0, ...vals);
  const hi = Math.max(0, ...vals);
  const span = hi - lo || 1;
  const n = vals.length;
  const innerW = W - PAD.left - PAD.right;
  const barW = Math.min(48, (innerW / n) * 0.7);
  const toX = (i: number) => PAD.left + ((i + 0.5) / n) * innerW;
  const toY = (v: number) => H - PAD.bottom - ((v - lo) / span) * (H - PAD.top - PAD.bottom);

  return (
    <figure className="rounded-xl border border-white/10 bg-white/[0.03] p-4">
      <figcaption className="mb-2 text-sm font-medium text-zinc-200">{chart.title}</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full select-none">
        <line
          x1={PAD.left}
          x2={W - PAD.right}
          y1={toY(0)}
          y2={toY(0)}
          stroke="#ffffff30"
        />
        {vals.map((v, i) => (
          <g key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
            <rect
              x={toX(i) - barW / 2}
              y={Math.min(toY(0), toY(v))}
              width={barW}
              height={Math.abs(toY(v) - toY(0))}
              rx={3}
              fill={hover === i ? "#fb923c" : "#f97316"}
              opacity={hover === null || hover === i ? 1 : 0.45}
            />
            <text
              x={toX(i)}
              y={H - PAD.bottom + 14}
              textAnchor="end"
              fontSize={10}
              fill="#a1a1aa"
              transform={`rotate(-30 ${toX(i)} ${H - PAD.bottom + 14})`}
            >
              {String(chart.x[i]).slice(0, 18)}
            </text>
            {hover === i && (
              <text
                x={toX(i)}
                y={Math.min(toY(0), toY(v)) - 6}
                textAnchor="middle"
                fontSize={11}
                fill="#fdba74"
                className="font-mono"
              >
                {fmt(v)}
              </text>
            )}
          </g>
        ))}
      </svg>
      <div className="mt-1 text-xs text-zinc-500">{series.name}</div>
    </figure>
  );
}
