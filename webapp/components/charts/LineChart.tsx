"use client";

import { useMemo, useRef, useState } from "react";
import type { HubChart } from "@/lib/types";

const PALETTE = ["#f97316", "#38bdf8", "#a78bfa", "#34d399", "#f472b6", "#facc15"];

const W = 720;
const H = 300;
const PAD = { top: 16, right: 16, bottom: 36, left: 56 };

function niceTicks(min: number, max: number, n = 5): number[] {
  if (!isFinite(min) || !isFinite(max) || min === max) return [min];
  const span = max - min;
  const step = Math.pow(10, Math.floor(Math.log10(span / n)));
  const err = (span / n) / step;
  const mult = err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1;
  const s = step * mult;
  const ticks: number[] = [];
  for (let v = Math.ceil(min / s) * s; v <= max + 1e-12; v += s) ticks.push(v);
  return ticks;
}

function fmt(v: number): string {
  if (Math.abs(v) >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (Math.abs(v) >= 1e4) return `${(v / 1e3).toFixed(0)}k`;
  if (Math.abs(v) >= 100) return v.toFixed(0);
  if (Math.abs(v) >= 1) return v.toFixed(2);
  return v.toPrecision(3);
}

export default function LineChart({ chart }: { chart: HubChart }) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  const { paths, yTicks, xTickIdxs, toX, toY, allY } = useMemo(() => {
    const allY = chart.series.flatMap((s) => s.data).filter((v) => v != null && isFinite(v));
    const yMin = Math.min(...allY);
    const yMax = Math.max(...allY);
    const yPad = (yMax - yMin || 1) * 0.05;
    const lo = yMin - yPad;
    const hi = yMax + yPad;
    const n = chart.x.length;
    const toX = (i: number) =>
      PAD.left + (n <= 1 ? 0 : (i / (n - 1)) * (W - PAD.left - PAD.right));
    const toY = (v: number) =>
      H - PAD.bottom - ((v - lo) / (hi - lo)) * (H - PAD.top - PAD.bottom);
    const paths = chart.series.map((s) =>
      s.data
        .map((v, i) => `${i === 0 ? "M" : "L"}${toX(i).toFixed(1)},${toY(v).toFixed(1)}`)
        .join("")
    );
    const yTicks = niceTicks(lo, hi);
    const xCount = Math.min(6, n);
    const xTickIdxs = Array.from({ length: xCount }, (_, k) =>
      Math.round((k / Math.max(1, xCount - 1)) * (n - 1))
    );
    return { paths, yTicks, xTickIdxs, toX, toY, allY };
  }, [chart]);

  if (allY.length === 0) return null;

  function onMove(e: React.MouseEvent<SVGSVGElement>) {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;
    const px = ((e.clientX - rect.left) / rect.width) * W;
    const n = chart.x.length;
    const frac = (px - PAD.left) / (W - PAD.left - PAD.right);
    const idx = Math.round(frac * (n - 1));
    setHoverIdx(idx >= 0 && idx < n ? idx : null);
  }

  return (
    <figure className="rounded-xl border border-white/10 bg-white/[0.03] p-4">
      <figcaption className="mb-2 text-sm font-medium text-zinc-200">{chart.title}</figcaption>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="w-full select-none"
        onMouseMove={onMove}
        onMouseLeave={() => setHoverIdx(null)}
      >
        {yTicks.map((t) => (
          <g key={t}>
            <line
              x1={PAD.left}
              x2={W - PAD.right}
              y1={toY(t)}
              y2={toY(t)}
              stroke="#ffffff14"
            />
            <text x={PAD.left - 8} y={toY(t) + 4} textAnchor="end" fontSize={11} fill="#a1a1aa">
              {fmt(t)}
            </text>
          </g>
        ))}
        {xTickIdxs.map((i) => (
          <text
            key={i}
            x={toX(i)}
            y={H - PAD.bottom + 18}
            textAnchor="middle"
            fontSize={11}
            fill="#a1a1aa"
          >
            {String(chart.x[i])}
          </text>
        ))}
        {paths.map((d, si) => (
          <path
            key={si}
            d={d}
            fill="none"
            stroke={PALETTE[si % PALETTE.length]}
            strokeWidth={1.8}
          />
        ))}
        {hoverIdx != null && (
          <g>
            <line
              x1={toX(hoverIdx)}
              x2={toX(hoverIdx)}
              y1={PAD.top}
              y2={H - PAD.bottom}
              stroke="#ffffff40"
              strokeDasharray="3 3"
            />
            {chart.series.map((s, si) =>
              s.data[hoverIdx] != null ? (
                <circle
                  key={si}
                  cx={toX(hoverIdx)}
                  cy={toY(s.data[hoverIdx])}
                  r={3.5}
                  fill={PALETTE[si % PALETTE.length]}
                />
              ) : null
            )}
          </g>
        )}
      </svg>
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-zinc-400">
        {chart.series.map((s, si) => (
          <span key={si} className="inline-flex items-center gap-1.5">
            <span
              className="inline-block h-2 w-2 rounded-full"
              style={{ background: PALETTE[si % PALETTE.length] }}
            />
            {s.name}
            {hoverIdx != null && s.data[hoverIdx] != null && (
              <span className="font-mono text-zinc-200">{fmt(s.data[hoverIdx])}</span>
            )}
          </span>
        ))}
        {hoverIdx != null && (
          <span className="ml-auto font-mono text-zinc-500">{String(chart.x[hoverIdx])}</span>
        )}
      </div>
    </figure>
  );
}
