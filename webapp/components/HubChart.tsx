import type { HubChart as HubChartType } from "@/lib/types";
import LineChart from "./charts/LineChart";
import BarChart from "./charts/BarChart";

export default function HubChart({ chart }: { chart: HubChartType }) {
  if (chart.type === "bar") return <BarChart chart={chart} />;
  return <LineChart chart={chart} />;
}
