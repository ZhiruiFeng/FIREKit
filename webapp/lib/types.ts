export interface HubMetric {
  label: string;
  value: string | number;
  unit?: string;
}

export interface HubSeries {
  name: string;
  data: number[];
}

export interface HubChart {
  id: string;
  title: string;
  type: "line" | "bar" | "scatter";
  x_label?: string;
  y_label?: string;
  x: (string | number)[];
  series: HubSeries[];
}

export interface HubTable {
  id: string;
  title: string;
  columns: string[];
  rows: (string | number)[][];
}

export interface HubData {
  schema_version: number;
  product: string;
  title: string;
  tagline: string;
  version: string;
  generated_at: string;
  status: string;
  summary_metrics: HubMetric[];
  charts: HubChart[];
  tables: HubTable[];
  notes?: string;
}
