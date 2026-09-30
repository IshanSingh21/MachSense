"use client";

import React from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { FeatureAttribution } from "@/types";

interface ShapWaterfallChartProps {
  escalators?: FeatureAttribution[];
  stabilizers?: FeatureAttribution[];
  height?: number;
}

export function ShapWaterfallChart({
  escalators = [],
  stabilizers = [],
  height = 260,
}: ShapWaterfallChartProps) {
  // Combine into a sorted feature list
  const combined = [
    ...escalators.map((e) => ({
      name: e.display_name || e.feature_name.replace(/_/g, " "),
      shap: Math.abs(e.attribution_value),
      rawShap: e.attribution_value,
      pct: e.percentage,
      type: "escalator" as const,
      color: "#EF4444",
    })),
    ...stabilizers.map((s) => ({
      name: s.display_name || s.feature_name.replace(/_/g, " "),
      shap: -Math.abs(s.attribution_value),
      rawShap: s.attribution_value,
      pct: s.percentage,
      type: "stabilizer" as const,
      color: "#10B981",
    })),
  ].sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));

  if (combined.length === 0) {
    return (
      <div className="flex items-center justify-center h-[180px] text-xs text-slate-400">
        Run a prediction with explainability enabled to inspect SHAP attributions.
      </div>
    );
  }

  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          layout="vertical"
          data={combined}
          margin={{ top: 10, right: 20, left: 40, bottom: 0 }}
        >
          <CartesianGrid stroke="#1E293B" strokeDasharray="3 3" horizontal={false} />
          <XAxis
            type="number"
            stroke="#64748B"
            fontSize={11}
            tickLine={false}
            tickFormatter={(v) => (v > 0 ? `+${v.toFixed(2)}` : v.toFixed(2))}
          />
          <YAxis
            type="category"
            dataKey="name"
            stroke="#94A3B8"
            fontSize={11}
            tickLine={false}
            width={140}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#111827",
              borderColor: "#334155",
              borderRadius: "8px",
              color: "#F8FAFC",
              fontSize: "12px",
            }}
            formatter={(value: unknown, _name, item) => [
              `${Number(value) > 0 ? "+" : ""}${Number(value).toFixed(4)} SHAP`,
              item.payload.type === "escalator"
                ? "Risk Driver (Increases Failure Prob)"
                : "Stabilizing Factor (Reduces Failure Prob)",
            ]}
          />
          <ReferenceLine x={0} stroke="#475569" strokeWidth={1.5} />
          <Bar dataKey="shap" radius={[4, 4, 4, 4]} barSize={16}>
            {combined.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
