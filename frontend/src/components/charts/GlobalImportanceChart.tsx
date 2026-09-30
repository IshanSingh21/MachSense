"use client";

import React from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface FeatureImportanceItem {
  name: string;
  key: string;
  importance: number;
  domain: string;
}

interface GlobalImportanceChartProps {
  data: FeatureImportanceItem[];
  height?: number;
}

export function GlobalImportanceChart({ data, height = 300 }: GlobalImportanceChartProps) {
  const sorted = [...data].sort((a, b) => b.importance - a.importance);

  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          layout="vertical"
          data={sorted}
          margin={{ top: 10, right: 30, left: 30, bottom: 0 }}
        >
          <CartesianGrid stroke="#1E293B" strokeDasharray="3 3" horizontal={false} />
          <XAxis
            type="number"
            stroke="#64748B"
            fontSize={11}
            tickLine={false}
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
          />
          <YAxis
            type="category"
            dataKey="name"
            stroke="#94A3B8"
            fontSize={11}
            tickLine={false}
            width={160}
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
              `${(Number(value) * 100).toFixed(1)}% Relative Importance (${item.payload.domain})`,
              "TreeSHAP Importance",
            ]}
          />
          <Bar dataKey="importance" radius={[0, 4, 4, 0]} barSize={18}>
            {sorted.map((_entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={index < 2 ? "#06B6D4" : index < 4 ? "#0284C7" : "#3B82F6"}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
