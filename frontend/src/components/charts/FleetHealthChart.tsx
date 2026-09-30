"use client";

import React from "react";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";

interface FleetHealthChartProps {
  healthy: number;
  warning: number;
  critical: number;
}

export function FleetHealthChart({ healthy, warning, critical }: FleetHealthChartProps) {
  const data = [
    { name: "Healthy", value: healthy, color: "#10B981" },
    { name: "Warning / At-Risk", value: warning, color: "#F59E0B" },
    { name: "Critical Failure", value: critical, color: "#EF4444" },
  ];

  const total = healthy + warning + critical;
  const healthyPct = total > 0 ? ((healthy / total) * 100).toFixed(1) : "100";

  return (
    <div className="relative w-full h-[220px] flex items-center justify-center">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Tooltip
            contentStyle={{
              backgroundColor: "#111827",
              borderColor: "#334155",
              borderRadius: "8px",
              color: "#F8FAFC",
              fontSize: "12px",
            }}
          />
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={60}
            outerRadius={85}
            paddingAngle={4}
            dataKey="value"
            stroke="#0B0F17"
            strokeWidth={3}
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Pie>
        </PieChart>
      </ResponsiveContainer>

      {/* Central Health Metric Overlay */}
      <div className="absolute flex flex-col items-center justify-center pointer-events-none text-center">
        <span className="text-2xl font-bold text-slate-100 font-mono">{healthyPct}%</span>
        <span className="text-[11px] uppercase tracking-wider text-slate-400 font-medium">
          Fleet Health
        </span>
      </div>
    </div>
  );
}
