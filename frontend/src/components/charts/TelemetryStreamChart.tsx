"use client";

import React from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface TelemetryPoint {
  time: string;
  air_temp?: number;
  proc_temp?: number;
  temp_diff?: number;
  speed?: number;
  torque?: number;
  power?: number;
  wear?: number;
  failure_prob?: number;
}

interface TelemetryStreamChartProps {
  data: TelemetryPoint[];
  metricKey?: "failure_prob" | "power" | "temp_diff" | "torque";
  height?: number;
}

export function TelemetryStreamChart({
  data,
  metricKey = "failure_prob",
  height = 240,
}: TelemetryStreamChartProps) {
  const configs = {
    failure_prob: {
      name: "Failure Risk Probability",
      dataKey: "failure_prob",
      stroke: "#EF4444",
      fill: "#EF4444",
      unit: "%",
      formatter: (val: number) => `${(val * 100).toFixed(1)}%`,
      domain: [0, 1],
    },
    power: {
      name: "Mechanical Power",
      dataKey: "power",
      stroke: "#06B6D4",
      fill: "#06B6D4",
      unit: " W",
      formatter: (val: number) => `${val.toLocaleString()} W`,
      domain: ["auto", "auto"],
    },
    temp_diff: {
      name: "Temperature Delta (ΔT)",
      dataKey: "temp_diff",
      stroke: "#F59E0B",
      fill: "#F59E0B",
      unit: " K",
      formatter: (val: number) => `${val.toFixed(1)} K`,
      domain: ["auto", "auto"],
    },
    torque: {
      name: "Applied Torque",
      dataKey: "torque",
      stroke: "#8B5CF6",
      fill: "#8B5CF6",
      unit: " Nm",
      formatter: (val: number) => `${val.toFixed(1)} Nm`,
      domain: ["auto", "auto"],
    },
  }[metricKey];

  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
          <defs>
            <linearGradient id={`gradient-${metricKey}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={configs.fill} stopOpacity={0.25} />
              <stop offset="95%" stopColor={configs.fill} stopOpacity={0.0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#1E293B" strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="time"
            stroke="#64748B"
            fontSize={11}
            tickLine={false}
            axisLine={{ stroke: "#1E293B" }}
          />
          <YAxis
            stroke="#64748B"
            fontSize={11}
            tickLine={false}
            axisLine={{ stroke: "#1E293B" }}
            tickFormatter={configs.formatter as (value: unknown) => string}
            domain={configs.domain as [number | string, number | string]}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#111827",
              borderColor: "#334155",
              borderRadius: "8px",
              color: "#F8FAFC",
              fontSize: "12px",
            }}
            formatter={(value: unknown) => [
              configs.formatter(Number(value)),
              configs.name,
            ]}
          />
          <Area
            type="monotone"
            dataKey={configs.dataKey}
            stroke={configs.stroke}
            strokeWidth={2}
            fillOpacity={1}
            fill={`url(#gradient-${metricKey})`}
          />
          <Line
            type="monotone"
            dataKey={configs.dataKey}
            stroke={configs.stroke}
            strokeWidth={2}
            dot={{ r: 3, fill: configs.stroke, strokeWidth: 0 }}
            activeDot={{ r: 5, fill: "#FFFFFF", stroke: configs.stroke, strokeWidth: 2 }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
