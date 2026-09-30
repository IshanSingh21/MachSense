"use client";

import React, { useState } from "react";
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  Activity,
  BarChart3,
  Calendar,
  Clock,
  Download,
  Flame,
  Gauge,
  Layers,
  RotateCw,
  Zap,
} from "lucide-react";
import { StatCard } from "@/components/ui/StatCard";
import { MOCK_TELEMETRY_SERIES } from "@/lib/mockData";

const FAILURE_MODE_DISTRIBUTION = [
  { mode: "Heat Dissipation (HDF)", count: 115, percentage: 33.9, color: "#F59E0B" },
  { mode: "Overstrain (OSF)", count: 98, percentage: 28.9, color: "#EF4444" },
  { mode: "Power Overload (PWF)", count: 95, percentage: 28.0, color: "#06B6D4" },
  { mode: "Tool Wear (TWF)", count: 45, percentage: 13.3, color: "#8B5CF6" },
  { mode: "Random Failures (RNF)", count: 18, percentage: 5.3, color: "#64748B" },
];

const SHIFT_AVAILABILITY_DATA = [
  { shift: "Shift 1 (06:00 - 14:00)", uptime: 98.2, alerts: 1, predictions: 450 },
  { shift: "Shift 2 (14:00 - 22:00)", uptime: 95.4, alerts: 4, predictions: 520 },
  { shift: "Shift 3 (22:00 - 06:00)", uptime: 97.8, alerts: 2, predictions: 380 },
];

export default function AnalyticsPage() {
  const [timeRange, setTimeRange] = useState<"24h" | "7d" | "30d">("24h");

  return (
    <div className="space-y-6">
      {/* 1. Analytics KPI Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Fleet Availability"
          value="97.8%"
          subtitle="Total operational uptime"
          trend={{ value: "+0.4%", isPositive: true }}
          icon={<Gauge className="w-5 h-5" />}
        />
        <StatCard
          title="Predictions Served"
          value="14,820"
          subtitle="Inference calls in 30 days"
          icon={<Activity className="w-5 h-5" />}
        />
        <StatCard
          title="Precursors Prevented"
          value="38"
          subtitle="Catastrophic shutdowns avoided"
          trend={{ value: "100% Caught", isPositive: true }}
          icon={<Flame className="w-5 h-5" />}
          variant="healthy"
        />
        <StatCard
          title="Mean Model Latency"
          value="6.4 ms"
          subtitle="P95 real-time scoring"
          icon={<Zap className="w-5 h-5" />}
        />
      </div>

      {/* 2. Historical Degradation & Failure Risk Stream */}
      <div className="bg-[#111827] border border-slate-800 p-6 rounded-2xl">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
          <div>
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200">
              Multi-Sensor Telemetry & Failure Probability Trend
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Continuous 6-hour condition profile leading up to mechanical overstrain
            </p>
          </div>

          <div className="flex items-center gap-1 bg-slate-900 border border-slate-800 p-1 rounded-xl text-xs">
            {(["24h", "7d", "30d"] as const).map((r) => (
              <button
                key={r}
                onClick={() => setTimeRange(r)}
                className={`px-3 py-1 rounded-lg font-medium transition-colors ${
                  timeRange === r ? "bg-cyan-600 text-white" : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {r.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        <div className="h-[280px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={MOCK_TELEMETRY_SERIES}>
              <CartesianGrid stroke="#1E293B" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="time" stroke="#64748B" fontSize={11} tickLine={false} />
              <YAxis yAxisId="left" stroke="#64748B" fontSize={11} tickLine={false} />
              <YAxis
                yAxisId="right"
                orientation="right"
                stroke="#EF4444"
                fontSize={11}
                tickLine={false}
                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                domain={[0, 1]}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#111827",
                  borderColor: "#334155",
                  borderRadius: "8px",
                  color: "#F8FAFC",
                  fontSize: "12px",
                }}
              />
              <Bar yAxisId="left" dataKey="power" fill="#06B6D4" opacity={0.3} barSize={20} name="Spindle Power (W)" />
              <Line yAxisId="left" type="monotone" dataKey="torque" stroke="#8B5CF6" strokeWidth={2} name="Torque (Nm)" dot />
              <Line yAxisId="left" type="monotone" dataKey="wear" stroke="#F59E0B" strokeWidth={2} name="Tool Wear (min)" dot />
              <Area yAxisId="right" type="monotone" dataKey="failure_prob" stroke="#EF4444" fill="#EF4444" fillOpacity={0.2} strokeWidth={2.5} name="Failure Risk %" />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 3. Failure Mode Historical Breakdown & Shift Availability */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Failure Mode Breakdown */}
        <div className="bg-[#111827] border border-slate-800 p-6 rounded-2xl">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200 mb-1">
            Historical Failure Modes Distribution (AI4I Dataset)
          </h3>
          <p className="text-xs text-slate-400 mb-4">
            339 failure events classified across physical breakdown domains
          </p>

          <div className="h-[220px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart layout="vertical" data={FAILURE_MODE_DISTRIBUTION} margin={{ left: 40, right: 20 }}>
                <CartesianGrid stroke="#1E293B" strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" stroke="#64748B" fontSize={11} tickLine={false} />
                <YAxis type="category" dataKey="mode" stroke="#94A3B8" fontSize={11} tickLine={false} width={150} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#111827",
                    borderColor: "#334155",
                    borderRadius: "8px",
                    color: "#F8FAFC",
                    fontSize: "12px",
                  }}
                  formatter={(val: unknown) => [`${val} occurrences`, "Total"]}
                />
                <Bar dataKey="count" radius={[0, 4, 4, 0]} barSize={16}>
                  {FAILURE_MODE_DISTRIBUTION.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Operational Shift Statistics */}
        <div className="bg-[#111827] border border-slate-800 p-6 rounded-2xl flex flex-col justify-between">
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200 mb-1">
              Shift-by-Shift Reliability Breakdown
            </h3>
            <p className="text-xs text-slate-400 mb-4">
              Operational metrics logged over the past 24 hours of production
            </p>

            <div className="space-y-3">
              {SHIFT_AVAILABILITY_DATA.map((shift, idx) => (
                <div key={idx} className="p-3.5 rounded-xl bg-slate-900 border border-slate-800/80">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-bold text-slate-200">{shift.shift}</span>
                    <span className="text-xs font-mono font-bold text-emerald-400">{shift.uptime}% Uptime</span>
                  </div>
                  <div className="flex items-center justify-between text-xs text-slate-400">
                    <span>Predictions: <strong className="text-slate-200 font-mono">{shift.predictions}</strong></span>
                    <span>Anomalies Flagged: <strong className="text-amber-400 font-mono">{shift.alerts}</strong></span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="pt-4 border-t border-slate-800 text-[11px] text-slate-400 flex items-center justify-between">
            <span>Automated Daily Report</span>
            <span className="text-cyan-400 font-mono">Logged to /logs/telemetry.log</span>
          </div>
        </div>
      </div>
    </div>
  );
}
