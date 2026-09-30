"use client";

import React, { useState } from "react";
import Link from "next/link";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Cpu,
  Flame,
  Gauge,
  Layers,
  Radio,
  RotateCw,
  Zap,
} from "lucide-react";
import { FleetHealthChart } from "@/components/charts/FleetHealthChart";
import { TelemetryStreamChart } from "@/components/charts/TelemetryStreamChart";
import { StatCard } from "@/components/ui/StatCard";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { TelemetryCard } from "@/components/ui/TelemetryCard";
import { MOCK_ALERTS, MOCK_MACHINES, MOCK_TELEMETRY_SERIES } from "@/lib/mockData";

export default function DashboardPage() {
  const [selectedMetric, setSelectedMetric] = useState<
    "failure_prob" | "power" | "temp_diff" | "torque"
  >("failure_prob");

  const totalMachines = MOCK_MACHINES.length;
  const healthyCount = MOCK_MACHINES.filter((m) => m.status === "HEALTHY").length;
  const warningCount = MOCK_MACHINES.filter((m) => m.status === "WARNING").length;
  const criticalCount = MOCK_MACHINES.filter((m) => m.status === "CRITICAL").length;

  const featuredMachine = MOCK_MACHINES.find((m) => m.status === "CRITICAL") || MOCK_MACHINES[0];

  return (
    <div className="space-y-8">
      {/* 1. Fleet Metric Stat Cards */}
      <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Monitored Fleet"
          value={totalMachines}
          subtitle="Active CNC Milling & Lathe Units"
          icon={<Cpu className="w-5 h-5" />}
        />
        <StatCard
          title="Healthy Machines"
          value={healthyCount}
          subtitle="Operating in nominal parameters"
          icon={<CheckCircle2 className="w-5 h-5" />}
          variant="healthy"
          trend={{ value: "95.8% Rate", isPositive: true }}
        />
        <StatCard
          title="At-Risk Warning"
          value={warningCount}
          subtitle="Approaching thermal/power limits"
          icon={<AlertTriangle className="w-5 h-5" />}
          variant="warning"
        />
        <StatCard
          title="Critical Alerts"
          value={criticalCount}
          subtitle="Immediate operator action required"
          icon={<Activity className="w-5 h-5" />}
          variant="critical"
        />
      </section>

      {/* 2. Main Visual Grid: Fleet Health Distribution & Featured Machine */}
      <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Fleet Health Breakdown */}
        <div className="bg-[#111827] border border-slate-800 rounded-2xl p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Gauge className="w-4 h-4 text-cyan-400" />
                <h2 className="text-sm font-bold uppercase tracking-wider text-slate-200">
                  Fleet Health Distribution
                </h2>
              </div>
              <span className="text-[11px] text-slate-400 font-mono">24/24 Online</span>
            </div>
            <p className="text-xs text-slate-400 mb-4">
              Continuous Bayesian risk aggregation across active shop-floor machining centers.
            </p>
          </div>

          <FleetHealthChart
            healthy={healthyCount}
            warning={warningCount}
            critical={criticalCount}
          />

          <div className="grid grid-cols-3 gap-2 pt-4 border-t border-slate-800/80 text-center text-xs">
            <div className="p-2 rounded-lg bg-emerald-950/20 border border-emerald-900/40">
              <span className="text-[10px] text-emerald-400 uppercase font-semibold block">
                Healthy
              </span>
              <span className="text-base font-bold text-slate-100 font-mono">{healthyCount}</span>
            </div>
            <div className="p-2 rounded-lg bg-amber-950/20 border border-amber-900/40">
              <span className="text-[10px] text-amber-400 uppercase font-semibold block">
                Warning
              </span>
              <span className="text-base font-bold text-slate-100 font-mono">{warningCount}</span>
            </div>
            <div className="p-2 rounded-lg bg-rose-950/20 border border-rose-900/40">
              <span className="text-[10px] text-rose-400 uppercase font-semibold block">
                Critical
              </span>
              <span className="text-base font-bold text-slate-100 font-mono">{criticalCount}</span>
            </div>
          </div>
        </div>

        {/* Featured Critical Machine Telemetry Card */}
        <div className="lg:col-span-2 bg-[#111827] border border-rose-900/40 rounded-2xl p-6 flex flex-col justify-between relative overflow-hidden">
          <div className="absolute top-0 right-0 w-32 h-32 bg-rose-500/5 rounded-full blur-2xl pointer-events-none" />

          <div>
            <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
              <div className="flex items-center gap-3">
                <span className="p-2 rounded-xl bg-rose-950/60 border border-rose-800/60 text-rose-400">
                  <Flame className="w-5 h-5" />
                </span>
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="text-base font-bold text-slate-100">{featuredMachine.name}</h3>
                    <span className="text-xs font-mono text-slate-400">({featuredMachine.id})</span>
                  </div>
                  <span className="text-xs text-slate-400">{featuredMachine.location}</span>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <StatusBadge status={featuredMachine.status} size="md" />
                <Link
                  href={`/machines/${featuredMachine.id}`}
                  className="px-3 py-1 text-xs font-medium text-cyan-400 hover:text-cyan-300 bg-cyan-950/40 border border-cyan-800/60 rounded-lg flex items-center gap-1 transition-colors"
                >
                  <span>Diagnostic View</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </Link>
              </div>
            </div>

            {/* Critical Alert Banner */}
            {featuredMachine.warning_message && (
              <div className="mb-5 p-3 rounded-xl bg-rose-950/30 border border-rose-800/60 flex items-start gap-2.5 text-xs text-rose-300">
                <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                <div className="flex-1">
                  <span className="font-semibold text-rose-200">Precursor Anomaly Detected: </span>
                  {featuredMachine.warning_message}
                </div>
              </div>
            )}

            {/* Real-time Sensor Metrics Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <TelemetryCard
                label="Air Temperature"
                value={featuredMachine.telemetry.air_temperature_k}
                unit="K"
                nominalRange="295 - 304 K"
              />
              <TelemetryCard
                label="Process Temp"
                value={featuredMachine.telemetry.process_temperature_k}
                unit="K"
                nominalRange="305 - 314 K"
              />
              <TelemetryCard
                label="Spindle Speed"
                value={featuredMachine.telemetry.rotational_speed_rpm}
                unit="RPM"
                nominalRange="1380 - 2800"
              />
              <TelemetryCard
                label="Mechanical Torque"
                value={featuredMachine.telemetry.torque_nm}
                unit="Nm"
                status="critical"
                nominalRange="10 - 55 Nm"
              />
            </div>
          </div>

          {/* Probability Gauge Strip */}
          <div className="mt-5 pt-4 border-t border-slate-800 flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <span className="text-xs uppercase font-semibold tracking-wider text-slate-400">
                Predicted Failure Risk:
              </span>
              <span className="text-xl font-bold font-mono text-rose-400">
                {(featuredMachine.failure_probability * 100).toFixed(1)}%
              </span>
              <span className="text-xs text-slate-400 font-mono">(Threshold: 0.5608)</span>
            </div>

            <div className="flex items-center gap-2">
              <Link
                href="/predictions"
                className="px-4 py-2 text-xs font-semibold rounded-lg bg-rose-600 hover:bg-rose-500 text-white shadow-lg shadow-rose-600/20 transition-all flex items-center gap-2"
              >
                <Activity className="w-3.5 h-3.5" />
                <span>Run Interactive SHAP Analysis</span>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* 3. Live Telemetry Stream Analytics & Recent Alerts */}
      <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Real-time Telemetry Multi-Trend Chart */}
        <div className="lg:col-span-2 bg-[#111827] border border-slate-800 rounded-2xl p-6">
          <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
            <div className="flex items-center gap-2">
              <Radio className="w-4 h-4 text-cyan-400 animate-pulse" />
              <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200">
                Telemetry Degradation Timeline ({featuredMachine.id})
              </h3>
            </div>

            {/* Metric Toggle Tabs */}
            <div className="flex items-center gap-1 bg-slate-900 border border-slate-800 rounded-lg p-1 text-xs">
              <button
                onClick={() => setSelectedMetric("failure_prob")}
                className={`px-2.5 py-1 rounded-md font-medium transition-colors ${
                  selectedMetric === "failure_prob"
                    ? "bg-rose-950 text-rose-300 border border-rose-800/60"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                Failure Risk
              </button>
              <button
                onClick={() => setSelectedMetric("power")}
                className={`px-2.5 py-1 rounded-md font-medium transition-colors ${
                  selectedMetric === "power"
                    ? "bg-cyan-950 text-cyan-300 border border-cyan-800/60"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                Power (W)
              </button>
              <button
                onClick={() => setSelectedMetric("temp_diff")}
                className={`px-2.5 py-1 rounded-md font-medium transition-colors ${
                  selectedMetric === "temp_diff"
                    ? "bg-amber-950 text-amber-300 border border-amber-800/60"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                Temp ΔT
              </button>
              <button
                onClick={() => setSelectedMetric("torque")}
                className={`px-2.5 py-1 rounded-md font-medium transition-colors ${
                  selectedMetric === "torque"
                    ? "bg-purple-950 text-purple-300 border border-purple-800/60"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                Torque
              </button>
            </div>
          </div>

          <TelemetryStreamChart data={MOCK_TELEMETRY_SERIES} metricKey={selectedMetric} />
        </div>

        {/* Recent Industrial Alerts */}
        <div className="bg-[#111827] border border-slate-800 rounded-2xl p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-400" />
                <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200">
                  Recent Anomaly Alerts
                </h3>
              </div>
              <Link
                href="/alerts"
                className="text-xs text-cyan-400 hover:text-cyan-300 flex items-center gap-1"
              >
                <span>View All</span>
                <ArrowRight className="w-3 h-3" />
              </Link>
            </div>

            <div className="space-y-3">
              {MOCK_ALERTS.slice(0, 3).map((alert) => (
                <div
                  key={alert.id}
                  className="p-3 rounded-xl bg-slate-900/60 border border-slate-800/80 hover:border-slate-700 transition-colors"
                >
                  <div className="flex items-center justify-between gap-2 mb-1.5">
                    <span className="text-xs font-bold text-slate-200">{alert.machine_id}</span>
                    <StatusBadge status={alert.severity} size="sm" />
                  </div>
                  <p className="text-xs text-slate-300 line-clamp-2">{alert.issue}</p>
                  <div className="mt-2 text-[10px] text-slate-400 flex items-center justify-between">
                    <span>{alert.timestamp}</span>
                    <span className="font-mono text-cyan-400 font-medium">
                      Type: {alert.failure_type}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="pt-4 mt-4 border-t border-slate-800 text-center">
            <Link
              href="/explainability"
              className="text-xs font-semibold text-slate-300 hover:text-white flex items-center justify-center gap-1.5"
            >
              <Layers className="w-3.5 h-3.5 text-cyan-400" />
              <span>Inspect Global TreeSHAP Feature Importance</span>
            </Link>
          </div>
        </div>
      </section>

      {/* 4. Complete Machine Fleet Overview Table */}
      <section className="bg-[#111827] border border-slate-800 rounded-2xl p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-cyan-400" />
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200">
              Active Machine Fleet Status
            </h3>
          </div>
          <Link
            href="/machines"
            className="text-xs text-cyan-400 hover:text-cyan-300 flex items-center gap-1"
          >
            <span>Full Fleet Directory</span>
            <ArrowRight className="w-3 h-3" />
          </Link>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-900/80 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800">
              <tr>
                <th className="py-3 px-4">Machine ID & Name</th>
                <th className="py-3 px-4">Quality Variant</th>
                <th className="py-3 px-4">Status</th>
                <th className="py-3 px-4">Failure Probability</th>
                <th className="py-3 px-4">Spindle Power</th>
                <th className="py-3 px-4">Thermal ΔT</th>
                <th className="py-3 px-4">Tool Wear</th>
                <th className="py-3 px-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {MOCK_MACHINES.map((m) => (
                <tr key={m.id} className="hover:bg-slate-800/40 transition-colors">
                  <td className="py-3.5 px-4">
                    <div className="font-semibold text-slate-200">{m.name}</div>
                    <div className="text-[11px] font-mono text-slate-400">
                      {m.id} • {m.location}
                    </div>
                  </td>
                  <td className="py-3.5 px-4">
                    <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono font-bold">
                      Type {m.type}
                    </span>
                  </td>
                  <td className="py-3.5 px-4">
                    <StatusBadge status={m.status} size="sm" />
                  </td>
                  <td className="py-3.5 px-4 font-mono font-bold">
                    <span
                      className={
                        m.failure_probability > 0.5
                          ? "text-rose-400"
                          : m.failure_probability > 0.2
                          ? "text-amber-400"
                          : "text-emerald-400"
                      }
                    >
                      {(m.failure_probability * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="py-3.5 px-4 font-mono text-slate-300">
                    {m.derived.power_w.toLocaleString()} W
                  </td>
                  <td className="py-3.5 px-4 font-mono text-slate-300">
                    {m.derived.temp_diff_k.toFixed(1)} K
                  </td>
                  <td className="py-3.5 px-4 font-mono text-slate-300">
                    {m.telemetry.tool_wear_min} min
                  </td>
                  <td className="py-3.5 px-4 text-right">
                    <Link
                      href={`/machines/${m.id}`}
                      className="px-3 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 text-[11px] font-medium transition-colors"
                    >
                      Details
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
