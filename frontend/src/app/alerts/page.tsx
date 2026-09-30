"use client";

import React, { useState } from "react";
import Link from "next/link";
import {
  AlertOctagon,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Clock,
  Filter,
  Info,
  ShieldAlert,
  Wrench,
} from "lucide-react";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { MOCK_ALERTS } from "@/lib/mockData";
import { Alert } from "@/types";

export default function AlertsPage() {
  const [alerts, setAlerts] = useState<Alert[]>(MOCK_ALERTS);
  const [severityFilter, setSeverityFilter] = useState<string>("ALL");
  const [failureTypeFilter, setFailureTypeFilter] = useState<string>("ALL");

  const handleAcknowledge = (id: string) => {
    setAlerts(
      alerts.map((a) => (a.id === id ? { ...a, acknowledged: !a.acknowledged } : a))
    );
  };

  const filteredAlerts = alerts.filter((a) => {
    const matchesSeverity = severityFilter === "ALL" || a.severity === severityFilter;
    const matchesType = failureTypeFilter === "ALL" || a.failure_type === failureTypeFilter;
    return matchesSeverity && matchesType;
  });

  const criticalCount = alerts.filter((a) => a.severity === "CRITICAL" && !a.acknowledged).length;
  const warningCount = alerts.filter((a) => a.severity === "WARNING" && !a.acknowledged).length;

  return (
    <div className="space-y-6">
      {/* 1. Header & Active Counts */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-[#111827] border border-rose-900/40 p-5 rounded-2xl flex items-center justify-between">
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
              Unacknowledged Critical
            </span>
            <div className="text-2xl font-bold font-mono text-rose-400 mt-1">{criticalCount}</div>
          </div>
          <div className="p-3 rounded-xl bg-rose-950/60 text-rose-400">
            <AlertOctagon className="w-6 h-6" />
          </div>
        </div>

        <div className="bg-[#111827] border border-amber-900/40 p-5 rounded-2xl flex items-center justify-between">
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
              Active Warnings
            </span>
            <div className="text-2xl font-bold font-mono text-amber-400 mt-1">{warningCount}</div>
          </div>
          <div className="p-3 rounded-xl bg-amber-950/60 text-amber-400">
            <AlertTriangle className="w-6 h-6" />
          </div>
        </div>

        <div className="bg-[#111827] border border-slate-800 p-5 rounded-2xl flex items-center justify-between">
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
              Total Logged Today
            </span>
            <div className="text-2xl font-bold font-mono text-slate-100 mt-1">{alerts.length}</div>
          </div>
          <div className="p-3 rounded-xl bg-slate-800 text-slate-400">
            <ShieldAlert className="w-6 h-6" />
          </div>
        </div>
      </div>

      {/* 2. Filters Strip */}
      <div className="bg-[#111827] border border-slate-800 p-4 rounded-2xl flex flex-wrap items-center justify-between gap-3">
        {/* Severity Tabs */}
        <div className="flex items-center gap-1 bg-slate-900 border border-slate-800 p-1 rounded-xl text-xs">
          {["ALL", "CRITICAL", "WARNING", "INFO"].map((s) => (
            <button
              key={s}
              onClick={() => setSeverityFilter(s)}
              className={`px-3 py-1 rounded-lg font-medium transition-colors ${
                severityFilter === s
                  ? "bg-cyan-950 text-cyan-400 border border-cyan-800/60"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {s}
            </button>
          ))}
        </div>

        {/* Failure Mode Filters */}
        <div className="flex items-center gap-1 bg-slate-900 border border-slate-800 p-1 rounded-xl text-xs">
          {["ALL", "OSF", "HDF", "PWF", "TWF"].map((t) => (
            <button
              key={t}
              onClick={() => setFailureTypeFilter(t)}
              className={`px-3 py-1 rounded-lg font-mono font-bold transition-colors ${
                failureTypeFilter === t
                  ? "bg-slate-800 text-white border border-slate-700"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {t === "ALL" ? "All Modes" : t}
            </button>
          ))}
        </div>
      </div>

      {/* 3. Alerts Feed */}
      <div className="space-y-4">
        {filteredAlerts.length === 0 ? (
          <div className="text-center py-16 bg-[#111827] border border-slate-800 rounded-2xl">
            <CheckCircle2 className="w-10 h-10 text-emerald-400 mx-auto mb-2" />
            <h3 className="text-sm font-bold text-slate-200">No Alerts Found</h3>
            <p className="text-xs text-slate-400 mt-1">All machines are operating within safe thermal and torque limits.</p>
          </div>
        ) : (
          filteredAlerts.map((alert) => (
            <div
              key={alert.id}
              className={`bg-[#111827] border rounded-2xl p-5 transition-all ${
                alert.severity === "CRITICAL"
                  ? "border-rose-900/60 bg-rose-950/10"
                  : alert.severity === "WARNING"
                  ? "border-amber-900/60 bg-amber-950/10"
                  : "border-slate-800"
              }`}
            >
              <div className="flex flex-wrap items-start justify-between gap-3 mb-3">
                <div className="flex items-start gap-3">
                  <span
                    className={`p-2.5 rounded-xl border mt-0.5 ${
                      alert.severity === "CRITICAL"
                        ? "bg-rose-950 text-rose-400 border-rose-800"
                        : alert.severity === "WARNING"
                        ? "bg-amber-950 text-amber-400 border-amber-800"
                        : "bg-cyan-950 text-cyan-400 border-cyan-800"
                    }`}
                  >
                    {alert.severity === "CRITICAL" ? (
                      <AlertOctagon className="w-5 h-5" />
                    ) : alert.severity === "WARNING" ? (
                      <AlertTriangle className="w-5 h-5" />
                    ) : (
                      <Info className="w-5 h-5" />
                    )}
                  </span>

                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-xs font-bold px-2 py-0.5 rounded bg-slate-900 text-slate-300">
                        {alert.id}
                      </span>
                      <span className="text-xs font-bold text-slate-100">{alert.machine_name}</span>
                      <span className="font-mono text-xs text-slate-400">({alert.machine_id})</span>
                    </div>
                    <p className="text-xs font-semibold text-slate-200 mt-1.5">{alert.issue}</p>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <span className="px-2 py-0.5 rounded bg-slate-900 font-mono text-xs font-bold text-cyan-400 border border-slate-800">
                    Mode: {alert.failure_type}
                  </span>
                  <StatusBadge status={alert.severity} size="sm" />
                </div>
              </div>

              {/* Recommended Action Box */}
              <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800/80 my-3 text-xs flex items-start gap-2.5">
                <Wrench className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                <div>
                  <span className="font-bold text-slate-200">Recommended Operator Action: </span>
                  <span className="text-slate-300">{alert.recommended_action}</span>
                </div>
              </div>

              {/* Bottom Footer Controls */}
              <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-slate-800/60 text-xs">
                <div className="flex items-center gap-2 text-slate-400 font-mono text-[11px]">
                  <Clock className="w-3.5 h-3.5" />
                  <span>Logged at: {alert.timestamp}</span>
                </div>

                <div className="flex items-center gap-3">
                  <button
                    onClick={() => handleAcknowledge(alert.id)}
                    className={`px-3 py-1.5 rounded-lg font-medium text-xs transition-colors cursor-pointer ${
                      alert.acknowledged
                        ? "bg-slate-800 text-emerald-400 border border-slate-700"
                        : "bg-slate-800 hover:bg-slate-700 text-slate-200"
                    }`}
                  >
                    {alert.acknowledged ? "✓ Acknowledged" : "Acknowledge Alert"}
                  </button>

                  <Link
                    href={`/machines/${alert.machine_id}`}
                    className="px-3 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white font-medium text-xs flex items-center gap-1 transition-colors"
                  >
                    <span>Inspect Machine</span>
                    <ArrowRight className="w-3 h-3" />
                  </Link>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
