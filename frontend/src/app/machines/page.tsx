"use client";

import React, { useState } from "react";
import Link from "next/link";
import {
  Activity,
  ArrowRight,
  CheckCircle2,
  Cpu,
  Filter,
  Search,
  SlidersHorizontal,
} from "lucide-react";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { TelemetryCard } from "@/components/ui/TelemetryCard";
import { MOCK_MACHINES } from "@/lib/mockData";
import { MachineStatus, MachineType } from "@/types";

export default function MachinesPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [typeFilter, setTypeFilter] = useState<string>("ALL");
  const [statusFilter, setStatusFilter] = useState<string>("ALL");
  const [viewMode, setViewMode] = useState<"cards" | "table">("cards");

  const filteredMachines = MOCK_MACHINES.filter((machine) => {
    const matchesSearch =
      machine.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      machine.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      machine.location.toLowerCase().includes(searchQuery.toLowerCase());

    const matchesType = typeFilter === "ALL" || machine.type === typeFilter;
    const matchesStatus = statusFilter === "ALL" || machine.status === statusFilter;

    return matchesSearch && matchesType && matchesStatus;
  });

  return (
    <div className="space-y-6">
      {/* 1. Header & Filters Control Strip */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 bg-[#111827] border border-slate-800 p-4 rounded-2xl">
        {/* Search Bar */}
        <div className="relative w-full md:w-80">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search by ID, name, or bay..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-4 py-2 bg-slate-900 border border-slate-800 rounded-xl text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-500 transition-colors"
          />
        </div>

        {/* Filter Badges & View Mode */}
        <div className="flex flex-wrap items-center gap-3 w-full md:w-auto">
          {/* Status Filter */}
          <div className="flex items-center gap-1 bg-slate-900 border border-slate-800 rounded-xl p-1 text-xs">
            {["ALL", "HEALTHY", "WARNING", "CRITICAL"].map((status) => (
              <button
                key={status}
                onClick={() => setStatusFilter(status)}
                className={`px-3 py-1 rounded-lg font-medium transition-colors ${
                  statusFilter === status
                    ? "bg-cyan-950 text-cyan-400 border border-cyan-800/60"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {status}
              </button>
            ))}
          </div>

          {/* Type Filter */}
          <div className="flex items-center gap-1 bg-slate-900 border border-slate-800 rounded-xl p-1 text-xs">
            {["ALL", "L", "M", "H"].map((type) => (
              <button
                key={type}
                onClick={() => setTypeFilter(type)}
                className={`px-3 py-1 rounded-lg font-mono font-bold transition-colors ${
                  typeFilter === type
                    ? "bg-slate-800 text-white border border-slate-700"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {type === "ALL" ? "All Types" : `Type ${type}`}
              </button>
            ))}
          </div>

          {/* View Toggle */}
          <div className="flex items-center gap-1 bg-slate-900 border border-slate-800 rounded-xl p-1 text-xs ml-auto">
            <button
              onClick={() => setViewMode("cards")}
              className={`px-3 py-1 rounded-lg font-medium transition-colors ${
                viewMode === "cards" ? "bg-slate-800 text-white" : "text-slate-400"
              }`}
            >
              Cards
            </button>
            <button
              onClick={() => setViewMode("table")}
              className={`px-3 py-1 rounded-lg font-medium transition-colors ${
                viewMode === "table" ? "bg-slate-800 text-white" : "text-slate-400"
              }`}
            >
              Table
            </button>
          </div>
        </div>
      </div>

      {/* 2. Machines Cards Grid */}
      {viewMode === "cards" ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredMachines.map((machine) => (
            <div
              key={machine.id}
              className="bg-[#111827] border border-slate-800 hover:border-slate-700/80 rounded-2xl p-5 flex flex-col justify-between transition-all duration-200 shadow-sm group"
            >
              <div>
                {/* Header */}
                <div className="flex items-start justify-between gap-2 mb-3">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-xs font-bold px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                        {machine.id}
                      </span>
                      <span className="text-xs font-semibold px-2 py-0.5 rounded bg-cyan-950/60 text-cyan-400 border border-cyan-900/60">
                        Type {machine.type}
                      </span>
                    </div>
                    <h3 className="font-bold text-slate-100 text-sm mt-2">{machine.name}</h3>
                    <p className="text-xs text-slate-400">{machine.location}</p>
                  </div>
                  <StatusBadge status={machine.status} size="sm" />
                </div>

                {/* Warning message if any */}
                {machine.warning_message && (
                  <div className="mb-4 p-2.5 rounded-xl bg-amber-950/20 border border-amber-800/40 text-[11px] text-amber-300">
                    {machine.warning_message}
                  </div>
                )}

                {/* Telemetry Snapshot */}
                <div className="grid grid-cols-2 gap-2 mb-4">
                  <div className="p-2.5 rounded-xl bg-slate-900 border border-slate-800/80">
                    <span className="text-[10px] text-slate-400 uppercase block font-medium">
                      Torque
                    </span>
                    <span className="text-sm font-bold font-mono text-slate-100">
                      {machine.telemetry.torque_nm} Nm
                    </span>
                  </div>
                  <div className="p-2.5 rounded-xl bg-slate-900 border border-slate-800/80">
                    <span className="text-[10px] text-slate-400 uppercase block font-medium">
                      Tool Wear
                    </span>
                    <span className="text-sm font-bold font-mono text-slate-100">
                      {machine.telemetry.tool_wear_min} min
                    </span>
                  </div>
                  <div className="p-2.5 rounded-xl bg-slate-900 border border-slate-800/80">
                    <span className="text-[10px] text-slate-400 uppercase block font-medium">
                      Spindle Speed
                    </span>
                    <span className="text-sm font-bold font-mono text-slate-100">
                      {machine.telemetry.rotational_speed_rpm} RPM
                    </span>
                  </div>
                  <div className="p-2.5 rounded-xl bg-slate-900 border border-slate-800/80">
                    <span className="text-[10px] text-slate-400 uppercase block font-medium">
                      Thermal ΔT
                    </span>
                    <span className="text-sm font-bold font-mono text-slate-100">
                      {machine.derived.temp_diff_k.toFixed(1)} K
                    </span>
                  </div>
                </div>
              </div>

              {/* Bottom Footer Action */}
              <div className="pt-3 border-t border-slate-800 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-[11px] text-slate-400">Risk Prob:</span>
                  <span
                    className={`font-mono text-xs font-bold ${
                      machine.failure_probability > 0.5
                        ? "text-rose-400"
                        : machine.failure_probability > 0.2
                        ? "text-amber-400"
                        : "text-emerald-400"
                    }`}
                  >
                    {(machine.failure_probability * 100).toFixed(1)}%
                  </span>
                </div>

                <Link
                  href={`/machines/${machine.id}`}
                  className="text-xs font-semibold text-cyan-400 hover:text-cyan-300 flex items-center gap-1 group-hover:translate-x-0.5 transition-transform"
                >
                  <span>Diagnostic View</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </Link>
              </div>
            </div>
          ))}
        </div>
      ) : (
        /* Table View */
        <div className="bg-[#111827] border border-slate-800 rounded-2xl overflow-hidden">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-900 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800">
              <tr>
                <th className="py-3.5 px-4">Machine</th>
                <th className="py-3.5 px-4">Type</th>
                <th className="py-3.5 px-4">Status</th>
                <th className="py-3.5 px-4">Failure Probability</th>
                <th className="py-3.5 px-4">Torque (Nm)</th>
                <th className="py-3.5 px-4">Spindle (RPM)</th>
                <th className="py-3.5 px-4">Wear (min)</th>
                <th className="py-3.5 px-4 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {filteredMachines.map((m) => (
                <tr key={m.id} className="hover:bg-slate-800/40 transition-colors">
                  <td className="py-3.5 px-4">
                    <div className="font-bold text-slate-200">{m.name}</div>
                    <div className="text-[11px] font-mono text-slate-400">{m.id}</div>
                  </td>
                  <td className="py-3.5 px-4">
                    <span className="font-mono font-bold px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                      {m.type}
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
                  <td className="py-3.5 px-4 font-mono text-slate-300">{m.telemetry.torque_nm}</td>
                  <td className="py-3.5 px-4 font-mono text-slate-300">
                    {m.telemetry.rotational_speed_rpm}
                  </td>
                  <td className="py-3.5 px-4 font-mono text-slate-300">
                    {m.telemetry.tool_wear_min}
                  </td>
                  <td className="py-3.5 px-4 text-right">
                    <Link
                      href={`/machines/${m.id}`}
                      className="px-3 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium text-xs"
                    >
                      Inspect
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
