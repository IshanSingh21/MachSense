"use client";

import React, { useState } from "react";
import {
  Activity,
  AlertTriangle,
  Brain,
  CheckCircle2,
  Cpu,
  Flame,
  Info,
  Layers,
  Sparkles,
  Zap,
} from "lucide-react";
import { GlobalImportanceChart } from "@/components/charts/GlobalImportanceChart";
import { ShapWaterfallChart } from "@/components/charts/ShapWaterfallChart";
import { StatCard } from "@/components/ui/StatCard";
import { GLOBAL_FEATURE_IMPORTANCE } from "@/lib/mockData";
import { FeatureAttribution } from "@/types";

const SAMPLE_XAI_ESCALATORS: FeatureAttribution[] = [
  { feature_name: "overstrain_index", display_name: "Overstrain Metric [Nm · min]", attribution_value: 0.1759, percentage: 29.3 },
  { feature_name: "tool_wear_rate", display_name: "Wear Degradation Rate [min / rpm]", attribution_value: 0.1483, percentage: 24.7 },
  { feature_name: "tool_wear_min", display_name: "Accumulated Cutting Time [min]", attribution_value: 0.0998, percentage: 16.6 },
  { feature_name: "torque_nm", display_name: "Spindle Torque [Nm]", attribution_value: 0.067, percentage: 11.2 },
];

const SAMPLE_XAI_STABILIZERS: FeatureAttribution[] = [
  { feature_name: "temp_ratio", display_name: "Process / Air Temp Ratio", attribution_value: -0.0405, percentage: 6.7 },
  { feature_name: "temp_difference_k", display_name: "Temperature Delta ΔT [K]", attribution_value: -0.0142, percentage: 2.4 },
  { feature_name: "power_w", display_name: "Mechanical Power [W]", attribution_value: -0.0109, percentage: 1.8 },
];

export default function ExplainabilityPage() {
  return (
    <div className="space-y-6">
      {/* 1. Header Banner */}
      <div className="bg-[#111827] border border-slate-800 p-6 rounded-2xl flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="p-2 rounded-xl bg-cyan-950 text-cyan-400 border border-cyan-800">
              <Brain className="w-5 h-5" />
            </span>
            <h2 className="text-base font-bold text-slate-100">
              TreeSHAP Explainable AI (XAI) Architecture
            </h2>
          </div>
          <p className="text-xs text-slate-400 max-w-2xl mt-1">
            MachSense decomposes complex non-linear Random Forest ensemble predictions into exact additive feature attributions (ϕ_i), guaranteeing mathematical fairness, symmetry, and efficiency.
          </p>
        </div>

        <div className="flex items-center gap-2 bg-slate-900 border border-slate-800 p-2 rounded-xl text-xs font-mono text-slate-300">
          <span className="text-cyan-400 font-bold">TreeSHAP:</span>
          <span>Exact Sub-10ms O(TLD²) Computation</span>
        </div>
      </div>

      {/* 2. Global Feature Importance Section */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-7 bg-[#111827] border border-slate-800 p-6 rounded-2xl">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-cyan-400" />
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
                Global Feature Importance Ranking (Mean |SHAP|)
              </h3>
            </div>
            <span className="text-[11px] font-mono text-slate-400">Tuned Random Forest v1.0.0</span>
          </div>
          <p className="text-xs text-slate-400 mb-4">
            Aggregated impact across all 10,000 training cycles in the AI4I industrial telemetry domain.
          </p>

          <GlobalImportanceChart data={GLOBAL_FEATURE_IMPORTANCE} height={320} />
        </div>

        {/* Feature Domain Breakdown Cards */}
        <div className="lg:col-span-5 space-y-3">
          <div className="p-4 bg-[#111827] border border-slate-800 rounded-2xl">
            <div className="flex items-center gap-2 text-rose-400 text-xs font-bold uppercase mb-1">
              <Zap className="w-4 h-4" />
              <span>1. Overstrain & Torque Stress (43.6%)</span>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              The interaction between mechanical torque (τ) and cumulative tool wear duration is the #1 predictor of catastrophic insert fracture.
            </p>
          </div>

          <div className="p-4 bg-[#111827] border border-slate-800 rounded-2xl">
            <div className="flex items-center gap-2 text-cyan-400 text-xs font-bold uppercase mb-1">
              <Activity className="w-4 h-4" />
              <span>2. Spindle Mechanical Power (22.8%)</span>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              Formulated as Power P = Torque · Speed · (2π/60). Flags abnormal inverter motor surge (&gt;9,000 W) and tool stalling.
            </p>
          </div>

          <div className="p-4 bg-[#111827] border border-slate-800 rounded-2xl">
            <div className="flex items-center gap-2 text-amber-400 text-xs font-bold uppercase mb-1">
              <Flame className="w-4 h-4" />
              <span>3. Thermal Dissipation Gradient (ΔT)</span>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              Captures cooling fluid starvation (ΔT &lt; 8.6 K) leading to heat dissipation failures (HDF).
            </p>
          </div>
        </div>
      </div>

      {/* 3. Sample Local Instance Attribution Waterfall Breakdown */}
      <div className="bg-[#111827] border border-slate-800 p-6 rounded-2xl">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
          <div>
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200">
              Local Instance Attribution Diagnostic (Sample OSF Failure)
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Exact additive contributions demonstrating how sensor parameters shifted baseline probability from 50.1% to 97.0%
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs font-mono px-2.5 py-1 rounded bg-rose-950 text-rose-300 border border-rose-800 font-bold">
              Failure Prob: 97.0%
            </span>
          </div>
        </div>

        <ShapWaterfallChart
          escalators={SAMPLE_XAI_ESCALATORS}
          stabilizers={SAMPLE_XAI_STABILIZERS}
          height={240}
        />

        {/* Translation Table */}
        <div className="mt-6 pt-4 border-t border-slate-800 grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 rounded-xl bg-rose-950/20 border border-rose-800/40">
            <span className="text-xs font-bold uppercase text-rose-400 block mb-2">
              Top Risk Escalators (Pushing Toward Failure)
            </span>
            <ul className="space-y-1.5 text-xs text-slate-300">
              {SAMPLE_XAI_ESCALATORS.map((e, idx) => (
                <li key={idx} className="flex justify-between font-mono">
                  <span>{e.display_name}</span>
                  <span className="text-rose-400 font-bold">+{e.attribution_value.toFixed(4)} SHAP</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="p-4 rounded-xl bg-emerald-950/20 border border-emerald-800/40">
            <span className="text-xs font-bold uppercase text-emerald-400 block mb-2">
              Stabilizing Factors (Reducing Failure Risk)
            </span>
            <ul className="space-y-1.5 text-xs text-slate-300">
              {SAMPLE_XAI_STABILIZERS.map((s, idx) => (
                <li key={idx} className="flex justify-between font-mono">
                  <span>{s.display_name}</span>
                  <span className="text-emerald-400 font-bold">{s.attribution_value.toFixed(4)} SHAP</span>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="mt-4 p-3 bg-slate-900 border border-slate-800 rounded-xl text-[11px] text-slate-400 italic">
          <strong>Non-Causal Statistical Attribution Disclaimer:</strong> SHAP values reflect mathematical contributions of features under the trained Random Forest decision trees. Attribution indicates statistical correlation with historical failures, not necessarily a direct physical root cause.
        </div>
      </div>
    </div>
  );
}
