import React from "react";
import clsx from "clsx";

interface TelemetryCardProps {
  label: string;
  value: string | number;
  unit: string;
  nominalRange?: string;
  status?: "normal" | "warning" | "critical";
  icon?: React.ReactNode;
}

export function TelemetryCard({
  label,
  value,
  unit,
  nominalRange,
  status = "normal",
  icon,
}: TelemetryCardProps) {
  const statusColor = {
    normal: "text-slate-100",
    warning: "text-amber-400",
    critical: "text-rose-400",
  }[status];

  const statusBg = {
    normal: "border-slate-800 bg-[#131B2E]",
    warning: "border-amber-900/40 bg-amber-950/10",
    critical: "border-rose-900/40 bg-rose-950/10",
  }[status];

  return (
    <div className={clsx("p-4 rounded-xl border transition-colors", statusBg)}>
      <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
        <span className="font-medium">{label}</span>
        {icon && <span className="text-slate-500">{icon}</span>}
      </div>
      <div className="flex items-baseline gap-1.5">
        <span className={clsx("text-2xl font-bold font-mono tracking-tight", statusColor)}>
          {typeof value === "number" ? value.toLocaleString() : value}
        </span>
        <span className="text-xs text-slate-400 font-medium">{unit}</span>
      </div>
      {nominalRange && (
        <div className="mt-2 text-[11px] text-slate-400 flex items-center justify-between pt-2 border-t border-slate-800/60">
          <span>Nominal Band:</span>
          <span className="font-mono text-slate-400">{nominalRange}</span>
        </div>
      )}
    </div>
  );
}
