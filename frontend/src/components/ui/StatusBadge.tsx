import React from "react";
import clsx from "clsx";

interface StatusBadgeProps {
  status: "HEALTHY" | "WARNING" | "CRITICAL" | "NOMINAL" | "MODERATE_WARNING" | "ELEVATED" | "INFO";
  label?: string;
  size?: "sm" | "md";
  showDot?: boolean;
}

export function StatusBadge({ status, label, size = "md", showDot = true }: StatusBadgeProps) {
  const displayLabel = label || status.replace("_", " ");

  const colorStyles = {
    HEALTHY: "bg-emerald-500/10 text-emerald-400 border-emerald-500/30",
    NOMINAL: "bg-emerald-500/10 text-emerald-400 border-emerald-500/30",
    WARNING: "bg-amber-500/10 text-amber-400 border-amber-500/30",
    MODERATE_WARNING: "bg-amber-500/10 text-amber-400 border-amber-500/30",
    ELEVATED: "bg-orange-500/10 text-orange-400 border-orange-500/30",
    CRITICAL: "bg-rose-500/10 text-rose-400 border-rose-500/30",
    INFO: "bg-cyan-500/10 text-cyan-400 border-cyan-500/30",
  }[status] || "bg-slate-500/10 text-slate-400 border-slate-500/30";

  const dotColor = {
    HEALTHY: "bg-emerald-400",
    NOMINAL: "bg-emerald-400",
    WARNING: "bg-amber-400",
    MODERATE_WARNING: "bg-amber-400",
    ELEVATED: "bg-orange-400",
    CRITICAL: "bg-rose-400 animate-pulse",
    INFO: "bg-cyan-400",
  }[status] || "bg-slate-400";

  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 font-medium border rounded-md uppercase tracking-wider",
        size === "sm" ? "px-2 py-0.5 text-[10px]" : "px-2.5 py-1 text-xs",
        colorStyles
      )}
    >
      {showDot && <span className={clsx("w-1.5 h-1.5 rounded-full", dotColor)} />}
      {displayLabel}
    </span>
  );
}
