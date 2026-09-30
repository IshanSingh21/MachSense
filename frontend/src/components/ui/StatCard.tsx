import React from "react";
import clsx from "clsx";

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: React.ReactNode;
  trend?: {
    value: string;
    isPositive?: boolean;
  };
  variant?: "default" | "healthy" | "warning" | "critical";
}

export function StatCard({
  title,
  value,
  subtitle,
  icon,
  trend,
  variant = "default",
}: StatCardProps) {
  const variantBorder = {
    default: "border-slate-800 hover:border-slate-700",
    healthy: "border-emerald-900/40 hover:border-emerald-700/50",
    warning: "border-amber-900/40 hover:border-amber-700/50",
    critical: "border-rose-900/40 hover:border-rose-700/50",
  }[variant];

  const variantAccent = {
    default: "text-slate-400",
    healthy: "text-emerald-400",
    warning: "text-amber-400",
    critical: "text-rose-400",
  }[variant];

  return (
    <div
      className={clsx(
        "bg-[#111827] border rounded-xl p-5 transition-all duration-200 shadow-sm",
        variantBorder
      )}
    >
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          {title}
        </span>
        {icon && <div className={clsx("p-2 rounded-lg bg-slate-800/60", variantAccent)}>{icon}</div>}
      </div>

      <div className="mt-3 flex items-baseline justify-between">
        <div className="text-2xl font-bold tracking-tight text-slate-100">{value}</div>
        {trend && (
          <span
            className={clsx(
              "text-xs font-medium px-1.5 py-0.5 rounded",
              trend.isPositive ? "bg-emerald-500/10 text-emerald-400" : "bg-rose-500/10 text-rose-400"
            )}
          >
            {trend.value}
          </span>
        )}
      </div>

      {subtitle && <p className="mt-1 text-xs text-slate-400">{subtitle}</p>}
    </div>
  );
}
