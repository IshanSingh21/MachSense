"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Bell, Clock, Menu, Shield, User } from "lucide-react";

interface TopNavProps {
  onOpenMobile?: () => void;
}

export function TopNav({ onOpenMobile }: TopNavProps) {
  const pathname = usePathname();
  const [time, setTime] = useState<string>("");
  const [timeZoneName, setTimeZoneName] = useState<string>("");
  const [timeZoneId, setTimeZoneId] = useState<string>("");

  useEffect(() => {
    function updateClock() {
      const now = new Date();
      try {
        const formattedTime = now.toLocaleTimeString(undefined, {
          hour: "numeric",
          minute: "2-digit",
          second: "2-digit",
          hour12: true,
        });

        const tz =
          Intl.DateTimeFormat(undefined, { timeZoneName: "short" })
            .formatToParts(now)
            .find((p) => p.type === "timeZoneName")?.value || "";

        const resolvedTz = Intl.DateTimeFormat().resolvedOptions().timeZone || "";

        setTime(formattedTime);
        setTimeZoneName(tz);
        setTimeZoneId(resolvedTz);
      } catch {
        setTime(now.toLocaleTimeString());
      }
    }

    updateClock();
    const interval = setInterval(updateClock, 1000);
    return () => clearInterval(interval);
  }, []);

  const pageTitles: Record<string, { title: string; subtitle: string }> = {
    "/": {
      title: "Fleet Telemetry Overview",
      subtitle: "Real-time CNC condition monitoring and failure risk assessment",
    },
    "/machines": {
      title: "Machine Fleet Directory",
      subtitle: "Multi-axis milling centers, lathes, and high-precision spindle units",
    },
    "/predictions": {
      title: "Predictive Inference Studio",
      subtitle: "Live sensor parameter scoring, failure mode presets, and batch testing",
    },
    "/analytics": {
      title: "Telemetry & Risk Analytics",
      subtitle: "Historical degradation trends, thermodynamic curves, and power loads",
    },
    "/alerts": {
      title: "Industrial Alert Center",
      subtitle: "Active anomaly warnings, threshold breaches, and operator corrective actions",
    },
    "/explainability": {
      title: "TreeSHAP Explainability Center",
      subtitle: "Additive feature attributions, risk escalators, and root-cause diagnostics",
    },
    "/settings": {
      title: "System Status & Environment",
      subtitle: "FastAPI endpoints, model metadata, sensor boundaries, and health logs",
    },
  };

  const currentPath = pathname.startsWith("/machines/") ? "/machines" : pathname;
  const { title, subtitle } = pageTitles[currentPath] || {
    title: "MachSense Console",
    subtitle: "Predictive Maintenance System",
  };

  return (
    <header className="h-16 bg-[#0E131F]/90 backdrop-blur border-b border-slate-800/80 px-4 sm:px-6 md:px-8 flex items-center justify-between sticky top-0 z-30">
      {/* Title & Hamburger Button */}
      <div className="flex items-center gap-3">
        {onOpenMobile && (
          <button
            onClick={onOpenMobile}
            className="lg:hidden p-2 rounded-lg bg-slate-900 border border-slate-800 text-slate-300 hover:text-white"
            aria-label="Open Navigation Menu"
          >
            <Menu className="w-4 h-4" />
          </button>
        )}

        <div>
          <h1 className="text-xs sm:text-sm font-bold text-slate-100 tracking-tight flex items-center gap-2">
            <span>{title}</span>
            <span className="hidden md:inline text-slate-600">/</span>
            <span className="hidden md:inline text-xs font-normal text-slate-400">
              {subtitle}
            </span>
          </h1>
        </div>
      </div>

      {/* Right Controls */}
      <div className="flex items-center gap-2.5 sm:gap-4">
        {/* Live System Clock */}
        <div
          className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs font-mono text-slate-300"
          title={timeZoneId ? `Local Browser Time (${timeZoneId})` : "Local System Clock"}
        >
          <Clock className="w-3.5 h-3.5 text-cyan-400" />
          <span>
            {time ? `${time}${timeZoneName ? ` ${timeZoneName}` : ""}` : "--:--:--"}
          </span>
        </div>

        {/* Fleet Health Badge */}
        <div className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-950/40 border border-emerald-800/50 text-emerald-400 text-xs font-medium">
          <Shield className="w-3.5 h-3.5" />
          <span>Fleet: 95.8% Nominal</span>
        </div>

        {/* Notifications Icon */}
        <Link
          href="/alerts"
          className="relative p-2 rounded-lg bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-700 transition-colors"
          title="Active Alerts"
          aria-label="View Active Alerts"
        >
          <Bell className="w-4 h-4" />
          <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-rose-500 animate-pulse" />
        </Link>

        {/* Operator Profile */}
        <div className="flex items-center gap-2.5 pl-2 border-l border-slate-800">
          <div className="w-7 h-7 rounded-lg bg-slate-800 border border-slate-700 flex items-center justify-center text-slate-300">
            <User className="w-4 h-4" />
          </div>
          <div className="hidden xl:flex flex-col text-left">
            <span className="text-xs font-medium text-slate-200">Lead Operator</span>
            <span className="text-[10px] text-slate-400 font-mono">ID: ENG-408</span>
          </div>
        </div>
      </div>
    </header>
  );
}
