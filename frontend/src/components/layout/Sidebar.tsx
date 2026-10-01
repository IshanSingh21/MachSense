"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import clsx from "clsx";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Cpu,
  Layers,
  LayoutDashboard,
  Radio,
  Settings,
  ShieldCheck,
  X,
} from "lucide-react";
import { api } from "@/lib/api";

const NAV_ITEMS = [
  { label: "Overview", href: "/", icon: LayoutDashboard },
  { label: "Machines", href: "/machines", icon: Cpu },
  { label: "Predictions", href: "/predictions", icon: Activity },
  { label: "Analytics", href: "/analytics", icon: BarChart3 },
  { label: "Alerts", href: "/alerts", icon: AlertTriangle, badge: "2" },
  { label: "Explainability", href: "/explainability", icon: Layers },
  { label: "Settings", href: "/settings", icon: Settings },
];

interface SidebarProps {
  mobileOpen?: boolean;
  onCloseMobile?: () => void;
}

export function Sidebar({ mobileOpen = false, onCloseMobile }: SidebarProps) {
  const pathname = usePathname();
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [modelVersion, setModelVersion] = useState<string>("v1.0.0");

  useEffect(() => {
    let mounted = true;
    async function checkApi() {
      try {
        const readiness = await api.getReadiness();
        if (mounted) {
          setApiOnline(readiness.model_loaded);
          if (readiness.model_version) {
            setModelVersion(readiness.model_version);
          }
        }
      } catch {
        if (mounted) {
          setApiOnline(false);
        }
      }
    }

    checkApi();
    const interval = setInterval(checkApi, 30000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <>
      {/* Mobile Backdrop */}
      {mobileOpen && (
        <div
          onClick={onCloseMobile}
          className="fixed inset-0 bg-black/70 backdrop-blur-sm z-40 lg:hidden transition-opacity"
        />
      )}

      <aside
        className={clsx(
          "w-64 bg-[#0E131F] border-r border-slate-800/80 flex flex-col justify-between shrink-0 select-none h-screen sticky top-0 z-50 transition-transform duration-200 lg:translate-x-0",
          mobileOpen ? "fixed inset-y-0 left-0 translate-x-0" : "hidden lg:flex"
        )}
      >
        {/* Brand Header */}
        <div>
          <div className="h-16 flex items-center justify-between px-6 border-b border-slate-800/60">
            <Link href="/" className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
                <Radio className="w-4 h-4 text-white animate-pulse" />
              </div>
              <div className="flex flex-col">
                <div className="flex items-center gap-2">
                  <span className="font-bold text-slate-100 tracking-tight text-base font-mono">
                    MACHSENSE
                  </span>
                  <span className="text-[10px] uppercase font-semibold px-1.5 py-0.2 rounded bg-cyan-950 text-cyan-400 border border-cyan-800/60">
                    PRO
                  </span>
                </div>
                <span className="text-[10px] text-slate-400 font-medium tracking-wide">
                  Industrial AI Telemetry
                </span>
              </div>
            </Link>

            {onCloseMobile && (
              <button
                onClick={onCloseMobile}
                className="lg:hidden p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>

          {/* Navigation Links */}
          <nav className="p-3 space-y-1">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const isActive =
                item.href === "/"
                  ? pathname === "/"
                  : pathname === item.href || pathname.startsWith(`${item.href}/`);

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  onClick={onCloseMobile}
                  className={clsx(
                    "flex items-center justify-between px-3.5 py-2.5 rounded-lg text-xs font-medium transition-all group",
                    isActive
                      ? "bg-slate-800/90 text-cyan-400 border border-slate-700/60 shadow-sm font-semibold"
                      : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/40"
                  )}
                >
                  <div className="flex items-center gap-3">
                    <Icon
                      className={clsx(
                        "w-4 h-4 transition-colors",
                        isActive ? "text-cyan-400" : "text-slate-400 group-hover:text-slate-300"
                      )}
                    />
                    <span>{item.label}</span>
                  </div>

                  {item.badge && (
                    <span
                      className={clsx(
                        "text-[10px] px-1.5 py-0.2 rounded-full font-semibold",
                        isActive
                          ? "bg-rose-500/20 text-rose-300 border border-rose-500/40"
                          : "bg-rose-950 text-rose-400 border border-rose-900/60"
                      )}
                    >
                      {item.badge}
                    </span>
                  )}
                </Link>
              );
            })}
          </nav>
        </div>

        {/* System Status Footer */}
        <div className="p-4 border-t border-slate-800/60 bg-[#0B0F17]/50 m-3 rounded-xl border">
          <div className="flex items-center justify-between text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-2">
            <span>System Status</span>
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
          </div>

          <div className="space-y-1.5 text-xs">
            <div className="flex items-center justify-between">
              <span className="text-slate-400 text-[11px]">FastAPI Backend</span>
              <div className="flex items-center gap-1.5 font-mono text-[11px]">
                <span
                  className={clsx(
                    "w-2 h-2 rounded-full",
                    apiOnline === true
                      ? "bg-emerald-400 animate-pulse"
                      : apiOnline === false
                      ? "bg-amber-400"
                      : "bg-slate-500"
                  )}
                />
                <span
                  className={
                    apiOnline === true
                      ? "text-emerald-400 font-medium"
                      : apiOnline === false
                      ? "text-amber-400 font-medium"
                      : "text-slate-400"
                  }
                >
                  {apiOnline === true ? "ONLINE" : apiOnline === false ? "STANDALONE" : "CHECKING"}
                </span>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-slate-400 text-[11px]">Model Engine</span>
              <span className="text-slate-300 font-mono text-[11px]">{modelVersion}</span>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}
