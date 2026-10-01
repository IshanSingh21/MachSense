"use client";

import React, { useState } from "react";
import { Sidebar } from "./Sidebar";
import { TopNav } from "./TopNav";

export function AppShell({ children }: { children: React.ReactNode }) {
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <div className="flex min-h-screen w-full bg-[#0B0F17] text-slate-100 antialiased selection:bg-cyan-500/20 selection:text-cyan-300">
      <Sidebar mobileOpen={mobileOpen} onCloseMobile={() => setMobileOpen(false)} />
      <div className="flex-1 flex flex-col min-w-0">
        <TopNav onOpenMobile={() => setMobileOpen(true)} />
        <main className="flex-1 p-4 sm:p-6 md:p-8 overflow-y-auto max-w-7xl w-full mx-auto">
          {children}
        </main>
      </div>
    </div>
  );
}
