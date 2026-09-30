import type { Metadata } from "next";
import "./globals.css";
import { Sidebar } from "@/components/layout/Sidebar";
import { TopNav } from "@/components/layout/TopNav";

export const metadata: Metadata = {
  title: "MachSense | Industrial AI Machine Monitoring & Diagnostics",
  description:
    "Enterprise AI-powered predictive maintenance, real-time machine telemetry validation, and TreeSHAP explainability control center.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-[#0B0F17] text-slate-100 min-h-screen flex antialiased selection:bg-cyan-500/20 selection:text-cyan-300">
        <Sidebar />
        <div className="flex-1 flex flex-col min-w-0">
          <TopNav />
          <main className="flex-1 p-6 md:p-8 overflow-y-auto max-w-7xl w-full mx-auto">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
