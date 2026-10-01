import React from "react";
import clsx from "clsx";

interface SkeletonProps {
  className?: string;
}

export function Skeleton({ className }: SkeletonProps) {
  return (
    <div
      className={clsx(
        "animate-pulse rounded-lg bg-slate-800/80 border border-slate-700/40",
        className
      )}
    />
  );
}

export function CardSkeleton() {
  return (
    <div className="bg-[#111827] border border-slate-800 rounded-2xl p-5 space-y-3">
      <div className="flex justify-between">
        <Skeleton className="h-4 w-28" />
        <Skeleton className="h-6 w-16" />
      </div>
      <Skeleton className="h-8 w-20" />
      <div className="grid grid-cols-2 gap-2 pt-2">
        <Skeleton className="h-12 w-full" />
        <Skeleton className="h-12 w-full" />
      </div>
    </div>
  );
}
