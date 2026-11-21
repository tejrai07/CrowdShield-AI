import type { ReactNode } from 'react';
import { cn } from '../../utils/cn';

interface StatCardProps {
  label: string;
  value: ReactNode;
  icon?: ReactNode;
  trend?: {
    value: string;
    direction: 'up' | 'down' | 'flat';
  };
  accent?: ReactNode;
  className?: string;
}

const trendColor: Record<'up' | 'down' | 'flat', string> = {
  up: 'text-[#DC2626]',
  down: 'text-[#16A34A]',
  flat: 'text-[#374151]',
};

export function StatCard({ label, value, icon, trend, accent, className }: StatCardProps) {
  return (
    <div
      className={cn(
        'flex flex-col gap-3 rounded-lg border border-[#E5E7EB] bg-white p-5',
        className,
      )}
    >
      <div className="flex items-center justify-between">
        <span className="text-small font-medium text-[#374151]">{label}</span>
        {icon ? <span className="text-[#374151]">{icon}</span> : null}
      </div>
      <div className="flex items-end justify-between gap-2">
        <span className="text-h2 font-semibold tabular-nums text-[#111827]">{value}</span>
        {accent}
      </div>
      {trend ? (
        <span className={cn('text-caption font-medium', trendColor[trend.direction])}>
          {trend.value}
        </span>
      ) : null}
    </div>
  );
}
