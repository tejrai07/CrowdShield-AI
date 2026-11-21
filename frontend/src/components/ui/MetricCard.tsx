import type { ReactNode } from 'react';
import { cn } from '../../utils/cn';

interface MetricCardProps {
  label: string;
  value: ReactNode;
  badge?: ReactNode;
  footnote?: string;
  className?: string;
}

export function MetricCard({ label, value, badge, footnote, className }: MetricCardProps) {
  return (
    <div className={cn('rounded-lg border border-[#E5E7EB] bg-white p-5', className)}>
      <div className="flex items-center justify-between">
        <span className="text-small font-medium text-[#374151]">{label}</span>
        {badge}
      </div>
      <div className="mt-2 text-h3 font-semibold text-[#111827]">{value}</div>
      {footnote ? <div className="mt-1 text-caption text-[#374151]">{footnote}</div> : null}
    </div>
  );
}
