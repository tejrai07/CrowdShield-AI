import type { PropsWithChildren, ReactNode } from 'react';
import { cn } from '../../utils/cn';

interface ChartCardProps {
  title: string;
  description?: string;
  actions?: ReactNode;
  className?: string;
  height?: number;
}

export function ChartCard({
  title,
  description,
  actions,
  className,
  height = 280,
  children,
}: PropsWithChildren<ChartCardProps>) {
  return (
    <div className={cn('rounded-lg border border-[#E5E7EB] bg-white p-5', className)}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-h4 font-semibold text-[#111827]">{title}</h3>
          {description ? (
            <p className="mt-0.5 text-caption text-[#374151]">{description}</p>
          ) : null}
        </div>
        {actions}
      </div>
      <div style={{ height }} className="mt-4">
        {children}
      </div>
    </div>
  );
}
