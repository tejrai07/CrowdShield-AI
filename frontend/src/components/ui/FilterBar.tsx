import type { PropsWithChildren, ReactNode } from 'react';
import { cn } from '../../utils/cn';

interface FilterBarProps {
  search?: ReactNode;
  className?: string;
}

export function FilterBar({ search, className, children }: PropsWithChildren<FilterBarProps>) {
  return (
    <div className={cn('flex flex-wrap items-center gap-3', className)}>
      {search}
      <div className="flex flex-wrap items-center gap-2">{children}</div>
    </div>
  );
}

interface FilterChipProps {
  label: string;
  active?: boolean;
  onClick?: () => void;
}

export function FilterChip({ label, active, onClick }: FilterChipProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={cn(
        'rounded-md border px-3 py-1.5 text-small font-medium transition-colors',
        'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#2563EB]',
        active
          ? 'border-[#2563EB] bg-[#2563EB]/10 text-[#2563EB]'
          : 'border-[#E5E7EB] bg-white text-[#374151] hover:border-[#374151]/40',
      )}
    >
      {label}
    </button>
  );
}
