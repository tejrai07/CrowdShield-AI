import type { ReactNode } from 'react';
import { Inbox } from 'lucide-react';
import { cn } from '../../utils/cn';

interface EmptyStateProps {
  title?: string;
  description?: string;
  icon?: ReactNode;
  action?: ReactNode;
  className?: string;
}

export function EmptyState({
  title = 'Nothing here yet',
  description = 'No data is available for this view.',
  icon,
  action,
  className,
}: EmptyStateProps) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center gap-3 rounded-lg border border-dashed border-[#E5E7EB] bg-white p-8 text-center',
        className,
      )}
    >
      <span className="flex h-10 w-10 items-center justify-center rounded-full bg-[#F8FAFC] text-[#374151]">
        {icon ?? <Inbox size={20} aria-hidden="true" />}
      </span>
      <div>
        <p className="text-h4 font-semibold text-[#111827]">{title}</p>
        <p className="mt-1 text-small text-[#374151]">{description}</p>
      </div>
      {action}
    </div>
  );
}
