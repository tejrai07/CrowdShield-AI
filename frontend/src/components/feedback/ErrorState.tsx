import { AlertTriangle } from 'lucide-react';
import { cn } from '../../utils/cn';

interface ErrorStateProps {
  title?: string;
  description?: string;
  onRetry?: () => void;
  className?: string;
}

export function ErrorState({
  title = 'Unable to load data',
  description = 'The request failed. Check the connection and try again.',
  onRetry,
  className,
}: ErrorStateProps) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center gap-3 rounded-lg border border-[#E5E7EB] bg-white p-8 text-center',
        className,
      )}
      role="alert"
    >
      <span className="flex h-10 w-10 items-center justify-center rounded-full bg-[#DC2626]/10 text-[#DC2626]">
        <AlertTriangle size={20} aria-hidden="true" />
      </span>
      <div>
        <p className="text-h4 font-semibold text-[#111827]">{title}</p>
        <p className="mt-1 text-small text-[#374151]">{description}</p>
      </div>
      {onRetry ? (
        <button
          type="button"
          onClick={onRetry}
          className="mt-1 rounded-md bg-[#2563EB] px-4 py-2 text-small font-medium text-white hover:bg-[#2563EB]/90 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#2563EB]"
        >
          Retry
        </button>
      ) : null}
    </div>
  );
}
