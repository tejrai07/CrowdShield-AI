import type { CSSProperties } from 'react';
import { cn } from '../../utils/cn';

interface LoadingStateProps {
  rows?: number;
  className?: string;
  variant?: 'card' | 'table' | 'chart';
}

function SkeletonBlock({
  className,
  style,
}: {
  className?: string;
  style?: CSSProperties;
}) {
  return <div className={cn('animate-pulse rounded-md bg-[#E5E7EB]', className)} style={style} />;
}

export function LoadingState({ rows = 3, className, variant = 'card' }: LoadingStateProps) {
  if (variant === 'chart') {
    return (
      <div className={cn('flex h-full w-full items-end gap-2 px-2', className)} role="status" aria-label="Loading chart">
        {Array.from({ length: 12 }).map((_, i) => (
          <SkeletonBlock key={i} className="flex-1" style={{ height: `${30 + ((i * 17) % 60)}%` }} />
        ))}
      </div>
    );
  }

  if (variant === 'table') {
    return (
      <div className={cn('space-y-2', className)} role="status" aria-label="Loading table">
        {Array.from({ length: rows }).map((_, i) => (
          <SkeletonBlock key={i} className="h-10 w-full" />
        ))}
      </div>
    );
  }

  return (
    <div
      className={cn(
        'space-y-3 rounded-lg border border-[#E5E7EB] bg-white p-5',
        className,
      )}
      role="status"
      aria-label="Loading"
    >
      <SkeletonBlock className="h-4 w-1/3" />
      <SkeletonBlock className="h-8 w-1/2" />
      <SkeletonBlock className="h-3 w-2/3" />
    </div>
  );
}
