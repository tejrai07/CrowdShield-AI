import type { RiskLevel } from '../../types/domain';
import { getRiskLevelStyle } from '../../utils/risk';
import { cn } from '../../utils/cn';

interface RiskBadgeProps {
  level: RiskLevel;
  className?: string;
}

export function RiskBadge({ level, className }: RiskBadgeProps) {
  const style = getRiskLevelStyle(level);
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-caption font-medium',
        style.textClass,
        style.bgClass,
        style.borderClass,
        className,
      )}
      role="status"
      aria-label={`Risk level: ${style.label}`}
    >
      <span className={cn('h-1.5 w-1.5 rounded-full', style.dotClass)} aria-hidden="true" />
      {style.label}
    </span>
  );
}
