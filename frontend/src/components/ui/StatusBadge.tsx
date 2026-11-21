import { cn } from '../../utils/cn';

interface StatusBadgeProps {
  label: string;
  textClass: string;
  bgClass: string;
  className?: string;
}

export function StatusBadge({ label, textClass, bgClass, className }: StatusBadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-md px-2 py-0.5 text-caption font-medium',
        textClass,
        bgClass,
        className,
      )}
    >
      {label}
    </span>
  );
}
