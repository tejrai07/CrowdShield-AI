import type { ReactNode } from 'react';
import { cn } from '../../utils/cn';

interface SectionHeaderProps {
  title: string;
  description?: string;
  actions?: ReactNode;
  className?: string;
  as?: 'h1' | 'h2' | 'h3';
}

export function SectionHeader({
  title,
  description,
  actions,
  className,
  as = 'h2',
}: SectionHeaderProps) {
  const Heading = as;
  const sizeClass =
    as === 'h1' ? 'text-h1 font-semibold' : as === 'h2' ? 'text-h2 font-semibold' : 'text-h3 font-semibold';

  return (
    <div className={cn('flex items-start justify-between gap-4', className)}>
      <div>
        <Heading className={cn(sizeClass, 'text-[#111827] tracking-tight')}>{title}</Heading>
        {description ? (
          <p className="mt-1 text-small text-[#374151]">{description}</p>
        ) : null}
      </div>
      {actions ? <div className="flex shrink-0 items-center gap-2">{actions}</div> : null}
    </div>
  );
}
