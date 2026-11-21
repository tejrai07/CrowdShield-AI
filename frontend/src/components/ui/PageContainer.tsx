import type { PropsWithChildren } from 'react';
import { cn } from '../../utils/cn';

interface PageContainerProps {
  className?: string;
}

export function PageContainer({ children, className }: PropsWithChildren<PageContainerProps>) {
  return (
    <div className={cn('mx-auto w-full max-w-[1600px] px-8 py-8', className)}>
      {children}
    </div>
  );
}
