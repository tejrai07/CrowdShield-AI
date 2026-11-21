import { Search } from 'lucide-react';
import type { InputHTMLAttributes } from 'react';
import { cn } from '../../utils/cn';

interface SearchInputProps extends InputHTMLAttributes<HTMLInputElement> {
  containerClassName?: string;
}

export function SearchInput({ containerClassName, className, ...props }: SearchInputProps) {
  return (
    <div className={cn('relative', containerClassName)}>
      <Search
        size={16}
        className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-[#374151]"
        aria-hidden="true"
      />
      <input
        type="search"
        className={cn(
          'h-9 w-full rounded-md border border-[#E5E7EB] bg-white pl-9 pr-3 text-small text-[#111827] placeholder:text-[#374151]/60',
          'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#2563EB]',
          className,
        )}
        {...props}
      />
    </div>
  );
}
