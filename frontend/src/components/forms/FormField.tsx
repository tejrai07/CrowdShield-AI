import type { InputHTMLAttributes } from 'react';
import { useId } from 'react';
import { cn } from '../../utils/cn';

interface FormFieldProps extends InputHTMLAttributes<HTMLInputElement> {
  label: string;
  hint?: string;
  containerClassName?: string;
}

export function FormField({ label, hint, containerClassName, className, ...props }: FormFieldProps) {
  const id = useId();
  return (
    <div className={cn('flex flex-col gap-1.5', containerClassName)}>
      <label htmlFor={id} className="text-small font-medium text-[#111827]">
        {label}
      </label>
      <input
        id={id}
        className={cn(
          'h-9 rounded-md border border-[#E5E7EB] bg-white px-3 text-small text-[#111827]',
          'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#2563EB]',
          'disabled:bg-[#F8FAFC] disabled:text-[#374151]',
          className,
        )}
        {...props}
      />
      {hint ? <p className="text-caption text-[#374151]">{hint}</p> : null}
    </div>
  );
}
