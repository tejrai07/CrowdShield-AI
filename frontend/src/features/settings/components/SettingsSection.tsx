import type { PropsWithChildren } from 'react';

interface SettingsSectionProps {
  title: string;
  description?: string;
}

export function SettingsSection({ title, description, children }: PropsWithChildren<SettingsSectionProps>) {
  return (
    <section className="grid grid-cols-1 gap-6 border-b border-[#E5E7EB] py-8 first:pt-0 last:border-0 lg:grid-cols-3">
      <div className="lg:col-span-1">
        <h2 className="text-h4 font-semibold text-[#111827]">{title}</h2>
        {description ? <p className="mt-1 text-small text-[#374151]">{description}</p> : null}
      </div>
      <div className="lg:col-span-2">{children}</div>
    </section>
  );
}
