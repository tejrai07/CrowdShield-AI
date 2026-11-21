import type { LucideIcon } from 'lucide-react';
import { PageContainer } from './PageContainer';
import { SectionHeader } from './SectionHeader';

interface PlaceholderPageProps {
  title: string;
  description: string;
  icon: LucideIcon;
}

// Used for routes wired up in Phase UI-1 but not yet fully implemented.
// Keeps navigation and layout intact without breaking on missing pages.
export function PlaceholderPage({ title, description, icon: Icon }: PlaceholderPageProps) {
  return (
    <PageContainer>
      <SectionHeader as="h1" title={title} description={description} />
      <div className="mt-8 flex flex-col items-center justify-center gap-3 rounded-lg border border-dashed border-[#E5E7EB] bg-white px-8 py-20 text-center">
        <span className="flex h-12 w-12 items-center justify-center rounded-full bg-[#2563EB]/10 text-[#2563EB]">
          <Icon size={22} aria-hidden="true" />
        </span>
        <p className="text-h4 font-semibold text-[#111827]">Coming in the next phase</p>
        <p className="max-w-md text-small text-[#374151]">
          This page is scaffolded and routed, but its full implementation lands in a later
          build phase.
        </p>
      </div>
    </PageContainer>
  );
}
