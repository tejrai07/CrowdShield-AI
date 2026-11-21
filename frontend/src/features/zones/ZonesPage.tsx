import { LayoutGrid, Table2 } from 'lucide-react';
import { useMemo, useState } from 'react';
import { PageContainer } from '../../components/ui/PageContainer';
import { SectionHeader } from '../../components/ui/SectionHeader';
import { SearchInput } from '../../components/ui/SearchInput';
import { FilterBar, FilterChip } from '../../components/ui/FilterBar';
import { LoadingState } from '../../components/feedback/LoadingState';
import { ErrorState } from '../../components/feedback/ErrorState';
import { EmptyState } from '../../components/feedback/EmptyState';
import { cn } from '../../utils/cn';
import type { Zone, ZoneStatus } from '../../types/domain';
import { useZones } from './hooks/useZones';
import { ZoneMetricsOverview } from './components/ZoneMetricsOverview';
import { ZoneCard } from './components/ZoneCard';
import { ZoneTable } from './components/ZoneTable';
import { ZoneDetailPanel } from './components/ZoneDetailPanel';

const STATUS_FILTERS: { label: string; value: ZoneStatus | 'ALL' }[] = [
  { label: 'All statuses', value: 'ALL' },
  { label: 'Normal', value: 'NORMAL' },
  { label: 'Elevated', value: 'ELEVATED' },
  { label: 'Restricted', value: 'RESTRICTED' },
  { label: 'Offline', value: 'OFFLINE' },
];

export function ZonesPage() {
  const { data, isLoading, isError, refetch } = useZones();
  const [view, setView] = useState<'cards' | 'table'>('cards');
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState<ZoneStatus | 'ALL'>('ALL');
  const [selectedZone, setSelectedZone] = useState<Zone | null>(null);

  const filteredZones = useMemo(() => {
    if (!data) return [];
    return data.filter((zone) => {
      const matchesSearch = zone.name.toLowerCase().includes(search.toLowerCase());
      const matchesStatus = statusFilter === 'ALL' || zone.status === statusFilter;
      return matchesSearch && matchesStatus;
    });
  }, [data, search, statusFilter]);

  return (
    <PageContainer>
      <SectionHeader
        as="h1"
        title="Zone Management"
        description="Venue spatial intelligence across all monitored zones."
      />

      <div className="mt-6">
        <ZoneMetricsOverview />
      </div>

      <div className="mt-6 flex flex-wrap items-center justify-between gap-3">
        <FilterBar
          search={
            <SearchInput
              placeholder="Search zones…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              containerClassName="w-64"
              aria-label="Search zones"
            />
          }
        >
          {STATUS_FILTERS.map((filter) => (
            <FilterChip
              key={filter.value}
              label={filter.label}
              active={statusFilter === filter.value}
              onClick={() => setStatusFilter(filter.value)}
            />
          ))}
        </FilterBar>

        <div className="flex items-center gap-1 rounded-md border border-[#E5E7EB] bg-white p-1">
          <button
            type="button"
            onClick={() => setView('cards')}
            aria-pressed={view === 'cards'}
            aria-label="Card view"
            className={cn(
              'flex h-7 w-7 items-center justify-center rounded focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#2563EB]',
              view === 'cards' ? 'bg-[#2563EB]/10 text-[#2563EB]' : 'text-[#374151]',
            )}
          >
            <LayoutGrid size={16} aria-hidden="true" />
          </button>
          <button
            type="button"
            onClick={() => setView('table')}
            aria-pressed={view === 'table'}
            aria-label="Table view"
            className={cn(
              'flex h-7 w-7 items-center justify-center rounded focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#2563EB]',
              view === 'table' ? 'bg-[#2563EB]/10 text-[#2563EB]' : 'text-[#374151]',
            )}
          >
            <Table2 size={16} aria-hidden="true" />
          </button>
        </div>
      </div>

      <div className="mt-4">
        {isLoading ? (
          view === 'cards' ? (
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {Array.from({ length: 8 }).map((_, i) => (
                <LoadingState key={i} />
              ))}
            </div>
          ) : (
            <LoadingState variant="table" rows={8} />
          )
        ) : isError ? (
          <ErrorState onRetry={() => refetch()} />
        ) : filteredZones.length === 0 ? (
          <EmptyState
            title="No zones match your filters"
            description="Try a different search term or clear the status filter."
          />
        ) : view === 'cards' ? (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {filteredZones.map((zone) => (
              <ZoneCard
                key={zone.id}
                zone={zone}
                onClick={() => setSelectedZone(zone)}
                isSelected={selectedZone?.id === zone.id}
              />
            ))}
          </div>
        ) : (
          <ZoneTable zones={filteredZones} onRowClick={setSelectedZone} />
        )}
      </div>

      <ZoneDetailPanel zone={selectedZone} onClose={() => setSelectedZone(null)} />
    </PageContainer>
  );
}
