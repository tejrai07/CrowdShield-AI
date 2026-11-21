import { useMemo, useState } from 'react';
import { PageContainer } from '../../components/ui/PageContainer';
import { SectionHeader } from '../../components/ui/SectionHeader';
import { SearchInput } from '../../components/ui/SearchInput';
import { FilterBar, FilterChip } from '../../components/ui/FilterBar';
import { ErrorState } from '../../components/feedback/ErrorState';
import type { AlertSeverity } from '../../types/domain';
import { useAlerts } from './hooks/useAlerts';
import { AlertsSummary } from './components/AlertsSummary';
import { AlertsTable } from './components/AlertsTable';

const SEVERITY_FILTERS: { label: string; value: AlertSeverity | 'ALL' }[] = [
  { label: 'All severities', value: 'ALL' },
  { label: 'Critical', value: 'CRITICAL' },
  { label: 'High', value: 'HIGH' },
  { label: 'Medium', value: 'MEDIUM' },
  { label: 'Low', value: 'LOW' },
];

export function AlertsPage() {
  const { data, isLoading, isError, refetch } = useAlerts();
  const [search, setSearch] = useState('');
  const [severityFilter, setSeverityFilter] = useState<AlertSeverity | 'ALL'>('ALL');

  const filtered = useMemo(() => {
    if (!data) return [];
    return data.filter((alert) => {
      const matchesSearch =
        alert.zone.toLowerCase().includes(search.toLowerCase()) ||
        alert.triggerReason.toLowerCase().includes(search.toLowerCase());
      const matchesSeverity = severityFilter === 'ALL' || alert.severity === severityFilter;
      return matchesSearch && matchesSeverity;
    });
  }, [data, search, severityFilter]);

  const activeAlerts = filtered.filter((a) => a.status === 'ACTIVE' || a.status === 'ACKNOWLEDGED');
  const resolvedAlerts = filtered.filter((a) => a.status === 'RESOLVED');

  return (
    <PageContainer>
      <SectionHeader
        as="h1"
        title="Alerts Center"
        description="Operational alert management across all monitored zones."
      />

      <div className="mt-6">
        <AlertsSummary />
      </div>

      <div className="mt-6">
        <FilterBar
          search={
            <SearchInput
              placeholder="Search by zone or trigger reason…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              containerClassName="w-80"
              aria-label="Search alerts"
            />
          }
        >
          {SEVERITY_FILTERS.map((filter) => (
            <FilterChip
              key={filter.value}
              label={filter.label}
              active={severityFilter === filter.value}
              onClick={() => setSeverityFilter(filter.value)}
            />
          ))}
        </FilterBar>
      </div>

      {isError ? (
        <div className="mt-6">
          <ErrorState onRetry={() => refetch()} />
        </div>
      ) : (
        <>
          <div className="mt-6">
            <SectionHeader as="h3" title="Active Alerts" className="mb-3" />
            <AlertsTable
              alerts={activeAlerts}
              isLoading={isLoading}
              emptyTitle="No active alerts"
              emptyDescription="Active and acknowledged alerts will appear here."
            />
          </div>

          <div className="mt-6">
            <SectionHeader as="h3" title="Resolved Alerts" className="mb-3" />
            <AlertsTable
              alerts={resolvedAlerts}
              isLoading={isLoading}
              emptyTitle="No resolved alerts"
              emptyDescription="Resolved alerts will appear here once closed out."
            />
          </div>
        </>
      )}
    </PageContainer>
  );
}
