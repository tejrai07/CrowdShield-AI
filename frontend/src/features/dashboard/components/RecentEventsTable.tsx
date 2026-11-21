import type { ColumnDef } from '@tanstack/react-table';
import { format } from 'date-fns';
import { DataTable } from '../../../components/tables/DataTable';
import { StatusBadge } from '../../../components/ui/StatusBadge';
import { getAlertSeverityStyle } from '../../../utils/risk';
import { useRecentEvents } from '../hooks/useDashboardData';
import type { OperationalEvent } from '../../../types/domain';
import { ErrorState } from '../../../components/feedback/ErrorState';

const columns: ColumnDef<OperationalEvent, any>[] = [
  {
    accessorKey: 'timestamp',
    header: 'Time',
    cell: ({ getValue }) => (
      <span className="tabular-nums text-[#374151]">
        {format(new Date(getValue<string>()), 'HH:mm:ss')}
      </span>
    ),
  },
  {
    accessorKey: 'zone',
    header: 'Zone',
    cell: ({ getValue }) => <span className="font-medium">{getValue<string>()}</span>,
  },
  {
    accessorKey: 'type',
    header: 'Event Type',
  },
  {
    accessorKey: 'description',
    header: 'Description',
    cell: ({ getValue }) => <span className="text-[#374151]">{getValue<string>()}</span>,
  },
  {
    accessorKey: 'severity',
    header: 'Severity',
    cell: ({ getValue }) => {
      const severity = getValue<OperationalEvent['severity']>();
      const style = getAlertSeverityStyle(severity);
      return (
        <StatusBadge
          label={severity[0] + severity.slice(1).toLowerCase()}
          textClass={style.textClass}
          bgClass={style.bgClass}
        />
      );
    },
  },
];

export function RecentEventsTable() {
  const { data, isLoading, isError, refetch } = useRecentEvents();

  if (isError) {
    return <ErrorState onRetry={() => refetch()} />;
  }

  return (
    <DataTable
      data={data ?? []}
      columns={columns}
      isLoading={isLoading}
      emptyTitle="No recent events"
      emptyDescription="Operational events will appear here as they occur."
    />
  );
}
