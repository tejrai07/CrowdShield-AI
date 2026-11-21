import type { ColumnDef } from '@tanstack/react-table';
import { format } from 'date-fns';
import { DataTable } from '../../../components/tables/DataTable';
import { StatusBadge } from '../../../components/ui/StatusBadge';
import { getAlertSeverityStyle } from '../../../utils/risk';
import type { Alert } from '../../../types/domain';

interface AlertsTableProps {
  alerts: Alert[];
  isLoading?: boolean;
  emptyTitle: string;
  emptyDescription: string;
}

const STATUS_STYLE: Record<Alert['status'], { label: string; textClass: string; bgClass: string }> = {
  ACTIVE: { label: 'Active', textClass: 'text-[#DC2626]', bgClass: 'bg-[#DC2626]/10' },
  ACKNOWLEDGED: { label: 'Acknowledged', textClass: 'text-[#D97706]', bgClass: 'bg-[#D97706]/10' },
  RESOLVED: { label: 'Resolved', textClass: 'text-[#16A34A]', bgClass: 'bg-[#16A34A]/10' },
};

const columns: ColumnDef<Alert, any>[] = [
  {
    accessorKey: 'severity',
    header: 'Severity',
    cell: ({ getValue }) => {
      const severity = getValue<Alert['severity']>();
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
  {
    accessorKey: 'timestamp',
    header: 'Timestamp',
    cell: ({ getValue }) => (
      <span className="tabular-nums text-[#374151]">
        {format(new Date(getValue<string>()), 'MMM d, HH:mm')}
      </span>
    ),
  },
  {
    accessorKey: 'zone',
    header: 'Zone',
    cell: ({ getValue }) => <span className="font-medium">{getValue<string>()}</span>,
  },
  {
    accessorKey: 'triggerReason',
    header: 'Trigger Reason',
    cell: ({ getValue }) => <span className="text-[#374151]">{getValue<string>()}</span>,
  },
  {
    accessorKey: 'status',
    header: 'Status',
    cell: ({ getValue }) => {
      const style = STATUS_STYLE[getValue<Alert['status']>()];
      return <StatusBadge label={style.label} textClass={style.textClass} bgClass={style.bgClass} />;
    },
  },
];

export function AlertsTable({ alerts, isLoading, emptyTitle, emptyDescription }: AlertsTableProps) {
  return (
    <DataTable
      data={alerts}
      columns={columns}
      isLoading={isLoading}
      emptyTitle={emptyTitle}
      emptyDescription={emptyDescription}
    />
  );
}
