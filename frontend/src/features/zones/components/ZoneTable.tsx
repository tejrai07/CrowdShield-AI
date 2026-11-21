import type { ColumnDef } from '@tanstack/react-table';
import { format } from 'date-fns';
import { DataTable } from '../../../components/tables/DataTable';
import { RiskBadge } from '../../../components/ui/RiskBadge';
import { StatusBadge } from '../../../components/ui/StatusBadge';
import { getZoneStatusStyle } from '../../../utils/risk';
import type { Zone } from '../../../types/domain';

interface ZoneTableProps {
  zones: Zone[];
  isLoading?: boolean;
  onRowClick: (zone: Zone) => void;
}

const columns: ColumnDef<Zone, any>[] = [
  {
    accessorKey: 'name',
    header: 'Zone Name',
    cell: ({ getValue }) => <span className="font-medium">{getValue<string>()}</span>,
  },
  {
    accessorKey: 'density',
    header: 'Density',
    cell: ({ getValue }) => <span className="tabular-nums">{getValue<number>()} p/m²</span>,
  },
  {
    accessorKey: 'occupancy',
    header: 'Occupancy %',
    cell: ({ getValue }) => <span className="tabular-nums font-medium">{getValue<number>()}%</span>,
  },
  {
    accessorKey: 'riskLevel',
    header: 'Risk Level',
    cell: ({ getValue }) => <RiskBadge level={getValue<Zone['riskLevel']>()} />,
  },
  {
    accessorKey: 'status',
    header: 'Status',
    cell: ({ getValue }) => {
      const style = getZoneStatusStyle(getValue<Zone['status']>());
      return <StatusBadge label={style.label} textClass={style.textClass} bgClass={style.bgClass} />;
    },
  },
  {
    accessorKey: 'updatedAt',
    header: 'Updated',
    cell: ({ getValue }) => (
      <span className="tabular-nums text-[#374151]">
        {format(new Date(getValue<string>()), 'HH:mm:ss')}
      </span>
    ),
  },
];

export function ZoneTable({ zones, isLoading, onRowClick }: ZoneTableProps) {
  return (
    <DataTable
      data={zones}
      columns={columns}
      isLoading={isLoading}
      emptyTitle="No zones match your filters"
      emptyDescription="Try clearing the search or filter selection."
      onRowClick={onRowClick}
    />
  );
}
