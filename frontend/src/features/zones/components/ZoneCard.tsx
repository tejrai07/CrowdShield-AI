import { format } from 'date-fns';
import { RiskBadge } from '../../../components/ui/RiskBadge';
import { StatusBadge } from '../../../components/ui/StatusBadge';
import { getZoneStatusStyle } from '../../../utils/risk';
import { cn } from '../../../utils/cn';
import type { Zone } from '../../../types/domain';

interface ZoneCardProps {
  zone: Zone;
  onClick: () => void;
  isSelected?: boolean;
}

function occupancyBarColor(occupancy: number): string {
  if (occupancy >= 90) return 'bg-[#DC2626]';
  if (occupancy >= 70) return 'bg-[#D97706]';
  return 'bg-[#2563EB]';
}

export function ZoneCard({ zone, onClick, isSelected }: ZoneCardProps) {
  const statusStyle = getZoneStatusStyle(zone.status);

  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={isSelected}
      className={cn(
        'flex flex-col gap-3 rounded-lg border bg-white p-5 text-left transition-colors',
        'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#2563EB]',
        isSelected ? 'border-[#2563EB] ring-1 ring-[#2563EB]' : 'border-[#E5E7EB] hover:border-[#374151]/40',
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <h3 className="text-h4 font-semibold text-[#111827]">{zone.name}</h3>
        <RiskBadge level={zone.riskLevel} />
      </div>

      <div>
        <div className="flex items-center justify-between text-caption text-[#374151]">
          <span>Occupancy</span>
          <span className="font-medium text-[#111827]">{zone.occupancy}%</span>
        </div>
        <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-[#F8FAFC]">
          <div
            className={cn('h-full rounded-full', occupancyBarColor(zone.occupancy))}
            style={{ width: `${zone.occupancy}%` }}
          />
        </div>
      </div>

      <div className="flex items-center justify-between text-small">
        <span className="text-[#374151]">
          Density <span className="font-medium text-[#111827]">{zone.density} p/m²</span>
        </span>
        <StatusBadge label={statusStyle.label} textClass={statusStyle.textClass} bgClass={statusStyle.bgClass} />
      </div>

      <p className="text-caption text-[#374151]">
        Updated {format(new Date(zone.updatedAt), 'HH:mm:ss')}
      </p>
    </button>
  );
}
