import { AlertTriangle, MapPin, PowerOff, ShieldAlert } from 'lucide-react';
import { StatCard } from '../../../components/ui/StatCard';
import { LoadingState } from '../../../components/feedback/LoadingState';
import { ErrorState } from '../../../components/feedback/ErrorState';
import { useZones } from '../hooks/useZones';

export function ZoneMetricsOverview() {
  const { data, isLoading, isError, refetch } = useZones();

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <LoadingState key={i} />
        ))}
      </div>
    );
  }

  if (isError || !data) {
    return <ErrorState onRetry={() => refetch()} />;
  }

  const elevatedOrWorse = data.filter((z) => z.status === 'ELEVATED' || z.status === 'RESTRICTED').length;
  const offline = data.filter((z) => z.status === 'OFFLINE').length;
  const avgOccupancy = Math.round(data.reduce((sum, z) => sum + z.occupancy, 0) / data.length);

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <StatCard
        label="Total Zones"
        value={data.length}
        icon={<MapPin size={18} aria-hidden="true" />}
      />
      <StatCard
        label="Average Occupancy"
        value={`${avgOccupancy}%`}
        icon={<ShieldAlert size={18} aria-hidden="true" />}
      />
      <StatCard
        label="Elevated / Restricted"
        value={elevatedOrWorse}
        icon={<AlertTriangle size={18} aria-hidden="true" />}
      />
      <StatCard
        label="Offline Zones"
        value={offline}
        icon={<PowerOff size={18} aria-hidden="true" />}
      />
    </div>
  );
}
