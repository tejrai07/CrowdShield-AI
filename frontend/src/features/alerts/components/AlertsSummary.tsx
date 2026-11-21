import { AlertOctagon, CheckCircle2, Clock, ShieldAlert } from 'lucide-react';
import { StatCard } from '../../../components/ui/StatCard';
import { LoadingState } from '../../../components/feedback/LoadingState';
import { ErrorState } from '../../../components/feedback/ErrorState';
import { useAlerts } from '../hooks/useAlerts';

export function AlertsSummary() {
  const { data, isLoading, isError, refetch } = useAlerts();

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

  const active = data.filter((a) => a.status === 'ACTIVE').length;
  const acknowledged = data.filter((a) => a.status === 'ACKNOWLEDGED').length;
  const resolved = data.filter((a) => a.status === 'RESOLVED').length;
  const critical = data.filter((a) => a.severity === 'CRITICAL' && a.status !== 'RESOLVED').length;

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <StatCard label="Active Alerts" value={active} icon={<AlertOctagon size={18} aria-hidden="true" />} />
      <StatCard label="Critical (unresolved)" value={critical} icon={<ShieldAlert size={18} aria-hidden="true" />} />
      <StatCard label="Acknowledged" value={acknowledged} icon={<Clock size={18} aria-hidden="true" />} />
      <StatCard label="Resolved" value={resolved} icon={<CheckCircle2 size={18} aria-hidden="true" />} />
    </div>
  );
}
