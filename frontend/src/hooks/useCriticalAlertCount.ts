import { useQuery } from '@tanstack/react-query';
import { alertsApi } from '../services/api/alertsApi';

// Lives in the shared hooks/ folder (not a feature) because it's
// consumed by the app shell (Header), not a specific feature page.
// Feature pages should not be imported by layouts — this keeps the
// dependency direction one-way: layouts -> hooks -> services.
export function useCriticalAlertCount() {
  const { data } = useQuery({
    queryKey: ['alerts', 'list'],
    queryFn: alertsApi.getAlerts,
    refetchInterval: 15_000,
  });

  const count = data?.filter((a) => a.severity === 'CRITICAL' && a.status !== 'RESOLVED').length ?? 0;
  return count;
}
