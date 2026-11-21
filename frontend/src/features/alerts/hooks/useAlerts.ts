import { useQuery } from '@tanstack/react-query';
import { alertsApi } from '../../../services/api/alertsApi';

const REFRESH_INTERVAL_MS = 15_000;

// Short refetch interval — this page is "real-time ready" per the spec:
// active incidents should feel current without a manual refresh.
export function useAlerts() {
  return useQuery({
    queryKey: ['alerts', 'list'],
    queryFn: alertsApi.getAlerts,
    refetchInterval: REFRESH_INTERVAL_MS,
  });
}
