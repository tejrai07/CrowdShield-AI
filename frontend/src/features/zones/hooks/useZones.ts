import { useQuery } from '@tanstack/react-query';
import { zoneApi } from '../../../services/api/zoneApi';

const REFRESH_INTERVAL_MS = 20_000;

export function useZones() {
  return useQuery({
    queryKey: ['zones', 'list'],
    queryFn: zoneApi.getZones,
    refetchInterval: REFRESH_INTERVAL_MS,
  });
}
