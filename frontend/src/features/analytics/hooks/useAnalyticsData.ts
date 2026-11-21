import { useQuery } from '@tanstack/react-query';
import { analyticsApi } from '../../../services/api/analyticsApi';
import type { TimeRange } from '../../../types/domain';

const REFRESH_INTERVAL_MS = 30_000;

export function useDensityOverTime(range: TimeRange) {
  return useQuery({
    queryKey: ['analytics', 'density', range],
    queryFn: () => analyticsApi.getDensityOverTime(range),
    refetchInterval: REFRESH_INTERVAL_MS,
  });
}

export function useMovementSpeedDistribution(range: TimeRange) {
  return useQuery({
    queryKey: ['analytics', 'movement-speed', range],
    queryFn: () => analyticsApi.getMovementSpeedDistribution(range),
    refetchInterval: REFRESH_INTERVAL_MS,
  });
}

export function useDirectionVariance(range: TimeRange) {
  return useQuery({
    queryKey: ['analytics', 'direction-variance', range],
    queryFn: () => analyticsApi.getDirectionVariance(range),
    refetchInterval: REFRESH_INTERVAL_MS,
  });
}

export function useInflowOutflow(range: TimeRange) {
  return useQuery({
    queryKey: ['analytics', 'inflow-outflow', range],
    queryFn: async () => {
      const [inflow, outflow] = await Promise.all([
        analyticsApi.getInflowRate(range),
        analyticsApi.getOutflowRate(range),
      ]);
      return { inflow, outflow };
    },
    refetchInterval: REFRESH_INTERVAL_MS,
  });
}

export function useBottleneckPressure() {
  return useQuery({
    queryKey: ['analytics', 'bottleneck-pressure'],
    queryFn: analyticsApi.getBottleneckPressure,
    refetchInterval: REFRESH_INTERVAL_MS,
  });
}
