import { useQuery } from '@tanstack/react-query';
import { dashboardApi } from '../../../services/api/dashboardApi';

const REFRESH_INTERVAL_MS = 30_000;

export function useDashboardSummary() {
  return useQuery({
    queryKey: ['dashboard', 'summary'],
    queryFn: dashboardApi.getSummary,
    refetchInterval: REFRESH_INTERVAL_MS,
  });
}

export function useCrowdDensityTrend() {
  return useQuery({
    queryKey: ['dashboard', 'crowd-density-trend'],
    queryFn: dashboardApi.getCrowdDensityTrend,
    refetchInterval: REFRESH_INTERVAL_MS,
  });
}

export function useRiskLevelTrend() {
  return useQuery({
    queryKey: ['dashboard', 'risk-level-trend'],
    queryFn: dashboardApi.getRiskLevelTrend,
    refetchInterval: REFRESH_INTERVAL_MS,
  });
}

export function useZoneOccupancyDistribution() {
  return useQuery({
    queryKey: ['dashboard', 'zone-occupancy'],
    queryFn: dashboardApi.getZoneOccupancyDistribution,
    refetchInterval: REFRESH_INTERVAL_MS,
  });
}

export function useFlowDirectionDistribution() {
  return useQuery({
    queryKey: ['dashboard', 'flow-direction'],
    queryFn: dashboardApi.getFlowDirectionDistribution,
    refetchInterval: REFRESH_INTERVAL_MS,
  });
}

export function useRecentEvents() {
  return useQuery({
    queryKey: ['dashboard', 'recent-events'],
    queryFn: dashboardApi.getRecentEvents,
    refetchInterval: REFRESH_INTERVAL_MS,
  });
}
