// Dashboard API abstraction.
// Connects to FastAPI backend via /api proxy, with automatic fallback to mock generators if offline.

import {
  getCrowdDensityTrend,
  getDashboardSummary,
  getFlowDirectionDistribution,
  getRecentEvents,
  getRiskLevelTrend,
  getZoneOccupancyDistribution,
} from '../mock/dashboardMock';
import { simulateLatency } from '../mock/generators';
import { fetchWithFallback } from './client';
import type {
  DashboardSummary,
  FlowDirectionSlice,
  OperationalEvent,
  TimeSeriesPoint,
  ZoneOccupancySlice,
} from '../../types/domain';

export const dashboardApi = {
  getSummary: (): Promise<DashboardSummary> =>
    fetchWithFallback('/api/dashboard/summary', () =>
      simulateLatency(getDashboardSummary(), 350),
    ),

  getCrowdDensityTrend: (): Promise<TimeSeriesPoint[]> =>
    fetchWithFallback('/api/dashboard/crowd-density-trend', () =>
      simulateLatency(getCrowdDensityTrend(), 450),
    ),

  getRiskLevelTrend: (): Promise<TimeSeriesPoint[]> =>
    fetchWithFallback('/api/dashboard/risk-level-trend', () =>
      simulateLatency(getRiskLevelTrend(), 450),
    ),

  getZoneOccupancyDistribution: (): Promise<ZoneOccupancySlice[]> =>
    fetchWithFallback('/api/dashboard/zone-occupancy-distribution', () =>
      simulateLatency(getZoneOccupancyDistribution(), 400),
    ),

  getFlowDirectionDistribution: (): Promise<FlowDirectionSlice[]> =>
    fetchWithFallback('/api/dashboard/flow-direction-distribution', () =>
      simulateLatency(getFlowDirectionDistribution(), 400),
    ),

  getRecentEvents: (): Promise<OperationalEvent[]> =>
    fetchWithFallback('/api/dashboard/recent-events', () =>
      simulateLatency(getRecentEvents(), 500),
    ),
};
