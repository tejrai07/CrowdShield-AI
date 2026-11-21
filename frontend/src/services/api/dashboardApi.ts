// Dashboard API abstraction.
//
// This module is the ONLY place dashboard data is fetched from.
// Components and hooks must never call the mock layer or an HTTP
// client directly — they call these functions. When the real backend
// is ready, swap the implementation below; call sites do not change.

import {
  getCrowdDensityTrend,
  getDashboardSummary,
  getFlowDirectionDistribution,
  getRecentEvents,
  getRiskLevelTrend,
  getZoneOccupancyDistribution,
} from '../mock/dashboardMock';
import { simulateLatency } from '../mock/generators';
import type {
  DashboardSummary,
  FlowDirectionSlice,
  OperationalEvent,
  TimeSeriesPoint,
  ZoneOccupancySlice,
} from '../../types/domain';

export const dashboardApi = {
  getSummary: (): Promise<DashboardSummary> => simulateLatency(getDashboardSummary(), 350),

  getCrowdDensityTrend: (): Promise<TimeSeriesPoint[]> =>
    simulateLatency(getCrowdDensityTrend(), 450),

  getRiskLevelTrend: (): Promise<TimeSeriesPoint[]> =>
    simulateLatency(getRiskLevelTrend(), 450),

  getZoneOccupancyDistribution: (): Promise<ZoneOccupancySlice[]> =>
    simulateLatency(getZoneOccupancyDistribution(), 400),

  getFlowDirectionDistribution: (): Promise<FlowDirectionSlice[]> =>
    simulateLatency(getFlowDirectionDistribution(), 400),

  getRecentEvents: (): Promise<OperationalEvent[]> =>
    simulateLatency(getRecentEvents(), 500),
};
