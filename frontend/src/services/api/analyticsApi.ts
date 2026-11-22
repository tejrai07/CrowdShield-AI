// Analytics API abstraction. Connects to FastAPI backend via /api proxy with fallback to mock.

import {
  getBottleneckPressure,
  getDensityOverTime,
  getDirectionVariance,
  getInflowRate,
  getMovementSpeedDistribution,
  getOutflowRate,
  type BottleneckPoint,
} from '../mock/analyticsMock';
import { simulateLatency } from '../mock/generators';
import { fetchWithFallback } from './client';
import type { TimeRange, TimeSeriesPoint } from '../../types/domain';

export const analyticsApi = {
  getDensityOverTime: (range: TimeRange): Promise<TimeSeriesPoint[]> =>
    fetchWithFallback(`/api/analytics/density-over-time?range=${range}`, () =>
      simulateLatency(getDensityOverTime(range), 400),
    ),

  getMovementSpeedDistribution: (range: TimeRange): Promise<TimeSeriesPoint[]> =>
    fetchWithFallback(`/api/analytics/movement-speed-distribution?range=${range}`, () =>
      simulateLatency(getMovementSpeedDistribution(range), 400),
    ),

  getDirectionVariance: (range: TimeRange): Promise<TimeSeriesPoint[]> =>
    fetchWithFallback(`/api/analytics/direction-variance?range=${range}`, () =>
      simulateLatency(getDirectionVariance(range), 400),
    ),

  getInflowRate: (range: TimeRange): Promise<TimeSeriesPoint[]> =>
    fetchWithFallback(`/api/analytics/inflow-rate?range=${range}`, () =>
      simulateLatency(getInflowRate(range), 400),
    ),

  getOutflowRate: (range: TimeRange): Promise<TimeSeriesPoint[]> =>
    fetchWithFallback(`/api/analytics/outflow-rate?range=${range}`, () =>
      simulateLatency(getOutflowRate(range), 400),
    ),

  getBottleneckPressure: (): Promise<BottleneckPoint[]> =>
    fetchWithFallback('/api/analytics/bottleneck-pressure', () =>
      simulateLatency(getBottleneckPressure(), 400),
    ),
};
