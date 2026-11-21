// Analytics API abstraction — the only place crowd analytics data is
// fetched from. Mock-backed today; swap the implementation when the
// real analytics service is ready. Call sites never change.

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
import type { TimeRange, TimeSeriesPoint } from '../../types/domain';

export const analyticsApi = {
  getDensityOverTime: (range: TimeRange): Promise<TimeSeriesPoint[]> =>
    simulateLatency(getDensityOverTime(range), 400),
  getMovementSpeedDistribution: (range: TimeRange): Promise<TimeSeriesPoint[]> =>
    simulateLatency(getMovementSpeedDistribution(range), 400),
  getDirectionVariance: (range: TimeRange): Promise<TimeSeriesPoint[]> =>
    simulateLatency(getDirectionVariance(range), 400),
  getInflowRate: (range: TimeRange): Promise<TimeSeriesPoint[]> =>
    simulateLatency(getInflowRate(range), 400),
  getOutflowRate: (range: TimeRange): Promise<TimeSeriesPoint[]> =>
    simulateLatency(getOutflowRate(range), 400),
  getBottleneckPressure: (): Promise<BottleneckPoint[]> =>
    simulateLatency(getBottleneckPressure(), 400),
};
