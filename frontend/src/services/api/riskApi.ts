// Risk API abstraction — the only place risk-monitoring data is fetched
// from. Mock-backed today; swap the implementation when the real
// prediction service is ready. Call sites never change.

import { getFeatureContributions, getRiskForecast, getRiskTimeline } from '../mock/riskMock';
import { simulateLatency } from '../mock/generators';
import type { FeatureContribution, RiskForecastPoint, TimeSeriesPoint } from '../../types/domain';

export const riskApi = {
  getForecast: (): Promise<RiskForecastPoint[]> => simulateLatency(getRiskForecast(), 400),
  getTimeline: (): Promise<TimeSeriesPoint[]> => simulateLatency(getRiskTimeline(), 450),
  getFeatureContributions: (): Promise<FeatureContribution[]> =>
    simulateLatency(getFeatureContributions(), 400),
};
