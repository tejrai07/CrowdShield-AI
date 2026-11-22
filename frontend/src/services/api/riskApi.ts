// Risk API abstraction. Connects to FastAPI backend via /api proxy with fallback to mock.

import { getFeatureContributions, getRiskForecast, getRiskTimeline } from '../mock/riskMock';
import { simulateLatency } from '../mock/generators';
import { fetchWithFallback } from './client';
import type { FeatureContribution, RiskForecastPoint, TimeSeriesPoint } from '../../types/domain';

export const riskApi = {
  getForecast: (): Promise<RiskForecastPoint[]> =>
    fetchWithFallback('/api/risk/forecast', () => simulateLatency(getRiskForecast(), 400)),

  getTimeline: (): Promise<TimeSeriesPoint[]> =>
    fetchWithFallback('/api/risk/timeline', () => simulateLatency(getRiskTimeline(), 450)),

  getFeatureContributions: (): Promise<FeatureContribution[]> =>
    fetchWithFallback('/api/risk/feature-contributions', () =>
      simulateLatency(getFeatureContributions(), 400),
    ),
};
