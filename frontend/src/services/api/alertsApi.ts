// Alerts API abstraction. Connects to FastAPI backend via /api proxy with fallback to mock.

import { getAlerts } from '../mock/alertsMock';
import { simulateLatency } from '../mock/generators';
import { fetchWithFallback } from './client';
import type { Alert } from '../../types/domain';

export const alertsApi = {
  getAlerts: (): Promise<Alert[]> =>
    fetchWithFallback('/api/alerts', () => simulateLatency(getAlerts(), 400)),
};
