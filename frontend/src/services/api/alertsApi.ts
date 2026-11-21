// Alerts API abstraction — the only place alert data is fetched from.
// Mock-backed today; swap the implementation when the real alerting
// backend is ready. Call sites never change.

import { getAlerts } from '../mock/alertsMock';
import { simulateLatency } from '../mock/generators';
import type { Alert } from '../../types/domain';

export const alertsApi = {
  getAlerts: (): Promise<Alert[]> => simulateLatency(getAlerts(), 400),
};
