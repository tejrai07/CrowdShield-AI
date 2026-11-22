// Zone API abstraction. Connects to FastAPI backend via /api proxy with fallback to mock.

import { getZoneById, getZones } from '../mock/zonesMock';
import { simulateLatency } from '../mock/generators';
import { fetchWithFallback } from './client';
import type { Zone } from '../../types/domain';

export const zoneApi = {
  getZones: (): Promise<Zone[]> =>
    fetchWithFallback('/api/zones', () => simulateLatency(getZones(), 400)),

  getZoneById: (id: string): Promise<Zone | undefined> =>
    fetchWithFallback(`/api/zones/${id}`, () => simulateLatency(getZoneById(id), 300)),
};
