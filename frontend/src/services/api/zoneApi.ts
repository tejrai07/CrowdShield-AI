// Zone API abstraction. Same contract rule as dashboardApi.ts: this is
// the only place zone data is fetched from. Swap the implementation
// when the real backend is ready — call sites never change.

import { getZoneById, getZones } from '../mock/zonesMock';
import { simulateLatency } from '../mock/generators';
import type { Zone } from '../../types/domain';

export const zoneApi = {
  getZones: (): Promise<Zone[]> => simulateLatency(getZones(), 400),
  getZoneById: (id: string): Promise<Zone | undefined> => simulateLatency(getZoneById(id), 300),
};
