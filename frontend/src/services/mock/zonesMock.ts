import type { Zone } from '../../types/domain';
import { pick, randomBetween, randomInt, weightedRiskLevel, isoMinutesAgo } from './generators';

const ZONE_DEFS: { name: string; capacity: number }[] = [
  { name: 'Main Concourse', capacity: 12000 },
  { name: 'North Gate', capacity: 4000 },
  { name: 'South Gate', capacity: 4000 },
  { name: 'Upper Tier — East', capacity: 6000 },
  { name: 'Upper Tier — West', capacity: 6000 },
  { name: 'Transit Hub A', capacity: 5000 },
  { name: 'Transit Hub B', capacity: 5000 },
  { name: 'VIP Concourse', capacity: 1500 },
];

const STATUS_POOL = ['NORMAL', 'NORMAL', 'NORMAL', 'ELEVATED', 'RESTRICTED', 'OFFLINE'] as const;

let cachedZones: Zone[] | null = null;

// Zones are cached in-module so a zone's identity stays stable across
// refetches within a session, while its metrics still drift — matching
// how a real backend would behave (same zone, updated readings).
export function getZones(): Zone[] {
  if (!cachedZones) {
    cachedZones = ZONE_DEFS.map((def, i) => ({
      id: `zone-${i + 1}`,
      name: def.name,
      capacity: def.capacity,
      density: Number(randomBetween(0.5, 5.5).toFixed(2)),
      occupancy: randomInt(10, 98),
      riskLevel: weightedRiskLevel(),
      status: pick(STATUS_POOL),
      updatedAt: isoMinutesAgo(randomInt(0, 4)),
    }));
  }

  // Drift metrics slightly on each call to simulate a live feed.
  cachedZones = cachedZones.map((zone) => {
    const occupancyDelta = randomInt(-4, 4);
    const nextOccupancy = Math.max(5, Math.min(100, zone.occupancy + occupancyDelta));
    return {
      ...zone,
      occupancy: nextOccupancy,
      density: Number(Math.max(0.2, zone.density + randomBetween(-0.2, 0.2)).toFixed(2)),
      updatedAt: isoMinutesAgo(randomInt(0, 2)),
    };
  });

  return cachedZones;
}

export function getZoneById(id: string): Zone | undefined {
  return getZones().find((zone) => zone.id === id);
}
