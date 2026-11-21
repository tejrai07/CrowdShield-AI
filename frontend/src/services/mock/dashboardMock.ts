import type {
  DashboardSummary,
  FlowDirectionSlice,
  OperationalEvent,
  TimeSeriesPoint,
  ZoneOccupancySlice,
} from '../../types/domain';
import { isoMinutesAgo, pick, randomBetween, randomInt, weightedRiskLevel } from './generators';

const ZONE_NAMES = [
  'Main Concourse',
  'North Gate',
  'South Gate',
  'Upper Tier — East',
  'Upper Tier — West',
  'Transit Hub A',
  'Transit Hub B',
  'VIP Concourse',
];

const EVENT_TYPES = [
  'Density threshold crossed',
  'Zone status changed',
  'Flow reversal detected',
  'Bottleneck forming',
  'Alert acknowledged',
  'Alert resolved',
  'Sensor reconnected',
  'Capacity updated',
];

export function getDashboardSummary(): DashboardSummary {
  const totalZoneCount = 8;
  return {
    crowdStatus: weightedRiskLevel(),
    activePeopleCount: randomInt(12_000, 48_000),
    currentRiskLevel: weightedRiskLevel(),
    activeZoneCount: randomInt(5, totalZoneCount),
    totalZoneCount,
    alertCount: randomInt(0, 9),
    predictionConfidence: randomInt(78, 98),
  };
}

export function getCrowdDensityTrend(points = 24): TimeSeriesPoint[] {
  let base = randomBetween(2, 3.5);
  return Array.from({ length: points }, (_, i) => {
    base += randomBetween(-0.3, 0.35);
    base = Math.max(0.5, Math.min(6.5, base));
    return {
      timestamp: isoMinutesAgo((points - i) * 5),
      value: Number(base.toFixed(2)),
    };
  });
}

export function getRiskLevelTrend(points = 24): TimeSeriesPoint[] {
  let base = randomBetween(15, 35);
  return Array.from({ length: points }, (_, i) => {
    base += randomBetween(-6, 7);
    base = Math.max(0, Math.min(100, base));
    return {
      timestamp: isoMinutesAgo((points - i) * 5),
      value: Number(base.toFixed(0)),
    };
  });
}

export function getZoneOccupancyDistribution(): ZoneOccupancySlice[] {
  return ZONE_NAMES.slice(0, 6).map((zoneName) => ({
    zoneName,
    occupancy: randomInt(15, 98),
  }));
}

export function getFlowDirectionDistribution(): FlowDirectionSlice[] {
  const raw = {
    NORTH: randomBetween(10, 40),
    SOUTH: randomBetween(10, 40),
    EAST: randomBetween(10, 40),
    WEST: randomBetween(10, 40),
  } as const;
  const total = Object.values(raw).reduce((a, b) => a + b, 0);
  return (Object.keys(raw) as (keyof typeof raw)[]).map((direction) => ({
    direction,
    percentage: Number(((raw[direction] / total) * 100).toFixed(1)),
  }));
}

export function getRecentEvents(count = 12): OperationalEvent[] {
  return Array.from({ length: count }, (_, i) => ({
    id: `evt-${i}-${Date.now()}`,
    timestamp: isoMinutesAgo(randomInt(0, 180)),
    zone: pick(ZONE_NAMES),
    type: pick(EVENT_TYPES),
    description: `${pick(EVENT_TYPES)} in ${pick(ZONE_NAMES)}`,
    severity: pick(['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'] as const),
  })).sort((a, b) => (a.timestamp < b.timestamp ? 1 : -1));
}

export { ZONE_NAMES };
