import type { TimeRange, TimeSeriesPoint } from '../../types/domain';
import { isoMinutesAgo, randomBetween, randomInt } from './generators';

const RANGE_TO_POINTS: Record<Exclude<TimeRange, 'custom'>, { points: number; stepMin: number }> = {
  '1h': { points: 12, stepMin: 5 },
  '6h': { points: 24, stepMin: 15 },
  '24h': { points: 24, stepMin: 60 },
};

function buildSeries(range: TimeRange, base: number, volatility: number, min: number, max: number): TimeSeriesPoint[] {
  const { points, stepMin } = RANGE_TO_POINTS[range === 'custom' ? '24h' : range];
  let value = base;
  return Array.from({ length: points }, (_, i) => {
    value += randomBetween(-volatility, volatility);
    value = Math.max(min, Math.min(max, value));
    return {
      timestamp: isoMinutesAgo((points - i) * stepMin),
      value: Number(value.toFixed(2)),
    };
  });
}

export function getDensityOverTime(range: TimeRange): TimeSeriesPoint[] {
  return buildSeries(range, randomBetween(2, 3.5), 0.35, 0.3, 6.5);
}

export function getMovementSpeedDistribution(range: TimeRange): TimeSeriesPoint[] {
  return buildSeries(range, randomBetween(0.8, 1.3), 0.15, 0.1, 2.2);
}

export function getDirectionVariance(range: TimeRange): TimeSeriesPoint[] {
  return buildSeries(range, randomBetween(15, 30), 6, 0, 90);
}

export function getInflowRate(range: TimeRange): TimeSeriesPoint[] {
  return buildSeries(range, randomBetween(200, 500), 90, 0, 1200);
}

export function getOutflowRate(range: TimeRange): TimeSeriesPoint[] {
  return buildSeries(range, randomBetween(180, 480), 90, 0, 1200);
}

export interface BottleneckPoint {
  location: string;
  pressure: number; // 0-100
}

const BOTTLENECK_LOCATIONS = [
  'North Gate Turnstiles',
  'Main Concourse Choke',
  'Transit Hub A Stairwell',
  'South Gate Ramp',
  'Upper Tier East Exit',
  'VIP Concourse Entrance',
];

export function getBottleneckPressure(): BottleneckPoint[] {
  return BOTTLENECK_LOCATIONS.map((location) => ({
    location,
    pressure: randomInt(10, 95),
  })).sort((a, b) => b.pressure - a.pressure);
}
