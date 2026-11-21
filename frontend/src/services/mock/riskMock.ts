import type { FeatureContribution, RiskForecastPoint, TimeSeriesPoint } from '../../types/domain';
import { isoMinutesAgo, isoMinutesFromNow, randomBetween, randomInt, riskLevelFromScore } from './generators';

const FEATURES: { feature: string; description: string }[] = [
  {
    feature: 'Density growth rate',
    description: 'Rate of change in crowd density over the last 5 minutes.',
  },
  {
    feature: 'Inflow / outflow imbalance',
    description: 'Difference between people entering and leaving monitored zones.',
  },
  {
    feature: 'Bottleneck pressure',
    description: 'Congestion buildup at known chokepoints and gates.',
  },
  {
    feature: 'Movement speed variance',
    description: 'Spread of individual movement speeds — a proxy for crowd stress.',
  },
  {
    feature: 'Historical event pattern',
    description: 'Similarity to past incidents at this venue and time of day.',
  },
  {
    feature: 'Weather / visibility conditions',
    description: 'Environmental factors known to correlate with crowd behavior.',
  },
];

export function getRiskForecast(): RiskForecastPoint[] {
  const currentScore = randomInt(15, 70);
  const horizons: { horizon: RiskForecastPoint['horizon']; offset: number; drift: number }[] = [
    { horizon: 'CURRENT', offset: 0, drift: 0 },
    { horizon: '+5min', offset: 5, drift: randomBetween(-10, 15) },
    { horizon: '+10min', offset: 10, drift: randomBetween(-15, 25) },
  ];

  return horizons.map(({ horizon, offset, drift }) => {
    const score = Math.max(0, Math.min(100, Math.round(currentScore + drift)));
    return {
      horizon,
      timestamp: offset === 0 ? isoMinutesAgo(0) : isoMinutesFromNow(offset),
      riskLevel: riskLevelFromScore(score),
      riskScore: score,
      confidence: randomInt(72, 97) - offset, // confidence decays with horizon
    };
  });
}

export function getRiskTimeline(points = 36): TimeSeriesPoint[] {
  let base = randomBetween(20, 40);
  return Array.from({ length: points }, (_, i) => {
    base += randomBetween(-7, 8);
    base = Math.max(0, Math.min(100, base));
    return {
      timestamp: isoMinutesAgo((points - i) * 5),
      value: Number(base.toFixed(0)),
    };
  });
}

export function getFeatureContributions(): FeatureContribution[] {
  return FEATURES.map((f) => ({
    ...f,
    contribution: Number(randomBetween(-35, 45).toFixed(0)),
  })).sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
}
