// Shared helpers for generating realistic, structured mock data.
// Isolated here so the mock layer can be deleted wholesale once real
// APIs are ready, with zero impact on UI components.

import type { RiskLevel } from '../../types/domain';

export function randomBetween(min: number, max: number): number {
  return Math.random() * (max - min) + min;
}

export function randomInt(min: number, max: number): number {
  return Math.floor(randomBetween(min, max + 1));
}

export function pick<T>(items: readonly T[]): T {
  return items[randomInt(0, items.length - 1)];
}

const RISK_LEVELS: readonly RiskLevel[] = ['SAFE', 'LOW', 'MODERATE', 'HIGH', 'CRITICAL'];

export function riskLevelFromScore(score: number): RiskLevel {
  if (score < 20) return 'SAFE';
  if (score < 40) return 'LOW';
  if (score < 65) return 'MODERATE';
  if (score < 85) return 'HIGH';
  return 'CRITICAL';
}

export function weightedRiskLevel(): RiskLevel {
  // Weighted so the demo dataset mostly looks calm, with occasional spikes.
  const roll = Math.random();
  if (roll < 0.5) return 'SAFE';
  if (roll < 0.75) return 'LOW';
  if (roll < 0.9) return 'MODERATE';
  if (roll < 0.98) return 'HIGH';
  return 'CRITICAL';
}

export { RISK_LEVELS };

export function isoMinutesAgo(minutes: number): string {
  return new Date(Date.now() - minutes * 60_000).toISOString();
}

export function isoMinutesFromNow(minutes: number): string {
  return new Date(Date.now() + minutes * 60_000).toISOString();
}

// Simulates realistic network latency so loading states are visible
// during development, matching how the real API will behave.
export function simulateLatency<T>(value: T, ms = 400): Promise<T> {
  return new Promise((resolve) => setTimeout(() => resolve(value), ms));
}
