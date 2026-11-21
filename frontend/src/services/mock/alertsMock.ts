import type { Alert, AlertSeverity, AlertStatus } from '../../types/domain';
import { isoMinutesAgo, pick, randomInt } from './generators';

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

const TRIGGER_REASONS = [
  'Density exceeded safe threshold',
  'Predicted risk crossed critical band',
  'Bottleneck pressure sustained > 5 minutes',
  'Flow reversal detected at exit',
  'Sensor coverage gap in zone',
  'Occupancy exceeded 95% of capacity',
  'Rapid inflow spike detected',
  'Model confidence dropped below threshold',
];

const SEVERITY_POOL: AlertSeverity[] = ['CRITICAL', 'CRITICAL', 'HIGH', 'HIGH', 'MEDIUM', 'MEDIUM', 'MEDIUM', 'LOW'];

let cachedAlerts: Alert[] | null = null;
let nextAlertSeq = 1;

function seedAlerts(): Alert[] {
  return Array.from({ length: 22 }, (_, i) => {
    nextAlertSeq = i + 2;
    const isResolved = i % 3 === 0;
    const status: AlertStatus = isResolved ? 'RESOLVED' : i % 5 === 0 ? 'ACKNOWLEDGED' : 'ACTIVE';
    return {
      id: `alert-${i + 1}`,
      severity: pick(SEVERITY_POOL),
      timestamp: isoMinutesAgo(randomInt(1, 720)),
      zone: pick(ZONE_NAMES),
      triggerReason: pick(TRIGGER_REASONS),
      status,
    };
  }).sort((a, b) => (a.timestamp < b.timestamp ? 1 : -1));
}

// Cached in-module so alert identity is stable across refetches, with
// status able to change over time (active -> acknowledged -> resolved)
// the way a real alerting backend would report it. Each call also has
// a small chance of appending a brand-new alert, simulating a live feed
// so the notification bell and toast watcher have something to react to.
export function getAlerts(): Alert[] {
  if (!cachedAlerts) {
    cachedAlerts = seedAlerts();
  }

  if (Math.random() < 0.15) {
    const newAlert: Alert = {
      id: `alert-${nextAlertSeq++}`,
      severity: pick(SEVERITY_POOL),
      timestamp: isoMinutesAgo(0),
      zone: pick(ZONE_NAMES),
      triggerReason: pick(TRIGGER_REASONS),
      status: 'ACTIVE',
    };
    cachedAlerts = [newAlert, ...cachedAlerts];
  }

  return cachedAlerts;
}

