// Core domain types shared across features.
// Kept independent of any backend/ML implementation — this is the UI's
// contract, not a mirror of internal model schemas.

export type RiskLevel = 'SAFE' | 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL';

export type AlertSeverity = 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';

export type AlertStatus = 'ACTIVE' | 'ACKNOWLEDGED' | 'RESOLVED';

export type ZoneStatus = 'NORMAL' | 'ELEVATED' | 'RESTRICTED' | 'OFFLINE';

export type FlowDirection = 'NORTH' | 'SOUTH' | 'EAST' | 'WEST';

export interface Zone {
  id: string;
  name: string;
  density: number; // people per m²
  occupancy: number; // percentage 0-100
  capacity: number;
  riskLevel: RiskLevel;
  status: ZoneStatus;
  updatedAt: string; // ISO timestamp
}

export interface TimeSeriesPoint {
  timestamp: string; // ISO timestamp
  value: number;
}

export interface RiskForecastPoint {
  horizon: 'CURRENT' | '+5min' | '+10min';
  timestamp: string;
  riskLevel: RiskLevel;
  riskScore: number; // 0-100
  confidence: number; // 0-100
}

export interface FeatureContribution {
  feature: string;
  contribution: number; // -100 to 100, signed impact
  description: string;
}

export interface OperationalEvent {
  id: string;
  timestamp: string;
  zone: string;
  type: string;
  description: string;
  severity: AlertSeverity;
}

export interface Alert {
  id: string;
  severity: AlertSeverity;
  timestamp: string;
  zone: string;
  triggerReason: string;
  status: AlertStatus;
}

export interface DashboardSummary {
  crowdStatus: RiskLevel;
  activePeopleCount: number;
  currentRiskLevel: RiskLevel;
  activeZoneCount: number;
  totalZoneCount: number;
  alertCount: number;
  predictionConfidence: number; // 0-100
}

export interface ZoneOccupancySlice {
  zoneName: string;
  occupancy: number;
}

export interface FlowDirectionSlice {
  direction: FlowDirection;
  percentage: number;
}

export type TimeRange = '1h' | '6h' | '24h' | 'custom';
