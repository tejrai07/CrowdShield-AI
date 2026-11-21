import type { AlertSeverity, RiskLevel, ZoneStatus } from '../types/domain';

// Single source of truth for the risk classification system.
// Every page that renders a risk level MUST read from here — never
// hardcode a color or label for a risk level elsewhere.

interface RiskLevelStyle {
  label: string;
  textClass: string;
  bgClass: string;
  borderClass: string;
  dotClass: string;
}

const RISK_LEVEL_STYLES: Record<RiskLevel, RiskLevelStyle> = {
  SAFE: {
    label: 'Safe',
    textClass: 'text-[#16A34A]',
    bgClass: 'bg-[#16A34A]/10',
    borderClass: 'border-[#16A34A]/30',
    dotClass: 'bg-[#16A34A]',
  },
  LOW: {
    label: 'Low',
    textClass: 'text-[#16A34A]',
    bgClass: 'bg-[#16A34A]/10',
    borderClass: 'border-[#16A34A]/30',
    dotClass: 'bg-[#16A34A]',
  },
  MODERATE: {
    label: 'Moderate',
    textClass: 'text-[#D97706]',
    bgClass: 'bg-[#D97706]/10',
    borderClass: 'border-[#D97706]/30',
    dotClass: 'bg-[#D97706]',
  },
  HIGH: {
    label: 'High',
    textClass: 'text-[#DC2626]',
    bgClass: 'bg-[#DC2626]/10',
    borderClass: 'border-[#DC2626]/30',
    dotClass: 'bg-[#DC2626]',
  },
  CRITICAL: {
    label: 'Critical',
    textClass: 'text-white',
    bgClass: 'bg-[#DC2626]',
    borderClass: 'border-[#DC2626]',
    dotClass: 'bg-white',
  },
};

export function getRiskLevelStyle(level: RiskLevel): RiskLevelStyle {
  return RISK_LEVEL_STYLES[level];
}

const ALERT_SEVERITY_TO_RISK: Record<AlertSeverity, RiskLevel> = {
  CRITICAL: 'CRITICAL',
  HIGH: 'HIGH',
  MEDIUM: 'MODERATE',
  LOW: 'LOW',
};

export function getAlertSeverityStyle(severity: AlertSeverity): RiskLevelStyle {
  return RISK_LEVEL_STYLES[ALERT_SEVERITY_TO_RISK[severity]];
}

interface StatusStyle {
  label: string;
  textClass: string;
  bgClass: string;
}

const ZONE_STATUS_STYLES: Record<ZoneStatus, StatusStyle> = {
  NORMAL: { label: 'Normal', textClass: 'text-[#16A34A]', bgClass: 'bg-[#16A34A]/10' },
  ELEVATED: { label: 'Elevated', textClass: 'text-[#D97706]', bgClass: 'bg-[#D97706]/10' },
  RESTRICTED: { label: 'Restricted', textClass: 'text-[#DC2626]', bgClass: 'bg-[#DC2626]/10' },
  OFFLINE: { label: 'Offline', textClass: 'text-[#374151]', bgClass: 'bg-[#374151]/10' },
};

export function getZoneStatusStyle(status: ZoneStatus): StatusStyle {
  return ZONE_STATUS_STYLES[status];
}
