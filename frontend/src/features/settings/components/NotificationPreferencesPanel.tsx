import { useState } from 'react';
import { FormToggle } from '../../../components/forms/FormToggle';

interface NotificationPrefs {
  criticalAlerts: boolean;
  highAlerts: boolean;
  dailySummary: boolean;
  systemStatus: boolean;
}

export function NotificationPreferencesPanel() {
  const [prefs, setPrefs] = useState<NotificationPrefs>({
    criticalAlerts: true,
    highAlerts: true,
    dailySummary: false,
    systemStatus: true,
  });

  const update = (key: keyof NotificationPrefs) => (value: boolean) =>
    setPrefs((prev) => ({ ...prev, [key]: value }));

  return (
    <div className="divide-y divide-[#E5E7EB] rounded-lg border border-[#E5E7EB] px-4">
      <FormToggle
        label="Critical alerts"
        description="Immediate notification for critical-severity alerts."
        checked={prefs.criticalAlerts}
        onChange={update('criticalAlerts')}
      />
      <FormToggle
        label="High-severity alerts"
        description="Notification for high-severity alerts."
        checked={prefs.highAlerts}
        onChange={update('highAlerts')}
      />
      <FormToggle
        label="Daily summary"
        description="A daily digest of zone and risk activity."
        checked={prefs.dailySummary}
        onChange={update('dailySummary')}
      />
      <FormToggle
        label="System status changes"
        description="Sensor connectivity and system health notices."
        checked={prefs.systemStatus}
        onChange={update('systemStatus')}
      />
    </div>
  );
}
