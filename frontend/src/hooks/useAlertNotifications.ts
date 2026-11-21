import { useEffect, useRef } from 'react';
import { toast } from 'sonner';
import { useQuery } from '@tanstack/react-query';
import { alertsApi } from '../services/api/alertsApi';
import type { Alert } from '../types/domain';

// Mounted once at the app root. Watches the alerts feed and surfaces a
// toast the moment a new critical or high-severity alert appears —
// this is the "real-time ready" behavior the Alerts Center spec calls
// for, made visible outside of that page too.
export function useAlertNotifications() {
  const { data } = useQuery({
    queryKey: ['alerts', 'list'],
    queryFn: alertsApi.getAlerts,
    refetchInterval: 15_000,
  });

  const knownIds = useRef<Set<string> | null>(null);

  useEffect(() => {
    if (!data) return;

    if (knownIds.current === null) {
      // First load: record existing alerts without notifying, so the
      // operator isn't spammed with toasts for pre-existing incidents.
      knownIds.current = new Set(data.map((a) => a.id));
      return;
    }

    const newOnes = data.filter((a) => !knownIds.current!.has(a.id));
    newOnes.forEach((alert: Alert) => {
      knownIds.current!.add(alert.id);
      if (alert.severity === 'CRITICAL') {
        toast.error(`Critical alert — ${alert.zone}`, { description: alert.triggerReason });
      } else if (alert.severity === 'HIGH') {
        toast.warning(`High-severity alert — ${alert.zone}`, { description: alert.triggerReason });
      }
    });

    data.forEach((a) => knownIds.current!.add(a.id));
  }, [data]);
}
