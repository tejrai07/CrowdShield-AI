import { X } from 'lucide-react';
import { format } from 'date-fns';
import { AnimatePresence, motion } from 'framer-motion';
import { RiskBadge } from '../../../components/ui/RiskBadge';
import { StatusBadge } from '../../../components/ui/StatusBadge';
import { getZoneStatusStyle } from '../../../utils/risk';
import type { Zone } from '../../../types/domain';

interface ZoneDetailPanelProps {
  zone: Zone | null;
  onClose: () => void;
}

function occupancyBarColor(occupancy: number): string {
  if (occupancy >= 90) return 'bg-[#DC2626]';
  if (occupancy >= 70) return 'bg-[#D97706]';
  return 'bg-[#2563EB]';
}

export function ZoneDetailPanel({ zone, onClose }: ZoneDetailPanelProps) {
  const isOpen = zone !== null;

  return (
    <AnimatePresence>
      {isOpen && zone ? (
        <>
          <motion.div
            className="fixed inset-0 z-30 bg-[#111827]/20"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            onClick={onClose}
            aria-hidden="true"
          />

          <motion.aside
            className="fixed inset-y-0 right-0 z-40 w-full max-w-sm border-l border-[#E5E7EB] bg-white shadow-xl"
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ duration: 0.2, ease: 'easeOut' }}
            role="dialog"
            aria-label="Zone detail"
          >
            <div className="flex h-full flex-col">
              <div className="flex items-center justify-between border-b border-[#E5E7EB] px-6 py-5">
                <h2 className="text-h4 font-semibold text-[#111827]">{zone.name}</h2>
                <button
                  type="button"
                  onClick={onClose}
                  aria-label="Close zone detail"
                  className="flex h-8 w-8 items-center justify-center rounded-md text-[#374151] hover:bg-[#F8FAFC] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#2563EB]"
                >
                  <X size={18} aria-hidden="true" />
                </button>
              </div>

              <div className="flex-1 space-y-6 overflow-y-auto px-6 py-6">
                <div className="flex items-center gap-2">
                  <RiskBadge level={zone.riskLevel} />
                  <StatusBadge
                    label={getZoneStatusStyle(zone.status).label}
                    textClass={getZoneStatusStyle(zone.status).textClass}
                    bgClass={getZoneStatusStyle(zone.status).bgClass}
                  />
                </div>

                <div>
                  <div className="flex items-center justify-between text-small text-[#374151]">
                    <span>Occupancy</span>
                    <span className="font-medium text-[#111827]">
                      {zone.occupancy}% of {zone.capacity.toLocaleString()}
                    </span>
                  </div>
                  <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-[#F8FAFC]">
                    <div
                      className={`h-full rounded-full ${occupancyBarColor(zone.occupancy)}`}
                      style={{ width: `${zone.occupancy}%` }}
                    />
                  </div>
                </div>

                <dl className="grid grid-cols-2 gap-4">
                  <div className="rounded-lg border border-[#E5E7EB] p-4">
                    <dt className="text-caption text-[#374151]">Density</dt>
                    <dd className="mt-1 text-h4 font-semibold text-[#111827]">{zone.density}</dd>
                    <dd className="text-caption text-[#374151]">people / m²</dd>
                  </div>
                  <div className="rounded-lg border border-[#E5E7EB] p-4">
                    <dt className="text-caption text-[#374151]">Est. Headcount</dt>
                    <dd className="mt-1 text-h4 font-semibold text-[#111827]">
                      {Math.round((zone.occupancy / 100) * zone.capacity).toLocaleString()}
                    </dd>
                    <dd className="text-caption text-[#374151]">of {zone.capacity.toLocaleString()}</dd>
                  </div>
                </dl>

                <div className="rounded-lg border border-[#E5E7EB] p-4">
                  <p className="text-caption text-[#374151]">Last updated</p>
                  <p className="mt-1 text-small font-medium text-[#111827]">
                    {format(new Date(zone.updatedAt), 'PPpp')}
                  </p>
                </div>

                <div className="rounded-lg border border-dashed border-[#E5E7EB] p-4 text-center">
                  <p className="text-caption text-[#374151]">
                    Map overlay is reserved for a future phase. This panel is designed to host it
                    without a layout change.
                  </p>
                </div>
              </div>
            </div>
          </motion.aside>
        </>
      ) : null}
    </AnimatePresence>
  );
}
