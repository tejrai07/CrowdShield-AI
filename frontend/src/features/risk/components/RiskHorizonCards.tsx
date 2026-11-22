import { format } from 'date-fns';
import { RiskBadge } from '../../../components/ui/RiskBadge';
import { LoadingState } from '../../../components/feedback/LoadingState';
import { ErrorState } from '../../../components/feedback/ErrorState';
import { cn } from '../../../utils/cn';
import { useRiskForecast } from '../hooks/useRiskData';

const HORIZON_LABEL: Record<string, string> = {
  CURRENT: 'Current',
  '+5min': 'In 5 minutes',
  '+10min': 'In 10 minutes',
};

export function RiskHorizonCards() {
  const { data, isLoading, isError, refetch } = useRiskForecast();

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        {Array.from({ length: 3 }).map((_, i) => (
          <LoadingState key={i} />
        ))}
      </div>
    );
  }

  if (isError || !data) {
    return <ErrorState onRetry={() => refetch()} />;
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        {data.map((point) => (
          <div
            key={point.horizon}
            className={cn(
              'rounded-lg border bg-white p-5',
              point.horizon === 'CURRENT' ? 'border-[#2563EB]' : 'border-[#E5E7EB]',
            )}
          >
            <div className="flex items-center justify-between">
              <span className="text-small font-medium text-[#374151]">
                {HORIZON_LABEL[point.horizon]}
              </span>
              {point.horizon === 'CURRENT' ? (
                <span className="text-caption font-medium text-[#2563EB]">Live</span>
              ) : (
                <span
                  className="rounded border border-amber-200 bg-amber-50 px-1.5 py-0.5 text-[11px] font-medium text-amber-700"
                  title="Low future escalation recall (~11.8%). Synthetic data only."
                >
                  Provisional (~12% recall)
                </span>
              )}
            </div>

            <div className="mt-3 flex items-end justify-between">
              <span className="text-h1 font-semibold tabular-nums text-[#111827]">
                {point.riskScore}
              </span>
              <RiskBadge level={point.riskLevel} />
            </div>

            <div className="mt-4">
              <div className="flex items-center justify-between text-caption text-[#374151]">
                <span>Model confidence</span>
                <span className="font-medium text-[#111827]">{point.confidence}%</span>
              </div>
              <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-[#F8FAFC]">
                <div
                  className="h-full rounded-full bg-[#2563EB]"
                  style={{ width: `${point.confidence}%` }}
                />
              </div>
            </div>

            {point.horizon !== 'CURRENT' && (
              <p className="mt-2 text-[11px] text-[#6B7280]">
                Predicted for {format(new Date(point.timestamp), 'HH:mm')} &middot; Synthetic
                baseline
              </p>
            )}
          </div>
        ))}
      </div>

      {/* Limitation Disclosure Banner */}
      <div className="rounded-md border border-amber-200/90 bg-amber-50/80 p-3 text-caption text-amber-800">
        <span className="font-semibold text-amber-900">Operational Advisory:</span> Future risk
        predictions (+5m, +10m) are trained exclusively on synthetic data with low escalation recall
        (~11.8%). Treat future horizons as advisory signals alongside live observations.
      </div>
    </div>
  );
}
