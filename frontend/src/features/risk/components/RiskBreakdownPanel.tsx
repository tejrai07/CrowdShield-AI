import { useRiskForecast } from '../hooks/useRiskData';
import { LoadingState } from '../../../components/feedback/LoadingState';
import { ErrorState } from '../../../components/feedback/ErrorState';
import { RiskBadge } from '../../../components/ui/RiskBadge';

export function RiskBreakdownPanel() {
  const { data, isLoading, isError, refetch } = useRiskForecast();
  const current = data?.find((p) => p.horizon === 'CURRENT');

  return (
    <div className="rounded-lg border border-[#E5E7EB] bg-white p-5">
      <h3 className="text-h4 font-semibold text-[#111827]">Risk Breakdown</h3>
      <p className="mt-0.5 text-caption text-[#374151]">
        Current classification and trajectory
      </p>

      <div className="mt-4">
        {isLoading ? (
          <LoadingState />
        ) : isError || !current ? (
          <ErrorState onRetry={() => refetch()} />
        ) : (
          <dl className="space-y-3">
            <div className="flex items-center justify-between">
              <dt className="text-small text-[#374151]">Classification</dt>
              <dd>
                <RiskBadge level={current.riskLevel} />
              </dd>
            </div>
            <div className="flex items-center justify-between">
              <dt className="text-small text-[#374151]">Composite score</dt>
              <dd className="text-small font-medium tabular-nums text-[#111827]">
                {current.riskScore} / 100
              </dd>
            </div>
            <div className="flex items-center justify-between">
              <dt className="text-small text-[#374151]">Model confidence</dt>
              <dd className="text-small font-medium tabular-nums text-[#111827]">
                {current.confidence}%
              </dd>
            </div>
            <div className="border-t border-[#E5E7EB] pt-3">
              <p className="text-caption text-[#374151]">
                Classification is derived from a weighted composite of live crowd density,
                flow, and historical pattern signals. See feature contributions for the
                individual drivers behind this score.
              </p>
            </div>
          </dl>
        )}
      </div>
    </div>
  );
}
