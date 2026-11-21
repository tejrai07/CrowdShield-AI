import { LoadingState } from '../../../components/feedback/LoadingState';
import { ErrorState } from '../../../components/feedback/ErrorState';
import { EmptyState } from '../../../components/feedback/EmptyState';
import { cn } from '../../../utils/cn';
import { useFeatureContributions } from '../hooks/useRiskData';

export function FeatureContributionView() {
  const { data, isLoading, isError, refetch } = useFeatureContributions();

  const maxAbs = data ? Math.max(...data.map((f) => Math.abs(f.contribution)), 1) : 1;

  return (
    <div className="rounded-lg border border-[#E5E7EB] bg-white p-5">
      <h3 className="text-h4 font-semibold text-[#111827]">Feature Contribution</h3>
      <p className="mt-0.5 text-caption text-[#374151]">
        What is driving the current risk score, most to least significant
      </p>

      <div className="mt-4">
        {isLoading ? (
          <LoadingState variant="table" rows={6} />
        ) : isError ? (
          <ErrorState onRetry={() => refetch()} />
        ) : !data || data.length === 0 ? (
          <EmptyState />
        ) : (
          <ul className="space-y-4">
            {data.map((item) => {
              const widthPct = (Math.abs(item.contribution) / maxAbs) * 100;
              const isPositive = item.contribution >= 0;
              return (
                <li key={item.feature}>
                  <div className="flex items-center justify-between text-small">
                    <span className="font-medium text-[#111827]">{item.feature}</span>
                    <span
                      className={cn(
                        'font-medium tabular-nums',
                        isPositive ? 'text-[#DC2626]' : 'text-[#16A34A]',
                      )}
                    >
                      {isPositive ? '+' : ''}
                      {item.contribution}
                    </span>
                  </div>
                  <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-[#F8FAFC]">
                    <div
                      className={cn('h-full rounded-full', isPositive ? 'bg-[#DC2626]' : 'bg-[#16A34A]')}
                      style={{ width: `${widthPct}%` }}
                    />
                  </div>
                  <p className="mt-1 text-caption text-[#374151]">{item.description}</p>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}
