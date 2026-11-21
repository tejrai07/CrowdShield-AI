import { LoadingState } from '../../../components/feedback/LoadingState';
import { ErrorState } from '../../../components/feedback/ErrorState';
import { EmptyState } from '../../../components/feedback/EmptyState';
import { cn } from '../../../utils/cn';
import { useBottleneckPressure } from '../hooks/useAnalyticsData';

function pressureColor(pressure: number): string {
  if (pressure >= 75) return 'bg-[#DC2626]';
  if (pressure >= 50) return 'bg-[#D97706]';
  return 'bg-[#2563EB]';
}

export function BottleneckPressurePanel() {
  const { data, isLoading, isError, refetch } = useBottleneckPressure();

  return (
    <div className="rounded-lg border border-[#E5E7EB] bg-white p-5">
      <h3 className="text-h4 font-semibold text-[#111827]">Bottleneck Pressure</h3>
      <p className="mt-0.5 text-caption text-[#374151]">
        Congestion buildup at known chokepoints, most pressured first
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
            {data.map((item) => (
              <li key={item.location}>
                <div className="flex items-center justify-between text-small">
                  <span className="font-medium text-[#111827]">{item.location}</span>
                  <span className="font-medium tabular-nums text-[#111827]">{item.pressure}%</span>
                </div>
                <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-[#F8FAFC]">
                  <div
                    className={cn('h-full rounded-full', pressureColor(item.pressure))}
                    style={{ width: `${item.pressure}%` }}
                  />
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
