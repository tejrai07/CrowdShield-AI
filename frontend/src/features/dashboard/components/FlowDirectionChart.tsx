import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts';
import { ChartCard } from '../../../components/ui/ChartCard';
import { LoadingState } from '../../../components/feedback/LoadingState';
import { ErrorState } from '../../../components/feedback/ErrorState';
import { EmptyState } from '../../../components/feedback/EmptyState';
import { useFlowDirectionDistribution } from '../hooks/useDashboardData';

const DIRECTION_COLORS: Record<string, string> = {
  NORTH: '#2563EB',
  SOUTH: '#111827',
  EAST: '#D97706',
  WEST: '#374151',
};

export function FlowDirectionChart() {
  const { data, isLoading, isError, refetch } = useFlowDirectionDistribution();

  return (
    <ChartCard title="Crowd Flow Direction" description="Share of movement, by heading">
      {isLoading ? (
        <LoadingState variant="chart" className="h-full" />
      ) : isError ? (
        <ErrorState onRetry={() => refetch()} />
      ) : !data || data.length === 0 ? (
        <EmptyState />
      ) : (
        <div className="flex h-full items-center gap-6">
          <ResponsiveContainer width="55%" height="100%">
            <PieChart>
              <Pie
                data={data}
                dataKey="percentage"
                nameKey="direction"
                innerRadius="60%"
                outerRadius="90%"
                paddingAngle={2}
                strokeWidth={0}
              >
                {data.map((entry) => (
                  <Cell key={entry.direction} fill={DIRECTION_COLORS[entry.direction]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{ borderRadius: 8, border: '1px solid #E5E7EB', fontSize: 12 }}
                formatter={(value) => [`${value}%`, 'Share']}
              />
            </PieChart>
          </ResponsiveContainer>
          <ul className="flex-1 space-y-2">
            {data.map((entry) => (
              <li key={entry.direction} className="flex items-center justify-between gap-3">
                <span className="flex items-center gap-2 text-small text-[#374151]">
                  <span
                    className="h-2.5 w-2.5 rounded-full"
                    style={{ backgroundColor: DIRECTION_COLORS[entry.direction] }}
                    aria-hidden="true"
                  />
                  {entry.direction[0] + entry.direction.slice(1).toLowerCase()}
                </span>
                <span className="text-small font-medium tabular-nums text-[#111827]">
                  {entry.percentage}%
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </ChartCard>
  );
}
