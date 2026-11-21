import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { format } from 'date-fns';
import { ChartCard } from '../../../components/ui/ChartCard';
import { LoadingState } from '../../../components/feedback/LoadingState';
import { ErrorState } from '../../../components/feedback/ErrorState';
import { EmptyState } from '../../../components/feedback/EmptyState';
import { useCrowdDensityTrend } from '../hooks/useDashboardData';

export function CrowdDensityTrendChart() {
  const { data, isLoading, isError, refetch } = useCrowdDensityTrend();

  return (
    <ChartCard
      title="Crowd Density Trend"
      description="People per square meter, last 2 hours"
    >
      {isLoading ? (
        <LoadingState variant="chart" className="h-full" />
      ) : isError ? (
        <ErrorState onRetry={() => refetch()} />
      ) : !data || data.length === 0 ? (
        <EmptyState />
      ) : (
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
            <defs>
              <linearGradient id="densityFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#2563EB" stopOpacity={0.18} />
                <stop offset="100%" stopColor="#2563EB" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#E5E7EB" vertical={false} />
            <XAxis
              dataKey="timestamp"
              tickFormatter={(v) => format(new Date(v), 'HH:mm')}
              tick={{ fontSize: 12, fill: '#374151' }}
              tickLine={false}
              axisLine={{ stroke: '#E5E7EB' }}
              minTickGap={32}
            />
            <YAxis
              tick={{ fontSize: 12, fill: '#374151' }}
              tickLine={false}
              axisLine={false}
              width={36}
            />
            <Tooltip
              contentStyle={{
                borderRadius: 8,
                border: '1px solid #E5E7EB',
                fontSize: 12,
              }}
              labelFormatter={(v) => format(new Date(v as string), 'HH:mm')}
              formatter={(value) => [`${value} p/m²`, 'Density']}
            />
            <Area
              type="monotone"
              dataKey="value"
              stroke="#2563EB"
              strokeWidth={2}
              fill="url(#densityFill)"
            />
          </AreaChart>
        </ResponsiveContainer>
      )}
    </ChartCard>
  );
}
