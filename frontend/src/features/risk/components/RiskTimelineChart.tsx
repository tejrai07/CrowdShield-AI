import { Area, AreaChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { format } from 'date-fns';
import { ChartCard } from '../../../components/ui/ChartCard';
import { LoadingState } from '../../../components/feedback/LoadingState';
import { ErrorState } from '../../../components/feedback/ErrorState';
import { EmptyState } from '../../../components/feedback/EmptyState';
import { useRiskTimeline } from '../hooks/useRiskData';

export function RiskTimelineChart() {
  const { data, isLoading, isError, refetch } = useRiskTimeline();

  return (
    <ChartCard title="Risk Timeline" description="Composite risk score, last 3 hours" height={320}>
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
              <linearGradient id="riskFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#D97706" stopOpacity={0.2} />
                <stop offset="100%" stopColor="#D97706" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#E5E7EB" vertical={false} />
            <XAxis
              dataKey="timestamp"
              tickFormatter={(v) => format(new Date(v), 'HH:mm')}
              tick={{ fontSize: 12, fill: '#374151' }}
              tickLine={false}
              axisLine={{ stroke: '#E5E7EB' }}
              minTickGap={40}
            />
            <YAxis
              domain={[0, 100]}
              tick={{ fontSize: 12, fill: '#374151' }}
              tickLine={false}
              axisLine={false}
              width={36}
            />
            <ReferenceLine y={65} stroke="#DC2626" strokeDasharray="4 4" strokeOpacity={0.6} />
            <Tooltip
              contentStyle={{ borderRadius: 8, border: '1px solid #E5E7EB', fontSize: 12 }}
              labelFormatter={(v) => format(new Date(v as string), 'HH:mm')}
              formatter={(value) => [`${value}`, 'Risk score']}
            />
            <Area
              type="monotone"
              dataKey="value"
              stroke="#D97706"
              strokeWidth={2}
              fill="url(#riskFill)"
            />
          </AreaChart>
        </ResponsiveContainer>
      )}
    </ChartCard>
  );
}
