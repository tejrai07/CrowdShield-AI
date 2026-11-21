import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { format } from 'date-fns';
import { ChartCard } from '../../../components/ui/ChartCard';
import { LoadingState } from '../../../components/feedback/LoadingState';
import { ErrorState } from '../../../components/feedback/ErrorState';
import { EmptyState } from '../../../components/feedback/EmptyState';
import { useInflowOutflow } from '../hooks/useAnalyticsData';
import type { TimeRange } from '../../../types/domain';

interface InflowOutflowChartProps {
  range: TimeRange;
}

export function InflowOutflowChart({ range }: InflowOutflowChartProps) {
  const { data, isLoading, isError, refetch } = useInflowOutflow(range);

  const merged =
    data?.inflow.map((point, i) => ({
      timestamp: point.timestamp,
      inflow: point.value,
      outflow: data.outflow[i]?.value ?? 0,
    })) ?? [];

  return (
    <ChartCard title="Inflow / Outflow Rate" description="People per minute, entering vs. leaving">
      {isLoading ? (
        <LoadingState variant="chart" className="h-full" />
      ) : isError ? (
        <ErrorState onRetry={() => refetch()} />
      ) : merged.length === 0 ? (
        <EmptyState />
      ) : (
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={merged} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
            <CartesianGrid stroke="#E5E7EB" vertical={false} />
            <XAxis
              dataKey="timestamp"
              tickFormatter={(v) => format(new Date(v), 'HH:mm')}
              tick={{ fontSize: 12, fill: '#374151' }}
              tickLine={false}
              axisLine={{ stroke: '#E5E7EB' }}
              minTickGap={32}
            />
            <YAxis tick={{ fontSize: 12, fill: '#374151' }} tickLine={false} axisLine={false} width={40} />
            <Tooltip
              contentStyle={{ borderRadius: 8, border: '1px solid #E5E7EB', fontSize: 12 }}
              labelFormatter={(v) => format(new Date(v as string), 'HH:mm')}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line type="monotone" dataKey="inflow" name="Inflow" stroke="#2563EB" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="outflow" name="Outflow" stroke="#111827" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      )}
    </ChartCard>
  );
}
