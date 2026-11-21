import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { format } from 'date-fns';
import { ChartCard } from '../../../components/ui/ChartCard';
import { LoadingState } from '../../../components/feedback/LoadingState';
import { ErrorState } from '../../../components/feedback/ErrorState';
import { EmptyState } from '../../../components/feedback/EmptyState';
import type { TimeSeriesPoint } from '../../../types/domain';

interface AnalyticsTimeSeriesPanelProps {
  title: string;
  description: string;
  data: TimeSeriesPoint[] | undefined;
  isLoading: boolean;
  isError: boolean;
  onRetry: () => void;
  color: string;
  unitLabel: string;
  valueSuffix?: string;
}

export function AnalyticsTimeSeriesPanel({
  title,
  description,
  data,
  isLoading,
  isError,
  onRetry,
  color,
  unitLabel,
  valueSuffix = '',
}: AnalyticsTimeSeriesPanelProps) {
  return (
    <ChartCard title={title} description={description}>
      {isLoading ? (
        <LoadingState variant="chart" className="h-full" />
      ) : isError ? (
        <ErrorState onRetry={onRetry} />
      ) : !data || data.length === 0 ? (
        <EmptyState />
      ) : (
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
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
              formatter={(value) => [`${value}${valueSuffix}`, unitLabel]}
            />
            <Line type="monotone" dataKey="value" stroke={color} strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
          </LineChart>
        </ResponsiveContainer>
      )}
    </ChartCard>
  );
}
