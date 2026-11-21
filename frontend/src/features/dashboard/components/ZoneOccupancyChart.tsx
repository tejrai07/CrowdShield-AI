import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis, Cell } from 'recharts';
import { ChartCard } from '../../../components/ui/ChartCard';
import { LoadingState } from '../../../components/feedback/LoadingState';
import { ErrorState } from '../../../components/feedback/ErrorState';
import { EmptyState } from '../../../components/feedback/EmptyState';
import { useZoneOccupancyDistribution } from '../hooks/useDashboardData';

function occupancyColor(value: number): string {
  if (value >= 90) return '#DC2626';
  if (value >= 70) return '#D97706';
  return '#2563EB';
}

export function ZoneOccupancyChart() {
  const { data, isLoading, isError, refetch } = useZoneOccupancyDistribution();

  return (
    <ChartCard title="Zone Occupancy Distribution" description="Percent of rated capacity, by zone">
      {isLoading ? (
        <LoadingState variant="chart" className="h-full" />
      ) : isError ? (
        <ErrorState onRetry={() => refetch()} />
      ) : !data || data.length === 0 ? (
        <EmptyState />
      ) : (
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="#E5E7EB" horizontal={false} />
            <XAxis
              type="number"
              domain={[0, 100]}
              tick={{ fontSize: 12, fill: '#374151' }}
              tickLine={false}
              axisLine={{ stroke: '#E5E7EB' }}
            />
            <YAxis
              type="category"
              dataKey="zoneName"
              tick={{ fontSize: 12, fill: '#111827' }}
              tickLine={false}
              axisLine={false}
              width={110}
            />
            <Tooltip
              contentStyle={{ borderRadius: 8, border: '1px solid #E5E7EB', fontSize: 12 }}
              formatter={(value) => [`${value}%`, 'Occupancy']}
              cursor={{ fill: '#F8FAFC' }}
            />
            <Bar dataKey="occupancy" radius={[0, 4, 4, 0]} barSize={16}>
              {data.map((entry) => (
                <Cell key={entry.zoneName} fill={occupancyColor(entry.occupancy)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
    </ChartCard>
  );
}
