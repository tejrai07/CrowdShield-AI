import { useState } from 'react';
import { PageContainer } from '../../components/ui/PageContainer';
import { SectionHeader } from '../../components/ui/SectionHeader';
import type { TimeRange } from '../../types/domain';
import { TimeRangeFilter } from './components/TimeRangeFilter';
import { AnalyticsTimeSeriesPanel } from './components/AnalyticsTimeSeriesPanel';
import { InflowOutflowChart } from './components/InflowOutflowChart';
import { BottleneckPressurePanel } from './components/BottleneckPressurePanel';
import {
  useDensityOverTime,
  useDirectionVariance,
  useMovementSpeedDistribution,
} from './hooks/useAnalyticsData';

export function AnalyticsPage() {
  const [range, setRange] = useState<TimeRange>('6h');

  const density = useDensityOverTime(range);
  const speed = useMovementSpeedDistribution(range);
  const direction = useDirectionVariance(range);

  return (
    <PageContainer>
      <SectionHeader
        as="h1"
        title="Crowd Analytics"
        description="Deep analytical insight into density, movement, and flow."
        actions={<TimeRangeFilter value={range} onChange={setRange} />}
      />

      <div className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-2">
        <AnalyticsTimeSeriesPanel
          title="Density Over Time"
          description="People per square meter"
          data={density.data}
          isLoading={density.isLoading}
          isError={density.isError}
          onRetry={() => density.refetch()}
          color="#2563EB"
          unitLabel="Density"
          valueSuffix=" p/m²"
        />
        <AnalyticsTimeSeriesPanel
          title="Movement Speed Distribution"
          description="Average pedestrian speed, meters per second"
          data={speed.data}
          isLoading={speed.isLoading}
          isError={speed.isError}
          onRetry={() => speed.refetch()}
          color="#16A34A"
          unitLabel="Speed"
          valueSuffix=" m/s"
        />
        <AnalyticsTimeSeriesPanel
          title="Direction Variance"
          description="Spread of movement heading, degrees"
          data={direction.data}
          isLoading={direction.isLoading}
          isError={direction.isError}
          onRetry={() => direction.refetch()}
          color="#D97706"
          unitLabel="Variance"
          valueSuffix="°"
        />
        <InflowOutflowChart range={range} />
      </div>

      <div className="mt-6">
        <BottleneckPressurePanel />
      </div>
    </PageContainer>
  );
}
