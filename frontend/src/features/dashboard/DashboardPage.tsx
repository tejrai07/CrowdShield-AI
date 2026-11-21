import { PageContainer } from '../../components/ui/PageContainer';
import { SectionHeader } from '../../components/ui/SectionHeader';
import { DashboardStatGrid } from './components/DashboardStatGrid';
import { CrowdDensityTrendChart } from './components/CrowdDensityTrendChart';
import { RiskLevelTrendChart } from './components/RiskLevelTrendChart';
import { ZoneOccupancyChart } from './components/ZoneOccupancyChart';
import { FlowDirectionChart } from './components/FlowDirectionChart';
import { RecentEventsTable } from './components/RecentEventsTable';

export function DashboardPage() {
  return (
    <PageContainer>
      <SectionHeader
        as="h1"
        title="Command Center"
        description="Real-time crowd status across all monitored zones."
      />

      <div className="mt-6">
        <DashboardStatGrid />
      </div>

      <div className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-2">
        <CrowdDensityTrendChart />
        <RiskLevelTrendChart />
        <ZoneOccupancyChart />
        <FlowDirectionChart />
      </div>

      <div className="mt-6">
        <SectionHeader as="h3" title="Recent Events" className="mb-3" />
        <RecentEventsTable />
      </div>
    </PageContainer>
  );
}
