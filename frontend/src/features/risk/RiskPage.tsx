import { PageContainer } from '../../components/ui/PageContainer';
import { SectionHeader } from '../../components/ui/SectionHeader';
import { RiskHorizonCards } from './components/RiskHorizonCards';
import { RiskTimelineChart } from './components/RiskTimelineChart';
import { RiskBreakdownPanel } from './components/RiskBreakdownPanel';
import { FeatureContributionView } from './components/FeatureContributionView';

export function RiskPage() {
  return (
    <PageContainer>
      <SectionHeader
        as="h1"
        title="Risk Monitoring"
        description="Current and predicted risk, with model explainability."
      />

      <div className="mt-6">
        <RiskHorizonCards />
      </div>

      <div className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <RiskTimelineChart />
        </div>
        <RiskBreakdownPanel />
      </div>

      <div className="mt-6">
        <FeatureContributionView />
      </div>
    </PageContainer>
  );
}
