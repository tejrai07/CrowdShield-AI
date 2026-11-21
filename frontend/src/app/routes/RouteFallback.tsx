import { LoadingState } from '../../components/feedback/LoadingState';
import { PageContainer } from '../../components/ui/PageContainer';

export function RouteFallback() {
  return (
    <PageContainer>
      <LoadingState />
    </PageContainer>
  );
}
