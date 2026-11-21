import { lazy, Suspense, type ReactNode } from 'react';
import { createBrowserRouter } from 'react-router-dom';
import { AppLayout } from '../layouts/AppLayout';
import { RouteFallback } from './RouteFallback';

// Each page is code-split by route. The dashboard loads eagerly since
// it's the landing page; the rest load on navigation.
import { DashboardPage } from '../../features/dashboard/DashboardPage';

const ZonesPage = lazy(() =>
  import('../../features/zones/ZonesPage').then((m) => ({ default: m.ZonesPage })),
);
const RiskPage = lazy(() =>
  import('../../features/risk/RiskPage').then((m) => ({ default: m.RiskPage })),
);
const AnalyticsPage = lazy(() =>
  import('../../features/analytics/AnalyticsPage').then((m) => ({ default: m.AnalyticsPage })),
);
const AlertsPage = lazy(() =>
  import('../../features/alerts/AlertsPage').then((m) => ({ default: m.AlertsPage })),
);
const SettingsPage = lazy(() =>
  import('../../features/settings/SettingsPage').then((m) => ({ default: m.SettingsPage })),
);

function withSuspense(element: ReactNode) {
  return <Suspense fallback={<RouteFallback />}>{element}</Suspense>;
}

export const router = createBrowserRouter([
  {
    path: '/',
    element: <AppLayout />,
    children: [
      { index: true, element: <DashboardPage /> },
      { path: 'zones', element: withSuspense(<ZonesPage />) },
      { path: 'risk', element: withSuspense(<RiskPage />) },
      { path: 'analytics', element: withSuspense(<AnalyticsPage />) },
      { path: 'alerts', element: withSuspense(<AlertsPage />) },
      { path: 'settings', element: withSuspense(<SettingsPage />) },
    ],
  },
]);
