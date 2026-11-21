import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { RouterProvider } from 'react-router-dom';
import { Toaster } from 'sonner';
import { router } from './app/routes/router';
import { useAlertNotifications } from './hooks/useAlertNotifications';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 15_000,
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

// Runs inside QueryClientProvider so it can share the query cache with
// every page (same query key as the Alerts Center and header badge).
function AlertNotificationsWatcher() {
  useAlertNotifications();
  return null;
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AlertNotificationsWatcher />
      <RouterProvider router={router} />
      <Toaster position="top-right" richColors />
    </QueryClientProvider>
  );
}
