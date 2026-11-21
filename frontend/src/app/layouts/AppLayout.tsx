import { Outlet } from 'react-router-dom';
import { TopNav } from './TopNav';
import { PageTransition } from './PageTransition';

export function AppLayout() {
  return (
    <div className="min-h-screen bg-[#F8FAFC]">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:bg-[#111827] focus:px-4 focus:py-2 focus:text-small focus:font-medium focus:text-white"
      >
        Skip to main content
      </a>
      <TopNav />
      <main id="main-content" className="min-h-[calc(100vh-64px)]">
        <PageTransition>
          <Outlet />
        </PageTransition>
      </main>
    </div>
  );
}
