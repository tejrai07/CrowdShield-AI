import { create } from 'zustand';

interface UiState {
  // Controls the off-canvas mobile navigation drawer (below the lg
  // breakpoint the sidebar is hidden by default and slides in over
  // a scrim; at lg and above the sidebar is always visible).
  mobileNavOpen: boolean;
  toggleMobileNav: () => void;
  closeMobileNav: () => void;
}

// UI state only. Server state (dashboard data, alerts, zones, etc.)
// lives in TanStack Query — never in this store.
export const useUiStore = create<UiState>((set) => ({
  mobileNavOpen: false,
  toggleMobileNav: () => set((state) => ({ mobileNavOpen: !state.mobileNavOpen })),
  closeMobileNav: () => set({ mobileNavOpen: false }),
}));
