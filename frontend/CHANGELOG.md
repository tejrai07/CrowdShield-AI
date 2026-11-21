# Changelog

## Phase UI-4 — Top navigation (this build)

### Changed

**Navigation moved from left sidebar to top nav**
- Removed `app/layouts/Sidebar.tsx` and `app/layouts/Header.tsx`
- Added `app/layouts/TopNav.tsx`: sticky header with brand mark, horizontal
  primary nav (desktop, `lg` and above), and a right-hand action cluster
  (Settings shortcut, live notification bell with critical-alert badge,
  operator identity)
- Added `app/layouts/MobileNavPanel.tsx`: animated (Framer Motion)
  dropdown nav for below `lg`, triggered by a hamburger button
- Added `app/layouts/navItems.ts`: nav item config shared between desktop
  and mobile nav so they can never drift out of sync
- `AppLayout`: removed the `pl-[280px]` / `lg:pl-[280px]` sidebar offset;
  every page now spans the full viewport width, with `PageContainer`
  still centering content at 1600px max width
- Active route indicator uses a solid dark (`#111827`) pill rather than
  the accent blue used elsewhere for links/actions, keeping the visual
  hierarchy the brief calls for ("color is secondary")

### Verification

- `npm run build` (tsc -b && vite build): passes, 0 type errors
- `npm run lint` (oxlint): 0 errors across 82 files (5 pre-existing
  fast-refresh style warnings on router.tsx — expected for the lazy-route
  pattern, not functional issues)
- Manual review: no stray references to the removed Sidebar/Header files;
  300-line file cap respected (largest file: 141 lines)

### Known limitations

See README → "Known limitations".

---

## Phase UI-3 — Interaction polish

### Added

**Responsive navigation**
- `store/useUiStore.ts`: renamed/clarified to `mobileNavOpen` +
  `toggleMobileNav` / `closeMobileNav` (was an unused `sidebarCollapsed`
  flag in Phase UI-1/2)
- `Sidebar`: now an off-canvas drawer below `lg`, with a scrim; always
  visible at `lg` and above
- `Header`: hamburger menu button (mobile only), responsive spacing,
  "Live" pill and operator identity hidden on narrow viewports

**Animation (Framer Motion, per the strict animation policy)**
- `app/layouts/PageTransition.tsx`: subtle fade/slide between routes
- `ZoneDetailPanel`: converted from manual CSS transitions to a proper
  Framer Motion drawer (slide + scrim fade, `AnimatePresence`)
- `DashboardStatGrid`: one-time entrance stagger on mount (not a loop)

**Live alerting**
- `services/mock/alertsMock.ts`: each refetch now has a small chance of
  appending a brand-new alert, simulating incidents arriving in real time
- `hooks/useAlertNotifications.ts`: global watcher mounted once at the
  app root; toasts (via Sonner) the moment a new critical or high
  alert appears, from any page
- `hooks/useCriticalAlertCount.ts`: shared hook powering a live badge
  count on the header's notification bell, linked to Alerts Center

**Accessibility**
- Skip-to-content link in `AppLayout`
- Notification bell has a dynamic `aria-label` reflecting unresolved
  critical alert count

### Verification

- `npm run build` (tsc -b && vite build): passes, 0 type errors
- `npm run lint` (oxlint): 0 errors across 81 files (5 pre-existing
  fast-refresh style warnings on router.tsx — expected for the lazy-route
  pattern, not functional issues)
- Manual review: no parallax/floating/decorative animation added; all
  motion is transition, drawer, or one-time entrance per policy; 300-line
  file cap respected (largest file: 141 lines)

### Known limitations

See README → "Known limitations".

---

## Phase UI-2 — Full page implementation

### Added

**Zone Management** (`features/zones/`)
- `services/mock/zonesMock.ts` + `services/api/zoneApi.ts`: stable
  in-session zone identities with drifting metrics, simulating a live feed
- `ZoneMetricsOverview`: total zones, average occupancy, elevated/restricted
  count, offline count
- `ZoneCard` / `ZoneTable`: card and table hybrid view, toggleable
- `ZoneDetailPanel`: slide-in side panel with occupancy, density,
  headcount, last-updated; explicitly reserves space for a future map
  overlay without a layout change
- Search + status filter (Normal / Elevated / Restricted / Offline)

**Risk Monitoring** (`features/risk/`) — the critical page
- `services/mock/riskMock.ts` + `services/api/riskApi.ts`: forecast,
  timeline, and feature-contribution generators
- `RiskHorizonCards`: Current / +5min / +10min, each with score, risk
  badge, and a decaying model-confidence bar
- `RiskTimelineChart`: 3-hour composite risk score with a critical
  reference line
- `RiskBreakdownPanel`: current classification summary
- `FeatureContributionView`: signed, magnitude-sorted explainability bars

**Crowd Analytics** (`features/analytics/`)
- `services/mock/analyticsMock.ts` + `services/api/analyticsApi.ts`,
  parametrized by time range (1h / 6h / 24h / custom)
- `TimeRangeFilter`, reusable `AnalyticsTimeSeriesPanel` (density, movement
  speed, direction variance), `InflowOutflowChart` (dual-line), and
  `BottleneckPressurePanel` (ranked chokepoints)

**Alerts Center** (`features/alerts/`)
- `services/mock/alertsMock.ts` + `services/api/alertsApi.ts`: stable
  alert identities with active/acknowledged/resolved status
- `AlertsSummary` stat row, `AlertsTable` (shared across Active/Resolved
  sections), search + severity filter, 15s refresh interval

**Settings** (`features/settings/`)
- `components/forms/FormField.tsx`, `FormToggle.tsx`: new shared RHF-driven
  form primitives
- General and Venue Configuration forms (React Hook Form, toast on save)
- Zone Configuration: read-only list backed by real zone data
- Model Settings: read-only placeholder panel per spec
- Notification Preferences: toggle panel
- System Information: read-only build/connection status

**Performance**
- Routes are now code-split via `React.lazy` + `Suspense`
  (`app/routes/router.tsx`, `RouteFallback.tsx`); per-route JS is
  2–36 KB gzipped instead of one monolithic bundle

### Verification

- `npm run build` (tsc -b && vite build): passes, 0 type errors
- `npm run lint` (oxlint): 0 errors across 78 files (5 pre-existing
  fast-refresh style warnings on router/hook files — expected for this
  pattern, not functional issues)
- Manual review against design brief: strict palette respected, type scale
  respected, no decorative gradients/animation, 300-line file cap respected
  (largest file: 141 lines)

### Known limitations

See README → "Known limitations".

---

## Phase UI-1 — Foundation

### Added

**Design system**
- `src/index.css`: strict color palette, type scale, and layout tokens
  mapped into Tailwind v4 via `@theme`
- `src/utils/risk.ts`: single source of truth for risk-level / status colors

**App shell**
- `AppLayout`, `Sidebar` (280px, fixed), `Header` (sticky) in `src/app/layouts/`
- Full route table in `src/app/routes/router.tsx` for all six pages

**Reusable components**
- `ui/`: StatCard, MetricCard, SectionHeader, RiskBadge, StatusBadge,
  PageContainer, ChartCard, SearchInput, FilterBar, PlaceholderPage
- `feedback/`: LoadingState (skeletons), ErrorState (recoverable), EmptyState
- `tables/`: DataTable (shared TanStack Table v8 base, sortable)

**Dashboard (Command Center) — fully implemented**
- Widgets: Crowd Status, Active People Count, Current Risk Level,
  Active Zones, Alert Count, Prediction Confidence
- Charts: Crowd Density Trend, Risk Level Trend, Zone Occupancy
  Distribution, Crowd Flow Direction Distribution
- Table: Recent Events (searchable columns, sortable, severity badges)
- Auto-refresh every 30s via TanStack Query

**Data layer**
- `services/mock/`: realistic generators for dashboard summary, trends,
  distributions, and events
- `services/api/dashboardApi.ts`: abstraction boundary — the only import
  point for dashboard data; mock-backed today, swappable to a real HTTP
  client with no call-site changes
- `features/dashboard/hooks/useDashboardData.ts`: TanStack Query hooks

**State management**
- `store/useUiStore.ts`: Zustand store, UI state only (sidebar)

**Other pages (routed, placeholder)**
- Zone Management, Risk Monitoring, Crowd Analytics, Alerts Center, Settings

### Verification

- `npm run build` (tsc -b && vite build): passes, 0 type errors
- `npm run lint` (oxlint): 0 warnings, 0 errors, 42 files
- Manual review against design brief: strict palette respected, type scale
  respected, no decorative gradients/animation, 300-line file cap respected

### Known limitations

See README → "Known limitations".
