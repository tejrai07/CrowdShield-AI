# CrowdShield AI — Enterprise Frontend

Operational intelligence dashboard for real-time crowd monitoring, analytics,
and predictive safety insight. This repository is **frontend only** — it has
no dependency on internal ML/backend implementation details and is designed
to swap its mock data layer for a real API with no UI changes.

## Status: Phase UI-4 (top navigation)

Navigation moved from a left sidebar to a top navigation bar:

- **TopNav** (`app/layouts/TopNav.tsx`) — a single sticky header: brand mark
  on the left, horizontal primary navigation (Dashboard, Zones, Risk,
  Analytics, Alerts), and a right-hand action cluster (Settings, live
  notification bell with unresolved-critical-alert badge, operator
  identity). The active route is shown as a solid dark pill, keeping the
  accent blue reserved for content-level actions and links.
- **Mobile** — below `lg`, the primary nav collapses behind a hamburger
  button that opens an animated dropdown panel (Framer Motion) with the
  full nav list, including Settings.
- Content is no longer offset by a fixed sidebar width — every page now
  uses the full viewport width, with the existing `PageContainer` still
  centering content at a 1600px max width.

All six pages remain fully implemented (Phase UI-2) with the same
interaction polish added in Phase UI-3 (page transitions, animated zone
detail drawer, live alert toasts, entrance-animated stat cards).

## Tech stack

React 19 · Vite · TypeScript (strict) · TailwindCSS v4 · Zustand ·
TanStack Query · TanStack Table v8 · Recharts · React Router · React Hook
Form · Lucide React · Sonner · Framer Motion

## Getting started

```bash
npm install
npm run dev       # start dev server
npm run build     # typecheck + production build
npm run lint      # oxlint
npm run preview   # preview the production build
```

## Project structure

```
src/
  app/
    layouts/        AppLayout, TopNav (top navigation bar), MobileNavPanel
                     (mobile dropdown), navItems.ts (shared nav config),
                     PageTransition
    routes/         router.tsx (code-split routes), RouteFallback.tsx
  components/
    ui/              StatCard, RiskBadge, StatusBadge, PageContainer,
                      SectionHeader, ChartCard, SearchInput, FilterBar,
                      PlaceholderPage
    forms/           FormField, FormToggle (React Hook Form primitives)
    tables/          DataTable (shared TanStack Table v8 base)
    feedback/        LoadingState, ErrorState, EmptyState
  features/
    dashboard/       Widgets (animated entrance), charts, recent events
    zones/           Card/table views, animated detail panel, hooks
    risk/            Forecast cards, timeline, breakdown, explainability
    analytics/       Time-series panels, time range filter, bottlenecks
    alerts/          Active/resolved tables, summary stats, filters
    settings/        General/venue/zone/model/notifications/system panels
  hooks/             Cross-feature hooks — useCriticalAlertCount (nav
                     badge), useAlertNotifications (global toast watcher)
  services/
    api/             One file per domain — the only import point for that
                     domain's data (dashboardApi, zoneApi, riskApi,
                     analyticsApi, alertsApi)
    mock/            Mock data generators, isolated from the API layer
  store/             useUiStore.ts (Zustand, UI state only — mobile nav)
  types/             domain.ts — shared domain types (frontend's own contract)
  utils/             cn.ts (class merge), risk.ts (single source of truth
                     for risk-level and status styling)
```

## Design system

Colors, typography, and spacing are defined once in `src/index.css` under
`@theme` and consumed everywhere via Tailwind utilities. Risk-level and
status color mappings live in `src/utils/risk.ts` — no page should hardcode
a risk color; they all read from that module. The top nav's active-route
indicator intentionally uses the dark neutral (`#111827`) rather than the
accent blue, so accent stays reserved for primary actions and links inside
page content, per the brief's "color is secondary" principle.

## Data flow

Each domain has its own `services/mock/*Mock.ts` generator and
`services/api/*Api.ts` abstraction (e.g. `zoneApi.ts`, `riskApi.ts`).
Components never import the mock layer directly — they go through
TanStack Query hooks in each feature's `hooks/` folder (or the shared
`src/hooks/` folder for cross-feature concerns like the nav alert badge
and toast watcher), which call the API layer. Swapping to a live backend
means changing the implementation inside each `*Api.ts` file only.

## Known limitations (Phase UI-4)

- No authentication layer yet.
- Zone map overlay is intentionally not implemented — the zone detail
  panel and zone card layout are built to host it without a layout
  change in a future phase, per the original brief.
- Settings forms (General, Venue) submit locally with a toast confirmation;
  there's no persistence layer since there's no backend yet.
- Model Settings and Zone Configuration are read-only placeholders,
  per spec, until their respective backend APIs exist.
- At narrow tablet widths (roughly 1024–1279px) the top nav switches to
  the mobile dropdown; this is a deliberate trade-off to keep the
  horizontal nav from crowding, consistent with "desktop-first, mobile
  basic functional support" in the brief.
- The vendor JS chunk (~939 KB / ~279 KB gzip, mostly React + Recharts +
  Framer Motion + TanStack) is shared across all routes; page-specific
  code is code-split and stays small (2–36 KB per route).
