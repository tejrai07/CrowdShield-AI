import { Bell, Gauge, Menu, X } from 'lucide-react';
import { Link, NavLink } from 'react-router-dom';
import { cn } from '../../utils/cn';
import { useUiStore } from '../../store/useUiStore';
import { useCriticalAlertCount } from '../../hooks/useCriticalAlertCount';
import { PRIMARY_NAV_ITEMS, SETTINGS_NAV_ITEM } from './navItems';
import { MobileNavPanel } from './MobileNavPanel';

const navLinkClass = ({ isActive }: { isActive: boolean }) =>
  cn(
    'flex items-center gap-2 rounded-md px-3 py-2 text-small font-medium transition-colors',
    'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#2563EB]',
    isActive
      ? 'bg-[#111827] text-white'
      : 'text-[#374151] hover:bg-[#F8FAFC] hover:text-[#111827]',
  );

export function TopNav() {
  const mobileNavOpen = useUiStore((s) => s.mobileNavOpen);
  const toggleMobileNav = useUiStore((s) => s.toggleMobileNav);
  const closeMobileNav = useUiStore((s) => s.closeMobileNav);
  const criticalCount = useCriticalAlertCount();

  return (
    <header className="sticky top-0 z-30 border-b border-[#E5E7EB] bg-white/95 shadow-sm backdrop-blur">
      <div className="mx-auto flex h-16 max-w-[1600px] items-center gap-3 px-4 sm:px-8">
        {/* Brand */}
        <Link to="/" className="flex shrink-0 items-center gap-2.5" onClick={closeMobileNav}>
          <span className="flex h-8 w-8 items-center justify-center rounded-md bg-[#111827] text-white">
            <Gauge size={18} aria-hidden="true" />
          </span>
          <span className="hidden leading-tight sm:block">
            <span className="block text-small font-semibold text-[#111827]">CrowdShield AI</span>
            <span className="block text-caption text-[#374151]">Operational Intelligence</span>
          </span>
        </Link>

        <div className="h-8 w-px shrink-0 bg-[#E5E7EB]" aria-hidden="true" />

        {/* Primary nav — desktop */}
        <nav className="hidden flex-1 items-center gap-1 lg:flex" aria-label="Primary navigation">
          {PRIMARY_NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
            <NavLink key={to} to={to} end={end} className={navLinkClass}>
              <Icon size={16} aria-hidden="true" />
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="flex-1 lg:hidden" />

        {/* Actions */}
        <div className="flex shrink-0 items-center gap-1.5 sm:gap-2">
          <span className="hidden items-center gap-1.5 rounded-md bg-[#16A34A]/10 px-2.5 py-1 text-caption font-medium text-[#16A34A] md:flex">
            <span className="h-1.5 w-1.5 rounded-full bg-[#16A34A]" aria-hidden="true" />
            Live
          </span>

          <NavLink
            to={SETTINGS_NAV_ITEM.to}
            className={navLinkClass}
            aria-label="Settings"
            title="Settings"
            end
          >
            <SETTINGS_NAV_ITEM.icon size={16} aria-hidden="true" />
            <span className="hidden xl:inline">Settings</span>
          </NavLink>

          <Link
            to="/alerts"
            aria-label={
              criticalCount > 0 ? `${criticalCount} unresolved critical alerts` : 'Notifications'
            }
            className="relative flex h-9 w-9 items-center justify-center rounded-md text-[#374151] hover:bg-[#F8FAFC] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#2563EB]"
          >
            <Bell size={18} aria-hidden="true" />
            {criticalCount > 0 ? (
              <span
                className="absolute -right-0.5 -top-0.5 flex h-4 min-w-4 items-center justify-center rounded-full bg-[#DC2626] px-1 text-[10px] font-semibold text-white"
                aria-hidden="true"
              >
                {criticalCount}
              </span>
            ) : null}
          </Link>

          <div className="hidden items-center gap-2.5 border-l border-[#E5E7EB] pl-3 sm:flex">
            <span className="flex h-8 w-8 items-center justify-center rounded-full bg-[#111827] text-caption font-semibold text-white">
              OP
            </span>
            <span className="hidden leading-tight xl:block">
              <span className="block text-small font-medium text-[#111827]">Operator</span>
              <span className="block text-caption text-[#374151]">Control Room</span>
            </span>
          </div>

          <button
            type="button"
            onClick={toggleMobileNav}
            aria-label="Toggle navigation menu"
            aria-expanded={mobileNavOpen}
            className="flex h-9 w-9 items-center justify-center rounded-md text-[#374151] hover:bg-[#F8FAFC] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#2563EB] lg:hidden"
          >
            {mobileNavOpen ? <X size={18} aria-hidden="true" /> : <Menu size={18} aria-hidden="true" />}
          </button>
        </div>
      </div>

      <MobileNavPanel isOpen={mobileNavOpen} onNavigate={closeMobileNav} />
    </header>
  );
}
