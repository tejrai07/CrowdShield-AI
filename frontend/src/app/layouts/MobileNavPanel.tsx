import { AnimatePresence, motion } from 'framer-motion';
import { NavLink } from 'react-router-dom';
import { cn } from '../../utils/cn';
import { PRIMARY_NAV_ITEMS, SETTINGS_NAV_ITEM } from './navItems';

interface MobileNavPanelProps {
  isOpen: boolean;
  onNavigate: () => void;
}

const linkClass = ({ isActive }: { isActive: boolean }) =>
  cn(
    'flex items-center gap-3 rounded-md px-3 py-2.5 text-small font-medium transition-colors',
    'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#2563EB]',
    isActive
      ? 'bg-[#2563EB]/10 text-[#2563EB]'
      : 'text-[#374151] hover:bg-[#F8FAFC] hover:text-[#111827]',
  );

export function MobileNavPanel({ isOpen, onNavigate }: MobileNavPanelProps) {
  return (
    <AnimatePresence>
      {isOpen ? (
        <motion.nav
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          transition={{ duration: 0.15, ease: 'easeOut' }}
          className="overflow-hidden border-t border-[#E5E7EB] bg-white lg:hidden"
          aria-label="Primary navigation"
        >
          <div className="space-y-1 px-4 py-3">
            {PRIMARY_NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
              <NavLink key={to} to={to} end={end} className={linkClass} onClick={onNavigate}>
                <Icon size={18} aria-hidden="true" />
                {label}
              </NavLink>
            ))}
            <div className="mt-2 border-t border-[#E5E7EB] pt-2">
              <NavLink to={SETTINGS_NAV_ITEM.to} className={linkClass} onClick={onNavigate}>
                <SETTINGS_NAV_ITEM.icon size={18} aria-hidden="true" />
                {SETTINGS_NAV_ITEM.label}
              </NavLink>
            </div>
          </div>
        </motion.nav>
      ) : null}
    </AnimatePresence>
  );
}
