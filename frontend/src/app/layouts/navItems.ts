import {
  AlertOctagon,
  LayoutGrid,
  LineChart,
  MapPin,
  Settings,
  ShieldAlert,
  type LucideIcon,
} from 'lucide-react';

export interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
  end?: boolean;
}

export const PRIMARY_NAV_ITEMS: NavItem[] = [
  { to: '/', label: 'Dashboard', icon: LayoutGrid, end: true },
  { to: '/zones', label: 'Zones', icon: MapPin },
  { to: '/risk', label: 'Risk', icon: ShieldAlert },
  { to: '/analytics', label: 'Analytics', icon: LineChart },
  { to: '/alerts', label: 'Alerts', icon: AlertOctagon },
];

export const SETTINGS_NAV_ITEM: NavItem = { to: '/settings', label: 'Settings', icon: Settings };
