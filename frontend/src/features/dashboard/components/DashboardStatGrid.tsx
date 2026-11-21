import { Activity, AlertTriangle, MapPin, ShieldCheck, Target } from 'lucide-react';
import { motion } from 'framer-motion';
import { StatCard } from '../../../components/ui/StatCard';
import { RiskBadge } from '../../../components/ui/RiskBadge';
import { LoadingState } from '../../../components/feedback/LoadingState';
import { ErrorState } from '../../../components/feedback/ErrorState';
import { useDashboardSummary } from '../hooks/useDashboardData';

// Minimal, one-time entrance stagger on mount — not a decorative loop,
// per the animation policy.
const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.04 },
  },
};

const item = {
  hidden: { opacity: 0, y: 6 },
  show: { opacity: 1, y: 0, transition: { duration: 0.15 } },
};

export function DashboardStatGrid() {
  const { data, isLoading, isError, refetch } = useDashboardSummary();

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
        {Array.from({ length: 6 }).map((_, i) => (
          <LoadingState key={i} />
        ))}
      </div>
    );
  }

  if (isError || !data) {
    return <ErrorState onRetry={() => refetch()} />;
  }

  const stats = [
    {
      label: 'Crowd Status',
      value: <RiskBadge level={data.crowdStatus} />,
      icon: <ShieldCheck size={18} aria-hidden="true" />,
    },
    {
      label: 'Active People Count',
      value: data.activePeopleCount.toLocaleString(),
      icon: <Activity size={18} aria-hidden="true" />,
    },
    {
      label: 'Current Risk Level',
      value: <RiskBadge level={data.currentRiskLevel} />,
      icon: <AlertTriangle size={18} aria-hidden="true" />,
    },
    {
      label: 'Active Zones',
      value: `${data.activeZoneCount} / ${data.totalZoneCount}`,
      icon: <MapPin size={18} aria-hidden="true" />,
    },
    {
      label: 'Alert Count',
      value: data.alertCount,
      icon: <AlertTriangle size={18} aria-hidden="true" />,
    },
    {
      label: 'Prediction Confidence',
      value: `${data.predictionConfidence}%`,
      icon: <Target size={18} aria-hidden="true" />,
    },
  ];

  return (
    <motion.div
      className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6"
      variants={container}
      initial="hidden"
      animate="show"
    >
      {stats.map((stat) => (
        <motion.div key={stat.label} variants={item}>
          <StatCard label={stat.label} value={stat.value} icon={stat.icon} />
        </motion.div>
      ))}
    </motion.div>
  );
}
