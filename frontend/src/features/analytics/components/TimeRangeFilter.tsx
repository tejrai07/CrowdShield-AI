import { FilterChip } from '../../../components/ui/FilterBar';
import type { TimeRange } from '../../../types/domain';

interface TimeRangeFilterProps {
  value: TimeRange;
  onChange: (range: TimeRange) => void;
}

const RANGES: { label: string; value: TimeRange }[] = [
  { label: '1 hour', value: '1h' },
  { label: '6 hours', value: '6h' },
  { label: '24 hours', value: '24h' },
  { label: 'Custom', value: 'custom' },
];

export function TimeRangeFilter({ value, onChange }: TimeRangeFilterProps) {
  return (
    <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Time range">
      {RANGES.map((range) => (
        <FilterChip
          key={range.value}
          label={range.label}
          active={value === range.value}
          onClick={() => onChange(range.value)}
        />
      ))}
    </div>
  );
}
