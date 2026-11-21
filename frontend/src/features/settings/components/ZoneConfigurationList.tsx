import { useZones } from '../../zones/hooks/useZones';
import { LoadingState } from '../../../components/feedback/LoadingState';
import { ErrorState } from '../../../components/feedback/ErrorState';

export function ZoneConfigurationList() {
  const { data, isLoading, isError, refetch } = useZones();

  if (isLoading) {
    return <LoadingState variant="table" rows={4} />;
  }

  if (isError || !data) {
    return <ErrorState onRetry={() => refetch()} />;
  }

  return (
    <div className="overflow-hidden rounded-lg border border-[#E5E7EB]">
      <ul>
        {data.map((zone, i) => (
          <li
            key={zone.id}
            className={`flex items-center justify-between px-4 py-3 text-small ${
              i !== data.length - 1 ? 'border-b border-[#E5E7EB]' : ''
            }`}
          >
            <span className="font-medium text-[#111827]">{zone.name}</span>
            <span className="text-[#374151]">Capacity {zone.capacity.toLocaleString()}</span>
          </li>
        ))}
      </ul>
      <p className="border-t border-[#E5E7EB] bg-[#F8FAFC] px-4 py-2 text-caption text-[#374151]">
        Zone editing arrives once zone provisioning is available on the backend.
      </p>
    </div>
  );
}
