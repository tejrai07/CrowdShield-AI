const SYSTEM_INFO = [
  { label: 'Frontend version', value: '0.1.0 (Phase UI-2)' },
  { label: 'Environment', value: 'Development (mock data)' },
  { label: 'API connection', value: 'Not connected — mock layer active' },
  { label: 'Last data refresh', value: 'Live, every 15–30s per page' },
];

export function SystemInformationPanel() {
  return (
    <dl className="grid grid-cols-1 gap-4 sm:grid-cols-2">
      {SYSTEM_INFO.map((item) => (
        <div key={item.label} className="rounded-lg border border-[#E5E7EB] p-4">
          <dt className="text-caption text-[#374151]">{item.label}</dt>
          <dd className="mt-1 text-small font-medium text-[#111827]">{item.value}</dd>
        </div>
      ))}
    </dl>
  );
}
