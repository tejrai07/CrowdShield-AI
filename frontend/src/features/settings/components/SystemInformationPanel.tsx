import { usePipelineStatus } from '../hooks/usePipelineData';

export function SystemInformationPanel() {
  const { data, isLoading } = usePipelineStatus();

  const systemItems = [
    { label: 'Frontend version', value: '0.1.0 (Phase UI-2)' },
    {
      label: 'API Connection',
      value: isLoading
        ? 'Checking connection...'
        : data
          ? `Connected (Pipeline: ${data.pipeline_status})`
          : 'Offline — fallback active',
    },
    {
      label: 'Schema Version',
      value: data?.schema_version ? `v${data.schema_version} (Pydantic / TS)` : 'v1.0.0',
    },
    {
      label: 'Processed Frames',
      value: data?.frame_id ? `Frame #${data.frame_id}` : '0 (Standby)',
    },
  ];

  return (
    <div className="space-y-4">
      <dl className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        {systemItems.map((item) => (
          <div key={item.label} className="rounded-lg border border-[#E5E7EB] p-4">
            <dt className="text-caption text-[#374151]">{item.label}</dt>
            <dd className="mt-1 text-small font-medium text-[#111827]">{item.value}</dd>
          </div>
        ))}
      </dl>

      {/* Model Transparency & Limitation Notice */}
      <div className="rounded-lg border border-amber-200 bg-amber-50 p-4">
        <h4 className="text-small font-semibold text-amber-900">
          ML Model Transparency & Operational Notice
        </h4>
        <p className="mt-1 text-caption text-amber-800">
          {data?.model_data_notice ||
            'Notice: Risk prediction models are trained on synthetic data. Future escalation recall is low (~11-12%). Live movement-dependent inputs may be flagged provisional if incomplete.'}
        </p>
        {data?.incomplete_input_zones && data.incomplete_input_zones.length > 0 && (
          <div className="mt-2 text-caption text-amber-900">
            <strong>Provisional Zones (Incomplete Movement Inputs):</strong>{' '}
            {data.incomplete_input_zones.join(', ')}
          </div>
        )}
      </div>
    </div>
  );
}
