const MODEL_FIELDS = [
  { label: 'Risk model version', value: '—' },
  { label: 'Last trained', value: '—' },
  { label: 'Inference latency target', value: '—' },
  { label: 'Confidence threshold', value: '—' },
];

// Read-only placeholders per spec: backend model settings are not yet
// wired up, so this panel must never break layout or navigation.
export function ModelSettingsPanel() {
  return (
    <div className="rounded-lg border border-dashed border-[#E5E7EB] bg-[#F8FAFC] p-4">
      <dl className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        {MODEL_FIELDS.map((field) => (
          <div key={field.label}>
            <dt className="text-caption text-[#374151]">{field.label}</dt>
            <dd className="mt-0.5 text-small font-medium text-[#111827]">{field.value}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-4 text-caption text-[#374151]">
        Model configuration is read-only until the model service exposes a settings API.
      </p>
    </div>
  );
}
