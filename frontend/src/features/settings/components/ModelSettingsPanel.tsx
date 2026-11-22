const MODEL_FIELDS = [
  { label: 'Risk model version', value: 'Phase C1-L1 Multi-horizon Ensemble (CatBoost + RF)' },
  { label: 'Training dataset', value: 'Synthetic crowd simulations (Evaluation set)' },
  { label: 'Evaluation metric', value: 'Future escalation recall ~11.8% (Advisory status)' },
  { label: 'Prediction horizons', value: 'CURRENT (live), +5min, +10min' },
  { label: 'Inference target', value: '5.0 FPS / 200ms per frame' },
  { label: 'Movement inputs', value: 'Active tracking & zone velocity (flagged if incomplete)' },
];

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
      <div className="mt-4 rounded-md border border-amber-200 bg-amber-50/70 p-3 text-caption text-amber-800">
        <span className="font-semibold text-amber-900">Model Limitation Disclosure:</span> Models
        are trained exclusively on synthetic data. Future risk forecasts (+5min, +10min) have low
        escalation recall (~11.8%) and must be treated as provisional advisory indicators.
      </div>
    </div>
  );
}
