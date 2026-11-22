import { fetchWithFallback } from './client';

export interface PipelineStatus {
  pipeline_status: string;
  schema_version: string;
  model_data_notice: string;
  frame_id: number;
  timestamp_s?: number | null;
  warnings: string[];
  incomplete_input_zones: string[];
  capabilities: Record<string, unknown>;
}

export interface DiversionRecommendation {
  source_zone: string;
  destination_zone: string | null;
  status: string;
  urgency: string;
  reason: string;
  confidence?: number | null;
  source_inputs_complete: boolean;
}

export interface WorkerStatus {
  is_running: boolean;
  type: string | null;
  frames_processed: number;
  fps: number;
  last_error: string | null;
  video_path?: string | null;
  jsonl_path?: string | null;
}

const DEFAULT_STATUS: PipelineStatus = {
  pipeline_status: 'standby',
  schema_version: '1.0.0',
  model_data_notice:
    'Synthetic Data Limitation: Models trained on synthetic data only. Future escalation recall is low (~11-12%). Live movement-dependent inputs may be provisional.',
  frame_id: 0,
  warnings: ['Synthetic Data Limitation: Models trained on synthetic data only.'],
  incomplete_input_zones: [],
  capabilities: {
    risk_prediction: true,
    diversions: true,
    explanations: true,
  },
};

export const pipelineApi = {
  getStatus: (): Promise<PipelineStatus> =>
    fetchWithFallback('/api/pipeline/status', () => DEFAULT_STATUS),

  getDiversions: (): Promise<DiversionRecommendation[]> =>
    fetchWithFallback('/api/pipeline/diversions', () => []),

  getWorkerStatus: (): Promise<WorkerStatus> =>
    fetchWithFallback('/api/pipeline/worker', () => ({
      is_running: false,
      type: null,
      frames_processed: 0,
      fps: 0,
      last_error: null,
    })),

  startVideo: async (options?: {
    video_path?: string;
    venue_yaml?: string;
    max_frames?: number;
    loop?: boolean;
    fps?: number;
  }) => {
    const res = await fetch('/api/pipeline/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(options ?? {}),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Failed to start video' }));
      throw new Error(err.detail || 'Failed to start video pipeline');
    }
    return res.json();
  },

  startReplay: async (options?: {
    jsonl_path?: string;
    loop?: boolean;
    fps?: number;
  }) => {
    const res = await fetch('/api/pipeline/replay', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(options ?? {}),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Failed to start replay' }));
      throw new Error(err.detail || 'Failed to start replay');
    }
    return res.json();
  },

  stop: async () => {
    const res = await fetch('/api/pipeline/stop', { method: 'POST' });
    if (!res.ok) {
      throw new Error('Failed to stop pipeline');
    }
    return res.json();
  },
};
