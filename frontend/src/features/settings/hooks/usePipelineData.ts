import { useQuery } from '@tanstack/react-query';
import { pipelineApi } from '../../../services/api/pipelineApi';

const REFRESH_INTERVAL_MS = 10_000;

export function usePipelineStatus() {
  return useQuery({
    queryKey: ['pipeline', 'status'],
    queryFn: pipelineApi.getStatus,
    refetchInterval: REFRESH_INTERVAL_MS,
  });
}

export function usePipelineWorker() {
  return useQuery({
    queryKey: ['pipeline', 'worker'],
    queryFn: pipelineApi.getWorkerStatus,
    refetchInterval: 3_000,
  });
}
