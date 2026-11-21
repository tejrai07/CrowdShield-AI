import { useQuery } from '@tanstack/react-query';
import { riskApi } from '../../../services/api/riskApi';

const REFRESH_INTERVAL_MS = 20_000;

export function useRiskForecast() {
  return useQuery({
    queryKey: ['risk', 'forecast'],
    queryFn: riskApi.getForecast,
    refetchInterval: REFRESH_INTERVAL_MS,
  });
}

export function useRiskTimeline() {
  return useQuery({
    queryKey: ['risk', 'timeline'],
    queryFn: riskApi.getTimeline,
    refetchInterval: REFRESH_INTERVAL_MS,
  });
}

export function useFeatureContributions() {
  return useQuery({
    queryKey: ['risk', 'feature-contributions'],
    queryFn: riskApi.getFeatureContributions,
    refetchInterval: REFRESH_INTERVAL_MS,
  });
}
