/**
 * Safe fetch wrapper that queries the backend with a fallback function.
 * If backend responds with an ok status (200-299), returns parsed JSON.
 * If backend is offline, unreachable, or returns an error, executes and returns fallbackFn().
 */
export async function fetchWithFallback<T>(
  url: string,
  fallbackFn: () => Promise<T> | T,
  init?: RequestInit,
): Promise<T> {
  try {
    const res = await fetch(url, {
      ...init,
      headers: {
        Accept: 'application/json',
        ...init?.headers,
      },
    });
    if (!res.ok) {
      console.warn(`[API] ${url} returned HTTP ${res.status}. Using fallback.`);
      return await fallbackFn();
    }
    return (await res.json()) as T;
  } catch {
    // Backend offline / network unreachable -> fall back seamlessly to mock
    return await fallbackFn();
  }
}
