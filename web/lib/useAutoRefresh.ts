"use client";

import { useState, useEffect, useCallback } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "";

interface UseAutoRefreshOptions<T> {
  path: string;
  fallback: T;
  intervalMs?: number;
  enabled?: boolean;
}

export function useAutoRefresh<T>({ path, fallback, intervalMs = 30000, enabled = true }: UseAutoRefreshOptions<T>) {
  const [data, setData] = useState<T>(fallback);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetchData = useCallback(async (isRefresh = false) => {
    if (isRefresh) setRefreshing(true);
    try {
      const res = await fetch(`${API_BASE}${path}`);
      if (!res.ok) throw new Error(`${res.status}`);
      const json = await res.json();
      setData(json);
      setError(null);
      setLastUpdated(new Date());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [path]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  useEffect(() => {
    if (!enabled) return;
    const interval = setInterval(() => fetchData(true), intervalMs);
    return () => clearInterval(interval);
  }, [fetchData, intervalMs, enabled]);

  const refresh = () => fetchData(true);

  return { data, loading, error, lastUpdated, refreshing, refresh };
}
