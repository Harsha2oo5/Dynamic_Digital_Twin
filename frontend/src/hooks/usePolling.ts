import { useCallback, useEffect, useRef, useState } from 'react'

interface PollState<T> {
  data: T | null
  error: string | null
  loading: boolean
  refresh: () => void
}

/** Calls `fn` immediately and then every `intervalMs` (0 = once). Keeps the last good data on error. */
export function usePolling<T>(fn: () => Promise<T>, intervalMs: number): PollState<T> {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const fnRef = useRef(fn)
  fnRef.current = fn

  const run = useCallback(async () => {
    try {
      setData(await fnRef.current())
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    run()
    if (!intervalMs) return
    const id = window.setInterval(run, intervalMs)
    return () => window.clearInterval(id)
  }, [run, intervalMs])

  return { data, error, loading, refresh: run }
}
