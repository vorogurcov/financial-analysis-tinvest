import type { ExperimentMeta, RunStatus } from './types'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    ...init,
  })
  if (!response.ok) {
    let detail = response.statusText
    try {
      const body = (await response.json()) as { detail?: string }
      if (body.detail) detail = body.detail
    } catch {
      // keep status text
    }
    throw new Error(detail)
  }
  return (await response.json()) as T
}

export function fetchExperiments(): Promise<ExperimentMeta[]> {
  return request('/api/experiments')
}

export function startRun(body: {
  experiment: string
  params: Record<string, unknown>
  sample?: Record<string, unknown>
}): Promise<{ run_id: string }> {
  return request('/api/runs', {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

export function fetchRun(runId: string): Promise<RunStatus> {
  return request(`/api/runs/${runId}`)
}
