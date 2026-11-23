import { useEffect, useMemo, useState } from 'react'
import { useParams } from 'react-router-dom'
import { fetchRun, startRun } from '../api'
import { ChartView } from '../components/ChartView'
import { defaultsFromFields, ParamForm } from '../components/ParamForm'
import { SummaryView } from '../components/SummaryView'
import type { ExperimentMeta, ParamMap, RunStatus } from '../types'

interface ExperimentPageProps {
  experiments: ExperimentMeta[]
  results: Record<string, RunStatus>
  onResult: (experimentId: string, run: RunStatus) => void
}

const STATUS_LABEL: Record<string, string> = {
  queued: 'В очереди',
  preparing_sample: 'Готовим выборку',
  running: 'Считаем',
  done: 'Готово',
  error: 'Ошибка',
}

export function ExperimentPage({ experiments, results, onResult }: ExperimentPageProps) {
  const { experimentId = '' } = useParams()
  const experiment = experiments.find((item) => item.id === experimentId)

  const [params, setParams] = useState<ParamMap>({})
  const [sample, setSample] = useState<ParamMap>({})
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeRunId, setActiveRunId] = useState<string | null>(null)
  const [liveStatus, setLiveStatus] = useState<string | null>(null)

  useEffect(() => {
    if (!experiment) return
    setParams(defaultsFromFields(experiment.fields))
    setSample(defaultsFromFields(experiment.sample_fields))
    setError(null)
    setActiveRunId(null)
    setLiveStatus(null)
  }, [experiment])

  useEffect(() => {
    if (!activeRunId || !experiment) return
    let cancelled = false
    const timer = window.setInterval(async () => {
      try {
        const run = await fetchRun(activeRunId)
        if (cancelled) return
        setLiveStatus(run.status)
        if (run.status === 'done' || run.status === 'error') {
          onResult(experiment.id, run)
          setBusy(false)
          setActiveRunId(null)
          if (run.status === 'error') setError(run.error ?? 'Ошибка запуска')
        }
      } catch (err) {
        if (cancelled) return
        setBusy(false)
        setActiveRunId(null)
        setError(err instanceof Error ? err.message : String(err))
      }
    }, 1000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [activeRunId, experiment, onResult])

  const stored = experiment ? results[experiment.id] : undefined
  const charts = stored?.status === 'done' ? stored.charts : []
  const summary = stored?.status === 'done' ? stored.summary : null
  const message = stored?.status === 'done' ? stored.message : null

  const statusText = useMemo(() => {
    if (liveStatus) return STATUS_LABEL[liveStatus] ?? liveStatus
    if (busy) return 'Запуск…'
    return null
  }, [busy, liveStatus])

  if (!experiments.length) {
    return <p className="muted">Загрузка каталога экспериментов…</p>
  }

  if (!experiment) {
    return <p className="muted">Эксперимент не найден.</p>
  }

  async function handleRun() {
    if (!experiment) return
    const current = experiment
    setError(null)
    setBusy(true)
    setLiveStatus('queued')
    try {
      const response = await startRun({
        experiment: current.id,
        params,
        sample,
      })
      setActiveRunId(response.run_id)
    } catch (err) {
      setBusy(false)
      setLiveStatus(null)
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <div className="experiment-page">
      <header className="page-header">
        <div>
          <h1>{experiment.title}</h1>
          <p className="muted">{experiment.description}</p>
        </div>
        <button type="button" className="primary" disabled={busy} onClick={handleRun}>
          {busy ? 'Считаем…' : 'Запустить'}
        </button>
      </header>

      <section className="panel">
        <h2>Выборка</h2>
        <ParamForm
          fields={experiment.sample_fields}
          values={sample}
          disabled={busy}
          onChange={(name, value) => setSample((prev) => ({ ...prev, [name]: value }))}
        />
      </section>

      <section className="panel">
        <h2>Параметры эксперимента</h2>
        <ParamForm
          fields={experiment.fields}
          values={params}
          disabled={busy}
          onChange={(name, value) => setParams((prev) => ({ ...prev, [name]: value }))}
        />
      </section>

      {statusText && <div className="banner info">{statusText}</div>}
      {error && <div className="banner error">{error}</div>}

      {(summary || message) && (
        <section className="panel">
          <h2>Сводка</h2>
          <SummaryView summary={summary} message={message} />
        </section>
      )}

      {charts.length > 0 && (
        <section className="charts">
          <h2>Графики</h2>
          {charts.map((chart) => (
            <ChartView key={chart.id} chart={chart} />
          ))}
        </section>
      )}
    </div>
  )
}
